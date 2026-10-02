"""
Load-dependent mesh power loss of an external spur gear pair.

Model (ISO-type, as used in ISO/TR 14179-2 and reviewed by Blech et al. 2023):
    P_VZP = P_in * mu_mZ * H_V
    H_V   = pi*(u+1)/(z1*u*cos(beta_b)) * (1 - eps_a + eps_1^2 + eps_2^2)      (Ohlendorf)
Mean coefficient of friction, three published variants (spur gears: K_Bgamma = 1, beta_b = 0):
    M1 ISO/TR 14179-2 (Schlenk):  0.048 * (w_Bt/(v_SumC*rho_redC))^0.2 * eta^-0.05 * X_L * Ra^0.25
    M2 ISO/TS 6336-20:            0.060 * (w_Bt/(v_SumC*rho_redC))^0.2 * X_L * eta^-0.05 * Ra^0.25
    M3 ISO/TS 6336-21 (main):     0.045 * (w_Bt/(v_SumC*rho_redC))^0.2 * eta^-0.05 * X_L * 2.2*(Ra/rho_redC)^0.25
Units: w_Bt [N/mm], v_SumC [m/s], rho_redC [mm], eta [mPa s], Ra [um] (mean of both flanks).
Lubricant factor X_L (Lohner & Paschold 2025): mineral 1.0, PAO 0.8, polyglycol 0.7.

Screening strength checks (simplified, K-factors = 1):
    contact stress  sigma_H = Z_E*Z_H*Z_eps*sqrt(F_t/(d1*b)*(u+1)/u)   (ISO 6336-2 structure)
    bending stress  sigma_F = F_t/(b*m*Y)                                 (Lewis, Y from Shigley table)
All functions are vectorised over numpy arrays.
"""
import numpy as np

ALPHA_N = np.radians(20.0)          # standard pressure angle
Z_E = 189.8                         # elasticity factor, steel/steel [sqrt(N/mm^2)]
XL_TABLE = {"mineral": 1.0, "pao": 0.8, "pag": 0.7}
OILS = ["mineral", "pao", "pag"]

# Lewis form factor Y for 20 deg full-depth teeth (Shigley, Table 14-2)
_LEWIS_Z = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 34, 38,
                     43, 50, 60, 75, 100, 150, 300, 400])
_LEWIS_Y = np.array([0.245, 0.261, 0.277, 0.290, 0.296, 0.303, 0.309, 0.314, 0.322, 0.328,
                     0.331, 0.337, 0.346, 0.353, 0.359, 0.371, 0.384, 0.397, 0.409, 0.422,
                     0.435, 0.447, 0.460, 0.472, 0.480])


def lewis_y(z):
    return np.interp(z, _LEWIS_Z, _LEWIS_Y)


def inv(a):
    return np.tan(a) - a


def inv_inverse(y):
    """Solve tan(a) - a = y for a in (0, pi/2) by Newton iteration (vectorised)."""
    y = np.asarray(y, dtype=float)
    a = np.cbrt(3.0 * y)                       # good start for small angles
    for _ in range(60):
        f = np.tan(a) - a - y
        a = a - f / np.tan(a) ** 2
    return a


def geometry(m, z1, z2, b, x1, x2, da1=None, da2=None, alpha=ALPHA_N):
    """Geometry of a profile-shifted spur pair. Lengths in mm.
    Tip diameters default to d + 2m(1+x) (no tip shortening); pass da1/da2 to override."""
    m, z1, z2, b, x1, x2 = map(lambda v: np.asarray(v, dtype=float), (m, z1, z2, b, x1, x2))
    d1, d2 = m * z1, m * z2
    db1, db2 = d1 * np.cos(alpha), d2 * np.cos(alpha)
    aw = inv_inverse(inv(alpha) + 2.0 * np.tan(alpha) * (x1 + x2) / (z1 + z2))
    a = 0.5 * (d1 + d2) * np.cos(alpha) / np.cos(aw)
    dw1, dw2 = 2 * a * z1 / (z1 + z2), 2 * a * z2 / (z1 + z2)
    if da1 is None:
        da1 = d1 + 2 * m * (1 + x1)
    if da2 is None:
        da2 = d2 + 2 * m * (1 + x2)
    aa1, aa2 = np.arccos(db1 / da1), np.arccos(db2 / da2)
    e1 = z1 / (2 * np.pi) * (np.tan(aa1) - np.tan(aw))
    e2 = z2 / (2 * np.pi) * (np.tan(aa2) - np.tan(aw))
    ea = e1 + e2
    u = z2 / z1
    Hv = np.pi * (u + 1) / (z1 * u) * (1 - ea + e1 ** 2 + e2 ** 2)
    rho1, rho2 = 0.5 * dw1 * np.sin(aw), 0.5 * dw2 * np.sin(aw)
    rho_red = rho1 * rho2 / (rho1 + rho2)
    # transverse tooth thickness at the tip circle
    sa1 = da1 * ((np.pi / 2 + 2 * x1 * np.tan(alpha)) / z1 + inv(alpha) - inv(aa1))
    sa2 = da2 * ((np.pi / 2 + 2 * x2 * np.tan(alpha)) / z2 + inv(alpha) - inv(aa2))
    # interference check: tip contact must not pass the interference point of the mate
    len_aw = a * np.sin(aw)
    g_a1 = np.sqrt(np.maximum((da1 / 2) ** 2 - (db1 / 2) ** 2, 0))
    g_a2 = np.sqrt(np.maximum((da2 / 2) ** 2 - (db2 / 2) ** 2, 0))
    no_interference = (g_a1 <= len_aw) & (g_a2 <= len_aw)
    zmin = 2.0 / np.sin(alpha) ** 2
    no_undercut = (x1 >= (zmin - z1) / zmin) & (x2 >= (zmin - z2) / zmin)
    return dict(d1=d1, d2=d2, db1=db1, db2=db2, a=a, alpha_w=aw, dw1=dw1, dw2=dw2,
                da1=da1, da2=da2, eps1=e1, eps2=e2, eps_a=ea, u=u, Hv=Hv,
                rho_red=rho_red, sa1=sa1, sa2=sa2, no_interference=no_interference,
                no_undercut=no_undercut)


def mesh_loss(m, z1, z2, b, x1, x2, T1, n1, Ra, eta, XL, alpha=ALPHA_N):
    """Return geometry, kinematics, friction (3 models) and loss ratios (%) for given inputs.
    T1 pinion torque [N m], n1 pinion speed [rpm], Ra [um], eta dynamic viscosity [mPa s]."""
    g = geometry(m, z1, z2, b, x1, x2, alpha=alpha)
    T1, n1, Ra, eta, XL, b = map(lambda v: np.asarray(v, dtype=float), (T1, n1, Ra, eta, XL, b))
    vt = np.pi * g["dw1"] * n1 / 60000.0                 # pitch-line velocity [m/s]
    v_sum = 2.0 * vt * np.sin(g["alpha_w"])              # sum velocity at pitch point [m/s]
    Fbt = 2000.0 * T1 / g["db1"]                         # force in plane of action [N]
    wBt = Fbt / b                                        # [N/mm]
    core = (wBt / (v_sum * g["rho_red"])) ** 0.2
    mu1 = 0.048 * core * eta ** -0.05 * XL * Ra ** 0.25
    mu2 = 0.060 * core * XL * eta ** -0.05 * Ra ** 0.25
    mu3 = 0.045 * core * eta ** -0.05 * XL * 2.2 * (Ra / g["rho_red"]) ** 0.25
    L1, L2, L3 = (100 * mu * g["Hv"] for mu in (mu1, mu2, mu3))   # loss ratio [% of input power]
    Ls = np.stack([L1, L2, L3])
    Lmean, Lmax, Lmin = Ls.mean(0), Ls.max(0), Ls.min(0)
    spread = 100 * (Lmax - Lmin) / Lmean                 # [% of mean]
    P_in = T1 * 2 * np.pi * n1 / 60.0                    # [W]
    # screening stresses (K factors = 1)
    Ft = 2000.0 * T1 / g["d1"]                           # nominal tangential force at d1 [N]
    ZH = np.sqrt(2 * np.cos(g["alpha_w"]) / (np.cos(alpha) ** 2 * np.sin(g["alpha_w"])))
    Zeps = np.sqrt((4 - g["eps_a"]) / 3)
    sigH = Z_E * ZH * Zeps * np.sqrt(Ft / (g["d1"] * b) * (g["u"] + 1) / g["u"])
    sigF1 = Ft / (b * m * lewis_y(z1))
    sigF2 = Ft / (b * m * lewis_y(z2))
    out = dict(g)
    out.update(vt=vt, v_sum=v_sum, Fbt=Fbt, wBt=wBt, mu1=mu1, mu2=mu2, mu3=mu3,
               L1=L1, L2=L2, L3=L3, Lmean=Lmean, Lmax=Lmax, spread=spread,
               P_in=P_in, P_loss1=P_in * L1 / 100, sigH=sigH, sigF1=sigF1, sigF2=sigF2)
    return out


def validity_mask(r, m):
    """Domain in which the H_V formula and the friction formulas are valid,
    plus basic manufacturability of the tooth form."""
    return ((r["eps1"] > 0) & (r["eps1"] < 1) & (r["eps2"] > 0) & (r["eps2"] < 1)
            & (r["eps_a"] >= 1.1) & (r["eps_a"] < 2)
            & (r["sa1"] >= 0.25 * m) & (r["sa2"] >= 0.25 * m)
            & r["no_undercut"] & r["no_interference"]
            & (r["vt"] >= 1) & (r["vt"] <= 50) & (r["wBt"] >= 150))
