"""Verification of the implementation against published reference values."""
import numpy as np
from gear_model import geometry, mesh_loss

# FZG type-C gear (Fernandes et al. 2015, Mech. Sci. 6:81-88): m=4.5, z=16/24, x=0.1817/0.1715,
# a=91.5 mm, da1=82.46 mm, da2=118.36 mm. Published Ohlendorf H_V (C40) = 0.1959.
g_pub = geometry(4.5, 16, 24, 40, 0.1817, 0.1715, da1=82.46, da2=118.36)
g_def = geometry(4.5, 16, 24, 40, 0.1817, 0.1715)
print("FZG C40 check")
print(f"  centre distance a      = {g_pub['a']:.3f} mm   (published 91.5)")
print(f"  working pressure angle = {np.degrees(g_pub['alpha_w']):.3f} deg")
print(f"  eps1, eps2, eps_a      = {g_pub['eps1']:.4f}, {g_pub['eps2']:.4f}, {g_pub['eps_a']:.4f}")
print(f"  H_V (published tips)   = {g_pub['Hv']:.4f}   (published Ohlendorf 0.1959)")
print(f"  H_V (default tips)     = {g_def['Hv']:.4f}")
print(f"  relative difference    = {100*(g_pub['Hv']-0.1959)/0.1959:+.2f} %")

# Typical operating point on the C40 gear: 1500 rpm, 200 N m, mineral oil 40 mPa s, Ra 0.35 um
r = mesh_loss(4.5, 16, 24, 40, 0.1817, 0.1715, T1=200, n1=1500, Ra=0.35, eta=40, XL=1.0)
print("\nC40 at 200 N m, 1500 rpm, eta=40 mPa s, Ra=0.35 um, mineral")
print(f"  vt={float(r['vt']):.2f} m/s  wBt={float(r['wBt']):.1f} N/mm  rho_red={float(r['rho_red']):.2f} mm")
print(f"  mu (M1, M2, M3) = {float(r['mu1']):.4f}, {float(r['mu2']):.4f}, {float(r['mu3']):.4f}")
print(f"  loss ratio %    = {float(r['L1']):.3f}, {float(r['L2']):.3f}, {float(r['L3']):.3f}")
print(f"  mesh efficiency (M1) = {100-float(r['L1']):.3f} %   spread = {float(r['spread']):.1f} %")
print(f"  sigma_H = {float(r['sigH']):.0f} MPa, sigma_F1 = {float(r['sigF1']):.0f} MPa")

# Size dependence of the disagreement (Blech et al. 2023: diverges for larger gears)
print("\nDisagreement vs module (z=20/40, x=0, b=10m, 1500 rpm, wBt held ~ 500 N/mm)")
for mm in [2, 3, 4.5, 6, 8, 12]:
    b = 10 * mm
    db1 = mm * 20 * np.cos(np.radians(20))
    T1 = 500 * b * db1 / 2000.0
    r = mesh_loss(mm, 20, 40, b, 0, 0, T1=T1, n1=1500, Ra=0.4, eta=40, XL=1.0)
    print(f"  m={mm:>4}: rho_red={float(r['rho_red']):6.2f} mm  mu1={float(r['mu1']):.4f} "
          f"mu2={float(r['mu2']):.4f} mu3={float(r['mu3']):.4f}  spread={float(r['spread']):5.1f} %")
