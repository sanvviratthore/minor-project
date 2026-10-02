"""Generate the synthetic dataset by Latin hypercube sampling of design and operating inputs,
evaluating the verified loss model, and keeping only points inside the model's validity domain."""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from gear_model import mesh_loss, validity_mask, XL_TABLE, OILS

SEED = 42
N_RAW = 120_000
N_KEEP = 6000

RANGES = {           # name: (low, high, scale)
    "m":   (1.5, 8.0, "lin"),       # module [mm]
    "z1":  (17, 45, "int"),         # pinion teeth
    "u":   (1.0, 4.0, "lin"),       # nominal ratio (z2 = round(u*z1))
    "bm":  (6.0, 20.0, "lin"),      # face width / module
    "x1":  (-0.2, 0.8, "lin"),      # profile shift pinion
    "x2":  (-0.2, 0.8, "lin"),      # profile shift wheel
    "T1":  (20.0, 3000.0, "log"),   # pinion torque [N m]
    "n1":  (300.0, 6000.0, "log"),  # pinion speed [rpm]
    "Ra":  (0.1, 1.2, "lin"),       # mean arithmetic roughness [um]
    "eta": (5.0, 400.0, "log"),     # dynamic oil viscosity at operating temp [mPa s]
    "oil": (0, 3, "cat"),           # 0 mineral, 1 PAO, 2 PAG
}


def sample(n, seed):
    s = qmc.LatinHypercube(d=len(RANGES), seed=seed).random(n)
    cols = {}
    for j, (k, (lo, hi, sc)) in enumerate(RANGES.items()):
        v = s[:, j]
        if sc == "lin":
            cols[k] = lo + v * (hi - lo)
        elif sc == "log":
            cols[k] = np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))
        elif sc == "int":
            cols[k] = np.floor(lo + v * (hi - lo + 1)).clip(lo, hi)
        elif sc == "cat":
            cols[k] = np.floor(v * hi).clip(0, hi - 1).astype(int)
    df = pd.DataFrame(cols)
    df["z2"] = np.maximum(np.round(df["u"] * df["z1"]), df["z1"])
    df["b"] = df["bm"] * df["m"]
    df["oil_type"] = [OILS[i] for i in df["oil"]]
    df["XL"] = df["oil_type"].map(XL_TABLE)
    return df


def evaluate(df):
    r = mesh_loss(df.m, df.z1, df.z2, df.b, df.x1, df.x2, df.T1, df.n1, df.Ra, df.eta, df.XL)
    keep = validity_mask(r, df.m.values) & (r["sigH"] <= 1800)
    out = df.copy()
    for k in ["a", "u", "Hv", "eps_a", "rho_red", "vt", "wBt", "mu1", "mu2", "mu3",
              "L1", "L2", "L3", "Lmean", "Lmax", "spread", "P_in", "sigH", "sigF1", "sigF2"]:
        out[k] = np.asarray(r[k])
    out["u"] = out["z2"] / out["z1"]
    return out[np.asarray(keep)]


if __name__ == "__main__":
    raw = sample(N_RAW, SEED)
    valid = evaluate(raw)
    print(f"raw samples: {len(raw)}, valid: {len(valid)} ({100*len(valid)/len(raw):.1f} %)")
    data = valid.sample(n=min(N_KEEP, len(valid)), random_state=SEED).reset_index(drop=True)
    cols = ["m", "z1", "z2", "u", "b", "x1", "x2", "T1", "n1", "Ra", "eta", "oil_type", "XL",
            "a", "Hv", "eps_a", "rho_red", "vt", "wBt", "mu1", "mu2", "mu3",
            "L1", "L2", "L3", "Lmean", "Lmax", "spread", "P_in", "sigH", "sigF1", "sigF2"]
    data[cols].to_csv("gear_power_loss_dataset.csv", index=False, float_format="%.6g")
    print(data[["L1", "Lmean", "Lmax", "spread", "Hv", "mu1", "vt", "wBt"]].describe().round(4).to_string())
    print(data["oil_type"].value_counts().to_string())
