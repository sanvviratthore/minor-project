"""SHAP explanation of the surrogates for the nominal loss L1 and the friction-model spread S.
Uses XGBoost with exact TreeSHAP (Lundberg & Lee 2017)."""
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from features import featurise, FEATURES

SEED = 42
df = pd.read_csv("gear_power_loss_dataset.csv")
X = featurise(df)
bp = json.load(open("results/best_params.json"))["best_params"]["XGBoost"]

NICE = {"m": "module m", "z1": "pinion teeth z1", "z2": "wheel teeth z2", "b": "face width b",
        "x1": "profile shift x1", "x2": "profile shift x2", "logT1": "torque T1 (log)",
        "logn1": "speed n1 (log)", "Ra": "roughness Ra", "logeta": "viscosity eta (log)",
        "oil_pao": "oil = PAO", "oil_pag": "oil = PAG"}
GROUPS = {"tooth geometry (z1, z2, x1, x2)": ["z1", "z2", "x1", "x2"],
          "size (m, b)": ["m", "b"], "operating (T1, n1)": ["logT1", "logn1"],
          "tribology (Ra, eta, oil)": ["Ra", "logeta", "oil_pao", "oil_pag"]}

out = {}
for tgt in ["L1", "spread"]:
    model = XGBRegressor(n_estimators=800, tree_method="hist", random_state=SEED, n_jobs=2, **bp)
    model.fit(X, np.log(df[tgt].values))
    sample = X.sample(2000, random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(sample)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    share = 100 * mean_abs / mean_abs.sum()
    grp = {g: float(share[f].sum()) for g, f in GROUPS.items()}
    out[tgt] = {"mean_abs_shap_log": mean_abs.round(5).to_dict(), "share_pct": share.round(2).to_dict(),
                "group_share_pct": {k: round(v, 2) for k, v in grp.items()}}
    print(f"\n=== {tgt} ===\n", share.round(1).to_string(), "\n", grp)
    plt.figure()
    shap.summary_plot(sv, sample.rename(columns=NICE), show=False, max_display=12, plot_size=(7.16, 4.6))
    ax = plt.gca()
    ax.set_xlabel(f"SHAP value (effect on log {tgt})", fontsize=8)
    ax.tick_params(labelsize=7.5)
    plt.tight_layout()
    plt.savefig(f"figures/Fig6_shap_{tgt}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close("all")

json.dump(out, open("results/shap.json", "w"), indent=2)
