"""Cross-check of the SHAP ranking with a model-agnostic permutation explainer on the ANN surrogate."""
import json
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from features import featurise, FEATURES

SEED = 42
df = pd.read_csv("gear_power_loss_dataset.csv")
X = featurise(df)
bp = json.load(open("results/best_params.json"))["best_params"]["ANN"]
p = {k.split("__")[1]: (tuple(v) if isinstance(v, list) else v) for k, v in bp.items()}
GROUPS = {"tooth geometry (z1, z2, x1, x2)": ["z1", "z2", "x1", "x2"], "size (m, b)": ["m", "b"],
          "operating (T1, n1)": ["logT1", "logn1"], "tribology (Ra, eta, oil)": ["Ra", "logeta", "oil_pao", "oil_pag"]}
out = {}
for tgt in ["L1", "spread"]:
    mdl = make_pipeline(StandardScaler(), MLPRegressor(max_iter=1500, early_stopping=True, n_iter_no_change=30,
                                                       random_state=SEED, **p))
    mdl.fit(X, np.log(df[tgt].values))
    bg = X.sample(100, random_state=1)
    ex = shap.PermutationExplainer(mdl.predict, bg)
    sv = ex(X.sample(400, random_state=SEED)).values
    s = pd.Series(np.abs(sv).mean(0), index=FEATURES)
    share = 100 * s / s.sum()
    out[tgt] = {"share_pct": share.round(2).to_dict(),
                "group_share_pct": {g: round(float(share[f].sum()), 2) for g, f in GROUPS.items()}}
    print(tgt, share.sort_values(ascending=False).round(1).to_dict(), out[tgt]["group_share_pct"])
json.dump(out, open("results/shap_ann_check.json", "w"), indent=2)
