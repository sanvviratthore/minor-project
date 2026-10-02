"""Train and benchmark ANN, SVR, Random Forest and XGBoost surrogates of the mesh loss ratio.
Outputs: results/metrics.json, results/best_params.json, results/*.csv and figures/*.png."""
import json, time, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from scipy.stats import loguniform, randint, uniform
from gear_model import mesh_loss

warnings.filterwarnings("ignore")
SEED = 42
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from features import featurise, FEATURES

df = pd.read_csv("gear_power_loss_dataset.csv")
X_all = featurise(df)


def models():
    return {
        "ANN": (make_pipeline(StandardScaler(), MLPRegressor(max_iter=1500, early_stopping=True,
                                                             n_iter_no_change=30, random_state=SEED)),
                {"mlpregressor__hidden_layer_sizes": [(64, 64), (128, 64), (128, 128), (128, 64, 32)],
                 "mlpregressor__alpha": loguniform(1e-6, 1e-2),
                 "mlpregressor__learning_rate_init": loguniform(3e-4, 3e-3),
                 "mlpregressor__activation": ["relu", "tanh"]}),
        "SVR": (make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                {"svr__C": loguniform(1, 1e3), "svr__gamma": loguniform(1e-3, 1),
                 "svr__epsilon": loguniform(1e-4, 5e-2)}),
        "RF": (RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=SEED),
               {"max_features": uniform(0.3, 0.7), "min_samples_leaf": randint(1, 6),
                "max_depth": [None, 20, 30]}),
        "XGBoost": (XGBRegressor(n_estimators=800, tree_method="hist", random_state=SEED, n_jobs=4),
                    {"max_depth": randint(3, 9), "learning_rate": loguniform(0.02, 0.15),
                     "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                     "min_child_weight": randint(1, 8)}),
    }


def metrics(y_true, y_pred):
    ape = 100 * np.abs(y_pred - y_true) / y_true
    return {"R2": float(r2_score(y_true, y_pred)),
            "RMSE_pp": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE_pct": float(ape.mean()), "P95_APE_pct": float(np.percentile(ape, 95)),
            "max_APE_pct": float(ape.max())}


def fit_predict(est, Xtr, ytr, Xte):
    est.fit(Xtr, np.log(ytr))
    return np.exp(est.predict(Xte))


# ------------------------------------------------------------------ splits
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=SEED)
splits = {
    "random": (tr_idx, te_idx),
    "ood_speed": (idx[df.n1 <= 3500], idx[df.n1 > 3500]),
    "ood_module": (idx[df.m <= 6.5], idx[df.m > 6.5]),
}
for k, (a, b) in splits.items():
    print(f"split {k}: train {len(a)}, test {len(b)}")

# ------------------------------------------------------------------ tuning on random-split training data
target = "L1"
y = df[target].values
best_params, tune_info = {}, {}
for name, (est, space) in models().items():
    t0 = time.time()
    rs = RandomizedSearchCV(est, space, n_iter=12, cv=KFold(3, shuffle=True, random_state=SEED),
                            scoring="neg_mean_squared_error", random_state=SEED, n_jobs=1 if name != "RF" else 1)
    rs.fit(X_all.iloc[tr_idx], np.log(y[tr_idx]))
    best_params[name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in rs.best_params_.items()}
    tune_info[name] = {"cv_rmse_log": float(np.sqrt(-rs.best_score_)), "tune_s": time.time() - t0}
    print(f"{name}: tuned in {time.time()-t0:.0f}s, best {rs.best_params_}")


def tuned(name):
    est, _ = models()[name]
    p = {k: (tuple(v) if isinstance(v, list) else v) for k, v in best_params[name].items()}
    est.set_params(**p)
    return est


json.dump({"best_params": best_params, "tuning": tune_info}, open("results/best_params.json", "w"),
          indent=2, default=float)

# ------------------------------------------------------------------ benchmark on all splits
results, preds = {}, {}
for split, (a, b) in splits.items():
    results[split] = {}
    for name in models():
        est = tuned(name)
        t0 = time.time()
        p = fit_predict(est, X_all.iloc[a], y[a], X_all.iloc[b])
        fit_s = time.time() - t0
        m_ = metrics(y[b], p)
        m_["fit_s"] = fit_s
        results[split][name] = m_
        if split == "random":
            preds[name] = p
            # inference timing on 10,000 rows
            Xbig = pd.concat([X_all] * 2).iloc[:10000]
            t0 = time.time(); est.predict(Xbig); m_["infer_us_per_row"] = 1e6 * (time.time() - t0) / 10000
        print(f"[{split}] {name}: " + ", ".join(f"{k}={v:.4g}" for k, v in m_.items()))

# analytical model timing for comparison
d = pd.concat([df] * 2).iloc[:10000]
t0 = time.time()
mesh_loss(d.m.values, d.z1.values, d.z2.values, d.b.values, d.x1.values, d.x2.values, d.T1.values,
          d.n1.values, d.Ra.values, d.eta.values, d.XL.values)
results["analytical_us_per_row"] = 1e6 * (time.time() - t0) / 10000
print("analytical model us/row:", results["analytical_us_per_row"])

pd.DataFrame({"true": y[te_idx], **{k: v for k, v in preds.items()}}).to_csv("results/parity_random.csv", index=False)

# ------------------------------------------------------------------ extra targets: worst case and spread
extra = {}
for tgt in ["Lmax", "spread"]:
    yt = df[tgt].values
    extra[tgt] = {}
    for name in ["ANN", "XGBoost"]:
        p = fit_predict(tuned(name), X_all.iloc[tr_idx], yt[tr_idx], X_all.iloc[te_idx])
        extra[tgt][name] = metrics(yt[te_idx], p)
        print(f"[target {tgt}] {name}: " + ", ".join(f"{k}={v:.4g}" for k, v in extra[tgt][name].items()))
results["extra_targets"] = extra

# ------------------------------------------------------------------ learning curves (random split)
sizes = [250, 500, 1000, 2000, len(tr_idx)]
lc = []
for name in models():
    for n in sizes:
        for rep in range(3 if n < len(tr_idx) else 1):
            rng = np.random.default_rng(SEED + rep)
            sub = rng.choice(tr_idx, size=n, replace=False)
            p = fit_predict(tuned(name), X_all.iloc[sub], y[sub], X_all.iloc[te_idx])
            lc.append({"model": name, "n_train": n, "rep": rep, **metrics(y[te_idx], p)})
    print("learning curve done:", name)
pd.DataFrame(lc).to_csv("results/learning_curve.csv", index=False)

json.dump(results, open("results/metrics.json", "w"), indent=2)
print("done")
