"""Constraint-aware two-objective design optimisation of a spur gear pair (NSGA-II, pymoo).
Objective 1: mesh loss ratio from the ML surrogate (nominal L1, or worst-case Lmax for the robust run).
Objective 2: centre distance a (exact). Constraints are evaluated with the exact analytical model.
Every Pareto design is re-evaluated with the exact model afterwards."""
import json, sys
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from gear_model import mesh_loss, geometry
from features import featurise

SEED = 42
SURROGATE = sys.argv[1] if len(sys.argv) > 1 else "ANN"

# ------------------------------------------------------------------ design requirement (case study)
REQ = dict(T1=250.0, n1=1500.0, u_target=2.5, eta=40.0, Ra=0.4, oil_type="mineral", XL=1.0)
LIMITS = dict(sigH=1250.0, sigF=350.0, eps_min=1.2, ratio_tol=0.03)
# bounds: m [mm], z1, x1, x2, b/m
XL_B = np.array([1.5, 17, -0.2, -0.2, 6.0])
XU_B = np.array([8.0, 45, 0.8, 0.8, 20.0])

df = pd.read_csv("gear_power_loss_dataset.csv")
bp = json.load(open("results/best_params.json"))["best_params"][SURROGATE]


def make_model():
    if SURROGATE == "ANN":
        p = {k.split("__")[1]: (tuple(v) if isinstance(v, list) else v) for k, v in bp.items()}
        return make_pipeline(StandardScaler(), MLPRegressor(max_iter=1500, early_stopping=True,
                                                            n_iter_no_change=30, random_state=SEED, **p))
    return XGBRegressor(n_estimators=800, tree_method="hist", random_state=SEED, n_jobs=2, **bp)


X = featurise(df)
sur = {}
for tgt in ["L1", "Lmax"]:
    mdl = make_model()
    mdl.fit(X, np.log(df[tgt].values))
    sur[tgt] = mdl


def design(v):
    m, z1, x1, x2, bm = v
    z1 = int(round(z1))
    z2 = int(round(REQ["u_target"] * z1))
    return m, z1, z2, x1, x2, bm * m


def exact(m, z1, z2, x1, x2, b):
    return mesh_loss(m, z1, z2, b, x1, x2, REQ["T1"], REQ["n1"], REQ["Ra"], REQ["eta"], REQ["XL"])


def constraints(r, m, z1, z2, x1, x2):
    zmin = 2 / np.sin(np.radians(20)) ** 2
    g_a1 = np.sqrt(max((r["da1"] / 2) ** 2 - (r["db1"] / 2) ** 2, 0))
    g_a2 = np.sqrt(max((r["da2"] / 2) ** 2 - (r["db2"] / 2) ** 2, 0))
    law = r["a"] * np.sin(r["alpha_w"])
    u = z2 / z1
    return [float(v) for v in (
        r["sigH"] - LIMITS["sigH"], r["sigF1"] - LIMITS["sigF"], r["sigF2"] - LIMITS["sigF"],
        LIMITS["eps_min"] - r["eps_a"], r["eps_a"] - 1.98, r["eps1"] - 0.99, r["eps2"] - 0.99,
        0.25 * m - r["sa1"], 0.25 * m - r["sa2"],
        (zmin - z1) / zmin - x1, (zmin - z2) / zmin - x2,
        g_a1 - law, g_a2 - law, 150 - r["wBt"], 1 - r["vt"],
        abs(u - REQ["u_target"]) / REQ["u_target"] - LIMITS["ratio_tol"])]


def predict(tgt, m, z1, z2, x1, x2, b):
    row = pd.DataFrame([dict(m=m, z1=z1, z2=z2, b=b, x1=x1, x2=x2, T1=REQ["T1"], n1=REQ["n1"],
                             Ra=REQ["Ra"], eta=REQ["eta"], oil_type=REQ["oil_type"])])
    return float(np.exp(sur[tgt].predict(featurise(row))[0]))


class GearProblem(ElementwiseProblem):
    def __init__(self, tgt):
        self.tgt = tgt
        super().__init__(n_var=5, n_obj=2, n_ieq_constr=16, xl=XL_B, xu=XU_B)

    def _evaluate(self, v, out, *a, **k):
        m, z1, z2, x1, x2, b = design(v)
        r = exact(m, z1, z2, x1, x2, b)
        out["F"] = [predict(self.tgt, m, z1, z2, x1, x2, b), float(r["a"])]
        out["G"] = constraints(r, m, z1, z2, x1, x2)


def run(tgt):
    res = minimize(GearProblem(tgt), NSGA2(pop_size=100), ("n_gen", 150), seed=SEED, verbose=False)
    rows = []
    for v in res.X:
        m, z1, z2, x1, x2, b = design(v)
        r = exact(m, z1, z2, x1, x2, b)
        rows.append(dict(m=m, z1=z1, z2=z2, x1=x1, x2=x2, b=b, a=float(r["a"]),
                         L_pred=predict(tgt, m, z1, z2, x1, x2, b),
                         L1=float(r["L1"]), L2=float(r["L2"]), L3=float(r["L3"]), Lmax=float(r["Lmax"]),
                         Hv=float(r["Hv"]), eps_a=float(r["eps_a"]), mu1=float(r["mu1"]),
                         sigH=float(r["sigH"]), sigF1=float(r["sigF1"]), sigF2=float(r["sigF2"]),
                         P_loss_W=float(r["P_loss1"])))
    out = pd.DataFrame(rows).sort_values("a").drop_duplicates(subset=["a", "L1"]).reset_index(drop=True)
    exact_col = "L1" if tgt == "L1" else "Lmax"
    out["surrogate_err_pct"] = 100 * (out.L_pred - out[exact_col]) / out[exact_col]
    return out


def baseline():
    """Conventional design: z1=20, z2=50, x1=x2=0, b/m=10, smallest standard module that passes."""
    for m in [1.5, 2, 2.5, 3, 4, 5, 6, 8]:
        z1, z2, b = 20, 50, 10 * m
        r = exact(m, z1, z2, 0.0, 0.0, b)
        if max(constraints(r, m, z1, z2, 0.0, 0.0)) <= 0:
            return dict(m=m, z1=z1, z2=z2, x1=0.0, x2=0.0, b=b, a=float(r["a"]), L1=float(r["L1"]),
                        L2=float(r["L2"]), L3=float(r["L3"]), Lmax=float(r["Lmax"]), Hv=float(r["Hv"]),
                        sigH=float(r["sigH"]), sigF1=float(r["sigF1"]), P_loss_W=float(r["P_loss1"]))
    return None


if __name__ == "__main__":
    base = baseline()
    print("baseline:", base)
    nom = run("L1"); nom.to_csv(f"results/pareto_nominal_{SURROGATE}.csv", index=False)
    rob = run("Lmax"); rob.to_csv(f"results/pareto_robust_{SURROGATE}.csv", index=False)
    summary = {"requirement": REQ, "limits": LIMITS, "baseline": base}
    for name, fr in [("nominal", nom), ("robust", rob)]:
        same = fr[fr.a <= base["a"] + 1e-9]
        best = same.loc[same.L1.idxmin() if name == "nominal" else same.Lmax.idxmin()]
        summary[name] = {"n_pareto": len(fr),
                         "surrogate_abs_err_pct_mean": float(fr.surrogate_err_pct.abs().mean()),
                         "surrogate_abs_err_pct_max": float(fr.surrogate_err_pct.abs().max()),
                         "best_at_or_below_baseline_size": best.to_dict()}
        print(f"\n{name}: {len(fr)} Pareto designs; surrogate |err| mean "
              f"{fr.surrogate_err_pct.abs().mean():.2f}% max {fr.surrogate_err_pct.abs().max():.2f}%")
        print(best.round(4).to_string())
    json.dump(summary, open(f"results/optimisation_{SURROGATE}.json", "w"), indent=2, default=float)
