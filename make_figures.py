"""Result figures from the saved outputs of generate_dataset.py, train_models.py and optimize.py."""
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.8,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white"})
INK, SLATE, BLUE, TEAL, AMBER, GREY = "#1c2430", "#5b6b7c", "#2f6f9f", "#2e8b83", "#c8862c", "#9aa5b1"
MODELS = ["ANN", "SVR", "RF", "XGBoost"]
COL = {"ANN": BLUE, "SVR": TEAL, "RF": AMBER, "XGBoost": "#7a5195"}
SUR = sys.argv[1] if len(sys.argv) > 1 else "ANN"

df = pd.read_csv("gear_power_loss_dataset.csv")
met = json.load(open("results/metrics.json"))


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------- Fig 2: friction formula disagreement
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
ax = axes[0]
for k, c, lab in [("mu2", AMBER, "M2  ISO/TS 6336-20"), ("mu3", BLUE, "M3  ISO/TS 6336-21")]:
    ax.scatter(df.rho_red, df[k] / df.mu1, s=3, alpha=0.35, color=c, label=lab, linewidths=0)
ax.axhline(1.0, color=SLATE, lw=0.8, ls=(0, (3, 2)))
ax.text(df.rho_red.min() * 1.05, 0.965, "M1  ISO/TR 14179-2 (reference)", ha="left", fontsize=6.5, color=SLATE)
ax.set_xscale("log"); ax.set_xlabel("equivalent radius of curvature  ρ_redC  [mm]")
ax.set_ylabel("friction coefficient relative to M1")
ax.legend(frameon=False, fontsize=6.5, markerscale=3, loc="upper right", bbox_to_anchor=(1, 0.93))
ax.set_title("(a) M2 is a fixed 1.25 × M1; M3 falls with size", fontsize=7.5, color=INK)
clean(ax)
ax = axes[1]
sc = ax.scatter(df.rho_red, df.spread, s=3, c=np.log10(df.m), cmap="viridis", alpha=0.6, linewidths=0)
ax.set_xscale("log"); ax.set_xlabel("equivalent radius of curvature  ρ_redC  [mm]")
ax.set_ylabel("spread between formulas  S  [% of mean]")
ax.set_title("(b) Disagreement is smallest for ρ ≈ 7–18 mm", fontsize=7.5, color=INK)
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("log10 module [mm]", fontsize=6.5); cb.ax.tick_params(labelsize=6)
clean(ax)
fig.tight_layout(); fig.savefig("figures/Fig1_friction_disagreement.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: parity plots (random split)
par = pd.read_csv("results/parity_random.csv")
fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0), sharex=True, sharey=True)
lo, hi = 0.08, par.true.max() * 1.1
for ax, mname in zip(axes, MODELS):
    ax.plot([lo, hi], [lo, hi], color=SLATE, lw=0.8, ls=(0, (3, 2)))
    ax.scatter(par.true, par[mname], s=3, color=COL[mname], alpha=0.5, linewidths=0)
    r = met["random"][mname]
    ax.set_title(f"{mname}\nR² = {r['R2']:.4f}, MAPE = {r['MAPE_pct']:.2f}%", fontsize=7, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("true L1 [%]"); clean(ax)
axes[0].set_ylabel("predicted L1 [%]")
fig.tight_layout(); fig.savefig("figures/Fig3_parity.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: interpolation vs extrapolation
fig, ax = plt.subplots(figsize=(7.16, 2.4))
splits = [("random", "random test"), ("ood_speed", "unseen speeds\n3,500–6,000 rpm"),
          ("ood_module", "unseen sizes\nm = 6.5–8 mm")]
w = 0.19
for i, mname in enumerate(MODELS):
    vals = [met[s][mname]["MAPE_pct"] for s, _ in splits]
    xs = np.arange(len(splits)) + (i - 1.5) * w
    bars = ax.bar(xs, vals, width=w * 0.92, color=COL[mname], label=mname)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=6.2, color=INK)
ax.set_xticks(np.arange(len(splits))); ax.set_xticklabels([s[1] for s in splits], fontsize=7)
ax.set_ylabel("MAPE on test set [%]")
ax.legend(frameon=False, ncol=4, fontsize=6.8, loc="upper left")
clean(ax)
fig.savefig("figures/Fig4_extrapolation.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: learning curves
lc = pd.read_csv("results/learning_curve.csv")
g = lc.groupby(["model", "n_train"]).MAPE_pct.agg(["mean", "min", "max"]).reset_index()
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for mname in MODELS:
    s = g[g.model == mname]
    ax.plot(s.n_train, s["mean"], marker="o", ms=3, lw=1.2, color=COL[mname], label=mname)
    ax.fill_between(s.n_train, s["min"], s["max"], color=COL[mname], alpha=0.15, linewidth=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([250, 500, 1000, 2000, 5100]); ax.set_xticklabels(["250", "500", "1,000", "2,000", "5,100"])
ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"])
ax.set_xlabel("training samples"); ax.set_ylabel("test MAPE [%]")
ax.legend(frameon=False, fontsize=6.5); clean(ax)
fig.savefig("figures/Fig5_learning_curve.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 7: Pareto fronts
try:
    nom = pd.read_csv(f"results/pareto_nominal_{SUR}.csv")
    rob = pd.read_csv(f"results/pareto_robust_{SUR}.csv")
    opt = json.load(open(f"results/optimisation_{SUR}.json"))
    base = opt["baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))
    ax = axes[0]
    ax.plot(nom.a, nom.L1, marker="o", ms=2.5, lw=1.0, color=BLUE, label="nominal front (exact L1)")
    ax.plot(rob.a, rob.L1, marker="o", ms=2.5, lw=1.0, color=AMBER, label="robust front (exact L1)")
    ax.scatter([base["a"]], [base["L1"]], marker="D", s=28, color=INK, zorder=5, label="conventional design")
    ax.set_xlabel("centre distance a [mm]"); ax.set_ylabel("mesh loss L1 [% of input power]")
    ax.set_title("(a) Judged by the ISO/TR 14179-2 formula", fontsize=7.5, color=INK)
    ax.legend(frameon=False, fontsize=6.3); clean(ax)
    ax = axes[1]
    ax.plot(nom.a, nom.Lmax, marker="o", ms=2.5, lw=1.0, color=BLUE, label="nominal front (exact worst case)")
    ax.plot(rob.a, rob.Lmax, marker="o", ms=2.5, lw=1.0, color=AMBER, label="robust front (exact worst case)")
    ax.scatter([base["a"]], [base["Lmax"]], marker="D", s=28, color=INK, zorder=5, label="conventional design")
    ax.set_xlabel("centre distance a [mm]"); ax.set_ylabel("worst-case loss L_max [%]")
    ax.set_title("(b) Judged by the worst of the three formulas", fontsize=7.5, color=INK)
    ax.legend(frameon=False, fontsize=6.3); clean(ax)
    fig.tight_layout(); fig.savefig("figures/Fig7_pareto.png"); plt.close(fig)
except FileNotFoundError as e:
    print("pareto figure skipped:", e)
print("figures written")
