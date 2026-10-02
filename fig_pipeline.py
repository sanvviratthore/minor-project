"""Figure: methodology pipeline (predict - explain - optimise)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})
INK, SLATE, BLUE, TEAL, AMBER = "#1c2430", "#5b6b7c", "#2f6f9f", "#2e8b83", "#c8862c"


def box(ax, x, y, w, h, title, body, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                linewidth=1.1, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=7.4, weight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 0.115, body, ha="center", va="top", fontsize=6.3, color=INK, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, c=SLATE, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0,
                                 color=c, linestyle=ls, zorder=4, shrinkA=0, shrinkB=1))


fig, ax = plt.subplots(figsize=(7.16, 3.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
W, H = 0.166, 0.36
ys = 0.56
xs = [0.012, 0.21, 0.408, 0.606, 0.804]
box(ax, xs[0], ys, W, H, "1  Physics model", "Ohlendorf H_V\n+ 3 friction formulas\n(ISO/TR 14179-2,\nISO/TS 6336-20, -21)", BLUE, "#eef3f7")
box(ax, xs[1], ys, W, H, "2  Dataset", "Latin hypercube,\n11 inputs, validity\nfiltering -> 6,000\nsamples; L1, Lmax, S", BLUE, "#eef3f7")
box(ax, xs[2], ys, W, H, "3  Surrogates", "ANN, SVR, RF,\nXGBoost; random +\nextrapolation\n(speed, module) tests", TEAL, "#e9f2f0")
box(ax, xs[3], ys, W, H, "4  Explain", "TreeSHAP on loss\nand on formula\nspread; groups of\ninputs ranked", TEAL, "#e9f2f0")
box(ax, xs[4], ys, W, H, "5  Optimise", "NSGA-II: loss vs\ncentre distance;\nnominal and robust\n(worst-case) runs", AMBER, "#faf1e2")
for i in range(4):
    arrow(ax, xs[i] + W, ys + H / 2, xs[i + 1], ys + H / 2)
# verification and re-evaluation loops
ax.add_patch(FancyBboxPatch((0.21, 0.08), 0.772, 0.24, boxstyle="round,pad=0.006,rounding_size=0.015",
                            linewidth=1.0, edgecolor=SLATE, facecolor="#f6f7f9", zorder=1))
ax.text(0.596, 0.27, "Checks that keep the surrogate honest", ha="center", fontsize=7.2, weight="bold", color=INK)
ax.text(0.596, 0.205, "H_V verified on FZG C40 (0.1950 vs 0.1959)   |   tests on unseen speeds and sizes\n"
        "every Pareto design re-evaluated with the exact model before it is reported",
        ha="center", va="top", fontsize=6.3, color=INK, linespacing=1.45)
arrow(ax, xs[0] + W / 2, ys, 0.21, 0.2, c=SLATE, ls=(0, (3, 2)))
arrow(ax, xs[4] + W / 2, ys, xs[4] + W / 2, 0.32, c=AMBER, ls=(0, (3, 2)))
fig.savefig("figures/Fig2_pipeline.png", bbox_inches="tight", facecolor="white")
print("ok")
