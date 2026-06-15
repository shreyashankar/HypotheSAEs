"""Write standalone PNGs of the key report plots, for embedding elsewhere
(e.g. Notion) via public URLs. Reuses the report's data + figure code."""
import os, sys, base64, importlib, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
m07 = importlib.import_module("agentic_interp.07_make_report")
COLORS = m07.COLORS

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)


def save_b64(b64, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    print("wrote", path)


def scatter_png(data, n_test, title, path):
    """Cost (log, cheaper to the right) vs F1; dot size = how many docs fire."""
    pts = []
    for j, d in data.items():
        for mkey in ("baseline", "gepa", "agent"):
            if mkey in d:
                _, cost = m07.method_cost(d, mkey, n_test)
                if cost > 0:
                    pts.append((cost, d[mkey]["f1"], mkey, d["n_pos_corpus"]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    npos = [p[3] for p in pts]
    lo, hi = math.log10(min(npos)), math.log10(max(npos) + 1)
    for cost, f1, mkey, n in sorted(pts, key=lambda t: -t[3]):
        r = 40 + 320 * (math.log10(n) - lo) / (hi - lo if hi > lo else 1)
        ax.scatter(cost, f1, s=r, c=COLORS[mkey], alpha=0.7, edgecolors="white", linewidths=0.6)
    for mkey, lbl in [("baseline", "baseline"), ("gepa", "GEPA"), ("agent", "agent")]:
        ax.scatter([], [], c=COLORS[mkey], label=lbl, s=80)
    ax.set_xscale("log")
    ax.invert_xaxis()  # cheaper to the right
    ax.set_xlabel("cost per feature (US dollars, log scale; cheaper to the right)")
    ax.set_ylabel("official-set F1")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, -0.18, "dot size = number of documents the feature fires on",
            transform=ax.transAxes, ha="right", fontsize=8, color="#777")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


for name, art, n_test, _ in m07.DATASETS:
    if not os.path.isdir(art):
        continue
    data = m07.collect(art, n_test)
    save_b64(m07.bars_fig(data, f"{name}: mean official-set F1 by method"),
             os.path.join(FIG, f"{name.lower()}_f1_bars.png"))
    scatter_png(data, n_test, f"{name}: cost vs F1 (one dot per feature per method)",
                os.path.join(FIG, f"{name.lower()}_cost_vs_f1.png"))
    tb64 = m07.trajectory_fig(data, name)
    if tb64:
        save_b64(tb64, os.path.join(FIG, f"{name.lower()}_trajectories.png"))
