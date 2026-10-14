#!/usr/bin/env python3
"""fig7_selfdebias_scatter.pdf: no-fix vs self-debias committed gap, per cell.

Color = model, marker shape = topology, dashed identity line. No in-figure
text annotations; the caption carries the reading.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

COLORS = {"Qwen3": "#0072B2", "Mistral": "#D55E00", "MedGemma": "#009E73",
          "Phi-4": "#CC79A7", "Llama": "#E69F00"}
MARKS = {"SharedPool": "o", "Decentralized": "s", "Layers": "^",
         "Centralized": "D"}

DATA = {  # model -> topo -> (agg, selfdebias)
    "Qwen3": {"SharedPool": (-0.047, -0.050), "Decentralized": (-0.058, -0.044),
              "Layers": (-0.064, -0.083), "Centralized": (-0.031, -0.067)},
    "Mistral": {"SharedPool": (-0.069, -0.050), "Decentralized": (-0.019, -0.064),
                "Layers": (-0.053, -0.083), "Centralized": (-0.025, -0.061)},
    "MedGemma": {"SharedPool": (-0.053, -0.047), "Decentralized": (-0.058, -0.078),
                 "Layers": (-0.039, -0.078), "Centralized": (-0.039, -0.092)},
    "Phi-4": {"SharedPool": (-0.025, -0.078), "Decentralized": (-0.025, -0.075),
              "Layers": (-0.039, -0.086), "Centralized": (-0.042, -0.072)},
    "Llama": {"SharedPool": (-0.039, -0.075), "Decentralized": (-0.044, -0.078),
              "Layers": (-0.061, -0.072), "Centralized": (-0.050, -0.078)},
}

fig, ax = plt.subplots(figsize=(3.15, 2.9))
lo, hi = -0.10, 0.0
ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls=(0, (4, 2)), zorder=1)

for model, topos in DATA.items():
    for topo, (x, y) in topos.items():
        ax.plot([x], [y], marker=MARKS[topo], ms=4.6, mfc=COLORS[model],
                mec="white", mew=0.6, lw=0, zorder=3)

ax.set_xlim(lo, hi + 0.002)
ax.set_ylim(lo, hi + 0.002)
ax.set_aspect("equal")
ax.tick_params(labelsize=7)
ax.set_xlabel("Committed gap, no intervention", fontsize=7.5)
ax.set_ylabel("Committed gap, self-debias", fontsize=7.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

mh = [Line2D([], [], marker="o", ms=4.6, mfc=c, mec="white", mew=0.6, lw=0,
             label=m) for m, c in COLORS.items()]
th = [Line2D([], [], marker=k, ms=4.2, mfc="0.45", mec="white", mew=0.6,
             lw=0, label=t) for t, k in MARKS.items()]
leg1 = ax.legend(handles=mh, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                 ncol=3, frameon=False, fontsize=6.6, handlelength=1.2,
                 columnspacing=0.9, handletextpad=0.4)
ax.add_artist(leg1)
ax.legend(handles=th, loc="upper left", bbox_to_anchor=(0.0, 0.99),
          frameon=False, fontsize=6.3, handlelength=1.2, labelspacing=0.3,
          handletextpad=0.4)

fig.subplots_adjust(left=0.185, right=0.975, top=0.86, bottom=0.135)
fig.savefig("fig7_selfdebias_scatter.pdf")
print("wrote fig7_selfdebias_scatter.pdf")
