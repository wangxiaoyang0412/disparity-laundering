#!/usr/bin/env python3
"""fig2_intervention_heatmap.pdf: interventions leave the gap in place.

Three aligned heatmap panels, rows = models, cols = topologies:
  (1) committed gap, no intervention   -> saturated (bias exists)
  (2) change under fairness instruction -> near-white (nothing moves)
  (3) change under demographic masking  -> near-white (nothing moves)
Shared diverging scale centered at 0; the blank intervention panels ARE the
message. Data = the agg/+fairness/+masking columns of tab:grid (all four topologies).
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 8,
    "pdf.fonttype": 42,
})

MODELS = ["Qwen3", "Mistral", "MedGemma", "Phi-4", "Llama"]
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]

AGG = np.array([
    [-0.047, -0.058, -0.064, -0.031],
    [-0.069, -0.019, -0.053, -0.025],
    [-0.053, -0.058, -0.039, -0.039],
    [-0.025, -0.025, -0.039, -0.042],
    [-0.039, -0.044, -0.061, -0.050],
])
FAIR = np.array([
    [-0.047, -0.058, -0.050, -0.044],
    [-0.072, -0.034, -0.072, -0.033],
    [-0.047, -0.053, -0.036, -0.031],
    [-0.014, -0.025, -0.058, -0.036],
    [-0.039, -0.044, -0.064, -0.050],
])
MASK = np.array([
    [-0.050, -0.061, -0.044, -0.039],
    [-0.072, -0.014, -0.049, -0.033],
    [-0.056, -0.056, -0.039, -0.033],
    [-0.025, -0.031, -0.044, -0.044],
    [-0.039, -0.044, -0.064, -0.050],
])

PANELS = [
    ("change under\nfairness instruction", FAIR - AGG),
    ("change under\ndemographic masking", MASK - AGG),
]

VLIM = 0.075


def fmt(v: float) -> str:
    s = f"{v:+.3f}".replace("0.", ".")
    return s if abs(v) > 1e-9 else "0"


fig, axes = plt.subplots(1, 2, figsize=(3.15, 2.35), sharey=True)

for k, (ax, (title, M)) in enumerate(zip(axes, PANELS)):
    ax.imshow(M, cmap="RdBu", vmin=-VLIM, vmax=VLIM, aspect="auto")
    ax.set_title(title, fontsize=6.8, pad=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["SP", "De", "La", "Ce"], fontsize=6.8)
    ax.set_yticks(range(5))
    if k == 0:
        ax.set_yticklabels(MODELS, fontsize=7)
    ax.tick_params(length=0)
    for i in range(5):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5.4,
                    color="white" if abs(v) > 0.052 else "black")
    for e in ax.spines.values():
        e.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

cbar = fig.colorbar(axes[0].images[0], ax=axes, orientation="horizontal",
                    fraction=0.05, pad=0.16, aspect=34)
cbar.ax.tick_params(labelsize=6)
cbar.outline.set_linewidth(0.4)
cbar.set_label("change in the committed gap (ESI levels);\n"
               "blue = toward parity, red = away from parity",
               fontsize=6.4, labelpad=2)

fig.subplots_adjust(left=0.20, right=0.94, top=0.85, bottom=0.315,
                    wspace=0.14)
fig.savefig("fig2_intervention_heatmap.pdf")
print("wrote fig2_intervention_heatmap.pdf")
