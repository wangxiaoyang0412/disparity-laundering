#!/usr/bin/env python3
"""fig9_map: consensus gain vs gap retention, 20 cells (analysis/compute_consensus_map.py).

x: drop in pairwise triage disagreement among the five agents, r0 -> r2,
   per cell, mean over (case, seed, arm in {W,B}). Positive = more agreement.
y: |committed gap| / |r0 gap| per cell. Above 1.0 = amplified.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAIN = {"Qwen3": [.077, .042, .004, .046], "Mistral": [.111, .047, -.029, .018],
        "MedGemma": [.074, .037, -.003, .032], "Phi-4": [.094, .046, -.009, .025],
        "Llama": [.031, .015, -.012, .016]}
RET = {"Qwen3": [1.06, 1.62, 1.92, 0.79], "Mistral": [1.19, 0.87, 0.61, 0.45],
       "MedGemma": [1.27, 1.11, 1.17, 0.61], "Phi-4": [1.00, 1.80, 1.27, 1.36],
       "Llama": [0.64, 0.84, 1.22, 1.06]}
TOPO = ["SharedPool", "Decentralized", "Layers", "Centralized"]
TCOL = {0: "#0072B2", 1: "#009E73", 2: "#D55E00", 3: "#CC79A7"}
TMRK = {0: "o", 1: "s", 2: "^", 3: "D"}
CHARCOAL = "#3a3a3a"

fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)
ax.axhline(1.0, color=CHARCOAL, lw=0.8, ls="--", zorder=1)
ax.axvline(0.0, color=CHARCOAL, lw=0.8, zorder=1)
for m in GAIN:
    for ti in range(4):
        ax.scatter(GAIN[m][ti], RET[m][ti], marker=TMRK[ti], s=26,
                   facecolor=TCOL[ti], edgecolor="white", linewidth=0.4, zorder=3)
ax.text(0.113, 1.03, "amplified", fontsize=6.4, color=CHARCOAL, ha="right", va="bottom")
ax.text(0.113, 0.97, "reduced", fontsize=6.4, color=CHARCOAL, ha="right", va="top")
ax.text(-0.001, 1.985, "consensus\nlost", fontsize=6.4, color=CHARCOAL, ha="right", va="top")
ax.text(0.002, 1.985, "consensus\ngained", fontsize=6.4, color=CHARCOAL, ha="left", va="top")
ax.set_xlabel("Consensus gain ($r_0\\rightarrow r_2$ drop in disagreement)", fontsize=8)
ax.set_ylabel("Gap retention $|$agg$|/|r_0|$", fontsize=8)
ax.set_xlim(-0.042, 0.12)
ax.set_ylim(0.35, 2.02)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], marker=TMRK[i], linestyle="", markersize=4,
                      markerfacecolor=TCOL[i], markeredgecolor="white",
                      label=TOPO[i]) for i in range(4)]
ax.legend(handles=handles, fontsize=6.0, frameon=False, loc="lower right",
          handletextpad=0.25, borderaxespad=0.2)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig9_map.{ext}", bbox_inches="tight")
print("wrote fig9_map")
