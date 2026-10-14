#!/usr/bin/env python3
"""fig6_rounds_trajectory.pdf v2: faceted stage trajectories.

One panel per model, four topology-colored lines each; every line stays below
the parity dash. Data = tab:rounds (published).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

TCOL = {"SharedPool": "#0072B2", "Decentralized": "#009E73",
        "Layers": "#D55E00", "Centralized": "#CC79A7"}
TOPOS = list(TCOL)

DATA = {
    "Qwen3":    {"SharedPool": (-0.044, -0.061, -0.050, -0.047),
                 "Decentralized": (-0.036, -0.033, -0.056, -0.058),
                 "Layers": (-0.033, -0.050, -0.039, -0.064),
                 "Centralized": (-0.039, -0.036, -0.036, -0.031)},
    "Mistral":  {"SharedPool": (-0.058, -0.069, -0.058, -0.069),
                 "Decentralized": (-0.022, -0.050, -0.025, -0.019),
                 "Layers": (-0.086, -0.039, -0.050, -0.053),
                 "Centralized": (-0.056, -0.025, -0.033, -0.025)},
    "MedGemma": {"SharedPool": (-0.042, -0.047, -0.047, -0.053),
                 "Decentralized": (-0.053, -0.047, -0.058, -0.058),
                 "Layers": (-0.033, -0.047, -0.036, -0.039),
                 "Centralized": (-0.064, -0.031, -0.028, -0.039)},
    "Phi-4":    {"SharedPool": (-0.025, -0.017, -0.019, -0.025),
                 "Decentralized": (-0.014, -0.033, -0.047, -0.025),
                 "Layers": (-0.031, -0.036, -0.050, -0.039),
                 "Centralized": (-0.031, -0.022, -0.042, -0.042)},
    "Llama":    {"SharedPool": (-0.061, -0.044, -0.039, -0.039),
                 "Decentralized": (-0.053, -0.056, -0.042, -0.044),
                 "Layers": (-0.050, -0.058, -0.058, -0.061),
                 "Centralized": (-0.047, -0.053, -0.053, -0.050)},
}

fig, axes = plt.subplots(2, 3, figsize=(3.15, 2.55), sharey=True)
xs = [0, 1, 2, 3]

for ax, (model, runs) in zip(axes.flat, DATA.items()):
    for topo in TOPOS:
        ax.plot(xs, runs[topo], color=TCOL[topo], lw=1.0, alpha=0.95,
                marker="o", ms=1.8)
    ax.axhline(0, color="0.3", lw=0.6, ls=(0, (3, 2)))
    ax.set_title(model, fontsize=6.8, pad=2)
    ax.set_ylim(-0.095, 0.01)
    ax.set_xlim(-0.25, 3.25)
    ax.set_xticks(xs)
    ax.set_xticklabels(["$r_0$", "$r_1$", "$r_2$", "agg"], fontsize=5.6)
    ax.tick_params(axis="y", labelsize=5.8, length=2)
    ax.tick_params(axis="x", length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

leg_ax = axes.flat[5]
leg_ax.set_axis_off()
handles = [Line2D([], [], color=c, lw=1.3, marker="o", ms=2.4, label=t)
           for t, c in TCOL.items()]
leg_ax.legend(handles=handles, loc="center", frameon=False, fontsize=6.4,
              handlelength=1.5, labelspacing=0.55)

fig.supylabel("Triage Black–White gap (ESI levels)", fontsize=7, x=0.015)
fig.subplots_adjust(left=0.15, right=0.985, top=0.93, bottom=0.075,
                    hspace=0.42, wspace=0.14)
fig.savefig("fig6_rounds_trajectory.pdf")
print("wrote fig6_rounds_trajectory.pdf")
