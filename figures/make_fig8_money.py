#!/usr/bin/env python3
"""Redesigned Figure 8: aggregate error contrast and paired triage gaps.

The left panel centers each model at its White-version MAE, so the plotted
quantity is the direct within-model contrast rather than a truncated absolute
MAE axis. The right panel encodes topology redundantly with both color and
marker shape for print and color-vision accessibility.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------ Data ---------------------------------------
MODELS = ["Qwen3", "Mistral", "MedGemma", "Phi-4", "Llama"]

MAE_WHITE = np.array([0.785, 0.958, 0.796, 0.710, 1.012])
MAE_BLACK = np.array([0.781, 0.933, 0.762, 0.733, 0.994])

GAPS = {
    "Qwen3": [-0.047, -0.058, -0.064, -0.031],
    "Mistral": [-0.069, -0.019, -0.053, -0.025],
    "MedGemma": [-0.053, -0.058, -0.039, -0.039],
    "Phi-4": [-0.025, -0.025, -0.039, -0.042],
    "Llama": [-0.039, -0.044, -0.061, -0.050],
}

# Optional clinical context: number of cases assigned more urgent triage than
# gold, shown as Black-version : White-version counts.
MORE_URGENT_COUNTS = ["11:1", "14:2", "14:1", "12:3", "7:0"]
SHOW_COUNTS = True

TOPOLOGIES = ["SharedPool", "Decentralized", "Layers", "Centralized"]
TOPOLOGY_COLORS = {
    "SharedPool": "#0072B2",       # blue
    "Decentralized": "#009E73",    # bluish green
    "Layers": "#D55E00",          # vermillion
    "Centralized": "#CC79A7",     # reddish purple
}
TOPOLOGY_MARKERS = {
    "SharedPool": "o",
    "Decentralized": "s",
    "Layers": "^",
    "Centralized": "D",
}


# ------------------------------ Style --------------------------------------
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.7,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def despine(ax: plt.Axes) -> None:
    """Remove top and right spines while keeping a paper-friendly frame."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.20,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def draw_mae_contrast(ax: plt.Axes) -> None:
    """Panel a: direct paired contrast, centered at each White-version MAE."""
    x = np.arange(len(MODELS))
    delta = MAE_BLACK - MAE_WHITE

    # A light neutral band keeps attention on the centered contrast, not on
    # the absolute magnitude of the original MAE values.
    ax.axhspan(-0.04, 0.04, color="#F5F7F8", zorder=0)
    ax.axhline(0, color="#30343B", lw=1.0, zorder=1)

    # White versions are the zero reference; Black versions are the endpoints.
    for xi, di in zip(x, delta):
        ax.plot([xi, xi], [0, di], color="#7A838A", lw=2.0, zorder=2)

    ax.scatter(
        x,
        np.zeros_like(x),
        s=41,
        facecolor="white",
        edgecolor="#30343B",
        linewidth=1.1,
        zorder=3,
        label="White (centered)",
    )
    ax.scatter(
        x,
        delta,
        s=45,
        facecolor="#5B8FB9",
        edgecolor="#203846",
        linewidth=0.8,
        zorder=4,
        label="Black",
    )

    for xi, di in zip(x, delta):
        offset = 4 if di >= 0 else -5
        va = "bottom" if di >= 0 else "top"
        ax.annotate(
            f"{di:+.3f}",
            (xi, di),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            color="#30343B",
            fontsize=7.4,
            fontweight="semibold",
        )

    ax.set_xticks(x, MODELS)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(-0.045, 0.035)
    ax.set_yticks([-0.04, -0.02, 0.00, 0.02])
    ax.set_ylabel("MAE difference (Black - White)")
    ax.set_title(
        "Aggregate error is nearly matched",
        loc="left",
        pad=32,
        fontweight="semibold",
    )
    ax.text(
        0.01,
        0.03,
        "Lower MAE for Black",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#5C6770",
        fontsize=7.1,
    )
    ax.text(
        0.01,
        0.97,
        "Higher MAE for Black",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#5C6770",
        fontsize=7.1,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.15, 1.015),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    ax.grid(axis="y", color="#D7DCE0", lw=0.55, alpha=0.8, zorder=0)
    despine(ax)
    add_panel_label(ax, "(a)")


def draw_triage_gaps(ax: plt.Axes) -> None:
    """Panel b: topology-specific paired gaps with redundant visual encoding."""
    x = np.arange(len(MODELS))
    offsets = np.array([-0.24, -0.08, 0.08, 0.24])

    # Alternating model bands make the four protocol markers visually belong
    # to their model without adding hard vertical separators.
    band_colors = ["#FFF6F2", "#F3F6F8"]
    for i in range(len(MODELS)):
        ax.axvspan(
            i - 0.5,
            i + 0.5,
            color=band_colors[i % 2],
            ec="none",
            zorder=-2,
        )

    ax.axhline(0, color="#30343B", lw=1.0, zorder=1)

    for j, topology in enumerate(TOPOLOGIES):
        y = np.array([GAPS[model][j] for model in MODELS])
        ax.scatter(
            x + offsets[j],
            y,
            s=38,
            marker=TOPOLOGY_MARKERS[topology],
            facecolor=TOPOLOGY_COLORS[topology],
            edgecolor="#30343B",
            linewidth=0.65,
            label=topology,
            zorder=3,
        )

    # A short dashed segment marks the mean across the four protocols for each
    # model while preserving the individual topology-specific observations.
    model_means = np.array([np.mean(GAPS[model]) for model in MODELS])
    for xi, mean_gap in zip(x, model_means):
        ax.plot(
            [xi - 0.34, xi + 0.34],
            [mean_gap, mean_gap],
            color="#30343B",
            lw=1.15,
            linestyle=(0, (3.0, 2.0)),
            solid_capstyle="butt",
            zorder=2,
        )

    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(-0.075, 0.008)
    ax.set_yticks([0.00, -0.02, -0.04, -0.06])
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("Committed triage gap (Black - White)")
    ax.set_title(
        "Allocation disparity persists across protocols",
        loc="left",
        pad=32,
        fontweight="semibold",
    )
    ax.text(
        0.99,
        0.86,
        "parity",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color="#4E555B",
        fontsize=7.4,
    )
    ax.text(
        0.02,
        0.82,
        "Black version assigned more urgent triage",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#8B3F2B",
        fontsize=7.2,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.58, 1.015),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.05,
    )
    ax.grid(axis="y", color="#E0D9D6", lw=0.55, alpha=0.9, zorder=0)
    despine(ax)
    add_panel_label(ax, "(b)")

    if SHOW_COUNTS:
        ax.text(
            0.5,
            -0.36,
            "Dashed: model mean; counts: more urgent than gold (B:W)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#5C6770",
            fontsize=6.8,
            clip_on=False,
        )
        for xi, count in zip(x, MORE_URGENT_COUNTS):
            ax.text(
                xi,
                -0.22,
                count,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                color="#30343B",
                fontsize=7.1,
                fontweight="semibold",
                clip_on=False,
            )


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 2.85),
        gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.33},
    )

    draw_mae_contrast(axes[0])
    draw_triage_gaps(axes[1])

    fig.subplots_adjust(left=0.085, right=0.995, top=0.71, bottom=0.22)

    output_stem = output_dir / "fig8_redesigned"
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        f"{output_stem}.png",
        dpi=360,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(f"{output_stem}.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
