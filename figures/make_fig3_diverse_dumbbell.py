#!/usr/bin/env python3
"""Figure 3: topology-dependent effect of DPE on committed triage gaps."""

from csv import writer
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# (homogeneous before, diverse/DPE after)
DATA = {
    "Qwen3": {
        "SharedPool": (-0.047, -0.003),
        "Decentralized": (-0.058, 0.000),
        "Layers": (-0.064, -0.042),
        "Centralized": (-0.031, -0.025),
    },
    "Mistral": {
        "SharedPool": (-0.069, +0.006),
        "Decentralized": (-0.019, -0.008),
        "Layers": (-0.053, -0.138),
        "Centralized": (-0.025, +0.039),
    },
    "MedGemma": {
        "SharedPool": (-0.053, -0.006),
        "Decentralized": (-0.058, +0.003),
        "Layers": (-0.039, -0.094),
        "Centralized": (-0.039, -0.003),
    },
    "Phi-4": {
        "SharedPool": (-0.025, -0.008),
        "Decentralized": (-0.025, -0.017),
        "Layers": (-0.039, -0.081),
        "Centralized": (-0.042, -0.031),
    },
    "Llama": {
        "SharedPool": (-0.039, -0.003),
        "Decentralized": (-0.044, -0.011),
        "Layers": (-0.061, -0.047),
        "Centralized": (-0.050, -0.025),
    },
}

MODELS = ["Qwen3", "Mistral", "MedGemma", "Phi-4", "Llama"]
DISPLAY_ORDER = [
    "SharedPool",
    "Decentralized",
    "Centralized",
    "Layers",
]

COLORS = {
    "before_edge": "#8C9499",
    "closer": "#277C6E",
    "farther": "#C45C4D",
    "dark": "#27343B",
}


mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 0,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def flatten_data():
    rows = []
    for model in MODELS:
        for topology in DISPLAY_ORDER:
            before, after = DATA[model][topology]
            closer = abs(after) < abs(before)
            rows.append(
                {
                    "model": model,
                    "topology": topology,
                    "before": before,
                    "after": after,
                    "closer_to_parity": closer,
                    "absolute_gap_change": abs(after) - abs(before),
                }
            )
    return rows


def validate(rows):
    assert len(rows) == 20
    assert all(row["before"] < 0 for row in rows)

    counts = {
        topology: sum(
            row["closer_to_parity"]
            for row in rows
            if row["topology"] == topology
        )
        for topology in DISPLAY_ORDER
    }
    assert counts == {
        "SharedPool": 5,
        "Decentralized": 5,
        "Centralized": 4,
        "Layers": 2,
    }

    sign_flips = {
        (row["model"], row["topology"])
        for row in rows
        if row["before"] * row["after"] < 0
    }
    assert sign_flips == {
        ("Mistral", "SharedPool"),
        ("Mistral", "Centralized"),
        ("MedGemma", "Decentralized"),
    }


def export_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = writer(stream)
        csv_writer.writerow(
            [
                "model",
                "topology",
                "before",
                "after",
                "closer_to_parity",
                "absolute_gap_change",
            ]
        )
        for row in rows:
            csv_writer.writerow(
                [
                    row["model"],
                    row["topology"],
                    f'{row["before"]:.3f}',
                    f'{row["after"]:.3f}',
                    str(row["closer_to_parity"]).lower(),
                    f'{row["absolute_gap_change"]:.3f}',
                ]
            )


def panel_summary(rows, topology):
    subset = [row for row in rows if row["topology"] == topology]
    closer = sum(row["closer_to_parity"] for row in subset)
    farther = len(subset) - closer
    if farther >= closer:
        return f"{farther}/5 farther", COLORS["farther"]
    return f"{closer}/5 closer", COLORS["closer"]


def draw_panel(ax, rows, topology, show_ylabels, show_xlabels):
    x_min, x_max = -0.150, 0.050
    y = np.arange(len(MODELS))[::-1]

    # Small multiples share the same scale; zero remains the only semantic
    # reference line. The light side tint makes sign reversals immediately clear.
    ax.axvspan(x_min, 0, color="#FAF4F2", zorder=-3)
    ax.axvspan(0, x_max, color="#F3F6F8", zorder=-3)
    ax.axvline(
        0,
        color="#4E575D",
        lw=1.05,
        linestyle=(0, (4, 2.5)),
        zorder=1,
    )

    subset = {
        row["model"]: row
        for row in rows
        if row["topology"] == topology
    }

    for yi, model in zip(y, MODELS):
        row = subset[model]
        color = (
            COLORS["closer"]
            if row["closer_to_parity"]
            else COLORS["farther"]
        )

        ax.plot(
            [row["before"], row["after"]],
            [yi, yi],
            color=color,
            lw=2.2,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            row["before"],
            yi,
            s=46,
            facecolor="white",
            edgecolor=COLORS["before_edge"],
            linewidth=1.25,
            zorder=3,
        )
        ax.scatter(
            row["after"],
            yi,
            s=55,
            facecolor=color,
            edgecolor=COLORS["dark"],
            linewidth=0.65,
            zorder=4,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.55, len(MODELS) - 0.45)
    ax.set_xticks([-0.15, -0.10, -0.05, 0.00, 0.05])
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS)
    ax.tick_params(
        axis="x",
        labelbottom=show_xlabels,
    )
    ax.tick_params(
        axis="y",
        labelleft=show_ylabels,
    )

    ax.grid(axis="x", color="#DDE1E2", lw=0.55, alpha=0.78)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(show_ylabels)
    ax.spines["bottom"].set_color("#AEB5B8")

    ax.set_title(
        topology,
        loc="left",
        pad=7,
        fontweight="semibold",
        color="#273139",
    )
    summary, summary_color = panel_summary(rows, topology)
    ax.text(
        0.99,
        1.045,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=summary_color,
        fontweight="semibold",
    )


def draw_figure(rows, output_dir):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 4.75),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        bottom=0.155,
        top=0.755,
        wspace=0.14,
        hspace=0.32,
    )

    for index, (ax, topology) in enumerate(
        zip(axes.flat, DISPLAY_ORDER)
    ):
        draw_panel(
            ax,
            rows,
            topology,
            show_ylabels=index % 2 == 0,
            show_xlabels=index >= 2,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["before_edge"],
            markeredgewidth=1.25,
            markersize=5.5,
            label="Homogeneous (before)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["closer"],
            markerfacecolor=COLORS["closer"],
            markeredgecolor=COLORS["dark"],
            markeredgewidth=0.6,
            linewidth=2.0,
            markersize=5.8,
            label="DPE after: closer to parity",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["farther"],
            markerfacecolor=COLORS["farther"],
            markeredgecolor=COLORS["dark"],
            markeredgewidth=0.6,
            linewidth=2.0,
            markersize=5.8,
            label="DPE after: farther from parity",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.875),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#D4D9DC",
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.15,
        borderpad=0.48,
    )
    legend.get_frame().set_linewidth(0.75)

    fig.text(
        0.50,
        0.075,
        "Committed triage gap (Black - White)",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#303A40",
    )
    output_stem = output_dir / "fig3_diverse_dumbbell_redesigned"
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{output_stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        f"{output_stem}.png",
        dpi=360,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        f"{output_stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = flatten_data()
    validate(rows)
    export_csv(rows, output_dir / "fig3_diverse_dumbbell_source_data.csv")
    draw_figure(rows, output_dir)


if __name__ == "__main__":
    main()
