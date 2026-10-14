#!/usr/bin/env python3
"""Intervention columns: +fairness / +masking per-cell committed B-W triage gaps.

Data: the re-aggregation runs (readouts agg_fair / agg_masked) in
data/grid_unified/<model>_reagg/decisions.csv, produced by scripts/reaggregate.py
on the stored base-grid traces.

Caliber mirrors the B-W master-table column: paired by (case, seed, topology),
mean of (Black - White) over pairs where both arms parsed. Printed next to the
paper's published column values for reference. Descriptive only.
"""
import csv
from collections import defaultdict
from pathlib import Path

GRID = Path("data/grid_unified")
MODELS = ["qwen3", "mistral", "medgemma", "phi4", "llama"]
LABEL = {"qwen3": "Qwen3", "mistral": "Mistral", "medgemma": "MedGemma",
         "phi4": "Phi-4", "llama": "Llama"}
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]

# published column values (order SP/De/La/Ce)
PUB_FAIR = {"Qwen3": [-0.047, -0.058, -0.050, -0.044],
            "Mistral": [-0.072, -0.034, -0.072, -0.033],
            "MedGemma": [-0.047, -0.053, -0.036, -0.031],
            "Phi-4": [-0.014, -0.025, -0.058, -0.036],
            "Llama": [-0.039, -0.044, -0.064, -0.050]}
PUB_MASK = {"Qwen3": [-0.050, -0.061, -0.044, -0.039],
            "Mistral": [-0.072, -0.014, -0.049, -0.033],
            "MedGemma": [-0.056, -0.056, -0.039, -0.033],
            "Phi-4": [-0.025, -0.031, -0.044, -0.044],
            "Llama": [-0.039, -0.044, -0.064, -0.050]}


def paired_gap(path, topo, readout):
    """Mean Black-minus-White committed triage over (case, seed) pairs at one cell."""
    vals = defaultdict(dict)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["topology"] != topo or row["readout"] != readout:
                continue
            if row["value"] not in ("Black", "White"):
                continue
            t = row["triage_acuity"]
            if t in ("", "None"):
                continue
            vals[(row["case_id"], row["seed"])][row["value"]] = float(t)
    d = [p["Black"] - p["White"] for p in vals.values() if len(p) == 2]
    return (sum(d) / len(d) if d else float("nan")), len(d)


def main() -> None:
    print(f"{'model':<9}{'topology':<14}{'fair':>9}{'pub':>8}{'n':>5}"
          f"{'masked':>9}{'pub':>8}{'n':>5}")
    min_n = 10 ** 9
    for m in MODELS:
        L = LABEL[m]
        p = GRID / f"{m}_reagg" / "decisions.csv"
        for i, topo in enumerate(TOPOS):
            f, nf = paired_gap(p, topo, "agg_fair")
            k, nk = paired_gap(p, topo, "agg_masked")
            min_n = min(min_n, nf, nk)
            print(f"{L:<9}{topo:<14}{f:>+9.4f}{PUB_FAIR[L][i]:>+8.3f}{nf:>5}"
                  f"{k:>+9.4f}{PUB_MASK[L][i]:>+8.3f}{nk:>5}")
        print()
    print(f"cells: {len(MODELS) * len(TOPOS)}; min pairs per cell: {min_n}")


if __name__ == "__main__":
    main()
