#!/usr/bin/env python3
"""Names-proxy preview: paired-caliber readout (descriptive only).

Paired gap = mean over (case, seed) of (NameBlack - NameWhite) per stage;
flip counts at the committed stage. Data:
data/grid_unified/qwen3_names_baseline_0730/.
"""
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

D = Path("data/grid_unified/qwen3_names_baseline_0730/decisions.csv")
FIELDS = [
    "triage_acuity",
    "testing_recommended",
    "imaging_recommended",
    "admission_recommended",
    "analgesia_strength",
    "mental_health_referral",
]


def to_num(v: str) -> float:
    if v in ("True", "False"):
        return 1.0 if v == "True" else 0.0
    return float(v)


def main() -> None:
    rows: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]] = defaultdict(dict)
    with open(D) as f:
        for row in csv.DictReader(f):
            key = (row["case_id"], row["seed"], row["readout"])
            rows[key][row["value"]] = {k: to_num(row[k]) for k in FIELDS if row[k] != ""}

    for stage in ["r0", "r1", "r2", "agg"]:
        diffs = []
        for (case, seed, st), pair in rows.items():
            if st == stage and "NameBlack" in pair and "NameWhite" in pair:
                diffs.append(
                    {k: pair["NameBlack"][k] - pair["NameWhite"][k] for k in FIELDS}
                )
        n = len(diffs)
        tri = [d["triage_acuity"] for d in diffs]
        line = f"{stage}: n={n} paired B-W triage mean={sum(tri)/n:+.4f}"
        if stage == "agg":
            more = sum(1 for t in tri if t < 0)
            eq = sum(1 for t in tri if t == 0)
            less = sum(1 for t in tri if t > 0)
            line += f" | flips: B-more-urgent {more}, equal {eq}, B-less-urgent {less}"
        print(line)
        if stage == "agg":
            for k in FIELDS[1:]:
                vals = [d[k] for d in diffs]
                print(f"   agg {k}: mean={sum(vals)/n:+.4f}")


if __name__ == "__main__":
    main()
