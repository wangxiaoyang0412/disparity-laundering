#!/usr/bin/env python3
"""Hispanic-White and Asian-White committed gaps, all 20 cells (paired means).

Caliber mirrors the B-W master-table column: agg readout, paired by
(case, seed, topology), mean of (variant - White). Data: frozen grid
decisions (data/grid_unified). Descriptive only.
"""
import csv
from collections import defaultdict
from pathlib import Path

BASE = Path("data/grid_unified")
MODELS = ["qwen3", "mistral", "medgemma", "phi4", "llama"]
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]


def main() -> None:
    for contrast in ["Hispanic", "Asian"]:
        print(f"==== {contrast} - White (agg, paired) ====")
        neg, tot, min_n = 0, 0, 10**9
        for m in MODELS:
            vals = defaultdict(dict)
            with open(BASE / m / "decisions.csv") as f:
                for r in csv.DictReader(f):
                    if (
                        r["readout"] == "agg"
                        and r["triage_acuity"]
                        and r["value"] in (contrast, "White")
                    ):
                        key = (r["case_id"], r["seed"], r["topology"])
                        vals[key][r["value"]] = float(r["triage_acuity"])
            row = []
            for t in TOPOS:
                d = [
                    p[contrast] - p["White"]
                    for (c, s, tp), p in vals.items()
                    if tp == t and len(p) == 2
                ]
                g = sum(d) / len(d)
                row.append(f"{g:+.4f}(n={len(d)})")
                tot += 1
                neg += g < 0
                min_n = min(min_n, len(d))
            print(f"  {m:<9}", " ".join(row))
        print(f"  negative cells: {neg}/{tot}; min pairs per cell: {min_n}")


if __name__ == "__main__":
    main()
