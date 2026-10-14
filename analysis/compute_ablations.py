#!/usr/bin/env python3
"""Ablation analyses over the frozen grid. Caliber = raw_triage: per-(case,race)
mean over rows/seeds, then per-case B-W diff, then mean.

(a) Round-depth: per-cell triage B-W gap at r0/r1/r2/agg; checks whether signs
    stay negative through deliberation.
(b) Per-seed sign stability: per-seed per-cell agg gap sign; checks whether
    (nearly) all 60 seed-cells are negative.
"""
import csv
from collections import defaultdict
from pathlib import Path

PULL = Path("data/grid_unified")
MODELS = [("qwen3", "Qwen3"), ("mistral", "Mistral"), ("medgemma", "MedGemma"),
          ("phi4", "Phi-4"), ("llama", "Llama")]
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]


def raw_triage(p, topo, readout, seed=None):
    acc = defaultdict(list)
    with open(p) as fh:
        for row in csv.DictReader(fh):
            if topo and row["topology"] != topo:
                continue
            if row["readout"] != readout:
                continue
            if seed is not None and int(row["seed"]) != seed:
                continue
            v = row["value"]
            if v not in ("Black", "White"):
                continue
            t = row["triage_acuity"]
            if t in ("", "None"):
                continue
            acc[(row["case_id"], v)].append(float(t))
    perc = defaultdict(dict)
    for (cid, v), vals in acc.items():
        perc[cid][v] = sum(vals) / len(vals)
    d = [perc[c]["Black"] - perc[c]["White"] for c in perc
         if "Black" in perc[c] and "White" in perc[c]]
    return (sum(d) / len(d) if d else float("nan")), len(d)


print("=== (a) round-depth: gap at r0 / r1 / r2 / agg per cell ===")
neg = tot = 0
for m, L in MODELS:
    p = PULL / m / "decisions.csv"
    for t in TOPOS:
        vals = [raw_triage(p, t, ro)[0] for ro in ("r0", "r1", "r2", "agg")]
        flags = "".join("-" if v < 0 else ("0" if v == 0 else "+") for v in vals)
        tot += 4
        neg += sum(1 for v in vals if v < 0)
        print(f"  {L:<9}{t:<14}" + "  ".join(f"{v:+.3f}" for v in vals) + f"   [{flags}]")
print(f"  negative stage-cells: {neg}/{tot}")

print("\n=== (b) per-seed agg sign per cell ===")
neg = tot = 0
worst = []
for m, L in MODELS:
    p = PULL / m / "decisions.csv"
    for t in TOPOS:
        signs = []
        for s in (42, 43, 44):
            v, n = raw_triage(p, t, "agg", seed=s)
            signs.append(v)
            tot += 1
            if v < 0:
                neg += 1
            else:
                worst.append((L, t, s, v))
        print(f"  {L:<9}{t:<14}" + "  ".join(f"s{s}={v:+.3f}" for s, v in zip((42, 43, 44), signs)))
print(f"  negative seed-cells: {neg}/{tot}")
if worst:
    print("  NON-NEGATIVE:", worst)
