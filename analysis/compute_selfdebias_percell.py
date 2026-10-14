#!/usr/bin/env python3
"""Per-cell self-debias triage B-W gap, validated against published pooled values.

Caliber = raw_triage(): per-(case,race) mean over rows/seeds, then per-case
Black-White diff, then mean.
Validation: pooled-across-topology value per model must reproduce
tab:repair_ladder (Qwen3 -0.061, Mistral -0.065, MedGemma -0.074,
Phi-4 -0.078, Llama -0.076); SharedPool cell must reproduce fig:forest data
(Qwen3 -0.050, Mistral -0.050, MedGemma -0.047, Phi-4 -0.078, Llama -0.075).
"""
import csv
from collections import defaultdict
from pathlib import Path

BASE = Path("data/grid_unified")
DIRS = {"Qwen3": "qwen3_selfdebias_bw_0618", "Mistral": "mistral_selfdebias_bw_0618",
        "MedGemma": "medgemma_selfdebias_bw_0619", "Phi-4": "phi4_selfdebias_bw_0618",
        "Llama": "llama_selfdebias_bw_0619"}
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]
PUB_POOLED = {"Qwen3": -0.061, "Mistral": -0.065, "MedGemma": -0.074,
              "Phi-4": -0.078, "Llama": -0.076}
PUB_SP = {"Qwen3": -0.050, "Mistral": -0.050, "MedGemma": -0.047,
          "Phi-4": -0.078, "Llama": -0.075}


def raw_triage(p, topo, readout):
    acc = defaultdict(list)
    with open(p) as fh:
        for row in csv.DictReader(fh):
            if topo and row["topology"] != topo:
                continue
            if row["readout"] != readout:
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


print(f"{'model':<9}{'pooled':>8} pub_ok  SP_ok   per-topology (SP/De/La/Ce)")
for L, d in DIRS.items():
    p = BASE / d / "decisions.csv"
    pooled, n = raw_triage(p, None, "agg")
    cells = {t: raw_triage(p, t, "agg")[0] for t in TOPOS}
    ok_pool = abs(pooled - PUB_POOLED[L]) < 0.0006
    ok_sp = abs(cells["SharedPool"] - PUB_SP[L]) < 0.0006
    per = "  ".join(f"{cells[t]:+.3f}" for t in TOPOS)
    print(f"{L:<9}{pooled:>+8.4f} {str(ok_pool):<7}{str(ok_sp):<7}{per}   n={n}")
