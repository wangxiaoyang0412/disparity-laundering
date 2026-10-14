#!/usr/bin/env python3
"""Consensus-Disparity Map — 20 cells, descriptive.

x  = within-arm consensus gain on triage: mean over (case, seed, arm in {W,B}) of
     [pairwise disagreement among the 5 agents at r0  minus  the same at r2]
     (pairwise disagreement = fraction of agent pairs with unequal triage answers)
y  = cross-version gap retention: |agg gap| / |r0 gap| per cell
     (gaps = committed/consensus triage B-W, per-case seed-mean then case-mean,
      from the frozen grid decisions.csv — same caliber as tab:grid)
Laundering quadrant: consensus gain > 0 AND retention >= 1 (gap not reduced).
"""
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

TR = Path("data/grid_unified")
DEC = Path("data/grid_unified")
DIRS = {"Qwen3": "qwen3", "Mistral": "mistral", "MedGemma": "medgemma",
        "Phi-4": "phi4", "Llama": "llama"}
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]


def gaps(path):
    acc = defaultdict(list)
    for row in csv.DictReader(open(path)):
        if row["value"] not in ("Black", "White") or row["readout"] not in ("r0", "agg"):
            continue
        t = row["triage_acuity"]
        if t in ("", "None"):
            continue
        acc[(row["topology"], row["readout"], row["case_id"], row["value"])].append(float(t))
    out = {}
    for topo in TOPOS:
        for ro in ("r0", "agg"):
            perc = defaultdict(dict)
            for (tp, r, c, v), vals in acc.items():
                if tp == topo and r == ro:
                    perc[c][v] = sum(vals) / len(vals)
            d = [perc[c]["Black"] - perc[c]["White"] for c in perc if len(perc[c]) == 2]
            out[(topo, ro)] = sum(d) / len(d)
    return out


def pairwise_disagreement(vals):
    pairs = list(combinations(vals, 2))
    return sum(1 for a, b in pairs if a != b) / len(pairs) if pairs else None


def consensus_gain(path):
    """traces -> {(topo): mean disagreement drop r0->r2 over (case,seed,arm W/B)}."""
    ans = defaultdict(dict)  # (topo,case,seed,value,round) -> {agent: triage}
    for line in open(path):
        t = json.loads(line)
        if t["agent_id"] == "aggregator" or t["value"] not in ("Black", "White"):
            continue
        if t["round"] not in (0, 2):
            continue
        v = (t.get("parsed") or {}).get("triage_acuity")
        if v is None:
            continue
        ans[(t["topology"], t["case_id"], t["seed"], t["value"], t["round"])][t["agent_id"]] = v
    drops = defaultdict(list)
    keys0 = {k[:4] for k in ans if k[4] == 0}
    for k in keys0:
        a0 = ans.get((*k, 0), {})
        a2 = ans.get((*k, 2), {})
        if len(a0) >= 2 and len(a2) >= 2:
            d0 = pairwise_disagreement(list(a0.values()))
            d2 = pairwise_disagreement(list(a2.values()))
            drops[k[0]].append(d0 - d2)
    return {topo: sum(v) / len(v) for topo, v in drops.items()}, \
           {topo: len(v) for topo, v in drops.items()}


print(f"{'model':<9}{'topo':<14}{'consens_gain':>13}{'|r0 gap|':>9}{'|agg gap|':>10}"
      f"{'retention':>10}{'quadrant':>10}{'n':>6}")
quad = 0
for m in DIRS:
    g = gaps(DEC / DIRS[m] / "decisions.csv")
    cg, ns = consensus_gain(TR / DIRS[m] / "traces.jsonl")
    for topo in TOPOS:
        x = cg[topo]
        r0g, aggg = abs(g[(topo, "r0")]), abs(g[(topo, "agg")])
        ret = aggg / r0g if r0g else float("inf")
        inq = x > 0 and ret >= 1.0
        quad += inq
        print(f"{m:<9}{topo:<14}{x:>+13.3f}{r0g:>9.3f}{aggg:>10.3f}"
              f"{ret:>10.2f}{'YES' if inq else 'no':>10}{ns[topo]:>6}")
    print()
print(f"laundering quadrant (consensus gain > 0 AND retention >= 1): {quad}/20")
print("retention >= 0.8:", end=" ")
cnt = 0
for m in DIRS:
    g = gaps(DEC / DIRS[m] / "decisions.csv")
    for topo in TOPOS:
        cnt += (abs(g[(topo, "agg")]) / abs(g[(topo, "r0")])) >= 0.8
print(f"{cnt}/20")
