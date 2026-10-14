#!/usr/bin/env python3
"""r0 triage B-W baselines for the 5-model spine.

raw_triage() is the canonical caliber: per-(case,race) mean over rows/seeds,
then per-case Black-White diff, then mean.
Validation: agg per cell must reproduce tab:grid before r0 is trusted.
"""
import csv
from collections import defaultdict
from pathlib import Path

PULL = Path("data/grid_unified")
MODELS = ["qwen3", "mistral", "medgemma", "phi4", "llama"]
LABEL = {"qwen3": "Qwen3", "mistral": "Mistral", "medgemma": "MedGemma",
         "phi4": "Phi-4", "llama": "Llama"}
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]

# published tab:grid agg values (paper, ARR version) for validation
PUB_AGG = {
    "Qwen3":    {"SharedPool": -0.047, "Decentralized": -0.058, "Layers": -0.064, "Centralized": -0.031},
    "Mistral":  {"SharedPool": -0.069, "Decentralized": -0.019, "Layers": -0.053, "Centralized": -0.025},
    "MedGemma": {"SharedPool": -0.053, "Decentralized": -0.058, "Layers": -0.039, "Centralized": -0.039},
    "Phi-4":    {"SharedPool": -0.025, "Decentralized": -0.025, "Layers": -0.039, "Centralized": -0.042},
    "Llama":    {"SharedPool": -0.039, "Decentralized": -0.044, "Layers": -0.061, "Centralized": -0.050},
}
# published amplify list (paper Section 5.2)
PUB_AMP = {
    "Qwen3": {"SharedPool", "Layers", "Decentralized"},
    "Mistral": {"SharedPool"},
    "MedGemma": {"SharedPool", "Layers", "Decentralized"},
    "Phi-4": {"SharedPool", "Layers", "Centralized", "Decentralized"},
    "Llama": {"Layers", "Centralized"},
}


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


print("=== VALIDATION: agg per cell vs published tab:grid ===")
bad = 0
agg_vals = {}
for m in MODELS:
    L = LABEL[m]
    agg_vals[L] = {}
    for t in TOPOS:
        v, n = raw_triage(PULL / m / "decisions.csv", t, "agg")
        agg_vals[L][t] = v
        pub = PUB_AGG[L][t]
        ok = abs(v - pub) < 0.0006
        bad += (not ok)
        print(f"  {L:<9}{t:<14} computed {v:+.4f}  published {pub:+.3f}  {'OK' if ok else 'MISMATCH'}")
print(f"validation mismatches: {bad}/20")

print("\n=== r0 triage B-W gap ===")
print(f"{'model':<9}{'r0 pooled':>10}{'n':>5}   per-topology r0")
r0_pooled = {}
r0_cell = {}
for m in MODELS:
    L = LABEL[m]
    v, n = raw_triage(PULL / m / "decisions.csv", None, "r0")
    r0_pooled[L] = v
    r0_cell[L] = {}
    parts = []
    for t in TOPOS:
        vt, nt = raw_triage(PULL / m / "decisions.csv", t, "r0")
        r0_cell[L][t] = vt
        parts.append(f"{t[:4]}={vt:+.4f}")
    print(f"{L:<9}{v:>+10.4f}{n:>5}   " + "  ".join(parts))

print("\n=== amplify check |agg|>=|r0|: which r0 caliber reproduces the published list ===")
for name, r0src in [("pooled-r0", lambda L, t: r0_pooled[L]),
                    ("per-cell-r0", lambda L, t: r0_cell[L][t])]:
    all_match = True
    for L in PUB_AMP:
        got = {t for t in TOPOS if abs(agg_vals[L][t]) >= abs(r0src(L, t)) - 1e-9}
        match = got == PUB_AMP[L]
        all_match &= match
        if not match:
            print(f"  [{name}] {L}: computed {sorted(got)} vs published {sorted(PUB_AMP[L])}  DIFF")
    print(f"  [{name}] reproduces published amplify list: {all_match}")
