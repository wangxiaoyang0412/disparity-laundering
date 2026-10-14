#!/usr/bin/env python3
"""Fairness-utility table, five methods at SharedPool, per model.

Methods (per model, SharedPool unless noted):
  homo-MAS   committed agg rows, W/B arms          (deployed baseline)
  r0-vote    readout=r0 consensus rows, W/B arms   (non-interactive majority vote)
  control    value=control agg rows                (upstream full masking; gap=0 by construction)
  cf-avg     mean of White-agg and Black-agg       (counterfactual average; gap=0 by construction)
  DPE        aggregator rows of the diverse traces (committed); gap = internal r2 d1-d0 (measured)

Utility on the gold_acuity subset (60/109 cases):
  MAE   per-case seed-mean prediction vs gold (float)
  exact / over(pred more acute: pred<gold) / under(pred>gold) / severe-under(pred-gold>=2)
        rates over per-(case,seed) integer predictions
Gap = committed triage Black-White (per-case seed-mean, case-mean), except DPE where the
published measure is the internal last-round gap (compute_dpe_percell.py).
Descriptive only; no inferential statistics.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

GRID = Path("data/grid_unified")
DPE = Path("data/grid_unified")
DIRS = {"Qwen3": "qwen3", "Mistral": "mistral", "MedGemma": "medgemma",
        "Phi-4": "phi4", "Llama": "llama"}
DPE_DIRS = {"Qwen3": "qwen3_dpe_bw_0618", "Mistral": "mistral_dpe_0618",
            "MedGemma": "medgemma_dpe_0619", "Phi-4": "phi4_dpe_0618",
            "Llama": "llama_dpe_0619"}
DPE_INTERNAL = {"Qwen3": -0.0028, "Mistral": +0.0056, "MedGemma": -0.0056,
                "Phi-4": -0.0083, "Llama": -0.0028}

GOLD = {}
for line in open(Path("data/gold_allocations.jsonl")):
    o = json.loads(line)
    g = o.get("gold_acuity")
    if g not in (None, "", "None"):
        GOLD[o["case_id"]] = float(g)


def load_rows(path, readout, values):
    """decisions.csv -> {(case,value): [triage per seed-row]} at SharedPool."""
    acc = defaultdict(list)
    for row in csv.DictReader(open(path)):
        if row["topology"] != "SharedPool" or row["readout"] != readout:
            continue
        if row["value"] not in values:
            continue
        t = row["triage_acuity"]
        if t in ("", "None"):
            continue
        acc[(row["case_id"], row["value"])].append(float(t))
    return acc


def dpe_committed(path):
    acc = defaultdict(list)
    for line in open(path):
        t = json.loads(line)
        if t["agent_id"] != "aggregator" or t["topology"] != "SharedPool":
            continue
        v = (t.get("parsed") or {}).get("triage_acuity")
        if v is not None:
            acc[(t["case_id"], "diverse")].append(float(v))
    return acc


def gap_bw(acc):
    perc = defaultdict(dict)
    for (c, v), vals in acc.items():
        perc[c][v] = sum(vals) / len(vals)
    d = [perc[c]["Black"] - perc[c]["White"] for c in perc
         if "Black" in perc[c] and "White" in perc[c]]
    return sum(d) / len(d) if d else float("nan")


def utility(preds_by_case, per_seed_int):
    """preds_by_case: {case: float pred}; per_seed_int: [(case, int_pred), ...]"""
    pairs = [(p, GOLD[c]) for c, p in preds_by_case.items() if c in GOLD]
    mae = sum(abs(p - g) for p, g in pairs) / len(pairs) if pairs else float("nan")
    ints = [(p, GOLD[c]) for c, p in per_seed_int if c in GOLD]
    n = len(ints)
    exact = sum(1 for p, g in ints if p == round(g)) / n
    over = sum(1 for p, g in ints if p < g) / n
    under = sum(1 for p, g in ints if p > g) / n
    sev = sum(1 for p, g in ints if p - g >= 2) / n
    return mae, exact, over, under, sev, len(pairs)


def acc_to_utility(acc):
    """Average across arms: prediction per case = mean over (arms x seeds)."""
    by_case = defaultdict(list)
    per_seed = []
    for (c, v), vals in acc.items():
        by_case[c].extend(vals)
        per_seed.extend((c, int(round(x))) for x in vals)
    return {c: sum(v) / len(v) for c, v in by_case.items()}, per_seed


print(f"{'model':<9}{'method':<9}{'gap':>8}{'MAE':>7}{'exact':>7}{'over':>7}"
      f"{'under':>7}{'sev>=2':>7}{'n_gold':>7}")
for m in DIRS:
    gp = GRID / DIRS[m] / "decisions.csv"
    methods = {}
    homo = load_rows(gp, "agg", {"Black", "White"})
    methods["homo"] = (gap_bw(homo), *acc_to_utility(homo))
    r0v = load_rows(gp, "r0", {"Black", "White"})
    methods["r0-vote"] = (gap_bw(r0v), *acc_to_utility(r0v))
    ctrl = load_rows(gp, "agg", {"control"})
    methods["control"] = (0.0, *acc_to_utility(ctrl))
    # counterfactual average: per case mean of White-mean and Black-mean
    perc = defaultdict(dict)
    for (c, v), vals in homo.items():
        perc[c][v] = sum(vals) / len(vals)
    cf_case = {c: (d["Black"] + d["White"]) / 2 for c, d in perc.items() if len(d) == 2}
    cf_seed = [(c, int(round(p))) for c, p in cf_case.items()]
    methods["cf-avg"] = (0.0, cf_case, cf_seed)
    dpe = dpe_committed(DPE / DPE_DIRS[m] / "traces.jsonl")
    methods["DPE"] = (DPE_INTERNAL[m], *acc_to_utility(dpe))
    for name, (gap, by_case, per_seed) in methods.items():
        mae, exact, over, under, sev, ng = utility(by_case, per_seed)
        print(f"{m:<9}{name:<9}{gap:>+8.3f}{mae:>7.3f}{exact:>7.2f}{over:>7.2f}"
              f"{under:>7.2f}{sev:>7.3f}{ng:>7}")
    print()
