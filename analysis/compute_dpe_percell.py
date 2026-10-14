#!/usr/bin/env python3
"""Per-cell DPE (diverse) measured after-gap, validated against published values.

Caliber = scripts/analyze_diverse.py "HOMOGENEOUS vs DIVERSE" block, verbatim:
  after  = mean over cases of [ mean over seeds of Black-seeing agent's triage
           minus White-seeing agent's triage, at the LAST debate round (r2);
           aggregator rows excluded ]  (d0=White, d1=Black per probe_axes.RACE)
  before = homogeneous agg B-W gap, raw_triage caliber (per-case seed-mean,
           then case-mean) from grid decisions.csv
  red%   = (1 - |after|/|before|) * 100

Validation targets:
  before  20/20 vs tab:grid agg column
  after    5/5  vs previously measured SharedPool values
  red%    20/20 vs published integers (tab:grid DPE red. column)
Also prints the previously recorded after values (15/20 back-derived) against
the measured ones, flagging 3-decimal or sign mismatches.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

BASE_DPE = Path("data/grid_unified")
BASE_GRID = Path("data/grid_unified")
DIRS_DPE = {"Qwen3": "qwen3_dpe_bw_0618", "Mistral": "mistral_dpe_0618",
            "MedGemma": "medgemma_dpe_0619", "Phi-4": "phi4_dpe_0618",
            "Llama": "llama_dpe_0619"}
DIRS_GRID = {"Qwen3": "qwen3", "Mistral": "mistral", "MedGemma": "medgemma",
             "Phi-4": "phi4", "Llama": "llama"}
TOPOS = ["SharedPool", "Decentralized", "Layers", "Centralized"]

# published anchors (order SP/De/La/Ce)
PUB_BEFORE = {"Qwen3": [-0.047, -0.058, -0.064, -0.031],
              "Mistral": [-0.069, -0.019, -0.053, -0.025],
              "MedGemma": [-0.053, -0.058, -0.039, -0.039],
              "Phi-4": [-0.025, -0.025, -0.039, -0.042],
              "Llama": [-0.039, -0.044, -0.061, -0.050]}
PUB_RED = {"Qwen3": [94, 85, 5, 18], "Mistral": [92, 57, 6, 56],
           "MedGemma": [89, 95, 4, 93], "Phi-4": [67, 33, 7, 27],
           "Llama": [93, 75, 13, 50]}
LEDGER_AFTER = {"Qwen3": [-0.003, -0.009, -0.061, -0.025],
                "Mistral": [+0.006, -0.008, -0.050, -0.011],
                "MedGemma": [-0.006, -0.003, -0.037, -0.003],
                "Phi-4": [-0.008, -0.017, -0.036, -0.031],
                "Llama": [-0.003, -0.011, -0.053, -0.025]}
MEASURED_SP = {"Qwen3": -0.003, "Mistral": +0.006, "MedGemma": -0.006,
               "Phi-4": -0.008, "Llama": -0.003}
RACE_BY_IDX = {"d0": "White", "d1": "Black", "d2": "Hispanic",
               "d3": "Asian", "d4": "control"}


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


def diverse_after(path, topo):
    """analyze_diverse.py caliber: last-round d1-minus-d0 triage, per case."""
    raw = defaultdict(lambda: defaultdict(dict))  # [cid][seed][round] -> {race: triage}
    rounds = set()
    with open(path) as fh:
        for line in fh:
            t = json.loads(line)
            if t["agent_id"] == "aggregator":
                continue
            if t.get("topology") != topo:
                continue
            parsed = t.get("parsed")
            if not parsed:
                continue
            race = RACE_BY_IDX.get(t["agent_id"])
            if race is None:
                continue
            val = parsed.get("triage_acuity")
            if val is None:
                continue
            raw[t["case_id"]][t["seed"]].setdefault(t["round"], {})[race] = float(val)
            rounds.add(t["round"])
    last = max(rounds)
    deltas = []
    for cid in raw:
        blacks, whites = [], []
        for seed in raw[cid]:
            rd = raw[cid][seed].get(last, {})
            if "Black" in rd:
                blacks.append(rd["Black"])
            if "White" in rd:
                whites.append(rd["White"])
        if blacks and whites:
            deltas.append(sum(blacks) / len(blacks) - sum(whites) / len(whites))
    return (sum(deltas) / len(deltas) if deltas else float("nan")), len(deltas), last


ok_before = ok_sp = ok_red = 0
mismatch_notes = []
print(f"{'model':<9}{'topo':<14}{'before':>8}{'b_ok':>5}{'after':>9}"
      f"{'ledger':>8}{'match':>6}{'red%':>6}{'pub':>5}{'r_ok':>5}{'n':>5}{'last_r':>7}")
for m in DIRS_DPE:
    tr = BASE_DPE / DIRS_DPE[m] / "traces.jsonl"
    gr = BASE_GRID / DIRS_GRID[m] / "decisions.csv"
    for i, topo in enumerate(TOPOS):
        before, _ = raw_triage(gr, topo, "agg")
        after, n, last = diverse_after(tr, topo)
        red = (1 - abs(after) / abs(before)) * 100
        b_ok = abs(before - PUB_BEFORE[m][i]) < 0.0006
        r_ok = abs(round(red) - PUB_RED[m][i]) <= 1
        led = LEDGER_AFTER[m][i]
        same3 = round(after, 3) == led
        sign_flip = (after > 0) != (led > 0) and abs(round(after, 3)) > 0
        tag = "ok" if same3 else ("SIGN!" if sign_flip else "DIFF")
        ok_before += b_ok
        ok_red += r_ok
        if topo == "SharedPool":
            ok_sp += abs(after - MEASURED_SP[m]) < 0.0006
        if not same3:
            mismatch_notes.append(
                f"{m} {topo}: measured {after:+.3f} vs ledger {led:+.3f}"
                f"{' [SIGN FLIP]' if sign_flip else ''}")
        print(f"{m:<9}{topo:<14}{before:>+8.3f}{str(b_ok):>5}{after:>+9.4f}"
              f"{led:>+8.3f}{tag:>6}{red:>6.1f}{PUB_RED[m][i]:>5}{str(r_ok):>5}"
              f"{n:>5}{last:>7}")

print(f"\nbefore vs tab:grid agg: {ok_before}/20"
      f"   SharedPool after vs ledger measured: {ok_sp}/5"
      f"   red% vs published: {ok_red}/20")
print("\n3-decimal mismatches vs previously recorded after values:")
for note in mismatch_notes or ["(none)"]:
    print(" ", note)
