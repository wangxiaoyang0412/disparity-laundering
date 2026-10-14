#!/usr/bin/env python3
"""L3 arbitration: what actually produced the committed (agg) rows of the
frozen grid -- the aggregator LLM call, or a per-field median of r2?

Per run (case, value, seed, topology):
  A = decisions.csv agg row (six fields)
  S = aggregator call's parsed output in traces.jsonl (round 3)
  M = per-field median of the five agents' round-2 parsed outputs
Count, per topology: A==S only / A==M only / A==S==M / neither / missing.

Data: data/grid_unified/<model>/ (decisions.csv + traces.jsonl, base arm).
Descriptive counting only.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

DEC = Path("data/grid_unified")
TRC = Path("data/grid_unified")
MODELS = ["qwen3", "mistral", "medgemma", "phi4", "llama"]
FIELDS = [
    "triage_acuity",
    "testing_recommended",
    "imaging_recommended",
    "admission_recommended",
    "analgesia_strength",
    "mental_health_referral",
]


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, str):
        if v in ("True", "False"):
            return 1.0 if v == "True" else 0.0
        try:
            return float(v)
        except ValueError:
            return None
    return float(v)


def main() -> None:
    tallies = defaultdict(lambda: defaultdict(int))
    for model in MODELS:
        agg_rows: Dict[Tuple, Dict[str, float]] = {}
        with open(DEC / model / "decisions.csv") as f:
            for row in csv.DictReader(f):
                if row["readout"] == "agg" and row["triage_acuity"]:
                    key = (row["case_id"], row["value"], row["seed"], row["topology"])
                    agg_rows[key] = {k: norm(row[k]) for k in FIELDS}

        synth: Dict[Tuple, Dict[str, float]] = {}
        r2: Dict[Tuple, list] = defaultdict(list)
        with open(TRC / model / "traces.jsonl") as f:
            for line in f:
                t = json.loads(line)
                p = t.get("parsed") or {}
                if not isinstance(p, dict) or "triage_acuity" not in p:
                    continue
                key = (t["case_id"], t["value"], str(t["seed"]), t["topology"])
                vals = {k: norm(p[k]) for k in FIELDS if k in p}
                if len(vals) != len(FIELDS) or any(v is None for v in vals.values()):
                    continue
                if t["agent_id"] == "aggregator":
                    synth[key] = vals
                elif int(t["round"]) == 2:
                    r2[key].append(vals)

        for key, a in agg_rows.items():
            topo = key[3]
            s = synth.get(key)
            agents = r2.get(key, [])
            if s is None or len(agents) != 5:
                tallies[topo]["missing"] += 1
                continue
            m = {k: statistics.median(x[k] for x in agents) for k in FIELDS}
            eq_s = all(abs(a[k] - s[k]) < 1e-9 for k in FIELDS)
            eq_m = all(abs(a[k] - m[k]) < 1e-9 for k in FIELDS)
            if eq_s and eq_m:
                tallies[topo]["both"] += 1
            elif eq_s:
                tallies[topo]["synth_only"] += 1
            elif eq_m:
                tallies[topo]["median_only"] += 1
            else:
                tallies[topo]["neither"] += 1

    print(f"{'topology':<14} {'synth_only':>10} {'median_only':>11} {'both':>6} {'neither':>8} {'missing':>8}")
    for topo in sorted(tallies):
        t = tallies[topo]
        print(
            f"{topo:<14} {t['synth_only']:>10} {t['median_only']:>11} "
            f"{t['both']:>6} {t['neither']:>8} {t['missing']:>8}"
        )


if __name__ == "__main__":
    main()
