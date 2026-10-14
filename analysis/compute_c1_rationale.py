#!/usr/bin/env python3
"""C1 full-run rationale audit.

Caliber:
- race-mention regex (word-boundary locked, case-insensitive; the pilot's
  "blackouts" false positive is excluded by \\b):
  race|racial|racially|black|white|african[- ]american|caucasian|ethnic|
  ethnicity|demographic|demographics
- divergent committed pair: same (case_id, seed), committed (agg) triage
  differs between the Black and White arms
- reported per model (0/N per model).

Descriptive counts only. Data: data/grid_unified/<model>_rationale_full_0730/.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BASE = Path("data/grid_unified")
RX = re.compile(
    r"\b(race|racial|racially|black|white|african[- ]american|caucasian|"
    r"ethnic|ethnicity|demographic|demographics)\b",
    re.IGNORECASE,
)


def load_committed(path: Path) -> Dict[Tuple[str, str, str], int]:
    """Committed (agg) triage keyed by (case_id, seed, value)."""
    out: Dict[Tuple[str, str, str], int] = {}
    skipped = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["readout"] == "agg":
                if not row["triage_acuity"]:
                    skipped += 1
                    continue
                out[(row["case_id"], row["seed"], row["value"])] = int(row["triage_acuity"])
    if skipped:
        print(f"   [note] agg rows with empty triage skipped: {skipped}")
    return out


def stage_gaps(path: Path) -> Dict[str, float]:
    """Mean Black-minus-White triage gap per readout stage (paired by case-seed)."""
    vals: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row["triage_acuity"]:
                continue
            vals[(row["case_id"], row["seed"], row["readout"])][row["value"]] = float(
                row["triage_acuity"]
            )
    sums: Dict[str, List[float]] = defaultdict(list)
    for (case, seed, stage), pair in vals.items():
        if "Black" in pair and "White" in pair:
            sums[stage].append(pair["Black"] - pair["White"])
    return {s: sum(v) / len(v) for s, v in sums.items()}


def main() -> None:
    for model in ["qwen3", "mistral"]:
        d = BASE / f"{model}_rationale_full_0730"
        committed = load_committed(d / "decisions.csv")

        rationales: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        total, nonempty, hits = 0, 0, []
        with open(d / "traces.jsonl") as f:
            for line in f:
                t = json.loads(line)
                total += 1
                parsed = t.get("parsed") or {}
                rat = parsed.get("rationale") if isinstance(parsed, dict) else None
                if not rat:
                    continue
                nonempty += 1
                key = (t["case_id"], str(t["seed"]), t["value"])
                rationales[key].append(rat)
                for m in RX.finditer(rat):
                    ctx = rat[max(0, m.start() - 45) : m.end() + 45]
                    hits.append((t["case_id"], t["seed"], t["value"], t["round"], t["agent_id"], m.group(0), ctx))

        pairs = defaultdict(dict)
        for (case, seed, value), tri in committed.items():
            pairs[(case, seed)][value] = tri
        divergent = [k for k, p in pairs.items() if len(p) == 2 and p["Black"] != p["White"]]

        div_rat_n, div_hits = 0, 0
        for case, seed in divergent:
            for value in ("Black", "White"):
                rats = rationales[(case, seed, value)]
                div_rat_n += len(rats)
                div_hits += sum(1 for r in rats if RX.search(r))

        gaps = stage_gaps(d / "decisions.csv")

        print(f"==== {model} ====")
        print(f"rationales non-empty: {nonempty}/{total}")
        print(f"race/demographic mentions (all rationales): {len(hits)}/{nonempty}")
        for h in hits[:20]:
            print("   HIT:", h)
        print(f"divergent committed pairs: {len(divergent)}/{len(pairs)}")
        print(f"rationales behind divergent pairs: {div_rat_n}; mentions among them: {div_hits}")
        print("stage gaps (B-W, paired means):", {k: round(v, 4) for k, v in sorted(gaps.items())})
        print()


if __name__ == "__main__":
    main()
