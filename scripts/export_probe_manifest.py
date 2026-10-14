"""Export the single-turn probe manifest: {case_id, scenario_text} per invariant case.

Bridges the gap between what build_pilot_set writes (equivalence_invariant.jsonl = verdicts
only, no text) and what scripts/probe_single_turn.py consumes (flat race-scrubbed vignette).

scenario_text = Demographics + History + Symptoms + Past_Medical_History — the same fields
the equivalence judge saw, and deliberately WITHOUT Physical_Examination_Findings or
Test_Results: testing/imaging recommendations are the probe's OUTCOME variables, so the
vignette must describe the pre-workup presentation only. Race is scrubbed by
load_scenarios(MIMICIV) before any text is assembled.

Run from repo root:
    python scripts/export_probe_manifest.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from fairmas.mas.scenarios import Scenario, load_scenarios
from fairmas.utils import get_logger

logger = get_logger("fairmas.export_probe_manifest")


def _to_text(value: object) -> str:
    """Flatten a Patient_Actor field (str | list | dict) into one readable string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(_to_text(v) for v in value if _to_text(v))
    if isinstance(value, dict):
        return "; ".join(f"{k.replace('_', ' ')}: {_to_text(v)}" for k, v in value.items()
                         if _to_text(v))
    return "" if value is None else str(value)


def scenario_text(scenario: Scenario) -> str:
    """Single-paragraph ED vignette; opens with the age-sex phrase the probe regex targets."""
    # Race-scrub can leave a stranded separator ("58-year-old / female") that breaks the
    # probe's inline insertion (10/120 pilot cases fell back to a prepended demographics
    # line). Normalize here.
    demo = re.sub(r"(year[s]?[-\s]old)\s*[/,]\s*", r"\1 ", scenario.demographics.strip())
    parts = [f"A {demo} presents to the emergency department."]
    actor = scenario.patient_actor
    for key, label in [("History", "History"), ("Symptoms", "Symptoms"),
                       ("Past_Medical_History", "Past medical history")]:
        text = _to_text(actor.get(key))
        if text:
            parts.append(f"{label}: {text}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export {case_id, scenario_text} probe manifest")
    ap.add_argument("--ambig", action="store_true",
                    help="preset: ambiguous cohort (pilot_mimic_ambig + AgentClinic_ambig)")
    ap.add_argument("--invariant", type=Path,
                    default=Path("run/pilot_mimic/equivalence_invariant.jsonl"))
    ap.add_argument("--dataset", default="MIMICIV")
    ap.add_argument("--agentclinic-root", default="external/AgentClinic")
    ap.add_argument("--out", type=Path, default=Path("run/pilot_mimic/invariant_manifest.jsonl"))
    args = ap.parse_args(argv)
    if args.ambig:
        args.invariant = Path("run/pilot_mimic_ambig/equivalence_invariant.jsonl")
        args.agentclinic_root = "external/AgentClinic_ambig"
        args.out = Path("run/pilot_mimic_ambig/invariant_manifest.jsonl")

    invariant_ids = []
    with args.invariant.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                invariant_ids.append(json.loads(line)["case_id"])
    id_set = set(invariant_ids)
    logger.info("invariant verdicts: %d cases (%s)", len(id_set), args.invariant)

    scenarios = load_scenarios(args.dataset, root=args.agentclinic_root)  # race scrubbed here
    kept = [s for s in scenarios if s.case_id in id_set]
    missing = id_set - {s.case_id for s in kept}
    if missing:
        logger.warning("%d invariant case_ids not found in scenario file: %s",
                       len(missing), sorted(missing)[:5])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for s in kept:
            fh.write(json.dumps({"case_id": s.case_id, "scenario_text": scenario_text(s)},
                                ensure_ascii=False) + "\n")
    logger.info("wrote %d cases -> %s", len(kept), args.out)
    if kept:
        logger.info("first vignette: %s", scenario_text(kept[0])[:200])
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
