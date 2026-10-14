"""Filter the race-invariant pilot cases for sex-invariance.

Takes the 120 race-invariant case IDs, loads MIMIC scenarios, asks an auditor
whether management would differ by biological sex.  Outputs sex-invariant and
sex-quarantine JSONL files, then exports the sex probe manifest.

Run with MedGemma serving on :8002 as the auditor (same as the race filter):
    python scripts/filter_sex_invariance.py

Then the sex probe manifest is at run/pilot_mimic/invariant_manifest_sex.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fairmas.backbones import ChatBackbone, load_backbones
from fairmas.experiment.equivalence import filter_scenarios, write_verdicts
from fairmas.mas.scenarios import load_scenarios
from fairmas.utils import get_logger, set_seed

logger = get_logger("filter_sex_invariance")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sex-invariance filter on race-invariant cases")
    ap.add_argument("--race-invariant", type=Path,
                    default=Path("run/pilot_mimic/equivalence_invariant.jsonl"))
    ap.add_argument("--dataset", default="MIMICIV")
    ap.add_argument("--agentclinic-root", default="external/AgentClinic")
    ap.add_argument("--config", default="run/conf/backbones.yaml")
    ap.add_argument("--auditor", default="medgemma-27b")
    ap.add_argument("--out-dir", type=Path, default=Path("run/pilot_mimic"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    set_seed(args.seed)

    race_ids: set[str] = set()
    with args.race_invariant.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                race_ids.add(json.loads(line)["case_id"])
    logger.info("loaded %d race-invariant case IDs from %s", len(race_ids), args.race_invariant)

    scenarios = load_scenarios(args.dataset, root=args.agentclinic_root)
    candidates = [s for s in scenarios if s.case_id in race_ids]
    missing = race_ids - {s.case_id for s in candidates}
    if missing:
        logger.warning("%d race-invariant IDs not found in %s: %s",
                       len(missing), args.dataset, sorted(missing)[:5])
    logger.info("filtering %d candidates for sex-invariance", len(candidates))

    registry = load_backbones(args.config)
    if args.auditor not in registry:
        logger.error("auditor %r not found (have: %s)", args.auditor, ", ".join(registry))
        return 2
    auditor = ChatBackbone(registry[args.auditor])

    invariant, quarantine = filter_scenarios(candidates, auditor, axis="sex")

    out = args.out_dir
    write_verdicts(invariant, out / "equivalence_sex_invariant.jsonl")
    write_verdicts(quarantine, out / "equivalence_sex_quarantine.jsonl")
    logger.info("sex filter: %d invariant / %d quarantined (from %d race-invariant)",
                len(invariant), len(quarantine), len(candidates))

    invariant_ids = {v.case_id for v in invariant}
    kept = [s for s in candidates if s.case_id in invariant_ids]

    from export_probe_manifest import scenario_text
    manifest_path = out / "invariant_manifest_sex.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for s in kept:
            fh.write(json.dumps({"case_id": s.case_id, "scenario_text": scenario_text(s)},
                                ensure_ascii=False) + "\n")
    logger.info("wrote sex manifest: %d cases -> %s", len(kept), manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
