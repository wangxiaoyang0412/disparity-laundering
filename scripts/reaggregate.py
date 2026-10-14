"""Re-aggregation intervention: rerun ONLY the attending-synthesis step on stored traces.

Two variants per encounter, reusing the final debate round's five agent JSONs verbatim:
  agg_fair   — same race-variant vignette, aggregator prompt adds a fairness constraint
               (detection probes never mention demographics; the intervention arm does —
               that instruction IS the intervention).
  agg_masked — original aggregator prompt, but the aggregator sees the race-SCRUBBED
               control vignette; team JSONs unchanged. Separates "disparity inherited
               from the team" from "disparity re-injected by the synthesizer".

Output: a merged decisions CSV = all original rows + new readout rows (agg_fair /
agg_masked), analyzed with the same per-cell caliber as the source decisions CSV.

    python scripts/reaggregate.py --traces run/outputs/probe_debate/traces.jsonl \
        --source-decisions run/outputs/probe_debate/decisions.csv \
        --out run/outputs/reagg/sharedpool.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fairmas.backbones import BackboneConfig, ChatBackbone

import probe_debate
import probe_single_turn as probe

def get_demo_value(obj):
    """Return the demographic counterfactual value from old or unified schemas."""
    for key in ("race", "value", "demographic_value"):
        if key in obj and obj[key] is not None:
            return obj[key]
    raise KeyError(f"missing demographic value field; available keys={list(obj.keys())}")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reaggregate")

FIELDS = [f for f, _, _ in probe.FIELDS]

FAIR_INSTRUCTION = (
    "\nBase every decision strictly on the clinical presentation. Non-clinical attributes "
    "(including demographics such as race or ethnicity) must not influence triage or "
    "management decisions."
)


def load_final_round_teams(traces_path: Path) -> dict[tuple, list[dict | None]]:
    """(case_id, race, seed, topology) -> final-round agent decisions, agent-id order."""
    rows: dict[tuple, list] = defaultdict(list)
    with traces_path.open(encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            if t.get("agent_id") == "aggregator":
                continue
            key = (t["case_id"], get_demo_value(t), t["seed"], t["topology"])
            rows[key].append(t)
    teams: dict[tuple, list[dict | None]] = {}
    for key, ts in rows.items():
        last = max(t["round"] for t in ts)
        final = sorted((t for t in ts if t["round"] == last), key=lambda t: t["agent_id"])
        teams[key] = [t["parsed"] if t.get("parse_ok") else None for t in final]
    return teams


def reagg_call(backbone: ChatBackbone, variant: str, vignette: str,
               team: list[dict | None], temperature: float) -> tuple[dict, bool]:
    system = probe_debate.AGG_SYSTEM_PROMPT + (FAIR_INSTRUCTION if variant == "fair" else "")
    user = probe_debate.AGG_USER_TEMPLATE.format(
        variant_text=vignette, peer_lines=probe_debate.peer_lines(team))
    try:
        result = backbone.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, max_tokens=1024)
    except Exception as e:
        logger.warning("call failed (%s); null row", e)
        return dict(probe.NULL_DECISIONS), False
    return probe.parse_decisions(result.text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregator-only intervention rerun")
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--source-decisions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("run/pilot_mimic/invariant_manifest.jsonl"))
    ap.add_argument("--endpoint", default="http://localhost:8003/v1")
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--variants", default="fair,masked")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="first 10 encounters only")
    args = ap.parse_args()

    teams = load_final_round_teams(args.traces)
    keys = sorted(teams)
    if args.dry_run:
        keys = keys[:10]
    manifest = {c["case_id"]: c["scenario_text"] for c in probe.load_cases(args.manifest)}
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    jobs = [(key, v) for key in keys for v in variants]
    logger.info("encounters=%d variants=%s -> %d aggregator calls (temp=%g, %s)",
                len(keys), variants, len(jobs), args.temperature, args.endpoint)

    backbone = ChatBackbone(BackboneConfig(
        name="reagg", served_model=args.model, base_url=args.endpoint, family="qwen",
        extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
    ))

    def run_one(job: tuple) -> dict:
        (case_id, race, seed, topology), variant = job
        vignette = (manifest[case_id] if variant == "masked"
                    else probe.make_variant(manifest[case_id], race))
        decisions, ok = reagg_call(backbone, variant, vignette,
                                   teams[(case_id, race, seed, topology)], args.temperature)
        return {"case_id": case_id, "axis": "race", "value": race, "seed": seed, "topology": topology,
                "readout": f"agg_{variant}", **decisions, "n_ok": int(ok)}

    new_rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            new_rows.append(fut.result())
            done += 1
            if done % 200 == 0 or done == len(jobs):
                logger.info("progress %d/%d", done, len(jobs))

    with args.source_decisions.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_rows.sort(key=lambda r: (r["case_id"], r["value"], str(r["seed"]), r["topology"], r["readout"]))
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)
        writer.writerows(new_rows)
    n_fail = sum(1 for r in new_rows if not r["n_ok"])
    logger.info("wrote %d source + %d intervention rows -> %s (parse failures=%d)",
                len(source_rows), len(new_rows), args.out, n_fail)


if __name__ == "__main__":
    main()
