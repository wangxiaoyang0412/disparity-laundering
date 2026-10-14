"""Compare diverse-perspective ensemble against homogeneous baseline.

In diverse mode each agent sees a different race variant. This script
extracts per-agent decisions from traces.jsonl, groups by the race each
agent saw, and computes the B-W (or M-F) gap at each round — showing
how cross-demographic exposure shrinks the gap through debate.

Also compares the diverse aggregate decision against each homogeneous
race arm from the baseline debate CSV.

    python scripts/analyze_diverse.py \
        --diverse-traces run/outputs/probe_diverse/traces.jsonl \
        --homo-csv run/outputs/probe_debate/decisions.csv

    python scripts/analyze_diverse.py \
        --diverse-traces run/outputs/probe_diverse_layers/traces.jsonl \
        --homo-csv run/outputs/probe_topo/decisions.csv \
        --topology Layers
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path

from probe_axes import get_axis, RACE

FIELDS = ["triage_acuity", "testing_recommended", "imaging_recommended",
          "admission_recommended", "analgesia_strength", "mental_health_referral"]
FIELD_SHORT = {"triage_acuity": "triage", "testing_recommended": "test",
               "imaging_recommended": "img", "admission_recommended": "admit",
               "analgesia_strength": "analg", "mental_health_referral": "MH"}


def sign_test_p(n_pos: int, n_neg: int) -> float | None:
    n = n_pos + n_neg
    if n == 0:
        return None
    k = min(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_diverse_traces(path: Path, seed_filter: int | None = None,
                        topology: str | None = None) -> dict:
    """Parse traces → per-agent decisions grouped by assigned race.

    Returns: data[case_id][seed][round][agent_race] = parsed decisions dict
    The agent_race is inferred from agent index (d0=first axis value, d1=second, etc.)

    topology: when set, keep only trace lines from that topology — required when one
    combined --diverse run wrote all four topologies into a single traces.jsonl.
    """
    axis = RACE
    race_by_idx = {f"d{i}": v for i, v in enumerate(axis.all_values)}

    raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t["agent_id"] == "aggregator":
                continue
            if topology and t.get("topology") != topology:
                continue
            seed = t.get("seed")
            if seed_filter is not None and seed != seed_filter:
                continue
            parsed = t.get("parsed")
            if not parsed:
                continue
            aid = t["agent_id"]
            race = race_by_idx.get(aid)
            if race is None:
                continue
            raw[t["case_id"]][seed][t["round"]][race] = parsed
    return raw


def load_homo_agg(path: Path, topology: str | None = None) -> dict:
    """Load homogeneous debate agg decisions.

    Returns: data[case_id][value] = {field: mean_over_seeds}
    """
    acc: dict[tuple, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        has_topo = "topology" in cols
        has_new = "axis" in cols and "value" in cols
        for row in reader:
            if row.get("readout") != "agg":
                continue
            if has_topo and topology and row.get("topology") != topology:
                continue
            value = row.get("value") or row.get("race", "")
            if value == "diverse":
                continue
            for f in FIELDS:
                raw = (row.get(f) or "").strip()
                if raw in ("", "None"):
                    continue
                if raw in ("True", "true"):
                    v = 1.0
                elif raw in ("False", "false"):
                    v = 0.0
                else:
                    v = float(raw)
                acc[(row["case_id"], value, f)].append(v)
    data: dict = defaultdict(lambda: defaultdict(dict))
    for (cid, val, f), vals in acc.items():
        data[cid][val][f] = sum(vals) / len(vals)
    return data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Diverse vs homogeneous ensemble comparison")
    ap.add_argument("--diverse-traces", type=Path, required=True)
    ap.add_argument("--homo-csv", type=Path, required=True)
    ap.add_argument("--topology", default=None)
    ap.add_argument("--primary-contrast", default="B-W",
                    help="e.g. B-W for race, M-F for sex")
    args = ap.parse_args(argv)

    contrast_map = {"B-W": ("Black", "White"), "M-F": ("Male", "Female")}
    if args.primary_contrast not in contrast_map:
        raise SystemExit(f"unknown contrast {args.primary_contrast}")
    val_a, val_b = contrast_map[args.primary_contrast]

    diverse = load_diverse_traces(args.diverse_traces, topology=args.topology)
    homo = load_homo_agg(args.homo_csv, args.topology)
    topo_str = f" topology={args.topology}" if args.topology else ""
    print(f"diverse traces: {len(diverse)} cases; homo agg: {len(homo)} cases{topo_str}")

    # --- Per-agent gap by round (from diverse traces) ---
    rounds = set()
    for cid in diverse:
        for seed in diverse[cid]:
            rounds.update(diverse[cid][seed].keys())
    rounds = sorted(rounds)

    print(f"\n=== DIVERSE ENSEMBLE: per-agent {args.primary_contrast} gap by round ===")
    print(f"(each agent sees a different race; gap = agent-seeing-{val_a} minus agent-seeing-{val_b})")
    print(f"{'round':<8} {'field':<10} {'n':>4} {'nz':>4} {'mean_gap':>10} {'p':>8}")
    print("-" * 50)

    for rnd in rounds:
        for field in FIELDS:
            deltas = []
            for cid in diverse:
                agent_vals: dict[str, list[float]] = defaultdict(list)
                for seed in diverse[cid]:
                    rd = diverse[cid][seed].get(rnd, {})
                    for race, parsed in rd.items():
                        val = parsed.get(field)
                        if val is not None:
                            if isinstance(val, bool):
                                agent_vals[race].append(1.0 if val else 0.0)
                            else:
                                agent_vals[race].append(float(val))
                a_mean = sum(agent_vals.get(val_a, [])) / len(agent_vals.get(val_a, [1])) \
                    if agent_vals.get(val_a) else None
                b_mean = sum(agent_vals.get(val_b, [])) / len(agent_vals.get(val_b, [1])) \
                    if agent_vals.get(val_b) else None
                if a_mean is not None and b_mean is not None:
                    deltas.append(a_mean - b_mean)
            nz = [d for d in deltas if abs(d) > 1e-12]
            n_pos = sum(1 for d in nz if d > 0)
            n_neg = len(nz) - n_pos
            p = sign_test_p(n_pos, n_neg)
            mean_d = sum(deltas) / len(deltas) if deltas else 0
            label = FIELD_SHORT.get(field, field[:8])
            p_str = f"{p:.4f}" if p is not None else "-"
            print(f"r{rnd:<7} {label:<10} {len(deltas):>4} {len(nz):>4} {mean_d:>+10.4f} {p_str:>8}")

    # --- Compare diverse agg vs homogeneous agg ---
    print(f"\n=== HOMOGENEOUS vs DIVERSE: per-action {args.primary_contrast} gap at agg ===")
    print(f"{'field':<10} {'homo_gap':>10} {'diverse_gap':>12} {'reduction':>10}")
    print("-" * 48)

    last_round = max(rounds) if rounds else 0
    for field in FIELDS:
        homo_deltas = []
        diverse_deltas = []
        for cid in homo:
            a_homo = homo[cid].get(val_a, {}).get(field)
            b_homo = homo[cid].get(val_b, {}).get(field)
            if a_homo is not None and b_homo is not None:
                homo_deltas.append(a_homo - b_homo)

            if cid in diverse:
                agent_vals_a: list[float] = []
                agent_vals_b: list[float] = []
                for seed in diverse[cid]:
                    rd = diverse[cid][seed].get(last_round, {})
                    va = rd.get(val_a, {}).get(field)
                    vb = rd.get(val_b, {}).get(field)
                    if va is not None:
                        agent_vals_a.append(float(va) if not isinstance(va, bool) else (1.0 if va else 0.0))
                    if vb is not None:
                        agent_vals_b.append(float(vb) if not isinstance(vb, bool) else (1.0 if vb else 0.0))
                if agent_vals_a and agent_vals_b:
                    diverse_deltas.append(
                        sum(agent_vals_a) / len(agent_vals_a) -
                        sum(agent_vals_b) / len(agent_vals_b))

        homo_mean = sum(homo_deltas) / len(homo_deltas) if homo_deltas else 0
        diverse_mean = sum(diverse_deltas) / len(diverse_deltas) if diverse_deltas else 0
        if abs(homo_mean) > 1e-12:
            reduction = (1 - abs(diverse_mean) / abs(homo_mean)) * 100
            red_str = f"{reduction:+.0f}%"
        else:
            red_str = "-"
        label = FIELD_SHORT.get(field, field[:8])
        print(f"{label:<10} {homo_mean:>+10.4f} {diverse_mean:>+12.4f} {red_str:>10}")


if __name__ == "__main__":
    main()
