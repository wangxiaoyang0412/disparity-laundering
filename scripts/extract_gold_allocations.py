"""Extract ground-truth triage acuity and disposition from MIMIC-IV-ED for pilot cases.

The OSCE JSONL produced by gen_mimic_local.py doesn't store subject_id.
This script re-derives the mapping by matching Correct_Diagnosis + demographics,
then joins to MIMIC-IV-ED triage/edstays for gold labels.

Run where the MIMIC-IV data lives:
    python scripts/extract_gold_allocations.py \
        --osce external/AgentClinic/agentclinic_mimiciv.jsonl \
        --mimic-hosp <MIMIC_IV_DIR>/hosp \
        --mimic-ed <MIMIC_IV_ED_DIR>/ed \
        --invariant run/pilot_mimic/equivalence_invariant.jsonl \
        --out run/pilot_mimic/gold_allocations.jsonl

Output per case: {case_id, subject_id, gold_acuity, gold_disposition, match_method}
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _read_dict(directory: Path, stem: str) -> list[dict]:
    for suffix in (".csv.gz", ".csv"):
        p = directory / f"{stem}{suffix}"
        if p.exists():
            opener = gzip.open if suffix.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    return []


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_age_gender(demographics: str) -> tuple[str, str]:
    m = re.match(r"(\d+)[-\s]year[-\s]old\s+(\w+)", demographics, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).lower()
    return "", ""


def build_source_index(hosp_dir: Path, ed_dir: Path | None) -> dict[str, dict]:
    """Build subject_id → {diagnosis, age, gender, acuity, disposition} from MIMIC tables."""
    patients = {r["subject_id"]: r for r in _read_dict(hosp_dir, "patients")}
    admits = _read_dict(hosp_dir, "admissions")
    icd = _read_dict(hosp_dir, "diagnoses_icd")
    dicd = {(r["icd_code"], r.get("icd_version", "")): r["long_title"]
            for r in _read_dict(hosp_dir, "d_icd_diagnoses")}

    subjects: dict[str, dict] = {}
    for a in admits:
        sid = a["subject_id"]
        if sid not in subjects:
            p = patients.get(sid, {})
            subjects[sid] = {
                "gender": p.get("gender", "").lower(),
                "age": str(p.get("anchor_age", "")),
                "diagnosis": None, "diag_seq": 10**9,
                "acuity": None, "disposition": None,
            }
    for d in icd:
        c = subjects.get(d["subject_id"])
        if c is None:
            continue
        title = dicd.get((d.get("icd_code", ""), d.get("icd_version", "")), "")
        if not title or "history" in title.lower():
            continue
        try:
            seq = int(d.get("seq_num", "999"))
        except ValueError:
            seq = 999
        if seq < c["diag_seq"]:
            c["diag_seq"], c["diagnosis"] = seq, title

    if ed_dir and ed_dir.exists():
        edstays = _read_dict(ed_dir, "edstays")
        triage = _read_dict(ed_dir, "triage")
        stay_to_acuity: dict[str, str] = {}
        for t in triage:
            stay_id = t.get("stay_id", "")
            acuity = t.get("acuity", "")
            if stay_id and acuity:
                stay_to_acuity[stay_id] = acuity
        subj_ed: dict[str, list[dict]] = defaultdict(list)
        for es in edstays:
            sid = es.get("subject_id", "")
            stay_id = es.get("stay_id", "")
            if sid:
                subj_ed[sid].append({
                    "stay_id": stay_id,
                    "disposition": (es.get("disposition") or "").strip(),
                    "acuity": stay_to_acuity.get(stay_id, ""),
                })
        for sid, visits in subj_ed.items():
            if sid in subjects:
                best = visits[0]
                subjects[sid]["acuity"] = best["acuity"]
                subjects[sid]["disposition"] = best["disposition"]

    return {sid: v for sid, v in subjects.items() if v["diagnosis"]}


def load_osce_cases(osce_path: Path) -> list[dict]:
    cases = []
    with osce_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            osce = obj.get("OSCE_Examination", obj)
            dx = str(osce.get("Correct_Diagnosis", "")).strip()
            demo = str(osce.get("Patient_Actor", {}).get("Demographics", "")).strip()
            age, gender = _extract_age_gender(demo)
            cases.append({
                "case_id": f"MIMICIV-{i}",
                "diagnosis": dx,
                "demographics": demo,
                "age": age,
                "gender": gender,
            })
    return cases


def match_cases(osce_cases: list[dict], source: dict[str, dict]) -> dict[str, dict]:
    dx_index: dict[str, list[str]] = defaultdict(list)
    for sid, v in source.items():
        key = _normalize(v["diagnosis"])
        dx_index[key].append(sid)

    matched: dict[str, dict] = {}
    unmatched = []
    for c in osce_cases:
        osce_dx = _normalize(c["diagnosis"])
        candidates = dx_index.get(osce_dx, [])
        best_sid = None
        method = "none"

        if len(candidates) == 1:
            best_sid = candidates[0]
            method = "exact_dx"
        elif len(candidates) > 1:
            for sid in candidates:
                s = source[sid]
                if s["age"] == c["age"] and s["gender"].startswith(c["gender"][:1]):
                    best_sid = sid
                    method = "dx+age+gender"
                    break
            if not best_sid:
                best_sid = candidates[0]
                method = "dx_first_match"

        if best_sid:
            s = source[best_sid]
            matched[c["case_id"]] = {
                "case_id": c["case_id"],
                "subject_id": best_sid,
                "gold_acuity": s["acuity"],
                "gold_disposition": s["disposition"],
                "match_method": method,
                "mimic_diagnosis": s["diagnosis"],
            }
        else:
            unmatched.append(c["case_id"])

    return matched


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract gold triage/disposition from MIMIC-IV-ED")
    ap.add_argument("--osce", type=Path, default=Path("external/AgentClinic/agentclinic_mimiciv.jsonl"))
    ap.add_argument("--mimic-hosp", type=Path, required=True)
    ap.add_argument("--mimic-ed", type=Path, default=None)
    ap.add_argument("--invariant", type=Path,
                    default=Path("run/pilot_mimic/equivalence_invariant.jsonl"),
                    help="only output gold for invariant cases (empty = all)")
    ap.add_argument("--out", type=Path, default=Path("run/pilot_mimic/gold_allocations.jsonl"))
    args = ap.parse_args(argv)

    if args.mimic_ed is None:
        args.mimic_ed = args.mimic_hosp.parent / "ed"

    inv_ids: set[str] | None = None
    if args.invariant and args.invariant.exists():
        inv_ids = set()
        with args.invariant.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    inv_ids.add(json.loads(line)["case_id"])
        print(f"invariant filter: {len(inv_ids)} cases")

    print("loading MIMIC source tables...")
    source = build_source_index(args.mimic_hosp, args.mimic_ed)
    print(f"source index: {len(source)} subjects with diagnosis")

    osce_cases = load_osce_cases(args.osce)
    print(f"OSCE cases: {len(osce_cases)}")

    matched = match_cases(osce_cases, source)
    print(f"matched: {len(matched)} / {len(osce_cases)}")

    if inv_ids:
        matched = {k: v for k, v in matched.items() if k in inv_ids}
        print(f"after invariant filter: {len(matched)}")

    has_acuity = sum(1 for v in matched.values() if v["gold_acuity"])
    has_disp = sum(1 for v in matched.values() if v["gold_disposition"])
    print(f"gold coverage: acuity={has_acuity}/{len(matched)}, "
          f"disposition={has_disp}/{len(matched)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for cid in sorted(matched, key=lambda x: int(x.split("-")[1])):
            fh.write(json.dumps(matched[cid], ensure_ascii=False) + "\n")
    print(f"wrote {len(matched)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
