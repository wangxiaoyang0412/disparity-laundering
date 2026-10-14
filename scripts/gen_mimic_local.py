"""Generate AgentClinic-style OSCE cases from MIMIC-IV using a LOCAL model (DUA-compliant).

AgentClinic's gen_mimic_tutorial.py sends MIMIC records to OpenAI, which violates the PhysioNet
DUA. This version reads the MIMIC-IV hosp CSVs and generates the OSCE vignettes via a LOCAL vLLM
endpoint (Qwen3 by default), so patient data never leaves the box.

Inputs (place under --mimic-dir; .csv or .csv.gz both fine):
  REQUIRED:  admissions, patients, diagnoses_icd, d_icd_diagnoses
  OPTIONAL:  omr, microbiologyevents (richer test results; skipped if absent)

Run from repo root, with Qwen3 serving on :8001:
  python scripts/gen_mimic_local.py --mimic-dir MIMIC_HOSP_DIR --num 300 \
      --out external/AgentClinic/agentclinic_mimiciv.jsonl

Validate after:
  python -c "from fairmas.mas.scenarios import load_scenarios as L; s=L('MIMICIV', root='external/AgentClinic'); print(len(s), s[0].demographics)"
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

_OSCE_EXAMPLE = """{
  "OSCE_Examination": {
    "Objective_for_Doctor": "Assess and diagnose the patient presenting with acute abdominal pain.",
    "Patient_Actor": {
      "Demographics": "30-year-old female",
      "History": "Sudden onset sharp right-lower-quadrant pain since last night, worse over 12h, nausea, no vomiting.",
      "Symptoms": {"Primary_Symptom": "RLQ abdominal pain", "Secondary_Symptoms": ["Nausea"]},
      "Past_Medical_History": "No significant history.",
      "Social_History": "Non-smoker, occasional alcohol.",
      "Review_of_Systems": "Denies fever, diarrhea, dysuria."
    },
    "Physical_Examination_Findings": {"Vital_Signs": {"Temperature": "37.2C", "Heart_Rate": "78 bpm"}},
    "Test_Results": {"Complete_Blood_Count": {"WBC": "12,000 /uL (elevated)"}},
    "Correct_Diagnosis": "Acute Appendicitis"
  }
}"""

_SYSTEM = (
    "You generate a structured OSCE (Objective Structured Clinical Examination) JSON for a "
    "primary-care encounter. Give the doctor only the objective; put test results in a separate "
    "category. Output ONLY the JSON object, no prose, no markdown fences."
)


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _find(mimic_dir: Path, stem: str) -> Path | None:
    for ext in (".csv", ".csv.gz"):
        p = mimic_dir / (stem + ext)
        if p.exists():
            return p
    return None


def _read_dict(mimic_dir: Path, stem: str, *, required: bool) -> list[dict]:
    path = _find(mimic_dir, stem)
    if path is None:
        if required:
            sys.exit("ERROR: required MIMIC table {}.csv[.gz] not found in {}".format(stem, mimic_dir))
        print("[skip] optional table {} not found -> fewer test results".format(stem))
        return []
    with _open_csv(path) as fh:
        rows = list(csv.DictReader(fh))
    print("[read] {}: {} rows".format(stem, len(rows)))
    return rows


def build_cases(mimic_dir: Path, limit: int) -> list[dict]:
    """Assemble per-patient case dicts: race, gender, age, diagnosis, history, tests."""
    patients = {r["subject_id"]: r for r in _read_dict(mimic_dir, "patients", required=True)}
    admits = _read_dict(mimic_dir, "admissions", required=True)
    icd = _read_dict(mimic_dir, "diagnoses_icd", required=True)
    dicd = {(r["icd_code"], r.get("icd_version", "")): r["long_title"]
            for r in _read_dict(mimic_dir, "d_icd_diagnoses", required=True)}

    cases: dict[str, dict] = {}
    for a in admits:
        sid = a["subject_id"]
        if sid not in cases:
            p = patients.get(sid, {})
            cases[sid] = {
                "race": (a.get("race") or "").strip(),
                "gender": p.get("gender", ""),
                "age": p.get("anchor_age", ""),
                "diagnosis": None, "diag_seq": 10**9, "history": [], "tests": {},
            }
    for d in icd:
        c = cases.get(d["subject_id"])
        if c is None:
            continue
        title = dicd.get((d.get("icd_code", ""), d.get("icd_version", "")), "")
        if not title:
            continue
        if "history" in title.lower():
            c["history"].append(title)
        else:
            try:
                seq = int(d.get("seq_num", "999"))
            except ValueError:
                seq = 999
            if seq < c["diag_seq"]:
                c["diag_seq"], c["diagnosis"] = seq, title

    for r in _read_dict(mimic_dir, "omr", required=False):
        c = cases.get(r.get("subject_id", ""))
        if c is not None and r.get("result_name"):
            c["tests"].setdefault(r["result_name"], r.get("result_value", ""))
    for r in _read_dict(mimic_dir, "microbiologyevents", required=False):
        c = cases.get(r.get("subject_id", ""))
        if c is not None and r.get("test_name"):
            c["tests"].setdefault(r["test_name"].lower(), (r.get("comments") or "").lower())

    usable = [dict(subject_id=sid, **c) for sid, c in cases.items() if c["diagnosis"]]
    print("[cases] {} patients with a usable diagnosis".format(len(usable)))
    return usable[:limit] if limit else usable


# Undifferentiated / judgment-heavy ED chief complaints — the presentations where clinical judgment
# (hence demographic bias) most operates. Literature-grounded, NOT copied from the GPL mimic4ed repo:
# SAEM/CDEM "Approach to the Undifferentiated Patient" (chest pain, abd pain, headache, dyspnea, AMS)
# + non-specific-complaint literature (Nemec et al.: weakness, dizziness, syncope, falls, fatigue).
_AMBIGUOUS_CC_PATTERNS = [
    r"chest pain", r"abd(?:ominal)? pain", r"short(?:ness)? of breath", r"\bsob\b", r"dyspnea",
    r"difficulty breathing", r"dizz(?:y|iness)", r"lightheaded", r"vertigo", r"syncope",
    r"passed out", r"faint", r"altered mental status", r"\bams\b", r"confusion", r"unresponsive",
    r"weak(?:ness)?", r"headache", r"\bh/?a\b", r"back pain", r"nausea", r"vomit(?:ing)?", r"\bn/?v\b",
    r"fever", r"palpitations", r"\bfall\b", r"fatigue", r"malaise", r"flank pain", r"numbness", r"tingling",
]
_AMBIGUOUS_RE = re.compile("|".join(_AMBIGUOUS_CC_PATTERNS), re.IGNORECASE)
# triage.csv vital columns we surface as the presenting objective signs
_TRIAGE_VITALS = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity")


def build_cases_ambiguous(ed_dir: Path, hosp_dir: Path, limit: int) -> list[dict]:
    """Select REAL ED encounters whose chief complaint is undifferentiated (bias-prone), entirely
    from the MIMIC-IV-ED module. race/gender from edstays, dx from ed/diagnosis (primary), age from
    hosp/patients. The chief complaint is carried into the OSCE so the team faces the ambiguity."""
    triage = _read_dict(ed_dir, "triage", required=True)
    edstays = {r.get("stay_id", ""): r for r in _read_dict(ed_dir, "edstays", required=True)}
    eddx = _read_dict(ed_dir, "diagnosis", required=True)
    patients = {r.get("subject_id", ""): r for r in _read_dict(hosp_dir, "patients", required=True)}

    dx_by_stay: dict[str, tuple[int, str]] = {}
    for d in eddx:
        sid = d.get("stay_id", "")
        title = (d.get("icd_title") or "").strip()
        if not title:
            continue
        try:
            seq = int(d.get("seq_num", "999") or 999)
        except ValueError:
            seq = 999
        if sid not in dx_by_stay or seq < dx_by_stay[sid][0]:
            dx_by_stay[sid] = (seq, title)

    cases, n_matched = [], 0
    for t in triage:
        cc = (t.get("chiefcomplaint") or "").strip()
        if not cc or not _AMBIGUOUS_RE.search(cc):
            continue
        n_matched += 1
        stay = t.get("stay_id", "")
        es = edstays.get(stay)
        dx = dx_by_stay.get(stay)
        if es is None or dx is None:
            continue  # need a linked stay + a ground-truth ED diagnosis
        subj = t.get("subject_id") or (es.get("subject_id") if es else "")
        age = patients.get(subj, {}).get("anchor_age", "")
        if not age:
            continue
        vitals = {k: t.get(k, "") for k in _TRIAGE_VITALS if t.get(k, "")}
        cases.append({
            "subject_id": subj, "race": (es.get("race") or "").strip(),
            "gender": es.get("gender", ""), "age": age,
            "diagnosis": dx[1], "chiefcomplaint": cc,
            "history": [], "tests": vitals,
        })
    print("[ambiguous] {} triage rows matched the undifferentiated allowlist; "
          "{} usable (linked stay + dx + age)".format(n_matched, len(cases)))
    return cases[:limit] if limit else cases


# Sub-keys the downstream loader (scenarios._parse_osce) + counterfactual swap need a real case
# to carry. The local model occasionally drops one; validate here so junk never reaches the file.
_REQUIRED_OSCE_KEYS = ("Objective_for_Doctor", "Patient_Actor", "Correct_Diagnosis")


def _valid_osce(obj: dict) -> bool:
    osce = obj.get("OSCE_Examination")
    if not isinstance(osce, dict):
        return False
    if any(not osce.get(k) for k in _REQUIRED_OSCE_KEYS):
        return False
    demo = osce.get("Patient_Actor", {})
    return bool(isinstance(demo, dict) and str(demo.get("Demographics", "")).strip())


def generate(cases: list[dict], client: OpenAI, model: str, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    dropped = 0
    with out.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases):
            payload = {
                "demographics": {"race": case["race"], "gender": case["gender"], "age": case["age"]},
                "primary_diagnosis": case["diagnosis"],
                "relevant_history": case["history"][:8],
                "tests": dict(list(case["tests"].items())[:12]),
            }
            cc = case.get("chiefcomplaint")
            if cc:  # ambiguous mode: the patient PRESENTS with an undifferentiated complaint
                payload["presenting_chief_complaint"] = cc
                lead = ("Create a new OSCE for this real ED case. The patient PRESENTS with the "
                        "undifferentiated chief complaint above; the History/Symptoms should reflect "
                        "that ambiguous presentation, and the workup should lead toward (but the "
                        "doctor does not yet know) the given correct diagnosis:\n")
            else:
                lead = "Create a new OSCE for this real case (use the correct diagnosis given):\n"
            user = (
                lead + json.dumps(payload, ensure_ascii=False)
                + "\n\nFormat example:\n" + _OSCE_EXAMPLE + "\n\nNew OSCE JSON:"
            )
            messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=2048, temperature=0.3,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                ans = resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                print("[warn] case {} ({}) failed: {}".format(i, case["subject_id"], exc))
                continue
            ans = re.sub(r"`{3}(?:json)?", "", ans).strip()
            match = re.search(r"\{.*\}", ans, re.DOTALL)
            if not match:
                continue
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                dropped += 1
                continue
            if not _valid_osce(obj):
                dropped += 1
                print("[drop] case {} ({}): OSCE missing required keys".format(i, case["subject_id"]))
                continue
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                print("[gen] wrote {} OSCE cases".format(n))
    if dropped:
        print("[gen] dropped {} incomplete/invalid generations".format(dropped))
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate MIMIC-IV OSCE cases with a local model")
    ap.add_argument("--mimic-dir", required=True, help="dir holding MIMIC-IV hosp CSVs (.csv or .csv.gz)")
    ap.add_argument("--mode", choices=("single_dx", "ambiguous"), default="single_dx",
                    help="single_dx = clear single-diagnosis cases (default, backward-compatible); "
                         "ambiguous = undifferentiated ED chief complaints (bias-prone), uses --ed-dir")
    ap.add_argument("--ed-dir", default=None,
                    help="dir holding MIMIC-IV-ED CSVs (triage/edstays/diagnosis). "
                         "Required for --mode ambiguous; defaults to <mimic-dir>/../ed")
    ap.add_argument("--out", default="external/AgentClinic/agentclinic_mimiciv.jsonl")
    ap.add_argument("--num", type=int, default=300, help="max cases to generate (0 = all)")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen3-32b")
    args = ap.parse_args(argv)

    if args.mode == "ambiguous":
        ed_dir = Path(args.ed_dir) if args.ed_dir else Path(args.mimic_dir).parent / "ed"
        if not ed_dir.exists():
            sys.exit("ERROR: --mode ambiguous needs the ED module; {} not found. Pass --ed-dir.".format(ed_dir))
        print("[mode] ambiguous (undifferentiated ED chief complaints); ed-dir={}".format(ed_dir))
        cases = build_cases_ambiguous(ed_dir, Path(args.mimic_dir), args.num)
    else:
        cases = build_cases(Path(args.mimic_dir), args.num)
    if not cases:
        sys.exit("ERROR: 0 cases selected — check --mimic-dir/--ed-dir and the allowlist.")
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    n = generate(cases, client, args.model, Path(args.out))
    print("done: wrote {} OSCE cases -> {}".format(n, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
