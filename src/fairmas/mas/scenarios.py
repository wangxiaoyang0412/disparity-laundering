"""AgentClinic OSCE clinical cases, parsed directly from their JSONL.

We read the case files rather than importing AgentClinic's module, which pulls in
heavyweight third-party SDKs and loads cases by bare relative path. The case directory
is resolved from an explicit `root`, then the `FAIRMAS_AGENTCLINIC` env var, then
`<cwd>/external/AgentClinic`; the driver passes an absolute root so loading is CWD-independent.

The patient demographic descriptor lives only in `Patient_Actor.Demographics`
(e.g. "35-year-old female"); `with_demographics` rewrites that one field and leaves every
clinical fact untouched.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Race/ethnicity tokens to strip from MIMIC cases before the counterfactual swap (real cases
# embed the patient's true race, e.g. "52-year-old female, White"). Longer phrases first so
# "African American" is removed before "American".
_RACE_TOKENS = [
    "african american", "black or african american", "native hawaiian or pacific islander",
    "american indian or alaska native", "american indian", "alaska native", "pacific islander",
    "native hawaiian", "middle eastern or north african", "middle eastern", "north african",
    "hispanic or latino", "south asian", "east asian", "native american", "caucasian", "white",
    "black", "hispanic", "latino", "latina", "latinx", "asian", "mena", "multiracial", "biracial",
]
_RACE_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in _RACE_TOKENS) + r")\b", re.IGNORECASE)

# OSCE-schema datasets (MedQA family). NEJM uses a different schema — added when needed.
DATASET_FILES: dict[str, str] = {
    "MedQA": "agentclinic_medqa.jsonl",
    "MedQA_Ext": "agentclinic_medqa_extended.jsonl",
    "MIMICIV": "agentclinic_mimiciv.jsonl",  # requires PhysioNet-credentialed local file
}


def _agentclinic_dir(root: str | Path | None = None) -> Path:
    """Resolve the AgentClinic case directory; never navigates above the given root."""
    if root is not None:
        return Path(root)
    env = os.environ.get("FAIRMAS_AGENTCLINIC")
    if env:
        return Path(env)
    return Path.cwd() / "external" / "AgentClinic"


@dataclass(frozen=True)
class Scenario:
    """One OSCE clinical case. Dicts mirror AgentClinic's `OSCE_Examination` block."""

    case_id: str
    objective_for_doctor: str
    patient_actor: dict[str, Any]
    physical_exams: dict[str, Any]
    test_results: dict[str, Any]
    correct_diagnosis: str

    @property
    def demographics(self) -> str:
        """The single demographic descriptor — the counterfactual-swap target."""
        return str(self.patient_actor.get("Demographics", ""))

    def patient_information(self) -> dict[str, Any]:
        return self.patient_actor

    def examiner_information(self) -> str:
        return self.objective_for_doctor

    def exam_information(self) -> dict[str, Any]:
        exams = dict(self.physical_exams)
        exams["tests"] = self.test_results
        return exams

    def diagnosis_information(self) -> str:
        return self.correct_diagnosis


def _parse_osce(case_id: str, record: dict[str, Any]) -> Scenario:
    """Parse one OSCE record. Tolerant to missing non-essential sub-keys: LLM-generated
    MIMIC cases sometimes omit a field (e.g. Correct_Diagnosis). Only a missing
    `OSCE_Examination` raises (KeyError) -> the caller skips that record. Allocation
    outcomes (analgesia/imaging/admission) do not depend on the ground-truth diagnosis.
    """
    osce = record["OSCE_Examination"]
    return Scenario(
        case_id=case_id,
        objective_for_doctor=str(osce.get("Objective_for_Doctor", "")),
        patient_actor=dict(osce.get("Patient_Actor", {})),
        physical_exams=dict(osce.get("Physical_Examination_Findings", {})),
        test_results=dict(osce.get("Test_Results", {})),
        correct_diagnosis=str(osce.get("Correct_Diagnosis", "")),
    )


def load_scenarios(
    dataset: str = "MedQA", root: str | Path | None = None, *, scrub_race: bool | None = None
) -> list[Scenario]:
    """Load all OSCE scenarios for a dataset. Raises if dataset name or file is missing.

    `scrub_race` removes the patient's real race before any counterfactual swap; default None
    means auto (on for MIMICIV, which embeds real race; off for the synthetic MedQA cases).
    """
    if dataset not in DATASET_FILES:
        raise KeyError(f"Unknown dataset {dataset!r}; known: {sorted(DATASET_FILES)}")
    if scrub_race is None:
        scrub_race = dataset == "MIMICIV"
    path = _agentclinic_dir(root) / DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}")
    scenarios: list[Scenario] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                scen = _parse_osce(f"{dataset}-{i}", json.loads(line))
            except (KeyError, json.JSONDecodeError) as exc:
                skipped += 1
                logger.warning("skip malformed %s record %d: %s", dataset, i, exc)
                continue
            # Counterfactual swap needs a demographic anchor (age+sex live in Demographics);
            # a record without it cannot produce race variants -> drop it.
            if not str(scen.patient_actor.get("Demographics", "")).strip():
                skipped += 1
                logger.warning("skip %s record %d: empty Demographics (no counterfactual anchor)", dataset, i)
                continue
            scenarios.append(scrub_race_scenario(scen) if scrub_race else scen)
    if skipped:
        logger.warning("%s: loaded %d scenarios, skipped %d malformed/empty", dataset, len(scenarios), skipped)
    return scenarios


def _scrub_text(text: str) -> str:
    """Remove race tokens from a free-text field and tidy whitespace/punctuation."""
    out = _RACE_RE.sub("", text)
    out = re.sub(r"\s*,\s*,", ",", out)        # collapse double commas left by removal
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out.strip(" ,")


def scrub_race_scenario(scenario: Scenario) -> Scenario:
    """Strip the patient's real race from Demographics + narrative (for MIMIC counterfactuals).

    "52-year-old female, White" -> "52-year-old female"; race words removed from History etc.
    Leaves age/sex and all clinical facts intact so the swap is the only demographic change.
    """
    pa = dict(scenario.patient_actor)
    demo = str(pa.get("Demographics", ""))
    demo = demo.split(",")[0]              # drop a trailing ", White"
    pa["Demographics"] = _scrub_text(demo)  # also remove an inline race word
    for key, value in list(pa.items()):
        if key != "Demographics" and isinstance(value, str):
            pa[key] = _scrub_text(value)
    return replace(scenario, patient_actor=pa,
                   objective_for_doctor=_scrub_text(scenario.objective_for_doctor))


def with_demographics(scenario: Scenario, new_demographics: str) -> Scenario:
    """Return a copy with ONLY `Patient_Actor.Demographics` rewritten (counterfactual swap).

    Every clinical fact (History/Symptoms/labs/exams/diagnosis) is held constant — the
    invariance that makes a disparity attributable to demographics, not to the case.
    """
    swapped = dict(scenario.patient_actor)
    swapped["Demographics"] = new_demographics
    return replace(scenario, patient_actor=swapped)


__all__ = ["Scenario", "DATASET_FILES", "load_scenarios", "with_demographics", "scrub_race_scenario"]
