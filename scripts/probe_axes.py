"""Axis configuration for demographic probes.

Each AxisConfig defines the variant set, display headers, variant-generation
function, and analysis contrasts for one demographic axis.  Probe scripts
import ``get_axis()`` and use it generically — adding a future axis requires
only a new AxisConfig here, no probe-logic changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

AGE_SEX_RE = re.compile(
    r"\b(\d{1,3}[-\s]year[-\s]old)\s*[/,]?\s+"
    r"(female|male|woman|man|lady|gentleman|girl|boy)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AxisConfig:
    name: str
    values: tuple[str, ...]
    control: str
    headers: dict[str, str]
    contrasts: tuple[str, ...]
    contrast_pairs: dict[str, tuple[str, str]]
    _make_variant: Callable = field(repr=False)

    @property
    def all_values(self) -> list[str]:
        return list(self.values) + [self.control]

    def make_variant(self, scenario_text: str, value: str) -> str:
        return self._make_variant(scenario_text, value)

    def case_delta(self, by_value: dict, contrast: str, field_name: str) -> float | None:
        def get(val: str) -> float | None:
            return by_value.get(val, {}).get(field_name)

        if contrast == "presence":
            vals = [get(v) for v in self.values]
            c = get(self.control)
            if c is None or any(v is None for v in vals):
                return None
            return sum(vals) / len(vals) - c
        pair = self.contrast_pairs.get(contrast)
        if pair is None:
            return None
        a, b = get(pair[0]), get(pair[1])
        if a is None or b is None:
            return None
        return a - b


def _make_race_variant(scenario_text: str, value: str) -> str:
    if value == "control":
        return scenario_text
    new_text, n = AGE_SEX_RE.subn(rf"\1 {value} \2", scenario_text, count=1)
    if n == 0:
        logger.warning("age-sex pattern not found; prepending (race=%s)", value)
        return f"Patient demographics: {value}.\n\n{scenario_text}"
    return new_text


def _make_sex_variant(scenario_text: str, value: str) -> str:
    if value == "control":
        new_text, n = AGE_SEX_RE.subn(r"\1 patient", scenario_text, count=1)
        if n == 0:
            return scenario_text
        return new_text
    new_text, n = AGE_SEX_RE.subn(rf"\1 {value.lower()}", scenario_text, count=1)
    if n == 0:
        logger.warning("age-sex pattern not found; prepending (sex=%s)", value)
        return f"Patient sex: {value}.\n\n{scenario_text}"
    return new_text


RACE = AxisConfig(
    name="race",
    values=("White", "Black", "Hispanic", "Asian"),
    control="control",
    headers={"White": "White", "Black": "Black", "Hispanic": "Hisp.",
             "Asian": "Asian", "control": "Ctrl"},
    contrasts=("presence", "B-W", "H-W", "A-W"),
    contrast_pairs={"B-W": ("Black", "White"),
                    "H-W": ("Hispanic", "White"),
                    "A-W": ("Asian", "White")},
    _make_variant=_make_race_variant,
)

SEX = AxisConfig(
    name="sex",
    values=("Male", "Female"),
    control="control",
    headers={"Male": "Male", "Female": "Female", "control": "Ctrl"},
    contrasts=("presence", "M-F"),
    contrast_pairs={"M-F": ("Male", "Female")},
    _make_variant=_make_sex_variant,
)

_REGISTRY: dict[str, AxisConfig] = {"race": RACE, "sex": SEX}


def get_axis(name: str) -> AxisConfig:
    ax = _REGISTRY.get(name)
    if ax is None:
        raise ValueError(f"unknown axis {name!r}; available: {sorted(_REGISTRY)}")
    return ax


def detect_axis_from_csv(fieldnames: list[str]) -> AxisConfig:
    if "axis" in fieldnames:
        return RACE  # caller should read row["axis"]; this is a fallback default
    if "race" in fieldnames:
        return RACE
    if "sex" in fieldnames:
        return SEX
    return RACE


def read_value_column(row: dict, axis: AxisConfig) -> str:
    if "value" in row and row["value"]:
        return row["value"]
    return row.get(axis.name, row.get("race", ""))
