"""Experiment construction for fairmas.

- `equivalence` — clinical-equivalence filter: keep only cases whose correct management is
  demographic-invariant; quarantine the rest.
"""

from __future__ import annotations

from fairmas.experiment.equivalence import (
    EquivalenceVerdict,
    filter_scenarios,
    judge_invariance,
)

__all__ = [
    "EquivalenceVerdict",
    "judge_invariance",
    "filter_scenarios",
]
