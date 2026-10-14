"""Measurement-side code for fairmas, kept separate from the `mas` generation substrate.

- `extractor` — LLM-based allocation-decision extraction (JSON-block parsing utilities).
"""

from __future__ import annotations

from fairmas.analysis.extractor import extract_allocation, parse_json_block

__all__ = [
    "extract_allocation",
    "parse_json_block",
]
