"""Multi-agent substrate for fairmas.

Building blocks used by the probe scripts:

- `topology`  — the four debate topologies + a matched-compute single-agent control.
- `scenarios` — AgentClinic-style OSCE cases + the counterfactual demographic swap.
- `actions`   — allocation-decision extraction + control-token parsing.
"""

from __future__ import annotations

from fairmas.mas.scenarios import Scenario, load_scenarios, with_demographics
from fairmas.mas.topology import TopologyFactory, register_topology

__all__ = [
    "TopologyFactory",
    "register_topology",
    "Scenario",
    "load_scenarios",
    "with_demographics",
]
