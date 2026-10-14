"""The agent-communication topologies: MedSentry's four + a single-agent control.

A topology defines the routing: for each agent, which peers' previous-round messages it can
see, and who aggregates the team decision. The doctor team applies these against the prior
round's utterances (round 0 sees nothing).

MedSentry's error-contamination vulnerability ordering (their p.8) is
    SharedPool  >  Centralized  >  Layers  >  Decentralized
(most exposed → most contained); we keep the names identical for direct comparison.
"""

from __future__ import annotations

from typing import Callable

TOPOLOGY_FACTORY: dict[str, "Topology"] = {}


class Topology:
    """Base routing rule. Subclasses set `name` and override visibility/aggregation."""

    name: str = "base"

    def visible_peers(self, agent_idx: int, agent_ids: list[str]) -> tuple[str, ...]:
        """Peer ids whose previous-round messages `agent_idx` may see this round."""
        raise NotImplementedError

    def aggregator_idx(self, agent_ids: list[str]) -> int | None:
        """Index of the agent that decides the team consensus, or None for majority vote."""
        return None


def register_topology(name: str) -> Callable[[type[Topology]], type[Topology]]:
    """Register a Topology subclass under `name` (instantiated once, eagerly)."""

    def decorator(cls: type[Topology]) -> type[Topology]:
        cls.name = name
        TOPOLOGY_FACTORY[name] = cls()
        return cls

    return decorator


def TopologyFactory(name: str) -> Topology:
    """Return the registered topology instance, or raise with the known names."""
    if name not in TOPOLOGY_FACTORY:
        raise KeyError(f"Unknown topology {name!r}; registered: {sorted(TOPOLOGY_FACTORY)}")
    return TOPOLOGY_FACTORY[name]


@register_topology("single")
class SingleAgent(Topology):
    """Matched-compute control: one agent, no cross-talk (self-consistency handled upstream)."""

    def visible_peers(self, agent_idx: int, agent_ids: list[str]) -> tuple[str, ...]:
        return ()

    def aggregator_idx(self, agent_ids: list[str]) -> int | None:
        return 0


@register_topology("SharedPool")
class SharedPool(Topology):
    """Fully connected blackboard: every agent sees every other agent. Most exposed."""

    def visible_peers(self, agent_idx: int, agent_ids: list[str]) -> tuple[str, ...]:
        return tuple(a for i, a in enumerate(agent_ids) if i != agent_idx)


@register_topology("Centralized")
class Centralized(Topology):
    """Hub-and-spoke: hub (idx 0) sees all spokes; each spoke sees only the hub."""

    def visible_peers(self, agent_idx: int, agent_ids: list[str]) -> tuple[str, ...]:
        if agent_idx == 0:  # hub sees every spoke
            return tuple(agent_ids[1:])
        return (agent_ids[0],)  # spoke sees only the hub

    def aggregator_idx(self, agent_ids: list[str]) -> int | None:
        return 0  # the hub decides


@register_topology("Layers")
class Layers(Topology):
    """Sequential pipeline: agent i sees only its predecessor i-1 (layer-by-layer flow)."""

    def visible_peers(self, agent_idx: int, agent_ids: list[str]) -> tuple[str, ...]:
        if agent_idx == 0:
            return ()
        return (agent_ids[agent_idx - 1],)

    def aggregator_idx(self, agent_ids: list[str]) -> int | None:
        return len(agent_ids) - 1  # the last layer emits the team decision


@register_topology("Decentralized")
class Decentralized(Topology):
    """Sparse ring: each agent sees only its two ring neighbours. Most contained."""

    def visible_peers(self, agent_idx: int, agent_ids: list[str]) -> tuple[str, ...]:
        n = len(agent_ids)
        if n <= 1:
            return ()
        left = agent_ids[(agent_idx - 1) % n]
        right = agent_ids[(agent_idx + 1) % n]
        # n == 2 → both neighbours are the same single peer; dedup.
        return (left,) if left == right else (left, right)


__all__ = ["Topology", "TopologyFactory", "register_topology", "TOPOLOGY_FACTORY"]
