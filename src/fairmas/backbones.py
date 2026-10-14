"""Backbone abstraction: an OpenAI-compatible chat client over a vLLM endpoint or an API.

One `BackboneConfig` = one served model (local vLLM or proprietary API). The same code
path serves both because vLLM exposes an OpenAI-compatible `/v1/chat/completions`.
Configs are loaded from a YAML registry (name -> backbone spec).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from fairmas.utils import get_logger

logger = get_logger(__name__)

_ENV_PREFIX = "${env:"


@dataclass(frozen=True)
class BackboneConfig:
    """Immutable description of one chat backbone."""

    name: str
    served_model: str
    base_url: str | None = None  # None -> default OpenAI endpoint
    api_key: str = "EMPTY"
    family: str = "unknown"  # gemma / qwen / llama / openai — used by homophily probe
    max_tokens: int = 1024
    temperature: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResult:
    """A single chat completion plus bookkeeping for token/latency accounting."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str


def _resolve_env(value: str) -> str:
    """Expand a ``${env:VAR}`` placeholder against the environment."""
    if isinstance(value, str) and value.startswith(_ENV_PREFIX) and value.endswith("}"):
        var = value[len(_ENV_PREFIX) : -1]
        resolved = os.environ.get(var)
        if not resolved:
            raise KeyError(f"Environment variable {var!r} required by backbone config is unset")
        return resolved
    return value


def load_backbones(config_path: str | Path) -> dict[str, BackboneConfig]:
    """Load the backbone registry from YAML into a name -> BackboneConfig mapping."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Backbone config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    registry: dict[str, BackboneConfig] = {}
    for name, spec in raw.items():
        spec = dict(spec)
        registry[name] = BackboneConfig(
            name=name,
            served_model=spec["served_model"],
            base_url=spec.get("base_url"),
            api_key=_resolve_env(spec.get("api_key", "EMPTY")),
            family=spec.get("family", "unknown"),
            max_tokens=int(spec.get("max_tokens", 1024)),
            temperature=float(spec.get("temperature", 0.2)),
            extra_body=spec.get("extra_body") or {},
        )
    logger.info("Loaded %d backbone(s): %s", len(registry), ", ".join(registry))
    return registry


class ChatBackbone:
    """Thin OpenAI-compatible chat wrapper with retry and usage accounting."""

    def __init__(self, cfg: BackboneConfig) -> None:
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True)
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Send a chat request and return text + token/latency accounting."""
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.cfg.served_model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
            extra_body=self.cfg.extra_body or None,
        )
        latency = time.perf_counter() - start
        usage = resp.usage
        return ChatResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_s=latency,
            model=self.cfg.served_model,
        )


__all__ = ["BackboneConfig", "ChatResult", "ChatBackbone", "load_backbones"]
