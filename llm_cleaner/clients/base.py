"""Common contract every cleaner (LLM or baseline) must satisfy.

A cleaner takes a value (or batch of values) plus the allowed scope and
returns one CleanResult per input value. The result mirrors the JSON
contract used by the LLMs:

    state   : "acceptance" | "decline" | "suggest"
    value   : the standardized name (or "" on decline)
    message : free-form explanation
    confidence (optional, P_with_confidence prompt only)

Adapters record what they actually saw: latency, tokens used, raw API
response, provider, and any failure information. The harness uses
those fields to populate the `runs` table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable


VALID_STATES = ("acceptance", "decline", "suggest")


class CleanerError(Exception):
    """Raised when a cleaner fails in a way the harness should record but not re-raise."""


@dataclass
class CleanResult:
    state: Optional[str]
    value: Optional[str]
    message: Optional[str]
    confidence: Optional[float] = None

    latency_ms: Optional[int] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    provider: Optional[str] = None
    response_format_used: Optional[str] = None
    seed: Optional[int] = None

    raw_response: Optional[dict[str, Any]] = None

    failure_mode: Optional[str] = None
    parse_error: Optional[str] = None

    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Cleaner(Protocol):
    """Anything that can map (value, scope) -> CleanResult."""

    name: str

    def clean(
        self,
        value: str,
        scope: Sequence[str],
        *,
        prompt_variant: str,
        prompt_messages: Optional[list[dict[str, Any]]] = None,
        response_format: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> CleanResult: ...

    def clean_batch(
        self,
        values: Sequence[str],
        scope: Sequence[str],
        *,
        prompt_variant: str,
        prompt_messages: Optional[list[dict[str, Any]]] = None,
        response_format: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[CleanResult]: ...


def normalize_state(state: Any) -> Optional[str]:
    """Map common state aliases (case, synonyms) to the canonical form."""
    if not isinstance(state, str):
        return None
    s = state.strip().lower()
    if s in {"acceptance", "accept", "accepted", "approve", "approved"}:
        return "acceptance"
    if s in {"decline", "declined", "reject", "rejected", "deny", "denied"}:
        return "decline"
    if s in {"suggest", "suggestion", "suggested", "add", "propose"}:
        return "suggest"
    return None


def coerce_confidence(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        c = float(raw)
        if c < 0 or c > 1:
            return None
        return c
    except (TypeError, ValueError):
        return None


def iter_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
