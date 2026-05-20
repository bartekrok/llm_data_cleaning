"""Non-LLM baseline: rapidfuzz string similarity.

Strategy:
- Score the input against each scope value with multiple metrics
  (ratio, token_sort_ratio, partial_ratio); keep the max.
- If score >= accept_threshold and the input is meaningfully short
  (not a paragraph), state="acceptance" and value=best_scope_item.
- If accept_threshold > score >= suggest_threshold, state="suggest".
  In the suggest case we return the cleaned input as the proposed
  new scope value.
- Otherwise state="decline".

Defaults are tuned conservatively; experiments can sweep them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from rapidfuzz import fuzz, process

from llm_cleaner.clients.base import Cleaner, CleanResult


@dataclass
class FuzzyThresholds:
    accept: float = 85.0
    suggest: float = 60.0
    max_input_len: int = 80


class FuzzyCleaner(Cleaner):
    """Deterministic, local. No network, no cost, no variance."""

    def __init__(
        self,
        thresholds: Optional[FuzzyThresholds] = None,
        *,
        name: str = "baseline/fuzzy",
    ):
        self.name = name
        self.thresholds = thresholds or FuzzyThresholds()

    def _score(self, value: str, scope_item: str) -> float:
        v = value.strip()
        s = scope_item.strip()
        return max(
            fuzz.ratio(v.lower(), s.lower()),
            fuzz.token_sort_ratio(v.lower(), s.lower()),
            fuzz.partial_ratio(v.lower(), s.lower()),
        )

    def _classify(self, value: str, scope: Sequence[str]) -> tuple[str, Optional[str], str, float]:
        if value is None or not value.strip():
            return "decline", "", "Empty or whitespace-only input.", 0.0

        if len(value) > self.thresholds.max_input_len:
            return (
                "decline",
                "",
                f"Input length {len(value)} exceeds max_input_len {self.thresholds.max_input_len}; treated as noise.",
                0.0,
            )

        choices = list(scope)
        best = process.extractOne(value, choices, scorer=fuzz.ratio)
        best_item, best_score = (best[0], best[1]) if best else (None, 0.0)
        if best_item is not None:
            best_score = max(best_score, self._score(value, best_item))

        if best_item is not None and best_score >= self.thresholds.accept:
            return (
                "acceptance",
                best_item,
                f"Fuzzy match to scope item '{best_item}' (score={best_score:.1f}).",
                best_score,
            )
        if best_score >= self.thresholds.suggest:
            return (
                "suggest",
                value.strip(),
                f"Closest scope match was '{best_item}' (score={best_score:.1f}); "
                "below accept threshold but above suggest threshold.",
                best_score,
            )
        return (
            "decline",
            "",
            f"No scope item scored above suggest threshold (best={best_score:.1f}).",
            best_score,
        )

    def clean(
        self,
        value: str,
        scope: Sequence[str],
        *,
        prompt_variant: str,
        prompt_messages: Any = None,
        response_format: Any = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> CleanResult:
        t0 = time.perf_counter()
        state, std_value, message, score = self._classify(value, scope)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        confidence = min(1.0, max(0.0, score / 100.0))
        return CleanResult(
            state=state,
            value=std_value,
            message=message,
            confidence=confidence,
            latency_ms=latency_ms,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            provider="local",
            response_format_used="n/a",
            seed=seed,
            raw_response={"score": score, "thresholds": self.thresholds.__dict__},
        )

    def clean_batch(
        self,
        values: Sequence[str],
        scope: Sequence[str],
        *,
        prompt_variant: str,
        prompt_messages: Any = None,
        response_format: Any = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[CleanResult]:
        return [
            self.clean(
                v,
                scope,
                prompt_variant=prompt_variant,
                temperature=temperature,
                seed=seed,
            )
            for v in values
        ]
