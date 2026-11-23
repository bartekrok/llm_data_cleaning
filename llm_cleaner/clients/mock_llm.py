"""Deterministic stand-in for OpenRouter during e2e / preflight.

Activated when ``LLM_MOCK=1`` (or true/yes). Uses the same fuzzy
classifier as the local baseline so parse → classify → insert paths
exercise real harness code without spending API money or needing a key.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from llm_cleaner.clients.base import CleanResult
from llm_cleaner.clients.fuzzy import FuzzyCleaner
from llm_cleaner.prompts import P_WITH_CONFIDENCE


class MockOpenRouterCleaner:
    """Drop-in OpenRouterCleaner replacement for offline e2e."""

    def __init__(self, model_id: str):
        self.name = model_id
        self.model_id = model_id
        self._fuzzy = FuzzyCleaner(name=f"mock/{model_id}")

    def _decorate(
        self,
        result: CleanResult,
        *,
        prompt_variant: str,
        seed: Optional[int],
        batch_index: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> CleanResult:
        conf = result.confidence
        if prompt_variant == P_WITH_CONFIDENCE and conf is None:
            conf = 0.85 if result.state else 0.1
        raw = {
            "mock": True,
            "model": self.model_id,
            "provider": "mock-openrouter",
            "choices": [
                {
                    "message": {
                        "content": (
                            f'{{"state":"{result.state}","value":{repr(result.value or "")},'
                            f'"message":{repr(result.message or "")}'
                            + (f',"confidence":{conf}' if conf is not None else "")
                            + "}"
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "cost": 0.0},
        }
        if batch_index is not None:
            raw["batch_index"] = batch_index
            raw["batch_size"] = batch_size
        return CleanResult(
            state=result.state,
            value=result.value,
            message=result.message,
            confidence=conf,
            latency_ms=max(result.latency_ms or 0, 1),
            tokens_in=12,
            tokens_out=8,
            cost_usd=0.0,
            provider="mock-openrouter",
            response_format_used="json_object",
            seed=seed,
            raw_response=raw,
            failure_mode=result.failure_mode,
            parse_error=result.parse_error,
            extras=dict(result.extras or {}),
        )

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
    ) -> CleanResult:
        t0 = time.perf_counter()
        base = self._fuzzy.clean(
            value,
            scope,
            prompt_variant=prompt_variant,
            temperature=temperature,
            seed=seed,
        )
        # Preserve measured fuzzy latency but keep mock metadata.
        out = self._decorate(base, prompt_variant=prompt_variant, seed=seed)
        out.latency_ms = max(int((time.perf_counter() - t0) * 1000), 1)
        return out

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
    ) -> list[CleanResult]:
        results: list[CleanResult] = []
        for idx, value in enumerate(values):
            base = self._fuzzy.clean(
                value,
                scope,
                prompt_variant=prompt_variant,
                temperature=temperature,
                seed=seed,
            )
            decorated = self._decorate(
                base,
                prompt_variant=prompt_variant,
                seed=seed,
                batch_index=idx,
                batch_size=len(values),
            )
            decorated.extras = {"batch_index": idx, "batch_size": len(values)}
            results.append(decorated)
        return results


def mock_mode_enabled() -> bool:
    import os

    return os.getenv("LLM_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


class MockEmbeddingCleaner(FuzzyCleaner):
    """Fuzzy stand-in for sentence-transformers during mock e2e (no model download)."""

    def __init__(self) -> None:
        super().__init__(name="baseline/embedding")
