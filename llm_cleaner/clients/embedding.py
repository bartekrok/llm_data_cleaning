"""Non-LLM baseline: sentence-transformers cosine similarity.

The model is loaded lazily on first .clean() call so importing this
module doesn't pay the cost. We cache scope embeddings keyed by the
tuple of scope values to avoid recomputing for the same scope.

Thresholds operate on cosine similarity in [0, 1].
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from llm_cleaner.clients.base import Cleaner, CleanResult


@dataclass
class EmbeddingThresholds:
    accept: float = 0.80
    suggest: float = 0.55
    max_input_len: int = 200


class EmbeddingCleaner(Cleaner):
    """Lazy-loaded sentence-transformers cleaner."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        thresholds: Optional[EmbeddingThresholds] = None,
        *,
        name: str = "baseline/embedding",
    ):
        self.name = name
        self.model_name = model_name
        self.thresholds = thresholds or EmbeddingThresholds()
        self._model = None
        self._scope_cache: dict[tuple[str, ...], Any] = {}

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def _scope_embeddings(self, scope: Sequence[str]):
        key = tuple(scope)
        cached = self._scope_cache.get(key)
        if cached is not None:
            return cached
        self._ensure_model()
        emb = self._model.encode(list(scope), normalize_embeddings=True, convert_to_numpy=True)
        self._scope_cache[key] = emb
        return emb

    def _classify(self, value: str, scope: Sequence[str]) -> tuple[str, Optional[str], str, float]:
        import numpy as np

        if value is None or not value.strip():
            return "decline", "", "Empty or whitespace-only input.", 0.0

        if len(value) > self.thresholds.max_input_len:
            return (
                "decline",
                "",
                f"Input length {len(value)} exceeds max_input_len {self.thresholds.max_input_len}; treated as noise.",
                0.0,
            )

        self._ensure_model()
        scope_emb = self._scope_embeddings(scope)
        val_emb = self._model.encode([value], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = scope_emb @ val_emb
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_item = scope[best_idx]

        if best_sim >= self.thresholds.accept:
            return (
                "acceptance",
                best_item,
                f"Embedding cosine similarity to '{best_item}' is {best_sim:.3f}.",
                best_sim,
            )
        if best_sim >= self.thresholds.suggest:
            return (
                "suggest",
                value.strip(),
                f"Closest scope item '{best_item}' has similarity {best_sim:.3f}; "
                "below accept threshold but above suggest threshold.",
                best_sim,
            )
        return (
            "decline",
            "",
            f"No scope item exceeded suggest threshold (best similarity={best_sim:.3f}).",
            best_sim,
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
        state, std_value, message, sim = self._classify(value, scope)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return CleanResult(
            state=state,
            value=std_value,
            message=message,
            confidence=min(1.0, max(0.0, sim)),
            latency_ms=latency_ms,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            provider="local",
            response_format_used="n/a",
            seed=seed,
            raw_response={
                "similarity": sim,
                "thresholds": self.thresholds.__dict__,
                "model": self.model_name,
            },
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
