"""Shared model lists for the experiment registry.

Centralizing the names here means swapping a model only requires editing
ONE place. The set is chosen for *type diversity* and cost, not a
frontier leaderboard:

  - dedicated reasoner (closed)
  - dedicated reasoner (open-weight / open ecosystem)
  - mid-tier thinking / flash (closed)
  - classic cheap chat without reasoning API
  - open instruct (non-reasoning)
  - one stronger closed generalist for product realism

Model identifiers follow OpenRouter's <vendor>/<model> convention.
Override at runtime with ``LLM_MODELS=a,b,c``.
"""

from __future__ import annotations

# Diverse, relatively cheap set (text-in → text-out only).
MANDATORY = [
    "openai/o4-mini",  # closed dedicated reasoner
    "deepseek/deepseek-r1",  # open reasoner
    "google/gemini-2.5-flash",  # mid-tier with thinking support
    "openai/gpt-4o-mini",  # classic cheap chat (no reasoning param)
]

RECOMMENDED = [
    "meta-llama/llama-3.3-70b-instruct",  # open instruct, non-reasoning
    "anthropic/claude-haiku-4.5",  # cheap closed generalist
]

ALL_MODELS = MANDATORY + RECOMMENDED

# Used by exp04_variance and exp06_scope_scaling to keep call counts down.
CHEAP_REPRESENTATIVES = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-r1",
    "google/gemini-2.5-flash",
]


def models_or_env_override() -> list[str]:
    """Allow `LLM_MODELS=a,b,c` env override for quick swaps without code edits."""
    import os

    raw = os.getenv("LLM_MODELS")
    if not raw:
        return list(ALL_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]
