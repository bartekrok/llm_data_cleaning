"""Shared model lists for the experiment registry.

Chosen for type diversity + low OpenRouter cost (live IDs as of 2026-11).
Text-in → text-out only in experiments; some models are multimodal-capable.

  - open reasoner (very cheap)
  - DeepSeek R1-class reasoner
  - cheap flash + reasoning + multimodal family
  - classic chat without reasoning API
  - open instruct + multimodal family
  - free reasoner + multimodal (rate-limited)

Override at runtime: ``LLM_MODELS=a,b,c``.
"""

from __future__ import annotations

MANDATORY = [
    "openai/gpt-oss-120b",  # open reasoner, very cheap
    "deepseek/deepseek-r1-0528",  # R1-class open reasoner
    "google/gemini-2.5-flash-lite",  # flash + reasoning + MM-capable
    "openai/gpt-4o-mini",  # classic cheap chat (no reasoning API)
]

RECOMMENDED = [
    "meta-llama/llama-4-scout",  # open instruct + MM-capable, non-reasoning
    "google/gemma-4-31b-it:free",  # free reasoner + MM-capable
]

ALL_MODELS = MANDATORY + RECOMMENDED

# Used by exp04_variance and exp06_scope_scaling.
CHEAP_REPRESENTATIVES = [
    "openai/gpt-4o-mini",
    "openai/gpt-oss-120b",
    "google/gemini-2.5-flash-lite",
]


def models_or_env_override() -> list[str]:
    """Allow `LLM_MODELS=a,b,c` env override for quick swaps without code edits."""
    import os

    raw = os.getenv("LLM_MODELS")
    if not raw:
        return list(ALL_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]
