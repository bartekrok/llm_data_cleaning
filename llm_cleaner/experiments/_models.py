"""Shared model lists for the experiment registry.

Centralizing the names here means swapping a model only requires editing
ONE place. The plan asks for 6 models total: 4 mandatory + 2 recommended.

Model identifiers follow OpenRouter's <vendor>/<model> convention.
"""

from __future__ import annotations

# Plan §2 mandatory set
MANDATORY = [
    "openai/gpt-5.5",
    "deepseek/deepseek-v4-pro",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
]

# Plan §2 recommended additions
RECOMMENDED = [
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o-mini",
]

ALL_MODELS = MANDATORY + RECOMMENDED

# Cheapest representative from each major provider (used by exp04_variance
# and exp06_scope_scaling to keep call counts reasonable).
CHEAP_REPRESENTATIVES = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.1-pro-preview",
]


def models_or_env_override() -> list[str]:
    """Allow `LLM_MODELS=a,b,c` env override for quick swaps without code edits."""
    import os

    raw = os.getenv("LLM_MODELS")
    if not raw:
        return list(ALL_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]
