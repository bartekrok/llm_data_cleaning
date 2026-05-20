"""exp07_baselines - RQ4 non-LLM comparison.

Runs the two local cleaners (rapidfuzz and sentence-transformers) over
the standard test set. Single repetition is enough - they are deterministic.

The harness exposes them as pseudo-models "baseline/fuzzy" and
"baseline/embedding"; the runner detects those names and constructs
local cleaners instead of OpenRouter clients.
"""

from __future__ import annotations

from llm_cleaner.core import BASELINE_EMBED, BASELINE_FUZZY, RunContext, load_cases, run_matrix
from llm_cleaner.prompts import P0_BASELINE


SCENARIOS = (
    "accept_rename",
    "accept_no_rename",
    "decline",
    "suggest",
    "mixed",
    "adversarial",
    "scope_scaling",
)


def run(ctx: RunContext) -> None:
    cases = load_cases(ctx, scenarios=SCENARIOS)
    run_matrix(
        ctx,
        models=[BASELINE_FUZZY, BASELINE_EMBED],
        prompts=[P0_BASELINE],  # ignored by baselines but the runner expects a value
        cases=cases,
        temperatures=[0.0],
        repetitions=1,
    )
