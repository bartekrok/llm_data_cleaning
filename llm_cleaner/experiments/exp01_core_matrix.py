"""exp01_core_matrix - RQ1 headline numbers.

All 6 models x standard test cases x P0_baseline x single x 3 repetitions.
This produces the per-model x per-scenario accuracy heatmap.
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix
from llm_cleaner.experiments._models import models_or_env_override
from llm_cleaner.prompts import P0_BASELINE


SCENARIOS = (
    "accept_rename",
    "accept_no_rename",
    "decline",
    "suggest",
    "mixed",
)


def run(ctx: RunContext) -> None:
    cases = load_cases(ctx, scenarios=SCENARIOS)
    run_matrix(
        ctx,
        models=models_or_env_override(),
        prompts=[P0_BASELINE],
        cases=cases,
        temperatures=[0.1],
        repetitions=3,
    )
