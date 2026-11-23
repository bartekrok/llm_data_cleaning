"""exp01_core_matrix - RQ1 headline numbers.

All 6 models x standard test cases x P0_baseline x single.

Repetitions: 3 on the small suite (variance for heatmaps); 1 on
``--suite large|all`` so a full 6k-row pass stays inside a ~$100 budget.
Override with ``harness.py run --repetitions N``.
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix, suite_repetitions
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
    reps = ctx.repetitions if ctx.repetitions is not None else suite_repetitions(ctx.suite)
    run_matrix(
        ctx,
        models=models_or_env_override(),
        prompts=[P0_BASELINE],
        cases=cases,
        temperatures=[0.1],
        repetitions=reps,
    )
