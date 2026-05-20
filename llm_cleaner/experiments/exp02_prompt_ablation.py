"""exp02_prompt_ablation - RQ2 prompt vs model.

All 6 models x ~20 representative cases x all 5 prompts x single x 3 reps.
We pick representative cases by taking the first N of each not-mixed
scenario so the matrix stays balanced.
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix
from llm_cleaner.experiments._models import models_or_env_override
from llm_cleaner.prompts import (
    P0_BASELINE,
    P1_FEW_SHOT,
    P2_COT,
    P3_STRICT_SCHEMA,
    P4_MINIMAL,
)


REPRESENTATIVE_SCENARIOS = (
    "accept_rename",
    "accept_no_rename",
    "decline",
    "suggest",
)


def run(ctx: RunContext) -> None:
    cases = load_cases(ctx, scenarios=REPRESENTATIVE_SCENARIOS)
    cases = cases[: ctx.limit or 24]
    run_matrix(
        ctx,
        models=models_or_env_override(),
        prompts=[P0_BASELINE, P1_FEW_SHOT, P2_COT, P3_STRICT_SCHEMA, P4_MINIMAL],
        cases=cases,
        temperatures=[0.1],
        repetitions=3,
    )
