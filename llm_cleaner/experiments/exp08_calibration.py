"""exp08_calibration - RQ5 confidence calibration.

All 6 models x ~20 representative cases x P_with_confidence x single x 3 reps.

The P_with_confidence prompt asks the model to add a "confidence" field
in [0, 1]. analyze.py will draw a reliability diagram and compute ECE.
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix
from llm_cleaner.experiments._models import models_or_env_override
from llm_cleaner.prompts import P_WITH_CONFIDENCE


SCENARIOS = (
    "accept_rename",
    "accept_no_rename",
    "decline",
    "suggest",
)


def run(ctx: RunContext) -> None:
    cases = load_cases(ctx, scenarios=SCENARIOS)
    cases = cases[: ctx.limit or 20]
    run_matrix(
        ctx,
        models=models_or_env_override(),
        prompts=[P_WITH_CONFIDENCE],
        cases=cases,
        temperatures=[0.1],
        repetitions=3,
    )
