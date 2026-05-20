"""exp05_adversarial - RQ1 robustness.

All 6 models x the adversarial mini-suite (prompt injection, homoglyphs,
empty/emoji) x P0_baseline x single x 3 reps.
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix
from llm_cleaner.experiments._models import models_or_env_override
from llm_cleaner.prompts import P0_BASELINE


def run(ctx: RunContext) -> None:
    cases = load_cases(ctx, scenarios=["adversarial"])
    run_matrix(
        ctx,
        models=models_or_env_override(),
        prompts=[P0_BASELINE],
        cases=cases,
        temperatures=[0.1],
        repetitions=3,
    )
