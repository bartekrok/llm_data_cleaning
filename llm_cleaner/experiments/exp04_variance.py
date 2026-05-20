"""exp04_variance - RQ5 reproducibility / determinism.

3 cheap representative models x ~10 cases x P0_baseline x single x
4 temperatures {0.0, 0.1, 0.7, 1.0} x 10 repetitions each.

The same case at the same temperature is asked 10 times, then we
compute per-(model, case, temperature) std-dev of the correctness
indicator and the per-prediction agreement rate downstream in analyze.py.
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix
from llm_cleaner.experiments._models import CHEAP_REPRESENTATIVES
from llm_cleaner.prompts import P0_BASELINE


def run(ctx: RunContext) -> None:
    cases = load_cases(
        ctx,
        scenarios=["accept_rename", "accept_no_rename", "decline", "suggest"],
    )
    cases = cases[: ctx.limit or 10]

    run_matrix(
        ctx,
        models=CHEAP_REPRESENTATIVES,
        prompts=[P0_BASELINE],
        cases=cases,
        temperatures=[0.0, 0.1, 0.7, 1.0],
        repetitions=10,
        seed=None,
    )
