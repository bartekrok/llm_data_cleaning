"""exp06_scope_scaling - RQ3b context shape.

3 cheap representative models x the scope_scaling test set
(scopes of size 5, 20, 50, 200) x P0_baseline x single x 3 reps.

Each scope-size case shares the same anchor input values, so accuracy
should ideally be flat across scope sizes. analyze.py will draw the
accuracy-vs-scope-size line plot.
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix
from llm_cleaner.experiments._models import CHEAP_REPRESENTATIVES
from llm_cleaner.prompts import P0_BASELINE


def run(ctx: RunContext) -> None:
    cases = load_cases(ctx, scenarios=["scope_scaling"])
    run_matrix(
        ctx,
        models=CHEAP_REPRESENTATIVES,
        prompts=[P0_BASELINE],
        cases=cases,
        temperatures=[0.1],
        repetitions=3,
    )
