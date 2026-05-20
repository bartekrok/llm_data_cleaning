"""exp09_smoke - end-to-end sanity check.

1 cheap model x 3 cases x P0_baseline x single x 1 rep.

Used after every refactor to verify that:
- DB connection works
- prompts.build_single returns valid messages
- OpenRouter call succeeds and returns parseable JSON
- failure-mode classifier runs without exceptions
- insert_run works
"""

from __future__ import annotations

from llm_cleaner.core import RunContext, load_cases, run_matrix
from llm_cleaner.prompts import P0_BASELINE


def run(ctx: RunContext) -> None:
    cases = load_cases(
        ctx,
        scenarios=["accept_rename", "decline", "suggest"],
    )
    cases = cases[: ctx.limit or 3]

    run_matrix(
        ctx,
        models=["openai/gpt-4o-mini"],
        prompts=[P0_BASELINE],
        cases=cases,
        temperatures=[0.1],
        repetitions=1,
    )
