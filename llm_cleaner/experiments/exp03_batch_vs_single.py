"""exp03_batch_vs_single - RQ3a context shape.

For each of the 4 mixed-test sets we run:
- single mode (one API call per value) - baseline
- batch_ordered (all values in one API call, original order)
- batch_shuffled (3 shuffled orderings) - tests position bias

Models: all 6. Prompt: P0_baseline. Repetitions: 3 per condition.
"""

from __future__ import annotations

from collections import defaultdict

from llm_cleaner.core import RunContext, load_cases, run_batch_matrix, run_matrix
from llm_cleaner.experiments._models import models_or_env_override
from llm_cleaner.prompts import P0_BASELINE


def run(ctx: RunContext) -> None:
    cases = load_cases(ctx, scenarios=["mixed"])

    # First: single-mode baseline over the mixed cases (so we can compare).
    run_matrix(
        ctx,
        models=models_or_env_override(),
        prompts=[P0_BASELINE],
        cases=cases,
        temperatures=[0.1],
        repetitions=3,
    )

    # Group cases by the parent batch (everything before ':<idx>').
    groups: dict[str, list] = defaultdict(list)
    for c in cases:
        key = c.test_name.rsplit(":", 1)[0]
        groups[key].append(c)

    batches = [(name, sorted(items, key=lambda c: c.test_name)) for name, items in groups.items()]

    run_batch_matrix(
        ctx,
        models=models_or_env_override(),
        prompts=[P0_BASELINE],
        batches=batches,
        temperatures=[0.1],
        repetitions=3,
        shuffle_orders=4,
        seed=42,
        mode_label="batch",
    )
