"""Shared experiment-runner machinery.

Every experiment is a function `run(ctx: RunContext) -> None`. Most
experiments just declare what to vary and call `run_matrix`. The
matrix runner does the iteration, calls the cleaner, classifies the
result, and inserts one row into `runs` per call.

Batch-mode experiments call `run_batch_matrix` instead - it groups
multiple raw values into a single API call and records one row per
input value, with `mode="batch_*"`.

Test cases are loaded from the DB via `etl.fetch_test_cases`. The
experiment specifies *which* cases to load by passing filters - the
runner does not pre-load everything.
"""

from __future__ import annotations

import os
import random
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

from llm_cleaner.clients.base import Cleaner, CleanResult
from llm_cleaner.clients.fuzzy import FuzzyCleaner
from llm_cleaner.clients.embedding import EmbeddingCleaner
from llm_cleaner.clients.openrouter import OpenRouterCleaner
from llm_cleaner.etl import (
    TestCaseRow,
    classify_failure,
    connect,
    fetch_test_cases,
    insert_run,
)
from llm_cleaner.prompts import (
    P0_BASELINE,
    P_WITH_CONFIDENCE,
    build_batch,
    build_single,
)


BASELINE_FUZZY = "baseline/fuzzy"
BASELINE_EMBED = "baseline/embedding"
BASELINE_NAMES = (BASELINE_FUZZY, BASELINE_EMBED)


@dataclass
class RunContext:
    experiment_id: str
    experiment_name: str
    conn: Any
    limit: Optional[int] = None
    dry_run: bool = False
    verbose: bool = True
    cleaner_cache: dict[str, Cleaner] = field(default_factory=dict)

    def get_cleaner(self, model_id: str) -> Cleaner:
        if model_id in self.cleaner_cache:
            return self.cleaner_cache[model_id]
        if model_id == BASELINE_FUZZY:
            cleaner: Cleaner = FuzzyCleaner()
        elif model_id == BASELINE_EMBED:
            cleaner = EmbeddingCleaner()
        else:
            cleaner = OpenRouterCleaner(model_id=model_id)
        self.cleaner_cache[model_id] = cleaner
        return cleaner

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


def _truncate(cases: Sequence[TestCaseRow], limit: Optional[int]) -> list[TestCaseRow]:
    if limit is None or limit <= 0:
        return list(cases)
    return list(cases)[:limit]


def _commit_safely(ctx: RunContext) -> None:
    if ctx.dry_run:
        return
    try:
        ctx.conn.commit()
    except Exception as e:  # pragma: no cover
        ctx.log(f"  WARN: commit failed: {e}")


def run_matrix(
    ctx: RunContext,
    *,
    models: Sequence[str],
    prompts: Sequence[str],
    cases: Sequence[TestCaseRow],
    temperatures: Sequence[float] = (0.1,),
    repetitions: int = 3,
    seed: Optional[int] = None,
    sleep_between_calls: float = 0.0,
    prompt_variant_label: Optional[Callable[[str, float], str]] = None,
) -> int:
    """Iterate (model, case, prompt, temperature, repetition) and persist each call.

    Returns the number of rows inserted.
    """
    cases = _truncate(cases, ctx.limit)
    total_planned = len(models) * len(cases) * len(prompts) * len(temperatures) * repetitions
    ctx.log(
        f"[{ctx.experiment_name}] run_matrix: models={len(models)} cases={len(cases)} "
        f"prompts={len(prompts)} temps={len(temperatures)} reps={repetitions} "
        f"=> {total_planned} planned API calls"
    )

    rows_inserted = 0
    counter = 0
    t_start = time.perf_counter()

    for model_id in models:
        cleaner = ctx.get_cleaner(model_id)
        for prompt_variant in prompts:
            for temperature in temperatures:
                effective_variant = (
                    prompt_variant_label(prompt_variant, temperature)
                    if prompt_variant_label
                    else prompt_variant
                )
                for case in cases:
                    for rep in range(repetitions):
                        counter += 1
                        messages, response_format = build_single(
                            prompt_variant, case.raw_value, case.scope
                        )
                        try:
                            result = cleaner.clean(
                                value=case.raw_value,
                                scope=case.scope,
                                prompt_variant=prompt_variant,
                                prompt_messages=messages,
                                response_format=response_format,
                                temperature=temperature,
                                seed=seed,
                            )
                        except Exception as e:  # pragma: no cover - safety net
                            result = CleanResult(
                                state=None,
                                value=None,
                                message=str(e),
                                failure_mode="api_error",
                                parse_error=type(e).__name__,
                                provider=None,
                                response_format_used=None,
                                seed=seed,
                                raw_response={"exception": repr(e)},
                            )

                        is_correct, failure_mode = classify_failure(
                            actual_state=result.state,
                            actual_value=result.value,
                            expected_state=case.expected_state,
                            expected_value=case.expected_value,
                            scope=case.scope,
                            pre_existing_failure=result.failure_mode,
                        )

                        if not ctx.dry_run:
                            insert_run(
                                ctx.conn,
                                experiment_id=ctx.experiment_id,
                                experiment_name=ctx.experiment_name,
                                model_name=model_id,
                                prompt_variant=effective_variant,
                                mode="single",
                                temperature=temperature,
                                repetition_idx=rep,
                                test_case=case,
                                result=result,
                                is_correct=is_correct,
                                failure_mode=failure_mode,
                            )
                            rows_inserted += 1
                            if rows_inserted % 25 == 0:
                                _commit_safely(ctx)

                        if ctx.verbose and counter % 10 == 0:
                            elapsed = time.perf_counter() - t_start
                            ctx.log(
                                f"  [{counter}/{total_planned}] model={model_id} "
                                f"case={case.test_name} prompt={effective_variant} "
                                f"T={temperature} rep={rep} ok={is_correct} "
                                f"({elapsed:.1f}s elapsed)"
                            )

                        if sleep_between_calls > 0:
                            time.sleep(sleep_between_calls)

    _commit_safely(ctx)
    ctx.log(
        f"[{ctx.experiment_name}] run_matrix done: {rows_inserted} rows inserted "
        f"in {time.perf_counter() - t_start:.1f}s"
    )
    return rows_inserted


def run_batch_matrix(
    ctx: RunContext,
    *,
    models: Sequence[str],
    prompts: Sequence[str],
    batches: Sequence[tuple[str, Sequence[TestCaseRow]]],
    temperatures: Sequence[float] = (0.1,),
    repetitions: int = 3,
    shuffle_orders: int = 1,
    seed: Optional[int] = None,
    sleep_between_calls: float = 0.0,
    mode_label: str = "batch",
) -> int:
    """Run batch-mode experiments.

    `batches` is a list of (batch_name, cases) tuples. All cases in a
    batch share the same scope (use the scope of the first case). For
    each batch we send one API call containing all raw values and
    record one row per case.

    `shuffle_orders` > 1 produces additional shuffled orderings of the
    same batch (with mode_label suffixed _shuffleN) to test position bias.
    """
    total_batches = len(models) * len(prompts) * len(temperatures) * repetitions * len(batches) * shuffle_orders
    ctx.log(
        f"[{ctx.experiment_name}] run_batch_matrix: models={len(models)} batches={len(batches)} "
        f"prompts={len(prompts)} temps={len(temperatures)} reps={repetitions} "
        f"shuffles={shuffle_orders} => {total_batches} planned API calls"
    )

    rows_inserted = 0
    t_start = time.perf_counter()
    rng = random.Random(seed if seed is not None else 0xC0FFEE)

    for model_id in models:
        cleaner = ctx.get_cleaner(model_id)
        for prompt_variant in prompts:
            for temperature in temperatures:
                for shuffle_idx in range(shuffle_orders):
                    for batch_name, cases in batches:
                        if not cases:
                            continue
                        scope = cases[0].scope

                        if shuffle_idx == 0:
                            ordered_cases = list(cases)
                            mode = f"{mode_label}_ordered"
                        else:
                            ordered_cases = list(cases)
                            rng.shuffle(ordered_cases)
                            mode = f"{mode_label}_shuffle{shuffle_idx}"

                        values = [c.raw_value for c in ordered_cases]

                        for rep in range(repetitions):
                            messages, response_format = build_batch(
                                prompt_variant, values, scope
                            )
                            try:
                                results = cleaner.clean_batch(
                                    values=values,
                                    scope=scope,
                                    prompt_variant=prompt_variant,
                                    prompt_messages=messages,
                                    response_format=response_format,
                                    temperature=temperature,
                                    seed=seed,
                                )
                            except Exception as e:  # pragma: no cover
                                results = [
                                    CleanResult(
                                        state=None,
                                        value=None,
                                        message=str(e),
                                        failure_mode="api_error",
                                        parse_error=type(e).__name__,
                                        raw_response={"exception": repr(e)},
                                    )
                                    for _ in values
                                ]

                            if len(results) != len(ordered_cases):
                                # pad or truncate so we always get one row per case
                                if len(results) < len(ordered_cases):
                                    results = list(results) + [
                                        CleanResult(
                                            state=None,
                                            value=None,
                                            message="missing batch element",
                                            failure_mode="parse_error",
                                        )
                                        for _ in range(len(ordered_cases) - len(results))
                                    ]
                                else:
                                    results = list(results[: len(ordered_cases)])

                            for case, result in zip(ordered_cases, results):
                                is_correct, failure_mode = classify_failure(
                                    actual_state=result.state,
                                    actual_value=result.value,
                                    expected_state=case.expected_state,
                                    expected_value=case.expected_value,
                                    scope=case.scope,
                                    pre_existing_failure=result.failure_mode,
                                )

                                if not ctx.dry_run:
                                    insert_run(
                                        ctx.conn,
                                        experiment_id=ctx.experiment_id,
                                        experiment_name=ctx.experiment_name,
                                        model_name=model_id,
                                        prompt_variant=prompt_variant,
                                        mode=mode,
                                        temperature=temperature,
                                        repetition_idx=rep,
                                        test_case=case,
                                        result=result,
                                        is_correct=is_correct,
                                        failure_mode=failure_mode,
                                        scope_size=len(scope),
                                    )
                                    rows_inserted += 1
                            if rows_inserted % 25 == 0:
                                _commit_safely(ctx)

                            if sleep_between_calls > 0:
                                time.sleep(sleep_between_calls)
    _commit_safely(ctx)
    ctx.log(
        f"[{ctx.experiment_name}] run_batch_matrix done: {rows_inserted} rows inserted "
        f"in {time.perf_counter() - t_start:.1f}s"
    )
    return rows_inserted


def load_cases(
    ctx: RunContext,
    *,
    scenarios: Optional[Sequence[str]] = None,
    name_prefixes: Optional[Sequence[str]] = None,
    name_in: Optional[Sequence[str]] = None,
) -> list[TestCaseRow]:
    """Convenience: read test cases from the DB and apply ctx.limit."""
    cases = fetch_test_cases(
        ctx.conn,
        scenarios=scenarios,
        name_prefixes=name_prefixes,
        name_in=name_in,
    )
    return _truncate(cases, ctx.limit)


@contextmanager
def make_context(experiment_name: str, *, limit: Optional[int] = None, dry_run: bool = False) -> Iterator[RunContext]:
    experiment_id = str(uuid.uuid4())
    with connect() as conn:
        ctx = RunContext(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            conn=conn,
            limit=limit,
            dry_run=dry_run,
        )
        ctx.log(f"=== Experiment {experiment_name} ({experiment_id}) ===")
        yield ctx
        ctx.log(f"=== Done. experiment_id={experiment_id} ===")
