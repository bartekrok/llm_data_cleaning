"""Offline self-check: exercise everything that doesn't need Postgres."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_cleaner.clients.base import normalize_state, coerce_confidence
from llm_cleaner.clients.fuzzy import FuzzyCleaner
from llm_cleaner.prompts import (
    P0_BASELINE,
    P1_FEW_SHOT,
    P2_COT,
    P3_STRICT_SCHEMA,
    P4_MINIMAL,
    P_WITH_CONFIDENCE,
    build_batch,
    build_single,
)
from llm_cleaner.etl import classify_failure
from llm_cleaner.experiments import EXPERIMENT_NAMES, get_experiment
from llm_cleaner.seed import iter_test_cases


def check_prompts() -> None:
    scope = ["Apple", "Banana", "Orange"]
    variants = (P0_BASELINE, P1_FEW_SHOT, P2_COT, P3_STRICT_SCHEMA, P4_MINIMAL, P_WITH_CONFIDENCE)
    for v in variants:
        m, rf = build_single(v, "aple", scope)
        assert isinstance(m, list) and len(m) >= 2, f"{v}: bad messages"
        if v == P3_STRICT_SCHEMA:
            assert rf and rf["type"] == "json_schema", f"{v}: missing schema"
        if v == P_WITH_CONFIDENCE:
            assert rf and rf["type"] == "json_schema", f"{v}: missing schema"
        m2, _ = build_batch(v, ["aple", "banan", "orng"], scope)
        assert isinstance(m2, list) and len(m2) >= 2
    print(f"  prompts: OK ({len(variants)} variants, single + batch)")


def check_fuzzy() -> None:
    fc = FuzzyCleaner()
    r1 = fc.clean("apple", ["Apple", "Banana"], prompt_variant=P0_BASELINE)
    assert r1.state == "acceptance"
    assert r1.value == "Apple"
    r2 = fc.clean("Elephant", ["Apple", "Banana"], prompt_variant=P0_BASELINE)
    assert r2.state == "decline"
    r3 = fc.clean("", ["Apple"], prompt_variant=P0_BASELINE)
    assert r3.state == "decline"
    print("  fuzzy:   OK")


def check_state_helpers() -> None:
    assert normalize_state("ACCEPTANCE") == "acceptance"
    assert normalize_state("rejected") == "decline"
    assert normalize_state("Suggestion") == "suggest"
    assert normalize_state("foo") is None
    assert coerce_confidence(0.5) == 0.5
    assert coerce_confidence(2.0) is None
    assert coerce_confidence("0.8") == 0.8
    assert coerce_confidence(None) is None
    print("  helpers: OK")


def check_failure_classifier() -> None:
    ok, fm = classify_failure(
        actual_state="acceptance",
        actual_value="Apple",
        expected_state="acceptance",
        expected_value="Apple",
        scope=["Apple", "Banana"],
        pre_existing_failure=None,
    )
    assert ok and fm is None

    ok, fm = classify_failure(
        actual_state="acceptance",
        actual_value="Banana",
        expected_state="acceptance",
        expected_value="Apple",
        scope=["Apple", "Banana"],
        pre_existing_failure=None,
    )
    assert not ok and fm == "wrong_rename"

    ok, fm = classify_failure(
        actual_state="acceptance",
        actual_value="Cherry",
        expected_state="acceptance",
        expected_value="Apple",
        scope=["Apple", "Banana"],
        pre_existing_failure=None,
    )
    assert not ok and fm == "hallucinated_scope"

    ok, fm = classify_failure(
        actual_state="decline",
        actual_value="",
        expected_state="acceptance",
        expected_value="Apple",
        scope=["Apple"],
        pre_existing_failure=None,
    )
    assert not ok and fm == "wrong_state"

    ok, fm = classify_failure(
        actual_state=None,
        actual_value=None,
        expected_state="acceptance",
        expected_value="Apple",
        scope=["Apple"],
        pre_existing_failure="parse_error",
    )
    assert not ok and fm == "parse_error"
    print("  classify_failure: OK")


def check_experiments() -> None:
    assert len(EXPERIMENT_NAMES) == 9
    for name in EXPERIMENT_NAMES:
        fn = get_experiment(name)
        assert callable(fn), f"{name} not callable"
    print(f"  experiments: OK ({len(EXPERIMENT_NAMES)} registered)")


def check_seed() -> None:
    cases = list(iter_test_cases(Path(__file__).resolve().parents[1]))
    assert len(cases) > 50, f"too few cases: {len(cases)}"
    by_scenario: dict[str, int] = {}
    for c in cases:
        by_scenario[c.scenario] = by_scenario.get(c.scenario, 0) + 1
    print(f"  seed: OK ({len(cases)} cases across {len(by_scenario)} scenarios)")
    for s, k in sorted(by_scenario.items()):
        print(f"    {s}: {k}")


def main() -> int:
    print("Running offline self-check...")
    check_prompts()
    check_fuzzy()
    check_state_helpers()
    check_failure_classifier()
    check_experiments()
    check_seed()
    print("\nAll offline checks passed.")
    print("To actually run an experiment you still need:")
    print("  1. docker compose up -d postgres   (or a local Postgres)")
    print("  2. python harness.py seed")
    print("  3. python harness.py run --experiment exp09_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
