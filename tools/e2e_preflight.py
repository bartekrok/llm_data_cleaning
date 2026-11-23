#!/usr/bin/env python3
"""Zero-cost end-to-end preflight before paid OpenRouter runs.

What it proves (against a real Postgres):
  1. Schema + seed work
  2. Every registered experiment runs without crashing
  3. Each experiment writes `runs` rows with the expected shape
     (models / prompts / modes / temperatures / confidence / batch)
  4. `analyze` can consume those rows

It never calls OpenRouter: set ``LLM_MOCK=1`` (this script forces it).

Usage:
    python tools/e2e_preflight.py
    python tools/e2e_preflight.py --limit 2 --keep-runs
    pytest tests/e2e/test_preflight.py -q
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["LLM_MOCK"] = "1"
# Avoid accidental live calls if something bypasses the mock flag.
os.environ.setdefault("OPENROUTER_API_KEY", "mock-e2e-no-spend")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from llm_cleaner.core import make_context
from llm_cleaner.etl import connect, ensure_schema, upsert_test_case
from llm_cleaner.experiments import EXPERIMENT_NAMES, get_experiment
from llm_cleaner.experiments._models import ALL_MODELS, CHEAP_REPRESENTATIVES
from llm_cleaner.prompts import (
    P0_BASELINE,
    P1_FEW_SHOT,
    P2_COT,
    P3_STRICT_SCHEMA,
    P4_MINIMAL,
    P_WITH_CONFIDENCE,
)
from llm_cleaner.seed import iter_test_cases


SCHEMA = ROOT / "db" / "init" / "01_schema.sql"


def seed_database() -> int:
    inserted = 0
    with connect() as conn:
        ensure_schema(conn, str(SCHEMA))
        for case in iter_test_cases(ROOT):
            if case.expected_state in (None, "unknown"):
                continue
            upsert_test_case(conn, case)
            inserted += 1
        conn.commit()
    return inserted


def _fetch_run_stats(conn, experiment_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS n,
                COUNT(DISTINCT model_name) AS n_models,
                COUNT(DISTINCT prompt_variant) AS n_prompts,
                COUNT(DISTINCT mode) AS n_modes,
                COUNT(DISTINCT temperature) AS n_temps,
                COUNT(*) FILTER (WHERE actual_state IS NOT NULL) AS n_with_state,
                COUNT(*) FILTER (WHERE raw_response IS NOT NULL) AS n_with_raw,
                COUNT(*) FILTER (WHERE actual_confidence IS NOT NULL) AS n_with_conf,
                COUNT(*) FILTER (WHERE failure_mode = 'api_error') AS n_api_errors,
                ARRAY_AGG(DISTINCT model_name ORDER BY model_name) AS models,
                ARRAY_AGG(DISTINCT prompt_variant ORDER BY prompt_variant) AS prompts,
                ARRAY_AGG(DISTINCT mode ORDER BY mode) AS modes,
                ARRAY_AGG(DISTINCT temperature ORDER BY temperature) AS temps
            FROM runs
            WHERE experiment_id = %s
            """,
            (experiment_id,),
        )
        row = cur.fetchone()
    keys = (
        "n",
        "n_models",
        "n_prompts",
        "n_modes",
        "n_temps",
        "n_with_state",
        "n_with_raw",
        "n_with_conf",
        "n_api_errors",
        "models",
        "prompts",
        "modes",
        "temps",
    )
    return dict(zip(keys, row))


def _assert(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def _expectations(name: str, stats: dict, limit: int) -> list[str]:
    """Return assertion failure messages for one experiment."""
    failures: list[str] = []
    models = set(stats["models"] or [])
    prompts = set(stats["prompts"] or [])
    modes = set(stats["modes"] or [])
    temps = set(float(t) for t in (stats["temps"] or []))

    _assert(stats["n"] > 0, f"{name}: expected >0 runs, got 0", failures)
    _assert(
        stats["n_api_errors"] == 0,
        f"{name}: unexpected api_error rows: {stats['n_api_errors']}",
        failures,
    )
    _assert(
        stats["n_with_state"] == stats["n"],
        f"{name}: missing actual_state on some rows "
        f"({stats['n_with_state']}/{stats['n']})",
        failures,
    )
    _assert(
        stats["n_with_raw"] == stats["n"],
        f"{name}: missing raw_response on some rows",
        failures,
    )

    if name == "exp01_core_matrix":
        _assert(set(ALL_MODELS) <= models, f"{name}: missing models {set(ALL_MODELS)-models}", failures)
        _assert(P0_BASELINE in prompts, f"{name}: missing P0", failures)
        _assert("single" in modes, f"{name}: missing single mode", failures)
        _assert(stats["n"] >= len(ALL_MODELS) * max(limit, 1), f"{name}: too few rows {stats['n']}", failures)

    elif name == "exp02_prompt_ablation":
        needed = {P0_BASELINE, P1_FEW_SHOT, P2_COT, P3_STRICT_SCHEMA, P4_MINIMAL}
        _assert(needed <= prompts, f"{name}: missing prompts {needed - prompts}", failures)
        _assert(set(ALL_MODELS) <= models, f"{name}: missing models", failures)

    elif name == "exp03_batch_vs_single":
        _assert("single" in modes, f"{name}: missing single", failures)
        batch_modes = {m for m in modes if m.startswith("batch")}
        _assert(len(batch_modes) >= 2, f"{name}: expected batch modes, got {modes}", failures)
        _assert(set(ALL_MODELS) <= models, f"{name}: missing models", failures)

    elif name == "exp04_variance":
        _assert(set(CHEAP_REPRESENTATIVES) <= models, f"{name}: missing cheap models", failures)
        _assert({0.0, 0.1, 0.7, 1.0} <= temps, f"{name}: missing temps {temps}", failures)

    elif name == "exp05_adversarial":
        _assert(set(ALL_MODELS) <= models, f"{name}: missing models", failures)
        _assert(stats["n"] >= len(ALL_MODELS), f"{name}: too few rows", failures)

    elif name == "exp06_scope_scaling":
        _assert(set(CHEAP_REPRESENTATIVES) <= models, f"{name}: missing cheap models", failures)

    elif name == "exp07_baselines":
        _assert({"baseline/fuzzy", "baseline/embedding"} <= models, f"{name}: missing baselines {models}", failures)

    elif name == "exp08_calibration":
        _assert(P_WITH_CONFIDENCE in prompts, f"{name}: missing confidence prompt", failures)
        _assert(
            stats["n_with_conf"] == stats["n"],
            f"{name}: expected confidence on all rows ({stats['n_with_conf']}/{stats['n']})",
            failures,
        )
        _assert(set(ALL_MODELS) <= models, f"{name}: missing models", failures)

    elif name == "exp09_smoke":
        _assert("openai/gpt-4o-mini" in models, f"{name}: expected gpt-4o-mini", failures)
        _assert(stats["n"] >= 1, f"{name}: no rows", failures)

    else:
        failures.append(f"{name}: no expectation rules registered")

    return failures


def prepare_db(*, reseeds: bool) -> int:
    if not SCHEMA.is_file():
        raise FileNotFoundError(SCHEMA)
    with connect() as conn:
        ensure_schema(conn, str(SCHEMA))
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM test_cases")
            n = cur.fetchone()[0]
    if reseeds or n == 0:
        return seed_database()
    return int(n)


def run_one(name: str, *, limit: int, suite: str) -> tuple[str, dict]:
    fn = get_experiment(name)
    with make_context(
        name,
        limit=limit,
        dry_run=False,
        suite=suite,
        repetitions=1,
    ) as ctx:
        # Keep e2e logs readable.
        ctx.verbose = True
        fn(ctx)
        eid = ctx.experiment_id
    with connect() as conn:
        stats = _fetch_run_stats(conn, eid)
    return eid, stats


def run_analyze_smoke(experiment_id: str, out_dir: Path) -> None:
    from llm_cleaner.analyze import analyze_experiment

    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_experiment(experiment_id=experiment_id, experiment_name=None, out_dir=out_dir)
    summary = next(out_dir.rglob("summary.json"), None)
    if summary is None or not summary.is_file():
        raise RuntimeError(f"analyze did not write summary.json under {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2, help="Cases per experiment (default 2)")
    parser.add_argument("--suite", default="small", choices=("small", "large", "all"))
    parser.add_argument("--skip-seed", action="store_true", help="Do not re-seed test_cases")
    parser.add_argument("--keep-runs", action="store_true", help="Do not delete prior e2e runs")
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENT_NAMES),
        help="Comma-separated subset (default: all)",
    )
    parser.add_argument("--out", default="reports/e2e_preflight", help="Analyze output dir")
    args = parser.parse_args(argv)

    print("=== e2e preflight (LLM_MOCK=1, $0 OpenRouter) ===")
    try:
        n_cases = prepare_db(reseeds=not args.skip_seed)
    except Exception as e:
        print(f"FAIL: database not ready: {e}", file=sys.stderr)
        print(
            "Start Postgres (docker compose up -d postgres) and set POSTGRES_* in .env.",
            file=sys.stderr,
        )
        return 2
    print(f"DB ready; test_cases≈{n_cases}")

    if not args.keep_runs:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM runs WHERE experiment_name LIKE 'exp%'")
            conn.commit()
        print("Cleared previous experiment runs")

    names = [n.strip() for n in args.experiments.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPERIMENT_NAMES]
    if unknown:
        print(f"Unknown experiments: {unknown}", file=sys.stderr)
        return 2

    results: list[tuple[str, str, dict]] = []
    all_failures: list[str] = []

    for name in names:
        print(f"\n--- {name} (limit={args.limit}, reps=1, suite={args.suite}) ---")
        try:
            eid, stats = run_one(name, limit=args.limit, suite=args.suite)
        except Exception:
            all_failures.append(f"{name}: crashed\n{traceback.format_exc()}")
            print(f"CRASHED: {name}")
            continue
        print(
            f"  experiment_id={eid} rows={stats['n']} models={stats['n_models']} "
            f"prompts={stats['n_prompts']} modes={stats['modes']}"
        )
        fails = _expectations(name, stats, args.limit)
        all_failures.extend(fails)
        results.append((name, eid, stats))
        if fails:
            for f in fails:
                print(f"  ASSERT: {f}")
        else:
            print("  OK")

    # Analyze one successful experiment (prefer exp01).
    analyze_target = next((r for r in results if r[0] == "exp01_core_matrix"), None)
    if analyze_target is None and results:
        analyze_target = results[0]
    if analyze_target is not None:
        name, eid, _ = analyze_target
        out = Path(args.out)
        print(f"\n--- analyze ({name} / {eid}) ---")
        try:
            run_analyze_smoke(eid, out)
            print(f"  analyze OK → {out}")
        except Exception:
            all_failures.append(f"analyze crashed\n{traceback.format_exc()}")
            print("  analyze FAILED")

    print("\n=== summary ===")
    print(f"experiments run: {len(results)}/{len(names)}")
    print(f"total run rows:  {sum(r[2]['n'] for r in results)}")
    if all_failures:
        print(f"FAILURES ({len(all_failures)}):")
        for f in all_failures:
            print(f"  - {f.splitlines()[0]}")
        return 1
    print("All e2e checks passed. Safe to spend API budget on real runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
