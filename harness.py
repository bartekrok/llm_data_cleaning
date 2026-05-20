"""harness.py - single entry point for the LLM-data-cleaning experiments.

Subcommands:
    seed         Walk tests data folders and upsert into the `test_cases` table.
    run          Execute one experiment by name (writes to the `runs` table).
    analyze      Compute metrics and write CSVs/PNGs for one experiment_id.
    list         List registered experiments and known scenarios.

Examples:
    python harness.py seed
    python harness.py run --experiment exp09_smoke
    python harness.py run --experiment exp01_core_matrix --limit 10
    python harness.py analyze --experiment-id <uuid> --out reports/
    python harness.py list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")


def cmd_seed(args: argparse.Namespace) -> int:
    from llm_cleaner.etl import connect, ensure_schema, upsert_test_case
    from llm_cleaner.seed import iter_test_cases

    schema_path = REPO_ROOT / "db" / "init" / "01_schema.sql"
    inserted = 0
    skipped = 0
    needs_review: list[str] = []

    with connect() as conn:
        ensure_schema(conn, str(schema_path))
        for case in iter_test_cases(REPO_ROOT):
            if case.expected_state in (None, "unknown"):
                needs_review.append(f"{case.test_name} (no expected_state)")
                skipped += 1
                continue
            try:
                upsert_test_case(conn, case)
                inserted += 1
            except Exception as e:
                print(f"  ERROR upserting {case.test_name}: {e}", file=sys.stderr)
                skipped += 1
        conn.commit()

    print(f"Seeded {inserted} test cases ({skipped} skipped).")
    if needs_review:
        print("Cases needing review:")
        for n in needs_review:
            print(f"  - {n}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from llm_cleaner.core import make_context
    from llm_cleaner.experiments import get_experiment

    experiment_fn = get_experiment(args.experiment)
    with make_context(args.experiment, limit=args.limit, dry_run=args.dry_run) as ctx:
        experiment_fn(ctx)
        print(f"\nDone. experiment_id={ctx.experiment_id}")
        if not args.dry_run:
            print(
                f"Analyze with: python harness.py analyze --experiment-id {ctx.experiment_id} --out reports/"
            )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from llm_cleaner.analyze import analyze_experiment

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_experiment(
        experiment_id=args.experiment_id,
        experiment_name=args.experiment_name,
        out_dir=out_dir,
    )
    print(f"Wrote analysis to {out_dir}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from llm_cleaner.experiments import EXPERIMENT_NAMES

    print("Registered experiments:")
    for name in EXPERIMENT_NAMES:
        print(f"  - {name}")

    print("\nKnown test-case scenarios:")
    for s in (
        "accept_rename",
        "accept_no_rename",
        "decline",
        "suggest",
        "mixed",
        "adversarial",
        "scope_scaling",
        "ambiguous",
    ):
        print(f"  - {s}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness.py", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seed", help="Seed the test_cases table from CSVs")
    p_seed.set_defaults(func=cmd_seed)

    p_run = sub.add_parser("run", help="Run a registered experiment")
    p_run.add_argument("--experiment", required=True, help="Experiment name, e.g. exp01_core_matrix")
    p_run.add_argument("--limit", type=int, default=None, help="Truncate test cases to N for dev runs")
    p_run.add_argument("--dry-run", action="store_true", help="Compute but do not insert runs")
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", help="Analyze runs for an experiment_id")
    p_analyze.add_argument("--experiment-id", required=False, help="UUID from a previous run")
    p_analyze.add_argument("--experiment-name", required=False, help="Or aggregate by experiment_name")
    p_analyze.add_argument("--out", default="reports", help="Output directory for CSVs/PNGs")
    p_analyze.set_defaults(func=cmd_analyze)

    p_list = sub.add_parser("list", help="List registered experiments")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
