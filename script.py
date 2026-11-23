"""Backwards-compatible shim.

The original CLI was `python script.py <test_folder>`. The new harness
lives in harness.py. Running script.py with no args triggers the smoke
experiment; passing a folder still works for the legacy one-off run.

For everything else use:
    python harness.py seed
    python harness.py run --experiment <name>
    python harness.py analyze --experiment-id <uuid>
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


def legacy_one_off(folder: str) -> int:
    """Re-implement the original script.py behavior using the new client."""
    import csv

    from llm_cleaner.clients.openrouter import OpenRouterCleaner
    from llm_cleaner.prompts import P0_BASELINE, build_single

    scope_file = Path(folder) / "scope.csv"
    input_file = Path(folder) / "input_data.csv"
    if not scope_file.exists() or not input_file.exists():
        print(f"Missing scope.csv or input_data.csv in {folder}", file=sys.stderr)
        return 1

    with scope_file.open("r", encoding="utf-8") as f:
        scope = [r[0].strip() for i, r in enumerate(csv.reader(f)) if r and (i > 0 or r[0].strip().lower() not in {"scope_value", "scope", "value"})]
    scope = [s for s in scope if s]

    model_id = os.getenv("LEGACY_MODEL", "google/gemini-2.5-flash")
    cleaner = OpenRouterCleaner(model_id=model_id)

    with input_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("raw_value") or "").strip()
            if not raw:
                continue
            messages, response_format = build_single(P0_BASELINE, raw, scope)
            res = cleaner.clean(
                value=raw,
                scope=scope,
                prompt_variant=P0_BASELINE,
                prompt_messages=messages,
                response_format=response_format,
                temperature=0.1,
            )
            print(f"\nEvaluating: '{raw}'")
            print(f"  state  : {res.state}")
            print(f"  value  : {res.value}")
            print(f"  message: {res.message}")
            print(f"  latency: {res.latency_ms} ms  tokens={res.tokens_in}/{res.tokens_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_folder", nargs="?", help="Legacy folder mode (scope.csv + input_data.csv)")
    args = parser.parse_args()

    if args.test_folder:
        return legacy_one_off(args.test_folder)

    print("Use `python harness.py --help`. Running smoke experiment by default.")
    from harness import main as harness_main
    return harness_main(["run", "--experiment", "exp09_smoke"])


if __name__ == "__main__":
    raise SystemExit(main())
