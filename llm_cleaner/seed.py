"""Test-case seeder.

Walks the workspace's test-data directories and produces TestCaseRow
records ready for `etl.upsert_test_case`.

Layout convention:
    <root>/<group>/<scenario>/<case>/
        scope.csv           - one column with the allowed scope values
        input_data.csv      - one column 'raw_value' with one or more inputs
        expected.csv        - OPTIONAL one row per input_data row with
                              columns: expected_state, expected_value, notes

If `expected.csv` is missing we infer:
    - expected_state from the scenario folder name
    - expected_value from a per-scenario heuristic + a hand-curated map
      maintained in EXPECTED_MAP below

Naming examples produced by this loader:
    not_mixed_tests/accept_with_renaming/2A:1
    mixed_tests/decline_mixed:3
    adversarial/prompt_injection:0
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from llm_cleaner.etl import TestCaseRow


SCENARIO_FROM_FOLDER = {
    "accept_with_renaming": "accept_rename",
    "accept_without_renaming": "accept_no_rename",
    "decline": "decline",
    "sugestion": "suggest",
    "suggestion": "suggest",
    "accept_with_renaming_mixed": "mixed",
    "accept_wthout_renaming_mixed": "mixed",
    "decline_mixed": "mixed",
    "sugestion_mixed": "mixed",
    "suggestion_mixed": "mixed",
    "adversarial": "adversarial",
    "scope_scaling": "scope_scaling",
    "typo_ladder": "accept_rename",
    "abbreviations": "accept_rename",
    "synonyms": "accept_rename",
    "noise": "accept_rename",
    "ambiguous": "ambiguous",
    "long_inputs": "decline",
    "edge_decline": "decline",
    "edge_suggest": "suggest",
}

EXPECTED_STATE_FROM_FOLDER = {
    "accept_with_renaming": "acceptance",
    "accept_without_renaming": "acceptance",
    "decline": "decline",
    "sugestion": "suggest",
    "suggestion": "suggest",
    "accept_with_renaming_mixed": None,
    "accept_wthout_renaming_mixed": None,
    "decline_mixed": "decline",
    "sugestion_mixed": "suggest",
    "suggestion_mixed": "suggest",
    "adversarial": None,
    "scope_scaling": None,
    "typo_ladder": "acceptance",
    "abbreviations": "acceptance",
    "synonyms": "acceptance",
    "noise": "acceptance",
    "ambiguous": None,
    "long_inputs": "decline",
    "edge_decline": "decline",
    "edge_suggest": "suggest",
}


def _load_csv_column(path: Path, header_aliases: Iterable[str]) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            val = row[0].strip()
            if i == 0 and val.lower() in {a.lower() for a in header_aliases}:
                continue
            if val:
                out.append(val)
    return out


def _load_expected_csv(path: Path) -> list[dict[str, Optional[str]]]:
    rows: list[dict[str, Optional[str]]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "expected_state": (row.get("expected_state") or "").strip() or None,
                    "expected_value": (row.get("expected_value") or "").strip() or None,
                    "notes": (row.get("notes") or "").strip() or None,
                }
            )
    return rows


def _normalize(s: str) -> str:
    return s.strip().lower()


def _heuristic_expected_value(
    raw_value: str,
    scope: list[str],
    expected_state: Optional[str],
) -> Optional[str]:
    """Best-guess expected_value when expected.csv is missing.

    For acceptance: pick the scope item whose lowercase form contains
    the input's lowercase form, or vice versa. If nothing matches we
    return None (the test case will be marked needs_review).
    """
    if expected_state == "decline":
        return None
    if expected_state == "suggest":
        return raw_value.strip()
    if expected_state == "acceptance":
        v = _normalize(raw_value)
        scope_norm = [_normalize(s) for s in scope]
        if v in scope_norm:
            return scope[scope_norm.index(v)]
        for i, s in enumerate(scope_norm):
            if v == s or v in s or s in v:
                return scope[i]
        return None
    return None


@dataclass
class SeedReport:
    seeded: int = 0
    skipped: int = 0
    needs_review: list[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.needs_review is None:
            self.needs_review = []


def iter_test_cases(repo_root: Path) -> Iterator[TestCaseRow]:
    """Walk the workspace for test-case folders and yield TestCaseRow objects."""
    candidates = [
        repo_root / "not_mixed_tests",
        repo_root / "mixed_tests",
        repo_root / "tests_data" / "adversarial",
        repo_root / "tests_data" / "scope_scaling",
        repo_root / "tests_data" / "extra",
    ]

    for root in candidates:
        if not root.exists():
            continue
        for scope_path in root.rglob("scope.csv"):
            input_path = scope_path.parent / "input_data.csv"
            expected_path = scope_path.parent / "expected.csv"
            if not input_path.exists():
                continue

            rel = scope_path.parent.relative_to(repo_root).as_posix()
            parts = scope_path.parent.parts

            scenario_folder = None
            for part in reversed(parts):
                if part in SCENARIO_FROM_FOLDER:
                    scenario_folder = part
                    break
            scenario = SCENARIO_FROM_FOLDER.get(scenario_folder or "", "unknown")
            default_state = EXPECTED_STATE_FROM_FOLDER.get(scenario_folder or "")

            scope = _load_csv_column(
                scope_path,
                header_aliases=("scope", "value", "scope_value", "allowed", "name", "raw_value"),
            )
            values = _load_csv_column(input_path, header_aliases=("raw_value", "value"))
            expected_rows = _load_expected_csv(expected_path)

            for idx, raw_value in enumerate(values):
                if idx < len(expected_rows):
                    e_state = expected_rows[idx]["expected_state"] or default_state
                    e_value = expected_rows[idx]["expected_value"]
                    notes = expected_rows[idx]["notes"]
                else:
                    e_state = default_state
                    e_value = _heuristic_expected_value(raw_value, scope, default_state)
                    notes = None

                if e_state is None:
                    notes = (notes or "") + " [needs expected_state]"

                test_name = f"{rel}:{idx}"
                yield TestCaseRow(
                    id=-1,
                    test_name=test_name,
                    scenario=scenario,
                    raw_value=raw_value,
                    scope=scope,
                    expected_state=e_state or "unknown",
                    expected_value=e_value,
                    notes=notes.strip() if notes else None,
                )
