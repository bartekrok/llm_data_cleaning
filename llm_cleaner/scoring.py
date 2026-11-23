"""Predicate gold scoring for analyze (strict / lenient).

Gold lives on disk as expected.jsonl (preferred) or expected.csv.
For suggest with mode ``same_referent_as_input`` we validate membership
in an open equivalence class via a deterministic cascade:

  1. normalized equality (casefold, strip non-alphanumerics except #+)
  2. alias table from the gold spec
  3. Levenshtein ratio >= fuzzy_threshold (default 0.85)

LLM-as-judge is intentionally omitted here (can be added later behind a flag).

Strict = first entry in ``allowed[]`` (preferred policy).
Lenient = any entry in ``allowed[]``.

Acceptance-with-renaming vs suggest: if the canonical scope item is the
same referent as the input, gold is acceptance + one_of(scope item).
Treating that as suggest is wrong_state.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_FUZZY_THRESHOLD = 0.85
CLEAN_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+#&'()/-]{0,79}$")

REPO_ROOT = Path(__file__).resolve().parents[1]


def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def ref_norm(s: Optional[str]) -> str:
    """Casefold and strip non-alphanumerics except '#' and '+' (C vs C++ vs C#)."""
    return re.sub(r"[^a-z0-9#+]", "", (s or "").casefold())


def lev_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def folder_from_test_name(test_name: str) -> str:
    """``large_tests/foo/bar:3`` -> ``large_tests/foo/bar``."""
    if ":" in test_name:
        return test_name.rsplit(":", 1)[0]
    return test_name


def _csv_entry(state: str, value: Optional[str]) -> dict[str, Any]:
    st = norm(state)
    if st == "decline":
        return {"state": "decline", "value": {"mode": "empty"}}
    if st == "suggest":
        return {"state": "suggest", "value": {"mode": "same_referent_as_input"}}
    return {"state": st or "acceptance", "value": {"mode": "one_of", "items": [value or ""]}}


@lru_cache(maxsize=512)
def load_expected_for_folder(folder_rel: str) -> dict[str, list[dict[str, Any]]]:
    """Map norm(raw_value) -> allowed[] for one test folder (relative to repo root)."""
    folder = REPO_ROOT / folder_rel
    jsonl_path = folder / "expected.jsonl"
    csv_path = folder / "expected.csv"
    expected: dict[str, list[dict[str, Any]]] = {}

    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                expected[norm(obj.get("raw_value"))] = list(obj.get("allowed") or [])
        return expected

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # Support both harness schema and legacy alt_* schema.
            for row in reader:
                raw = row.get("raw_value")
                if raw is None:
                    # harness expected.csv is aligned by row index; caller may key by raw from input
                    continue
                state = row.get("expected_state") or row.get("state") or ""
                value = row.get("expected_value") if "expected_value" in row else row.get("value")
                allowed = [_csv_entry(state, value)]
                alt_state = row.get("alt_state")
                if alt_state and norm(alt_state):
                    allowed.append(_csv_entry(alt_state, row.get("alt_value")))
                expected[norm(raw)] = allowed
        return expected

    return expected


def load_expected_rows_by_index(folder_rel: str) -> list[list[dict[str, Any]]]:
    """For harness expected.csv without raw_value column: list of allowed[] by row index."""
    folder = REPO_ROOT / folder_rel
    jsonl_path = folder / "expected.jsonl"
    csv_path = folder / "expected.csv"
    rows: list[list[dict[str, Any]]] = []

    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append(list(obj.get("allowed") or []))
        return rows

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = row.get("expected_state") or row.get("state") or ""
                value = row.get("expected_value") if "expected_value" in row else row.get("value")
                allowed = [_csv_entry(state, value)]
                alt_state = row.get("alt_state")
                if alt_state and norm(alt_state):
                    allowed.append(_csv_entry(alt_state, row.get("alt_value")))
                rows.append(allowed)
        return rows

    return rows


def resolve_allowed(
    test_name: str,
    raw_value: str,
) -> list[dict[str, Any]]:
    folder = folder_from_test_name(test_name)
    by_raw = load_expected_for_folder(folder)
    key = norm(raw_value)
    if key in by_raw:
        return by_raw[key]

    # Fallback: row index from test_name suffix
    if ":" in test_name:
        try:
            idx = int(test_name.rsplit(":", 1)[1])
        except ValueError:
            idx = -1
        rows = load_expected_rows_by_index(folder)
        if 0 <= idx < len(rows):
            return rows[idx]

    return []


def validate_value(
    mode_obj: dict[str, Any],
    pred_value: Optional[str],
    raw_value: str,
    scope: Sequence[str],
) -> tuple[bool, str]:
    """Returns (ok, tier_or_reason)."""
    mode = (mode_obj.get("mode") or "exact").strip().lower()

    if mode == "empty":
        return (norm(pred_value) == "", "exact")

    if mode == "exact":
        return (norm(pred_value) == norm(str(mode_obj.get("item", ""))), "exact")

    if mode == "one_of":
        items = {norm(i) for i in mode_obj.get("items", [])}
        return (norm(pred_value) in items, "exact")

    if mode == "same_referent_as_input":
        value = (pred_value or "").strip()
        if not value:
            return (False, "empty_value")
        if not CLEAN_VALUE_RE.match(value):
            return (False, "cleanliness")
        if mode_obj.get("not_in_scope", True):
            scope_refs = {ref_norm(s) for s in scope}
            if ref_norm(value) in scope_refs:
                return (False, "not_in_scope")
        if ref_norm(value) == ref_norm(raw_value):
            return (True, "normalized")
        aliases = {ref_norm(a) for a in mode_obj.get("aliases", [])}
        if ref_norm(value) in aliases:
            return (True, "alias")
        threshold = float(mode_obj.get("fuzzy_threshold", DEFAULT_FUZZY_THRESHOLD))
        if lev_ratio(ref_norm(value), ref_norm(raw_value)) >= threshold:
            return (True, "fuzzy")
        return (False, "fuzzy")

    return (False, f"unknown_mode:{mode}")


def validate_entry(
    entry: dict[str, Any],
    pred_state: Optional[str],
    pred_value: Optional[str],
    raw_value: str,
    scope: Sequence[str],
) -> tuple[bool, str]:
    if norm(pred_state) != norm(entry.get("state")):
        return (False, "state")
    return validate_value(entry.get("value") or {}, pred_value, raw_value, scope)


@dataclass
class ScoreResult:
    is_correct_strict: bool
    is_correct_lenient: bool
    matched_tier: Optional[str]
    failure_mode: Optional[str]
    n_allowed: int


def _failure_from_miss(
    pred_state: Optional[str],
    pred_value: Optional[str],
    primary: Optional[dict[str, Any]],
    scope: Sequence[str],
) -> str:
    if pred_state is None:
        return "invalid_state"
    # Acceptance of a value not in scope is always a hallucination, even when
    # gold expected suggest/decline (e.g. Pear accepted though Pear ∉ scope).
    if norm(pred_state) == "acceptance":
        scope_lower = {s.strip().lower() for s in scope}
        if (pred_value or "").strip().lower() not in scope_lower:
            return "hallucinated_scope"
        if primary and norm(pred_state) != norm(primary.get("state")):
            return "wrong_state"
        return "wrong_rename"
    if primary and norm(pred_state) != norm(primary.get("state")):
        return "wrong_state"
    if norm(pred_state) == "suggest":
        return "wrong_referent"
    if norm(pred_state) == "decline":
        return "wrong_rename"
    return "wrong_state"


def score_prediction(
    *,
    test_name: str,
    raw_value: str,
    pred_state: Optional[str],
    pred_value: Optional[str],
    scope: Sequence[str],
    fallback_expected_state: Optional[str] = None,
    fallback_expected_value: Optional[str] = None,
    pre_existing_failure: Optional[str] = None,
) -> ScoreResult:
    """Score one prediction against on-disk gold (jsonl/csv) or DB fallbacks."""
    if pre_existing_failure in {
        "parse_error",
        "api_error",
        "invalid_state",
        "timeout",
        "refusal",
    }:
        return ScoreResult(False, False, None, pre_existing_failure, 0)

    allowed = resolve_allowed(test_name, raw_value)
    if not allowed and fallback_expected_state:
        allowed = [_csv_entry(fallback_expected_state, fallback_expected_value)]

    if not allowed:
        return ScoreResult(False, False, None, "missing_gold", 0)

    primary = allowed[0]
    ok_strict, tier_strict = validate_entry(primary, pred_state, pred_value, raw_value, scope)

    ok_lenient = ok_strict
    tier_lenient = tier_strict if ok_strict else None
    if not ok_lenient:
        for entry in allowed[1:]:
            ok, tier = validate_entry(entry, pred_state, pred_value, raw_value, scope)
            if ok:
                ok_lenient = True
                tier_lenient = tier
                break

    if ok_strict:
        return ScoreResult(True, True, tier_strict, None, len(allowed))
    if ok_lenient:
        return ScoreResult(False, True, tier_lenient, "wrong_policy", len(allowed))

    return ScoreResult(
        False,
        False,
        None,
        _failure_from_miss(pred_state, pred_value, primary, scope),
        len(allowed),
    )
