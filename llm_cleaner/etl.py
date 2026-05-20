"""ETL helpers: Postgres connection, insertion, failure-mode classification.

Connection details come from environment variables (see .env.example).
We try psycopg3 first; if it isn't installed (e.g. someone is running
the harness without the optional Postgres deps) we surface a clear
error.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Sequence

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "psycopg[binary]>=3.2 is required. Install with `pip install -r requirements.txt`."
    ) from e

from llm_cleaner.clients.base import CleanResult


@dataclass
class TestCaseRow:
    id: int
    test_name: str
    scenario: str
    raw_value: str
    scope: list[str]
    expected_state: str
    expected_value: Optional[str]
    notes: Optional[str] = None


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return val


def get_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "llm_cleaning")
    user = os.getenv("POSTGRES_USER", "llm")
    pwd = os.getenv("POSTGRES_PASSWORD", "llm")
    return f"host={host} port={port} dbname={db} user={user} password={pwd}"


@contextmanager
def connect() -> Iterator["psycopg.Connection"]:
    conn = psycopg.connect(get_dsn(), autocommit=False)
    try:
        yield conn
    finally:
        conn.close()


def upsert_test_case(conn: "psycopg.Connection", row: TestCaseRow) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO test_cases (test_name, scenario, raw_value, scope, expected_state, expected_value, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (test_name) DO UPDATE SET
                scenario = EXCLUDED.scenario,
                raw_value = EXCLUDED.raw_value,
                scope = EXCLUDED.scope,
                expected_state = EXCLUDED.expected_state,
                expected_value = EXCLUDED.expected_value,
                notes = EXCLUDED.notes
            RETURNING id
            """,
            (
                row.test_name,
                row.scenario,
                row.raw_value,
                row.scope,
                row.expected_state,
                row.expected_value,
                row.notes,
            ),
        )
        new_id = cur.fetchone()[0]
    return new_id


def fetch_test_cases(
    conn: "psycopg.Connection",
    *,
    scenarios: Optional[Sequence[str]] = None,
    name_prefixes: Optional[Sequence[str]] = None,
    name_in: Optional[Sequence[str]] = None,
) -> list[TestCaseRow]:
    sql = (
        "SELECT id, test_name, scenario, raw_value, scope, expected_state, expected_value, notes "
        "FROM test_cases"
    )
    clauses: list[str] = []
    params: list[Any] = []
    if scenarios:
        clauses.append("scenario = ANY(%s)")
        params.append(list(scenarios))
    if name_prefixes:
        sub = []
        for p in name_prefixes:
            sub.append("test_name LIKE %s")
            params.append(f"{p}%")
        clauses.append("(" + " OR ".join(sub) + ")")
    if name_in:
        clauses.append("test_name = ANY(%s)")
        params.append(list(name_in))
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        TestCaseRow(
            id=r[0],
            test_name=r[1],
            scenario=r[2],
            raw_value=r[3],
            scope=list(r[4]),
            expected_state=r[5],
            expected_value=r[6],
            notes=r[7],
        )
        for r in rows
    ]


def classify_failure(
    *,
    actual_state: Optional[str],
    actual_value: Optional[str],
    expected_state: str,
    expected_value: Optional[str],
    scope: Sequence[str],
    pre_existing_failure: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Compute (is_correct, failure_mode) from a single run's outputs.

    A run is correct iff:
    - state matches expected_state, AND
    - when expected_state in {acceptance, suggest}, the value matches
      expected_value (case-insensitive). On decline we require value=="".
    """
    if pre_existing_failure in {"parse_error", "api_error", "invalid_state", "timeout", "refusal"}:
        return False, pre_existing_failure

    if actual_state is None:
        return False, "invalid_state"

    if actual_state != expected_state:
        return False, "wrong_state"

    av = (actual_value or "").strip()

    if expected_state == "acceptance":
        ev = (expected_value or "").strip().lower()
        if av.lower() != ev:
            scope_lower = {s.strip().lower() for s in scope}
            if av.lower() not in scope_lower:
                return False, "hallucinated_scope"
            return False, "wrong_rename"
        return True, None

    if expected_state == "decline":
        if av != "":
            return False, "wrong_rename"
        return True, None

    if expected_state == "suggest":
        if not av:
            return False, "wrong_rename"
        if expected_value:
            ev = expected_value.strip().lower()
            if av.lower() != ev:
                return True, None
        return True, None

    return False, "unknown_expected_state"


def insert_run(
    conn: "psycopg.Connection",
    *,
    experiment_id: str,
    experiment_name: str,
    model_name: str,
    prompt_variant: str,
    mode: str,
    temperature: float,
    repetition_idx: int,
    test_case: TestCaseRow,
    result: CleanResult,
    is_correct: bool,
    failure_mode: Optional[str],
    raw_value_used: Optional[str] = None,
    scope_size: Optional[int] = None,
) -> int:
    raw_response_json: Any = None
    if result.raw_response is not None:
        try:
            raw_response_json = Jsonb(result.raw_response)
        except (TypeError, ValueError):
            raw_response_json = Jsonb({"unserializable": str(type(result.raw_response))})

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO runs (
                experiment_id, experiment_name, model_name, prompt_variant, mode, temperature,
                repetition_idx, test_case_id, test_name, raw_value, scope_size,
                actual_state, actual_value, actual_message, actual_confidence,
                expected_state, expected_value, is_correct, failure_mode,
                latency_ms, tokens_in, tokens_out, cost_usd,
                response_format_used, provider, seed, raw_response
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                experiment_id,
                experiment_name,
                model_name,
                prompt_variant,
                mode,
                temperature,
                repetition_idx,
                test_case.id,
                test_case.test_name,
                raw_value_used if raw_value_used is not None else test_case.raw_value,
                scope_size if scope_size is not None else len(test_case.scope),
                result.state,
                result.value,
                result.message,
                result.confidence,
                test_case.expected_state,
                test_case.expected_value,
                is_correct,
                failure_mode,
                result.latency_ms,
                result.tokens_in,
                result.tokens_out,
                result.cost_usd,
                result.response_format_used,
                result.provider,
                result.seed,
                raw_response_json,
            ),
        )
        new_id = cur.fetchone()[0]
    return new_id


def ensure_schema(conn: "psycopg.Connection", schema_path: str) -> None:
    """Run the schema SQL file (idempotent thanks to IF NOT EXISTS clauses)."""
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
