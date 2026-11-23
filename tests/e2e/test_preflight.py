"""Pytest wrapper around the zero-cost e2e preflight.

Skipped automatically when Postgres is unreachable so unit CI without
a DB still passes. Run explicitly:

    LLM_MOCK=1 pytest tests/e2e/test_preflight.py -q
    python tools/e2e_preflight.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["LLM_MOCK"] = "1"
os.environ.setdefault("OPENROUTER_API_KEY", "mock-e2e-no-spend")


def _postgres_ready() -> bool:
    try:
        from llm_cleaner.etl import connect

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_ready(), reason="Postgres not available")
def test_all_experiments_write_runs(tmp_path: Path) -> None:
    from tools.e2e_preflight import main

    rc = main(
        [
            "--limit",
            "2",
            "--suite",
            "small",
            "--out",
            str(tmp_path / "reports"),
        ]
    )
    assert rc == 0


@pytest.mark.skipif(not _postgres_ready(), reason="Postgres not available")
def test_mock_cleaner_no_network() -> None:
    from llm_cleaner.clients.mock_llm import MockOpenRouterCleaner
    from llm_cleaner.prompts import P0_BASELINE, build_single

    cleaner = MockOpenRouterCleaner("test/mock-model")
    messages, rf = build_single(P0_BASELINE, "Apple", ["Apple", "Banana"])
    result = cleaner.clean(
        "Apple",
        ["Apple", "Banana"],
        prompt_variant=P0_BASELINE,
        prompt_messages=messages,
        response_format=rf,
    )
    assert result.state == "acceptance"
    assert result.value == "Apple"
    assert result.provider == "mock-openrouter"
    assert result.cost_usd == 0.0
    assert result.raw_response and result.raw_response.get("mock") is True
