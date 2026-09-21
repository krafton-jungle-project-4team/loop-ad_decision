from __future__ import annotations

from pathlib import Path

from scripts.backfill_expedia_raw_events import (
    UNBOUNDED_SORT,
    build_query,
)


SQL = Path("scripts/expedia_to_raw_events.sql").read_text(encoding="utf-8")


def test_unlimited_full_backfill_streams_without_global_source_sort() -> None:
    query = build_query(SQL, mode="execute", max_source_rows=0)
    assert UNBOUNDED_SORT not in query
    assert "LIMIT source_row_limit" in query
    assert "INSERT INTO raw_events" in query


def test_finite_backfill_keeps_deterministic_source_prefix() -> None:
    query = build_query(SQL, mode="execute", max_source_rows=1_000_000)
    assert UNBOUNDED_SORT in query


def test_unlimited_preview_uses_the_same_streaming_transform() -> None:
    query = build_query(SQL, mode="preview", max_source_rows=0)
    assert UNBOUNDED_SORT not in query
    assert "GROUP BY event_name" in query
