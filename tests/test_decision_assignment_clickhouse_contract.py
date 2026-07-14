from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

import pytest

from app.decision.repositories import UserBehaviorVectorRepository


CUTOFF = datetime(2026, 7, 14, 1, 2, 3, 456789, tzinfo=UTC)
WINDOW_START = datetime(2026, 7, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 7, 7, tzinfo=UTC)
UPDATED_AT = datetime(2026, 7, 8, 9, 10, 11, 123000, tzinfo=UTC)
VECTOR_ROW_ID = "a" * 64


@dataclass(frozen=True)
class ClickHouseCall:
    query: str
    parameters: Mapping[str, Any]


class FakeClickHouseResult:
    def __init__(self, rows: list[Any]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.calls: list[ClickHouseCall] = []

    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> FakeClickHouseResult:
        self.calls.append(ClickHouseCall(query, parameters or {}))
        return FakeClickHouseResult(self.rows)


def compact_sql(query: str) -> str:
    return " ".join(query.lower().split())


def revision_row(
    *,
    vector_dim: int = 64,
    vector_row_id: Any = VECTOR_ROW_ID,
    window_start: Any = WINDOW_START,
    window_end: Any = WINDOW_END,
    updated_at: Any = UPDATED_AT,
) -> tuple[Any, ...]:
    return (
        "hotel-client-a",
        "user_001",
        "v1",
        vector_dim,
        [1.0] + [0.0] * 63,
        "booking_profile",
        window_start,
        window_end,
        updated_at,
        vector_row_id,
    )


def test_get_source_cutoff_uses_clickhouse_utc_microsecond_clock() -> None:
    driver_value = CUTOFF.replace(tzinfo=None)
    client = FakeClickHouseClient(rows=[(driver_value,)])

    cutoff = UserBehaviorVectorRepository(client).get_source_cutoff()

    assert cutoff == CUTOFF
    assert client.calls[0].query == "SELECT now64(6, 'UTC') AS source_cutoff_at"
    assert client.calls[0].parameters == {}


def test_driver_naive_revision_timestamps_are_normalized_from_typed_utc_columns() -> None:
    client = FakeClickHouseClient(
        rows=[
            revision_row(
                window_start=WINDOW_START.replace(tzinfo=None),
                window_end=WINDOW_END.replace(tzinfo=None),
                updated_at=UPDATED_AT.replace(tzinfo=None),
            )
        ]
    )

    record = UserBehaviorVectorRepository(client).list_by_user_ids(
        project_id="hotel-client-a",
        user_ids=["user_001"],
        vector_version="v1",
        source_cutoff_at=CUTOFF,
    )[0]

    assert record.window_start == WINDOW_START
    assert record.window_end == WINDOW_END
    assert record.updated_at == UPDATED_AT


def test_explicit_users_use_canonical_revision_winner_and_strict_cutoff() -> None:
    client = FakeClickHouseClient(rows=[revision_row(vector_dim=63)])
    repository = UserBehaviorVectorRepository(client)

    records = repository.list_by_user_ids(
        project_id="hotel-client-a",
        user_ids=["user_001"],
        vector_version="v1",
        source_cutoff_at=CUTOFF,
        source="booking_profile",
    )

    assert records[0].vector_dim == 63
    assert records[0].vector_row_id == VECTOR_ROW_ID
    assert records[0].window_start == WINDOW_START
    assert records[0].window_end == WINDOW_END
    assert records[0].updated_at == UPDATED_AT

    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from user_behavior_vector_revisions" in sql
    assert (
        "argmax( tuple( vector_dim, vector_values, cast(source, 'string'), "
        "window_start, window_end, updated_at, vector_row_id ), "
        "tuple(updated_at, vector_row_id) ) as selected_payload"
    ) in sql
    assert "ingested_at < {source_cutoff_at:datetime64(6, 'utc')}" in sql
    assert "ingested_at <=" not in sql
    assert "and source = {source:string}" in sql
    assert sql.index("and source = {source:string}") < sql.index(
        "group by project_id, user_id, vector_version"
    )
    assert "and user_id in {user_ids:array(string)}" in sql
    assert "where tupleelement(selected_payload, 1)" not in sql
    assert call.parameters == {
        "project_id": "hotel-client-a",
        "vector_version": "v1",
        "source_cutoff_at": CUTOFF,
        "user_ids": ["user_001"],
        "source": "booking_profile",
    }


def test_project_page_uses_same_canonical_query_with_tuple_keyset() -> None:
    client = FakeClickHouseClient(rows=[])
    repository = UserBehaviorVectorRepository(client)

    records = repository.list_for_project(
        project_id="hotel-client-a",
        vector_version="v1",
        limit=10_000,
        source_cutoff_at=CUTOFF,
        source="booking_profile",
        after_user_id="user_000999",
    )

    assert records == []
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from user_behavior_vector_revisions" in sql
    assert "tuple(updated_at, vector_row_id)" in sql
    assert (
        "tuple(user_id, vector_version) > "
        "tuple({after_user_id:string}, {vector_version:string})"
    ) in sql
    assert "order by user_id asc, vector_version asc" in sql
    assert "limit {limit:uint32}" in sql
    assert "user_id in" not in sql
    assert call.parameters == {
        "project_id": "hotel-client-a",
        "vector_version": "v1",
        "source_cutoff_at": CUTOFF,
        "after_user_id": "user_000999",
        "source": "booking_profile",
        "limit": 10_000,
    }


@pytest.mark.parametrize(
    ("row", "field_name"),
    [
        (revision_row(vector_row_id="A" * 64), "vector_row_id"),
        (revision_row(window_start=None), "window_start"),
        (revision_row(window_end="2026-07-07"), "window_end"),
        (revision_row(updated_at=None), "updated_at"),
    ],
)
def test_malformed_revision_provenance_fails_the_read(
    row: tuple[Any, ...],
    field_name: str,
) -> None:
    repository = UserBehaviorVectorRepository(FakeClickHouseClient(rows=[row]))

    with pytest.raises(ValueError, match=field_name):
        repository.list_by_user_ids(
            project_id="hotel-client-a",
            user_ids=["user_001"],
            vector_version="v1",
            source_cutoff_at=CUTOFF,
        )


def test_source_cutoff_must_be_timezone_aware() -> None:
    repository = UserBehaviorVectorRepository(FakeClickHouseClient(rows=[]))

    with pytest.raises(ValueError, match="source_cutoff_at"):
        repository.list_for_project(
            project_id="hotel-client-a",
            vector_version="v1",
            limit=10,
            source_cutoff_at=datetime(2026, 7, 14),
        )
