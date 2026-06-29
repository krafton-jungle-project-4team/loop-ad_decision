from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.analysis.models import SegmentAggregate
from app.analysis.models import UserPrimarySegmentCandidate
from app.analysis.postgres_repository import PostgresAnalysisRepository
import pytest


class FakeCursor:
    def __init__(self, rows: list[tuple[object, ...] | None]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, parameters: tuple[object, ...] = ()) -> None:
        self.executed.append((query, parameters))

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows.pop(0) if self.rows else None


class FakeConnection:
    def __init__(self, rows: list[tuple[object, ...] | None]) -> None:
        self.cursor_instance = FakeCursor(rows)

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def aggregate(segment_key: str = "age_30s__gender_male__device_mobile__channel_kakao__category_fresh") -> SegmentAggregate:
    return SegmentAggregate(
        project_id=1,
        segment_key=segment_key,
        name="30s / male / mobile / kakao / fresh",
        dimensions={
            "age_group": "30s",
            "gender": "male",
            "device_type": "mobile",
            "acquisition_channel": "kakao",
            "primary_category": "fresh",
        },
        user_count=30,
        session_count=40,
        page_view_count=100,
        product_view_count=100,
        add_to_cart_count=20,
        checkout_start_count=10,
        purchase_count=5,
        ad_impression_count=0,
        ad_click_count=0,
        revenue=Decimal("1000"),
        view_to_cart_rate=Decimal("0.2"),
        cart_to_checkout_rate=Decimal("0.5"),
        checkout_to_purchase_rate=Decimal("0.5"),
        view_to_purchase_rate=Decimal("0.05"),
        ctr=None,
        cvr=Decimal("0.05"),
    )


def test_upsert_segments_uses_schema_unique_key_and_nullable_run_id() -> None:
    connection = FakeConnection(rows=[(10, aggregate().segment_key)])
    repository = PostgresAnalysisRepository(connection)

    stored_segments = repository.upsert_segments(
        project_id=1,
        aggregates=[aggregate()],
        run_id=None,
    )

    query, parameters = connection.cursor_instance.executed[0]
    assert "ON CONFLICT (project_id, segment_key)" in query
    assert "created_run_id" in query
    assert parameters[-1] is None
    assert stored_segments[aggregate().segment_key].id == 10


def test_get_project_timezone_falls_back_to_seoul_when_empty() -> None:
    connection = FakeConnection(rows=[("  ",)])
    repository = PostgresAnalysisRepository(connection)

    assert repository.get_project_timezone(1) == "Asia/Seoul"


def test_get_project_timezone_rejects_invalid_timezone_before_window_build() -> None:
    connection = FakeConnection(rows=[("Mars/Seoul",)])
    repository = PostgresAnalysisRepository(connection)

    with pytest.raises(ValueError, match="invalid timezone"):
        repository.get_project_timezone(1)


def test_get_project_key_returns_non_empty_key_for_clickhouse_lookup() -> None:
    connection = FakeConnection(rows=[(" demo-shop ",)])
    repository = PostgresAnalysisRepository(connection)

    project_key = repository.get_project_key(1)

    query, parameters = connection.cursor_instance.executed[0]
    assert project_key == "demo-shop"
    assert "SELECT project_key FROM projects WHERE id = %s" in query
    assert parameters == (1,)


def test_get_project_key_rejects_empty_key() -> None:
    connection = FakeConnection(rows=[(None,)])
    repository = PostgresAnalysisRepository(connection)

    with pytest.raises(ValueError, match="project_key is required"):
        repository.get_project_key(1)


def test_upsert_segments_skips_default_segment_key() -> None:
    connection = FakeConnection(rows=[])
    repository = PostgresAnalysisRepository(connection)

    assert repository.upsert_segments(project_id=1, aggregates=[aggregate("default")], run_id=7) == {}
    assert connection.cursor_instance.executed == []


def test_upsert_segment_daily_metrics_uses_schema_unique_key() -> None:
    connection = FakeConnection(rows=[(10, aggregate().segment_key)])
    repository = PostgresAnalysisRepository(connection)
    stored_segments = repository.upsert_segments(1, [aggregate()], run_id=7)

    metric_count = repository.upsert_segment_daily_metrics(
        project_id=1,
        analysis_date=date(2021, 1, 4),
        aggregates=[aggregate()],
        stored_segments=stored_segments,
        run_id=None,
    )

    query, parameters = connection.cursor_instance.executed[1]
    assert metric_count == 1
    assert "ON CONFLICT (project_id, segment_id, analysis_date)" in query
    assert "baseline_view_to_purchase_rate" in query
    assert parameters[-1] is None


def test_upsert_user_segment_memberships_only_uses_stored_segments() -> None:
    connection = FakeConnection(rows=[(10, aggregate().segment_key)])
    repository = PostgresAnalysisRepository(connection)
    stored_segments = repository.upsert_segments(1, [aggregate()], run_id=7)

    membership_count = repository.upsert_user_segment_memberships(
        project_id=1,
        analysis_date=date(2021, 1, 4),
        candidates=[
            UserPrimarySegmentCandidate(
                external_user_id="user-1",
                segment_key=aggregate().segment_key,
                dimensions=aggregate().dimensions,
            ),
            UserPrimarySegmentCandidate(
                external_user_id="user-2",
                segment_key="missing",
                dimensions={},
            ),
        ],
        stored_segments=stored_segments,
        run_id=None,
    )

    delete_query, _ = connection.cursor_instance.executed[1]
    upsert_query, upsert_parameters = connection.cursor_instance.executed[2]
    assert membership_count == 1
    assert "DELETE FROM user_segment_memberships" in delete_query
    assert "ON CONFLICT (project_id, external_user_id, segment_id, analysis_date)" in upsert_query
    assert upsert_parameters[-1] is None
