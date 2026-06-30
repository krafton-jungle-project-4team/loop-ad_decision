from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.analysis.models import BaselineMetrics
from app.analysis.models import SegmentAggregate
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

    def fetchall(self) -> list[tuple[object, ...]]:
        rows = [row for row in self.rows if row is not None]
        self.rows.clear()
        return rows


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


def test_refresh_segment_centroids_reads_active_non_default_segments_then_cleans_stale_rows() -> None:
    connection = FakeConnection(
        rows=[
            ("segment_match_v2",),
            (
                10,
                {
                    "age_group": "30s",
                    "gender": "male",
                    "device_type": "mobile",
                    "acquisition_channel": "kakao",
                    "primary_category": "fresh",
                },
            ),
            (
                20,
                {
                    "age_group": "40s",
                    "gender": "female",
                    "device_type": "pc",
                    "acquisition_channel": "naver",
                    "primary_category": "beauty",
                },
            ),
        ]
    )
    repository = PostgresAnalysisRepository(connection)

    refreshed_count = repository.refresh_segment_centroids(
        project_id=1,
        analysis_date=date(2021, 1, 4),
    )

    config_query, config_parameters = connection.cursor_instance.executed[0]
    source_query, source_parameters = connection.cursor_instance.executed[1]
    first_upsert_query, first_upsert_parameters = connection.cursor_instance.executed[2]
    cleanup_query, cleanup_parameters = connection.cursor_instance.executed[4]
    assert refreshed_count == 2
    assert "FROM segment_matching_configs" in config_query
    assert "analysis_date = %s" in config_query
    assert config_parameters == (1, date(2021, 1, 4))
    assert "FROM segments" in source_query
    assert "is_default = false" in source_query
    assert "status = 'active'" in source_query
    assert "INSERT INTO segments" not in source_query
    assert source_parameters == (1,)
    assert "INSERT INTO segment_centroids" in first_upsert_query
    assert first_upsert_parameters[:4] == (
        1,
        10,
        date(2021, 1, 4),
        "segment_match_v2",
    )
    assert "DELETE FROM segment_centroids" in cleanup_query
    assert "embedding_version = %s" in cleanup_query
    assert "NOT (segment_id = ANY(%s))" in cleanup_query
    assert cleanup_parameters == (
        1,
        date(2021, 1, 4),
        "segment_match_v2",
        [10, 20],
    )


def test_refresh_segment_centroids_does_not_delete_other_embedding_versions() -> None:
    connection = FakeConnection(rows=[("segment_match_v2",)])
    repository = PostgresAnalysisRepository(connection)

    repository.refresh_segment_centroids(
        project_id=1,
        analysis_date=date(2021, 1, 4),
    )

    cleanup_query, cleanup_parameters = connection.cursor_instance.executed[-1]
    assert "DELETE FROM segment_centroids" in cleanup_query
    assert "embedding_version = %s" in cleanup_query
    assert cleanup_parameters == (1, date(2021, 1, 4), "segment_match_v2")


def test_refresh_segment_centroids_removes_today_version_rows_when_segment_becomes_inactive() -> None:
    connection = FakeConnection(rows=[("segment_match_v1",)])
    repository = PostgresAnalysisRepository(connection)

    refreshed_count = repository.refresh_segment_centroids(
        project_id=1,
        analysis_date=date(2021, 1, 4),
    )

    cleanup_query, cleanup_parameters = connection.cursor_instance.executed[-1]
    assert refreshed_count == 0
    assert "AND NOT (segment_id = ANY(%s))" not in cleanup_query
    assert cleanup_parameters == (1, date(2021, 1, 4), "segment_match_v1")


def test_fetch_segment_metric_baselines_uses_previous_seven_days_only() -> None:
    connection = FakeConnection(rows=[(10, Decimal("0.06"))])
    repository = PostgresAnalysisRepository(connection)

    baselines = repository.fetch_segment_metric_baselines(
        project_id=1,
        analysis_date=date(2021, 1, 8),
        stored_segments={"segment": type("Stored", (), {"id": 10})()},
    )

    query, parameters = connection.cursor_instance.executed[0]
    assert "analysis_date >= %s" in query
    assert "analysis_date <= %s" in query
    assert parameters[2] == date(2021, 1, 1)
    assert parameters[3] == date(2021, 1, 7)
    assert baselines[10].view_to_purchase_rate == Decimal("0.06")


def test_update_segment_daily_metric_baselines_updates_current_analysis_date_rows() -> None:
    connection = FakeConnection(rows=[])
    repository = PostgresAnalysisRepository(connection)

    updated_count = repository.update_segment_daily_metric_baselines(
        project_id=1,
        analysis_date=date(2021, 1, 8),
        baselines={
            10: BaselineMetrics(
                segment_id=10,
                view_to_purchase_rate=Decimal("0.06"),
            )
        },
    )

    query, parameters = connection.cursor_instance.executed[0]
    assert updated_count == 1
    assert "UPDATE segment_daily_metrics" in query
    assert "baseline_view_to_purchase_rate = %s" in query
    assert parameters == (Decimal("0.06"), 1, 10, date(2021, 1, 8))


def test_upsert_segment_anomalies_and_root_causes_use_schema_keys() -> None:
    connection = FakeConnection(rows=[(501, 10)])
    repository = PostgresAnalysisRepository(connection)

    stored_anomalies = repository.upsert_segment_anomalies(
        project_id=1,
        analysis_date=date(2021, 1, 8),
        anomalies=[
            type(
                "Anomaly",
                (),
                {
                    "segment_id": 10,
                    "metric_name": "view_to_purchase_rate",
                    "actual_value": Decimal("0.01"),
                    "expected_value": Decimal("0.05"),
                    "target_value": Decimal("0.05"),
                    "difference_value": Decimal("0.04"),
                    "difference_rate": Decimal("0.8"),
                    "severity": "medium",
                    "impact_score": Decimal("4"),
                    "evidence_json": {},
                },
            )()
        ],
        run_id=None,
    )

    root_count = repository.upsert_root_cause_candidates(
        [
            type(
                "RootCause",
                (),
                {
                    "anomaly_id": 501,
                    "cause_type": "funnel_step_drop",
                    "cause_key": "view_to_cart",
                    "title": "상품 조회 후 장바구니 전환 낮음",
                    "description": "전환율이 가장 낮습니다.",
                    "confidence_score": Decimal("0.7"),
                    "impact_score": Decimal("0.9"),
                    "rank_no": 1,
                    "evidence_json": {},
                },
            )()
        ]
    )

    anomaly_query, anomaly_parameters = connection.cursor_instance.executed[0]
    root_query, _ = connection.cursor_instance.executed[1]
    assert stored_anomalies[0].id == 501
    assert stored_anomalies[0].segment_id == 10
    assert anomaly_parameters[-1] is None
    assert "ON CONFLICT (project_id, segment_id, analysis_date, metric_name)" in anomaly_query
    assert "ON CONFLICT (anomaly_id, cause_type, cause_key)" in root_query
    assert root_count == 1
