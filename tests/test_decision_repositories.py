from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from app.decision.models import Experiment, ExperimentVariant
from app.decision.repositories import (
    ClickHouseExperimentResultRepository,
    ClickHouseUserSegmentCandidateRepository,
    PostgresDecisionRepository,
)


class FakeClickHouseResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict[str, object]]] = []

    def query(self, sql: str, parameters: dict[str, object]) -> FakeClickHouseResult:
        self.queries.append((sql, parameters))
        return FakeClickHouseResult(self.rows)


class FakeCursor:
    def __init__(
        self,
        row: dict[str, object] | None = None,
        rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.row = row
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, parameters: tuple[object, ...]) -> None:
        self.executed.append((query, parameters))

    def fetchone(self) -> dict[str, object] | None:
        return self.row

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(
        self,
        row: dict[str, object] | None = None,
        rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.cursor_instance = FakeCursor(row=row, rows=rows)

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def experiment() -> Experiment:
    return Experiment(
        id=42,
        project_id=1,
        segment_id=10,
        recommendation_action_id=7,
        name="experiment",
        objective_metric="view_to_purchase_rate",
        target_value=Decimal("0.05"),
        allocation_policy="fixed_split",
        status="running",
        start_date=date(2021, 1, 4),
    )


def variant(*, variant_id: int, variant_key: str) -> ExperimentVariant:
    return ExperimentVariant(
        id=variant_id,
        experiment_id=42,
        project_id=1,
        variant_key=variant_key,
        name=variant_key,
        generated_content_id=variant_id + 100,
        is_control=variant_key == "control",
        traffic_weight=Decimal("0.5"),
        impression_count=0,
        click_count=0,
        conversion_count=0,
        ctr=Decimal("0"),
        conversion_rate=Decimal("0"),
        status="active",
    )


def test_clickhouse_experiment_results_use_loopad_events_contract() -> None:
    client = FakeClickHouseClient(
        rows=[
            ("42", "11", "111", 410, 28, 1),
            ("42", "12", "999", 999, 999, 999),
            ("42", "unknown", "111", 999, 999, 999),
            ("999", "11", "111", 999, 999, 999),
        ]
    )
    repository = ClickHouseExperimentResultRepository(client)
    control = variant(variant_id=11, variant_key="control")
    treatment = variant(variant_id=12, variant_key="treatment_a")

    results = repository.fetch_variant_results(
        project_id="demo-shop",
        experiment=experiment(),
        variants=[control, treatment],
        window_start=datetime(2021, 1, 4, tzinfo=ZoneInfo("Asia/Seoul")),
        window_end=datetime(2021, 1, 5, tzinfo=ZoneInfo("Asia/Seoul")),
    )

    sql, parameters = client.queries[0]
    compact_sql = " ".join(sql.split())
    assert "experiment_variant_id" not in sql
    assert "generated_content_id" not in sql
    assert "creative_id" in sql
    assert "project_id = {project_id:String}" in sql
    assert "experiment_id = {experiment_id:String}" in sql
    assert "variant_id IN {variant_ids:Array(String)}" in sql
    assert "event_name IN ('ad_impression', 'ad_click', 'purchase')" in compact_sql
    assert "GROUP BY experiment_id, variant_id, creative_id" in compact_sql
    assert parameters["project_id"] == "demo-shop"
    assert parameters["experiment_id"] == "42"
    assert parameters["variant_ids"] == ("11", "12")
    assert parameters["window_start_utc"] == "2021-01-03T15:00:00+00:00"
    assert parameters["window_end_utc"] == "2021-01-04T15:00:00+00:00"
    assert results[control.id].ad_impression_count == 410
    assert results[control.id].ad_click_count == 28
    assert results[control.id].attributed_purchase_count == 1
    assert results[treatment.id].ad_impression_count == 0
    assert results[treatment.id].ad_click_count == 0
    assert results[treatment.id].attributed_purchase_count == 0


def test_clickhouse_experiment_results_skip_variants_without_generated_content() -> None:
    client = FakeClickHouseClient(rows=[])
    repository = ClickHouseExperimentResultRepository(client)
    control = variant(variant_id=11, variant_key="control")
    draft = variant(variant_id=12, variant_key="treatment_a")
    draft.generated_content_id = None

    results = repository.fetch_variant_results(
        project_id="demo-shop",
        experiment=experiment(),
        variants=[control, draft],
        window_start=datetime(2021, 1, 4, tzinfo=ZoneInfo("Asia/Seoul")),
        window_end=datetime(2021, 1, 5, tzinfo=ZoneInfo("Asia/Seoul")),
    )

    _, parameters = client.queries[0]
    assert parameters["variant_ids"] == ("11",)
    assert results[draft.id].ad_impression_count == 0
    assert results[draft.id].ad_click_count == 0
    assert results[draft.id].attributed_purchase_count == 0


def test_postgres_decision_repository_returns_project_key_for_clickhouse_lookup() -> None:
    connection = FakeConnection({"project_key": " demo-shop "})
    repository = PostgresDecisionRepository(connection)

    assert repository.get_project_key(project_id=1) == "demo-shop"
    query, parameters = connection.cursor_instance.executed[0]
    assert query == "SELECT project_key FROM projects WHERE id = %s"
    assert parameters == (1,)


def test_postgres_decision_repository_rejects_empty_project_key() -> None:
    connection = FakeConnection({"project_key": "  "})
    repository = PostgresDecisionRepository(connection)

    with pytest.raises(ValueError, match="project_key is required"):
        repository.get_project_key(project_id=1)


def test_postgres_decision_repository_upserts_experiment_with_active_partial_index() -> None:
    connection = FakeConnection(
        {
            "id": 42,
            "project_id": 1,
            "segment_id": 10,
            "recommendation_action_id": 7,
            "name": "experiment",
            "objective_metric": "view_to_purchase_rate",
            "target_value": Decimal("0.05"),
            "allocation_policy": "fixed_split",
            "status": "running",
            "start_date": date(2021, 1, 4),
            "winner_variant_id": None,
        }
    )
    repository = PostgresDecisionRepository(connection)

    repository.upsert_experiment(
        project_id=1,
        segment_id=10,
        recommendation_action_id=7,
        name="experiment",
        objective_metric="view_to_purchase_rate",
        target_value=Decimal("0.05"),
        allocation_policy="fixed_split",
        status="running",
        start_date=date(2021, 1, 4),
        run_id=100,
    )

    query, parameters = connection.cursor_instance.executed[0]
    assert "ON CONFLICT (project_id, recommendation_action_id)" in query
    assert "WHERE recommendation_action_id IS NOT NULL" in query
    assert "AND status IN ('draft', 'running', 'paused')" in query
    assert parameters == (
        1,
        10,
        7,
        "experiment",
        "view_to_purchase_rate",
        Decimal("0.05"),
        "fixed_split",
        "running",
        date(2021, 1, 4),
        100,
    )


def test_postgres_decision_repository_lists_existing_segments_read_only() -> None:
    connection = FakeConnection(
        rows=[
            {
                "id": 10,
                "segment_key": "age_30s__gender_male__device_mobile__channel_kakao__category_fresh",
            }
        ]
    )
    repository = PostgresDecisionRepository(connection)

    segments = repository.list_existing_segments(project_id=1)

    query, parameters = connection.cursor_instance.executed[0]
    assert "SELECT id, segment_key" in query
    assert "FROM segments" in query
    assert "INSERT INTO segments" not in query
    assert "UPDATE segments" not in query
    assert "DELETE FROM segments" not in query
    assert parameters == (1,)
    assert segments["age_30s__gender_male__device_mobile__channel_kakao__category_fresh"].id == 10


def test_postgres_decision_repository_replaces_primary_membership_in_one_transaction() -> None:
    connection = FakeConnection()
    repository = PostgresDecisionRepository(connection)

    repository.replace_primary_membership(
        project_id=1,
        external_user_id="user-1",
        segment_id=10,
        analysis_date=date(2021, 1, 4),
        confidence=Decimal("1.0"),
        reason_json={"segment_key": "segment"},
        run_id=None,
    )

    delete_query, delete_parameters = connection.cursor_instance.executed[0]
    upsert_query, upsert_parameters = connection.cursor_instance.executed[1]
    assert "DELETE FROM user_segment_memberships" in delete_query
    assert "INSERT INTO user_segment_memberships" in upsert_query
    assert "ON CONFLICT (project_id, external_user_id, segment_id, analysis_date)" in upsert_query
    assert "INSERT INTO segments" not in delete_query + upsert_query
    assert "UPDATE segments" not in delete_query + upsert_query
    assert "DELETE FROM segments" not in delete_query + upsert_query
    assert delete_parameters == (1, "user-1", date(2021, 1, 4), 10)
    assert upsert_parameters[-1] is None


def test_clickhouse_user_segment_candidates_use_real_event_columns_and_internal_names() -> None:
    client = FakeClickHouseClient(
        rows=[
            (
                "user-1",
                "30s",
                "Male",
                "Mobile Web",
                "Kakao",
                "Fresh Food",
            )
        ]
    )
    repository = ClickHouseUserSegmentCandidateRepository(client)

    candidates = repository.fetch_user_segment_candidates(
        project_id="demo-shop",
        window=type(
            "Window",
            (),
            {
                "window_start": datetime(2021, 1, 4, tzinfo=ZoneInfo("Asia/Seoul")),
                "window_end": datetime(2021, 1, 5, tzinfo=ZoneInfo("Asia/Seoul")),
            },
        )(),
    )

    sql, parameters = client.queries[0]
    assert "events.device" in sql
    assert "events.channel" in sql
    assert "events.category" in sql
    assert "device_type" in sql
    assert "acquisition_channel" in sql
    assert "primary_category" in sql
    assert "events.device_type" not in sql
    assert "events.acquisition_channel" not in sql
    assert "events.primary_category" not in sql
    assert candidates[0].dimensions == {
        "age_group": "30s",
        "gender": "Male",
        "device_type": "Mobile Web",
        "acquisition_channel": "Kakao",
        "primary_category": "Fresh Food",
    }
    assert parameters["project_id"] == "demo-shop"
