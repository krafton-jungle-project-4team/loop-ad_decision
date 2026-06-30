from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from app.decision.models import Experiment, ExperimentVariant
from app.decision.repositories import ClickHouseExperimentResultRepository, PostgresDecisionRepository


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
    def __init__(self, row: dict[str, object] | None) -> None:
        self.row = row
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, parameters: tuple[object, ...]) -> None:
        self.executed.append((query, parameters))

    def fetchone(self) -> dict[str, object] | None:
        return self.row

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, row: dict[str, object] | None) -> None:
        self.cursor_instance = FakeCursor(row)

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def experiment() -> Experiment:
    return Experiment(
        id=42,
        project_id=1,
        segment_id=10,
        recommendation_action_id=7,
        name="experiment",
        objective_metric="click_to_purchase_rate",
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
            ("42", "control", 410, 28, 1),
            ("42", "unknown", 999, 999, 999),
            ("999", "treatment_a", 999, 999, 999),
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
    assert "project_id = {project_id:String}" in sql
    assert "experiment_id = {experiment_id:String}" in sql
    assert "variant_id IN {variant_ids:Array(String)}" in sql
    assert "event_name IN ('ad_impression', 'ad_click', 'purchase')" in compact_sql
    assert "GROUP BY experiment_id, variant_id" in compact_sql
    assert parameters["project_id"] == "demo-shop"
    assert parameters["experiment_id"] == "42"
    assert parameters["variant_ids"] == ("control", "treatment_a")
    assert parameters["window_start_utc"] == "2021-01-03T15:00:00+00:00"
    assert parameters["window_end_utc"] == "2021-01-04T15:00:00+00:00"
    assert results[control.id].ad_impression_count == 410
    assert results[control.id].ad_click_count == 28
    assert results[control.id].attributed_purchase_count == 1
    assert results[treatment.id].ad_impression_count == 0
    assert results[treatment.id].ad_click_count == 0
    assert results[treatment.id].attributed_purchase_count == 0


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
