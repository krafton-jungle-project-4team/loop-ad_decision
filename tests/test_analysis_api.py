from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.main import create_app


def valid_env() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "15432",
            "LOOPAD_OPENAI_CONTENT_MODEL": "gpt-test",
        }
    )
    return values


def analysis_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "operator_instruction": None,
    }
    payload.update(overrides)
    return payload


def promotion_row() -> dict[str, Any]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "channel": "onsite_banner",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.030000"),
        "goal_basis": "all_segments",
        "min_sample_size": 1000,
        "landing_url": "https://demo-stay.example.com/summer",
        "message_brief": "Drive summer hotel booking.",
    }


def segment_rows() -> list[dict[str, Any]]:
    return [
        {
            "segment_id": "seg_repeat_hotel_no_booking",
            "project_id": "hotel-client-a",
            "segment_name": "Repeat hotel viewers without booking",
            "source": "system_default",
            "query_preview_id": None,
            "natural_language_query": "same hotel views without booking",
            "generated_sql": None,
            "rule_json": {"segment_id": "seg_repeat_hotel_no_booking"},
            "profile_json": {"primary_segment": "seg_repeat_hotel_no_booking"},
            "sample_size": 1342,
            "total_eligible_user_count": 74200,
            "sample_ratio": Decimal("0.018000"),
            "status": "active",
        }
    ]


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def test_analysis_api_wires_data_repositories(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            fetchone_results=[promotion_row()],
            fetchall_results=[segment_rows()],
        )
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient(rows=[])
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.analysis.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.analysis.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/analysis",
        json=analysis_payload(),
    )

    assert response.status_code == 200
    segment_vector_id = response.json()["target_segments"][0]["segment_vector_id"]
    assert segment_vector_id.startswith("segvec_seg_repeat_hotel_no_booking_v1_")
    assert len(segment_vector_id) <= 100
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1

    executed_queries = [compact_sql(query) for query, _params in connection.executed]
    assert (
        sum("insert into promotion_analyses" in query for query in executed_queries)
        == 1
    )
    assert (
        sum("insert into segment_vectors" in query for query in executed_queries)
        == 1
    )
    assert (
        sum(
            "insert into promotion_target_segments" in query
            for query in executed_queries
        )
        == 1
    )


def test_analysis_api_rolls_back_when_repository_write_fails(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            fetchone_results=[promotion_row()],
            fetchall_results=[segment_rows()],
            fail_on_execute="INSERT INTO segment_vectors",
        )
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient(rows=[])
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.analysis.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.analysis.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/analysis",
        json=analysis_payload(),
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1


@dataclass(frozen=True)
class RecordingClickHouseResult:
    result_rows: list[Any]


class RecordingClickHouseClient:
    def __init__(self, *, rows: list[Any]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.close_count = 0

    def query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> RecordingClickHouseResult:
        self.queries.append((query, parameters or {}))
        return RecordingClickHouseResult(self.rows)

    def close(self) -> None:
        self.close_count += 1


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self._connection = connection

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: object = ()) -> None:
        if (
            self._connection.fail_on_execute
            and self._connection.fail_on_execute in query
        ):
            raise RuntimeError("forced repository write failure")
        self._connection.executed.append((query, params))

    def fetchone(self) -> dict[str, Any] | None:
        if not self._connection.fetchone_results:
            return None
        return self._connection.fetchone_results.pop(0)

    def fetchall(self) -> list[dict[str, Any]]:
        if not self._connection.fetchall_results:
            return []
        return self._connection.fetchall_results.pop(0)


class RecordingConnection:
    def __init__(
        self,
        *,
        fetchone_results: list[dict[str, Any]] | None = None,
        fetchall_results: list[list[dict[str, Any]]] | None = None,
        fail_on_execute: str | None = None,
    ) -> None:
        self.fetchone_results = fetchone_results or []
        self.fetchall_results = fetchall_results or []
        self.fail_on_execute = fail_on_execute
        self.executed: list[tuple[str, object]] = []
        self.row_factories: list[object] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self, *, row_factory: object = None) -> RecordingCursor:
        self.row_factories.append(row_factory)
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1
