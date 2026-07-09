from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.main import create_app


DEFAULT_ROW = object()


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


def analysis_payload() -> dict[str, Any]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "operator_instruction": None,
    }


def test_analysis_api_wires_real_service_and_commits(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
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
    payload = response.json()
    assert re.fullmatch(
        r"analysis_promo_banner_001_run_[0-9a-f]{8}",
        payload["analysis_id"],
    )
    assert payload["promotion_id"] == "promo_banner_001"
    assert payload["status"] == "completed"
    assert [segment["segment_id"] for segment in payload["target_segments"]] == [
        "seg_family_trip",
        "seg_mobile_user",
        "seg_repeat_hotel_no_booking",
        "seg_near_checkin",
    ]

    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1

    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("from promotions" in query for query in executed_sql)
    assert any("from segment_definitions" in query for query in executed_sql)
    assert any("insert into promotion_analyses" in query for query in executed_sql)
    assert any("insert into segment_vectors" in query for query in executed_sql)
    assert any(
        "insert into promotion_segment_suggestions" in query for query in executed_sql
    )
    assert not any(
        "insert into promotion_target_segments" in query for query in executed_sql
    )
    assert any(
        "from user_behavior_vectors" in compact_sql(query)
        for query, _parameters in clickhouse_clients[0].queries
    )


def test_analysis_api_rolls_back_when_promotion_is_missing(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(promotion_row=None)
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
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

    assert response.status_code == 404
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1


def test_analysis_api_rolls_back_when_segment_vector_data_is_unavailable(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient(user_vector_rows=[])
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

    assert response.status_code == 422
    assert response.json()["detail"] == "segment vector data unavailable"
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self._connection = connection
        self._last_query = ""

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: Any = None) -> None:
        self._last_query = query
        self._connection.executed.append((query, params))

    def fetchone(self) -> dict[str, object] | None:
        sql = compact_sql(self._last_query)
        if "from promotions" in sql:
            return self._connection.promotion_row
        if "from segment_vectors" in sql:
            return None
        return None

    def fetchall(self) -> list[dict[str, object]]:
        sql = compact_sql(self._last_query)
        if "from segment_definitions" in sql:
            return segment_definition_rows()
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        promotion_row: dict[str, object] | None | object = DEFAULT_ROW,
    ) -> None:
        self.promotion_row = (
            promotion_record_row() if promotion_row is DEFAULT_ROW else promotion_row
        )
        self.executed: list[tuple[str, Any]] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self, *_args: object, **_kwargs: object) -> RecordingCursor:
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1


class RecordingClickHouseClient:
    def __init__(
        self,
        *,
        user_vector_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.queries: list[tuple[str, Any]] = []
        self.close_count = 0
        self.user_vector_rows = (
            user_behavior_vector_rows()
            if user_vector_rows is None
            else user_vector_rows
        )

    def query(self, query: str, parameters: Any = None) -> "ClickHouseResult":
        self.queries.append((query, parameters))
        sql = compact_sql(query)
        if "from user_behavior_vectors" in sql and "user_id in" in sql:
            return ClickHouseResult(self.user_vector_rows)
        return ClickHouseResult([])

    def close(self) -> None:
        self.close_count += 1


class ClickHouseResult:
    def __init__(self, rows: list[Any]) -> None:
        self.result_rows = rows


def promotion_record_row() -> dict[str, object]:
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


def segment_definition_rows() -> list[dict[str, object]]:
    return [
        segment_definition_row("seg_mobile_user", 2400),
        segment_definition_row("seg_family_trip", 3200),
        segment_definition_row("seg_near_checkin", 1800),
        segment_definition_row("seg_repeat_hotel_no_booking", 1342),
        segment_definition_row("seg_existing_all", 9000),
    ]


def segment_definition_row(segment_id: str, sample_size: int) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "project_id": "hotel-client-a",
        "campaign_id": None,
        "promotion_id": None,
        "segment_name": segment_id.replace("_", " ").title(),
        "source": "system_default",
        "query_preview_id": None,
        "natural_language_query": f"{segment_id} hotel audience",
        "generated_sql": None,
        "rule_json": {
            "segment_id": segment_id,
            "candidate_user_ids": [
                f"user_{segment_id}_001",
                f"user_{segment_id}_002",
            ],
        },
        "profile_json": {"primary_segment": segment_id},
        "sample_size": sample_size,
        "total_eligible_user_count": 74200,
        "sample_ratio": Decimal("0.020000"),
        "status": "active",
    }


def user_behavior_vector_rows() -> list[dict[str, object]]:
    return [
        {
            "project_id": "hotel-client-a",
            "user_id": "user_vector_fixture_001",
            "vector_dim": 64,
            "vector_values": [1.0, *([0.0] * 63)],
            "vector_version": "v1",
            "source": "batch_profile",
        }
    ]


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()
