from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.internal.schemas import UserBehaviorVectorBuildRequest
from app.internal.user_behavior_vectors import (
    UserBehaviorVectorBatchService,
    UserBehaviorVectorBuildRepository,
)
from app.main import create_app


@dataclass(frozen=True)
class ClickHouseCall:
    operation: str
    query: str
    parameters: Mapping[str, Any]


class FakeClickHouseResult:
    def __init__(self, rows: list[Any]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(self, *, processed_user_count: int = 3) -> None:
        self.processed_user_count = processed_user_count
        self.calls: list[ClickHouseCall] = []
        self.close_count = 0

    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> FakeClickHouseResult:
        self.calls.append(ClickHouseCall("query", query, parameters or {}))
        return FakeClickHouseResult(
            [{"processed_user_count": self.processed_user_count}],
        )

    def command(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.calls.append(ClickHouseCall("command", query, parameters or {}))

    def close(self) -> None:
        self.close_count += 1


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


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def test_internal_user_behavior_vector_batch_requires_internal_key() -> None:
    client = TestClient(create_app(settings=load_settings(valid_env())))

    response = client.post(
        "/internal/decision/v1/batches/user-behavior-vectors/build",
        json={
            "project_id": "demo_project",
            "vector_version": "v1",
            "source": "expedia_hotel_events",
            "window_days": 90,
        },
    )

    assert response.status_code == 401


def test_internal_user_behavior_vector_batch_rejects_invalid_source() -> None:
    env = valid_env()
    client = TestClient(create_app(settings=load_settings(env)))

    response = client.post(
        "/internal/decision/v1/batches/user-behavior-vectors/build",
        headers={"X-Loop-Ad-Internal-Key": env["LOOPAD_INTERNAL_API_KEY"]},
        json={
            "project_id": "demo_project",
            "vector_version": "v1",
            "source": "raw_events",
            "window_days": 90,
        },
    )

    assert response.status_code == 422


def test_internal_user_behavior_vector_batch_wires_clickhouse_and_closes(
    monkeypatch,
) -> None:
    env = valid_env()
    fake_client = FakeClickHouseClient(processed_user_count=12)

    def fake_create_clickhouse_client(_settings) -> FakeClickHouseClient:
        return fake_client

    monkeypatch.setattr(
        "app.internal.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    client = TestClient(create_app(settings=load_settings(env)))

    response = client.post(
        "/internal/decision/v1/batches/user-behavior-vectors/build",
        headers={"X-Loop-Ad-Internal-Key": env["LOOPAD_INTERNAL_API_KEY"]},
        json={
            "project_id": "demo_project",
            "vector_version": "v1",
            "source": "expedia_hotel_events",
            "window_days": 90,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == "demo_project"
    assert payload["vector_version"] == "v1"
    assert payload["source"] == "expedia_hotel_events"
    assert payload["vector_dim"] == 64
    assert payload["processed_user_count"] == 12
    assert payload["status"] == "completed"
    assert fake_client.close_count == 1
    assert [call.operation for call in fake_client.calls] == ["query", "command"]


def test_user_behavior_vector_build_repository_counts_and_inserts_expedia_vectors() -> None:
    fake_client = FakeClickHouseClient(processed_user_count=2)
    repository = UserBehaviorVectorBuildRepository(fake_client)
    service = UserBehaviorVectorBatchService(
        repository,
        now=datetime(2026, 7, 5, 12, 0, 0, tzinfo=UTC),
    )

    result = service.build(
        UserBehaviorVectorBuildRequest(
            project_id="demo_project",
            vector_version="v1",
            source="expedia_hotel_events",
            window_days=30,
        ),
    )

    assert result.processed_user_count == 2
    assert result.vector_dim == 64
    count_call, insert_call = fake_client.calls
    count_sql = compact_sql(count_call.query)
    insert_sql = compact_sql(insert_call.query)
    assert count_call.operation == "query"
    assert "from expedia_hotel_events" in count_sql
    assert "countdistinct(tostring(user_id))" in count_sql
    assert count_call.parameters == {
        "window_start": "2026-06-05 12:00:00",
        "window_end": "2026-07-05 12:00:00",
    }
    assert insert_call.operation == "command"
    assert "insert into user_behavior_vectors" in insert_sql
    assert "from expedia_hotel_events" in insert_sql
    assert "arraymap" in insert_sql
    assert "touint16({vector_dim:uint16})" in insert_sql
    assert "modulo(touint32(hotel_cluster), 32)" in insert_sql
    assert "modulo(touint32(srch_destination_id), 16)" in insert_sql
    assert "modulo(touint32(channel), 10)" in insert_sql
    assert insert_call.parameters == {
        "project_id": "demo_project",
        "vector_dim": 64,
        "vector_version": "v1",
        "source": "expedia_hotel_events",
        "window_start": "2026-06-05 12:00:00",
        "window_end": "2026-07-05 12:00:00",
    }


def test_user_behavior_vector_batch_skips_insert_when_no_source_users() -> None:
    fake_client = FakeClickHouseClient(processed_user_count=0)
    repository = UserBehaviorVectorBuildRepository(fake_client)
    service = UserBehaviorVectorBatchService(
        repository,
        now=datetime(2026, 7, 5, 12, 0, 0, tzinfo=UTC),
    )

    result = service.build(
        UserBehaviorVectorBuildRequest(
            project_id="demo_project",
            vector_version="v1",
            source="expedia_hotel_events",
            window_days=30,
        ),
    )

    assert result.processed_user_count == 0
    assert [call.operation for call in fake_client.calls] == ["query"]
