from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Any, Mapping

from fastapi.testclient import TestClient

from app.config import load_settings
from tests.config_env import required_env_values
from app.internal.schemas import UserBehaviorVectorBuildRequest
from app.analysis.behavior_vector_schema import HotelBookingBehaviorSchemaV2
from app.analysis.repositories import RawEventUserSignalRecord
from app.internal.user_behavior_vectors import (
    UserBehaviorVectorBatchService,
    UserBehaviorVectorBuildRepository,
    _build_hotel_behavior_v2_insert_sql,
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
    def __init__(
        self,
        *,
        processed_user_count: int = 3,
        revision_cutoff: datetime | None = None,
    ) -> None:
        self.processed_user_count = processed_user_count
        self.revision_cutoff = revision_cutoff
        self.calls: list[ClickHouseCall] = []
        self.close_count = 0

    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> FakeClickHouseResult:
        self.calls.append(ClickHouseCall("query", query, parameters or {}))
        if "max(ingested_at)" in query:
            return FakeClickHouseResult(
                [{"source_revision_cutoff": self.revision_cutoff}],
            )
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
    values = required_env_values()
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


def test_hotel_behavior_v2_sql_uses_shared_sha256_destination_space() -> None:
    sql = compact_sql(_build_hotel_behavior_v2_insert_sql())
    assert "sha256" in sql
    assert "destination_bucket_15" in sql
    assert "booking_start_count > booking_complete_count" in sql
    assert "raw_vector_values" in sql
    assert "sqrt(arraysum(value -> value * value, raw_vector_values))" in sql
    assert "value / vector_norm" in sql


def test_python_anchor_vectorizer_and_clickhouse_sql_share_formula_golden() -> None:
    profile = RawEventUserSignalRecord(
        project_id="project",
        user_id="golden",
        event_count=20,
        page_view_count=2,
        hotel_search_count=6,
        hotel_click_count=3,
        hotel_detail_view_count=4,
        promotion_impression_count=8,
        promotion_click_count=2,
        campaign_redirect_click_count=1,
        campaign_landing_count=1,
        booking_start_count=2,
        booking_complete_count=1,
        booking_cancel_count=0,
        deal_event_count=2,
        free_cancellation_count=1,
        breakfast_included_count=1,
        price_event_count=4,
        avg_price=200_000.0,
        destination_values=("제주", "jeju"),
        checkin_dates=("2026-07-10", "2026-07-11"),
        hotel_market_values=(),
        hotel_cluster_values=(),
        age_group_values=(),
        gender_values=(),
        preferred_category_values=(),
        destination_match_count=2,
        season_match_count=2,
        hotel_search_recency_days=7,
        hotel_detail_recency_days=14,
        booking_start_recency_days=21,
        deal_recency_days=28,
        promotion_response_recency_days=7,
        lead_time_8_30_count=2,
        budget_price_count=1,
        premium_price_count=2,
    )
    vector = HotelBookingBehaviorSchemaV2().vectorize_user(profile)
    assert len(vector) == 64
    assert math.isclose(sum(value * value for value in vector), 1.0)
    assert math.isclose(
        vector[7] / vector[8],
        math.exp(-7 / 30) / math.exp(-14 / 30),
        rel_tol=1e-9,
    )
    assert vector[36] == 0
    assert vector[37] > 0
    assert math.isclose(vector[43] / vector[44], 0.5, rel_tol=1e-9)
    assert math.isclose(vector[54] / vector[7], 1.0, rel_tol=1e-9)
    assert vector[62] > 0

    sql = compact_sql(_build_hotel_behavior_v2_insert_sql())
    assert "exp(-tofloat64(datediff('day'" in sql
    assert "tofloat64(lead_8_30_count) / tofloat64(checkin_date_count)" in sql
    assert "tofloat64(budget_price_count) / tofloat64(price_count)" in sql
    assert "tofloat64(premium_price_count) / tofloat64(price_count)" in sql
    assert "<= 30, 1.0, 0.0" in sql


def test_internal_user_behavior_vector_batch_requires_internal_key() -> None:
    client = TestClient(create_app(settings=load_settings(valid_env())))

    response = client.post(
        "/internal/decision/v1/batches/user-behavior-vectors/build",
        json={
            "project_id": "demo_project",
            "vector_version": "v1",
            "window_days": 90,
        },
    )

    assert response.status_code == 401


def test_user_behavior_vector_search_sync_has_no_feature_switch() -> None:
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/internal/decision/v1/batches/user-behavior-vector-search/sync",
        json={
            "project_id": "demo_project",
            "vector_generation_id": "uvgen_contract_test",
        },
    )

    assert response.status_code == 401
    assert not hasattr(app.state, "analysis_audience_v2_enabled")


def test_internal_user_behavior_vector_batch_rejects_source_override() -> None:
    env = valid_env()
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
            "window_days": 90,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == "demo_project"
    assert payload["vector_version"] == "v1"
    assert payload["source"] == "raw_events"
    assert payload["vector_dim"] == 64
    assert payload["processed_user_count"] == 12
    assert payload["expected_user_count"] == 12
    assert payload["vector_generation_id"].startswith("uvgen_")
    assert len(payload["manifest_hash"]) == 64
    assert payload["status"] == "completed"
    assert fake_client.close_count == 1
    assert [call.operation for call in fake_client.calls] == ["query", "command"]
    assert "from raw_events" in compact_sql(fake_client.calls[0].query)
    assert "from raw_events" in compact_sql(fake_client.calls[1].query)


def test_user_behavior_vector_build_repository_counts_and_inserts_raw_event_vectors() -> None:
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
            window_days=30,
        ),
    )

    assert result.source == "raw_events"
    assert result.processed_user_count == 2
    assert result.vector_dim == 64
    count_call, insert_call = fake_client.calls
    count_sql = compact_sql(count_call.query)
    insert_sql = compact_sql(insert_call.query)
    assert count_call.operation == "query"
    assert "from raw_events" in count_sql
    assert "project_id = {project_id:string}" in count_sql
    assert "validation_status = 'valid'" in count_sql
    assert "countdistinct(user_id)" in count_sql
    assert count_call.parameters == {
        "project_id": "demo_project",
        "window_start": "2026-06-05 12:00:00",
        "window_end": "2026-07-05 12:00:00",
    }
    assert insert_call.operation == "command"
    assert "insert into user_behavior_vectors" in insert_sql
    assert "from raw_events" in insert_sql
    assert "jsonextractstring(properties_json, 'hotel_cluster')" in insert_sql
    assert "jsonextractstring(properties_json, 'hotel_market')" in insert_sql
    assert "jsonextractstring(properties_json, 'page_path')" in insert_sql
    assert "event_name = 'booking_complete'" in insert_sql
    assert "event_name = 'promotion_click'" in insert_sql
    assert "touint16({vector_dim:uint16})" in insert_sql
    assert insert_call.parameters == {
        "project_id": "demo_project",
        "vector_dim": 64,
        "vector_version": "v1",
        "source": "raw_events",
        "window_start": "2026-06-05 12:00:00",
        "window_end": "2026-07-05 12:00:00",
    }


def test_hotel_v2_build_uses_clickhouse_revision_cutoff() -> None:
    revision_cutoff = datetime(2026, 7, 5, 12, 0, 1, 123456, tzinfo=UTC)
    fake_client = FakeClickHouseClient(
        processed_user_count=2,
        revision_cutoff=revision_cutoff,
    )
    service = UserBehaviorVectorBatchService(
        UserBehaviorVectorBuildRepository(fake_client),
        now=datetime(2026, 7, 5, 12, 0, 0, tzinfo=UTC),
    )

    result = service.build(
        UserBehaviorVectorBuildRequest(
            project_id="demo_project",
            vector_version="hotel_behavior.v2",
            window_days=30,
        ),
    )

    assert result.source_revision_cutoff == revision_cutoff
    assert [call.operation for call in fake_client.calls] == [
        "query",
        "command",
        "query",
    ]
    assert "user_behavior_vector_revisions" in compact_sql(
        fake_client.calls[-1].query
    )


def test_vector_generation_window_uses_clickhouse_second_precision() -> None:
    revision_cutoff = datetime(2026, 7, 5, 12, 0, 2, tzinfo=UTC)
    service = UserBehaviorVectorBatchService(
        UserBehaviorVectorBuildRepository(
            FakeClickHouseClient(
                processed_user_count=1,
                revision_cutoff=revision_cutoff,
            )
        ),
        now=datetime(2026, 7, 5, 12, 0, 0, 987654, tzinfo=UTC),
    )

    result = service.build(
        UserBehaviorVectorBuildRequest(
            project_id="demo_project",
            vector_version="hotel_behavior.v2",
            window_days=30,
        )
    )

    assert result.window_end.microsecond == 0
    assert result.window_start.microsecond == 0


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
            window_days=30,
        ),
    )

    assert result.processed_user_count == 0
    assert [call.operation for call in fake_client.calls] == ["query"]
