import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from tests.config_env import required_env_values
from app.generation.router import get_generation_service
from app.generation.schemas import (
    GenerationAcceptedResponse,
    GenerationRequest,
    GenerationStatus,
)
from app.generation.submission import (
    GenerationIdempotencyConflict,
    GenerationSubmissionUnavailable,
)
from app.main import create_app


FORBIDDEN_PUBLIC_KEYS = {"variant_id", "experiment_id", "creative_id"}
DEFAULT_FETCHONE_RESULT = object()
CONFIRMED_TARGET_SEGMENT_STATUSES = {"approved"}


@pytest.fixture(autouse=True)
def stub_s3_brand_context_loader(monkeypatch) -> None:
    class NoBrandContextLoader:
        def __init__(self, **_kwargs) -> None:
            pass

        def resolve_snapshot(self, *, project_id: str):
            del project_id
            return None

    monkeypatch.setattr(
        "app.generation.router.S3BrandContextLoader",
        NoBrandContextLoader,
    )


def valid_env() -> dict[str, str]:
    values = required_env_values()
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "15432",
        }
    )
    return values


def make_generation_client(service=None) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_generation_service] = (
        lambda: service or FakeGenerationSubmissionService()
    )
    return TestClient(app)


def test_generation_api_returns_durable_acceptance_contract() -> None:
    client = make_generation_client()

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 3,
            "operator_instruction": None,
        },
        headers={"Idempotency-Key": "generation-api-001"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["generation_id"] == "generation_fake"
    assert payload["promotion_id"] == "promo_banner_001"
    assert payload["status"] == "requested"
    assert set(payload) == {"generation_id", "promotion_id", "status"}
    assert_no_forbidden_public_keys(payload)


def test_generation_api_requires_idempotency_key() -> None:
    client = make_generation_client()

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json=generation_payload(),
    )

    assert response.status_code == 400
    assert "Idempotency-Key" in response.json()["detail"]


def test_generation_api_maps_idempotency_conflict_to_409() -> None:
    client = make_generation_client(
        RaisingGenerationSubmissionService(
            GenerationIdempotencyConflict("idempotency conflict")
        )
    )

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json=generation_payload(),
        headers={"Idempotency-Key": "generation-conflict-001"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "idempotency conflict"


def test_generation_api_rejects_new_work_with_503_during_shutdown() -> None:
    client = make_generation_client(
        RaisingGenerationSubmissionService(
            GenerationSubmissionUnavailable("generation worker is shutting down")
        )
    )

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json=generation_payload(),
        headers={"Idempotency-Key": "generation-shutdown-001"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "generation worker is shutting down"


def test_generation_api_rejects_path_body_promotion_mismatch() -> None:
    client = make_generation_client()

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_other_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": None,
        },
    )

    assert response.status_code == 400
    assert "promotion_id" in response.json()["detail"]


def test_generation_api_rejects_non_positive_content_option_count() -> None:
    client = make_generation_client()

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 0,
            "operator_instruction": None,
        },
    )

    assert response.status_code == 400


def test_generation_api_calls_generation_service() -> None:
    app = create_app()
    fake_service = FakeGenerationSubmissionService()
    app.dependency_overrides[get_generation_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "segment_ids": ["seg_mobile_user"],
            "content_option_count": 1,
            "operator_instruction": "Keep it short.",
        },
        headers={"Idempotency-Key": "generation-api-call-001"},
    )

    assert response.status_code == 202
    assert response.json()["generation_id"] == "generation_fake"
    assert len(fake_service.requests) == 1
    assert fake_service.requests[0].analysis_id == "analysis_banner_001"
    assert fake_service.requests[0].segment_ids == ["seg_mobile_user"]
    assert fake_service.requests[0].operator_instruction == "Keep it short."
    assert fake_service.idempotency_keys == ["generation-api-call-001"]


@pytest.mark.parametrize(
    "segment_ids",
    [[], ["seg_family_trip", "seg_family_trip"], ["   "]],
)
def test_generation_api_rejects_empty_duplicate_or_blank_segment_ids(
    segment_ids: list[str],
) -> None:
    client = make_generation_client()

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "segment_ids": segment_ids,
            "content_option_count": 1,
            "operator_instruction": None,
        },
    )

    assert response.status_code == 400


def test_generation_api_wires_postgres_repositories(monkeypatch) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.generation.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 2,
            "operator_instruction": None,
        },
        headers={"Idempotency-Key": "generation-db-001"},
    )

    assert response.status_code == 202
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1

    executed_queries = [query for query, _params in connection.executed]
    assert sum("INSERT INTO generation_runs" in query for query in executed_queries) == 1
    assert all("INSERT INTO content_candidates" not in query for query in executed_queries)
    assert any("FROM promotion_target_segments" in query for query in executed_queries)
    assert all("promotion_segment_suggestions" not in query for query in executed_queries)


def test_generation_api_rejects_without_confirmed_target_segments(monkeypatch) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            target_segment_rows=[
                target_segment_row(
                    segment_id="seg_family_trip",
                    segment_name="Family trip planners",
                    status="planned",
                )
            ]
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.generation.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": None,
        },
        headers={"Idempotency-Key": "generation-no-target-001"},
    )

    assert response.status_code == 409
    assert "promotion_target_segments" in response.json()["detail"]

    connection = connections[0]
    executed_queries = [query for query, _params in connection.executed]
    assert any("FROM promotion_target_segments" in query for query in executed_queries)
    assert all("INSERT INTO generation_runs" not in query for query in executed_queries)
    assert all("promotion_segment_suggestions" not in query for query in executed_queries)
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


def test_generation_api_does_not_invoke_provider_or_artifact_work(monkeypatch) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.generation.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": None,
        },
        headers={"Idempotency-Key": "generation-no-provider-001"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "requested"
    assert len(connections) == 1
    queries = [query for query, _params in connections[0].executed]
    assert all("content_candidates" not in query for query in queries)


def test_generation_api_snapshots_only_confirmed_segments(monkeypatch) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            target_segment_rows=[
                target_segment_row(
                    segment_id="seg_repeat_hotel_no_booking",
                    segment_name="Repeat hotel viewers without booking",
                    status="approved",
                ),
                target_segment_row(
                    segment_id="seg_family_trip",
                    segment_name="Family trip planners",
                    status="planned",
                ),
            ]
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.generation.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": None,
        },
        headers={"Idempotency-Key": "generation-confirmed-001"},
    )

    assert response.status_code == 202

    connection = connections[0]
    executed_queries = [query for query, _params in connection.executed]
    assert any(
        "pts.status = 'approved'" in query
        for query in executed_queries
    )
    insert_params = next(
        params
        for query, params in connection.executed
        if "INSERT INTO generation_runs" in query
    )
    snapshot = insert_params["input_json"].obj
    assert [target["segment_id"] for target in snapshot["target_segments"]] == [
        "seg_repeat_hotel_no_booking"
    ]


def test_generation_api_rolls_back_when_repository_write_fails(monkeypatch) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(fetchone_result=None)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.generation.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": None,
        },
        headers={"Idempotency-Key": "generation-write-fail-001"},
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 2
    assert connection.close_count == 1


def test_health_returns_ok() -> None:
    client = TestClient(create_app(settings=load_settings(valid_env())))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "decision-api",
        "env": "test",
    }


def assert_no_forbidden_public_keys(value) -> None:
    if isinstance(value, dict):
        assert not (set(value) & FORBIDDEN_PUBLIC_KEYS)
        for item in value.values():
            assert_no_forbidden_public_keys(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_forbidden_public_keys(item)


class FakeGenerationSubmissionService:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.idempotency_keys: list[str] = []

    def submit(
        self,
        request: GenerationRequest,
        *,
        idempotency_key: str,
    ) -> GenerationAcceptedResponse:
        self.requests.append(request)
        self.idempotency_keys.append(idempotency_key)
        return GenerationAcceptedResponse(
            generation_id="generation_fake",
            promotion_id=request.promotion_id,
            status=GenerationStatus.REQUESTED,
        )


class RaisingGenerationSubmissionService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def submit(
        self,
        request: GenerationRequest,
        *,
        idempotency_key: str,
    ) -> GenerationAcceptedResponse:
        del request, idempotency_key
        raise self._error


def generation_payload() -> dict[str, object]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "content_option_count": 1,
        "operator_instruction": None,
    }


def target_segment_row(
    *,
    segment_id: str = "seg_repeat_hotel_no_booking",
    segment_name: str = "Repeat hotel viewers without booking",
    status: str = "approved",
    estimated_size: int = 1342,
    priority: str = "high",
) -> dict[str, object]:
    return {
        "analysis_id": "analysis_banner_001",
        "promotion_id": "promo_banner_001",
        "segment_id": segment_id,
        "segment_name": segment_name,
        "content_brief_json": {
            "message_direction": (
                "Emphasize refundable rooms and clear booking steps."
            ),
            "keywords": ["refundable rooms", "hotel deals"],
        },
        "data_evidence_json": {
            "source": "ai_suggested",
            "sample_ratio": "0.018000",
        },
        "segment_vector_id": f"segvec_{segment_id.removeprefix('seg_')}_v1",
        "estimated_size": estimated_size,
        "priority": priority,
        "status": status,
        "segment_source": "ai_suggested",
        "query_preview_id": None,
        "natural_language_query": "repeat hotel viewers who did not book",
        "generated_sql": None,
        "segment_sample_size": estimated_size,
        "segment_sample_ratio": "0.018000",
    }


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self._connection = connection
        self._last_query = ""
        self._last_params: dict[str, object] | None = None

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: dict[str, object] | None = None) -> None:
        self._last_query = query
        self._last_params = params
        self._connection.executed.append((query, params))

    def fetchone(self) -> dict[str, object] | None:
        if "FROM promotions" in self._last_query:
            return self._connection.promotion_row
        if "INSERT INTO generation_runs" in self._last_query:
            if self._connection.insert_fetchone_result is None:
                return None
            assert self._last_params is not None
            return {
                **self._last_params,
                "input_json": self._last_params["input_json"].obj,
                "generation_report_json": self._last_params[
                    "generation_report_json"
                ].obj,
                "output_json": None,
            }
        return self._connection.insert_fetchone_result

    def fetchall(self) -> list[dict[str, object]]:
        if "FROM promotion_target_segments" in self._last_query:
            rows = self._connection.target_segment_rows
            if "pts.status = 'approved'" in self._last_query:
                return [
                    row
                    for row in rows
                    if row.get("status") in CONFIRMED_TARGET_SEGMENT_STATUSES
                ]
            return rows
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        fetchone_result: dict[str, object] | None | object = DEFAULT_FETCHONE_RESULT,
        target_segment_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.insert_fetchone_result = (
            {"ok": True}
            if fetchone_result is DEFAULT_FETCHONE_RESULT
            else fetchone_result
        )
        self.promotion_row = {
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "channel": "onsite_banner",
            "goal_metric": "booking_conversion_rate",
            "goal_target_value": "0.030000",
            "goal_basis": "all_segments",
            "message_brief": "Drive hotel booking conversion for summer stays.",
            "landing_url": "https://demo-stay.example.com/summer",
        }
        self.target_segment_rows = (
            target_segment_rows
            if target_segment_rows is not None
            else [target_segment_row()]
        )
        self.executed: list[tuple[str, dict[str, object] | None]] = []
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
