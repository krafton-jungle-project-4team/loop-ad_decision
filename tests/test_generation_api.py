from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.generation.router import get_generation_service
from app.generation.schemas import (
    ContentCandidateResponse,
    ContentCandidateStatus,
    ContentChannel,
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
)
from app.generation.service import GenerationService
from app.main import create_app


FORBIDDEN_PUBLIC_KEYS = {"creative_id", "variant_id", "experiment_id"}
DEFAULT_FETCHONE_RESULT = object()


def valid_env() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
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
        lambda: service or GenerationService()
    )
    return TestClient(app)


def test_generation_api_returns_v1_6_final_names() -> None:
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
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["generation_id"] == "generation_banner_001"
    assert payload["promotion_id"] == "promo_banner_001"
    assert payload["status"] == "completed"
    assert len(payload["content_candidates"]) == 3
    assert "content_candidates" in payload
    assert_no_forbidden_public_keys(payload)

    first_candidate = payload["content_candidates"][0]
    assert first_candidate["content_id"] == "content_banner_repeat_hotel_001"
    assert first_candidate["content_option_id"] == "banner_repeat_hotel_option_001"
    assert first_candidate["segment_id"] == "seg_repeat_hotel_no_booking"
    assert first_candidate["channel"] == "onsite_banner"
    assert first_candidate["status"] == "draft"
    assert first_candidate["title"]
    assert first_candidate["body"]
    assert first_candidate["cta"]
    assert first_candidate["image_prompt"]
    assert first_candidate["image_url"] is None
    assert first_candidate["landing_url"] == "https://demo-stay.example.com/summer"


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
    fake_service = FakeGenerationService()
    app.dependency_overrides[get_generation_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": "Keep it short.",
        },
    )

    assert response.status_code == 200
    assert response.json()["generation_id"] == "generation_fake"
    assert len(fake_service.requests) == 1
    assert fake_service.requests[0].analysis_id == "analysis_banner_001"
    assert fake_service.requests[0].operator_instruction == "Keep it short."


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
    )

    assert response.status_code == 200
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1

    executed_queries = [query for query, _params in connection.executed]
    assert sum("INSERT INTO generation_runs" in query for query in executed_queries) == 1
    assert (
        sum("INSERT INTO content_candidates" in query for query in executed_queries)
        == 2
    )


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
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
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


class FakeGenerationService:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        return GenerationResponse(
            generation_id="generation_fake",
            promotion_id=request.promotion_id,
            status=GenerationStatus.COMPLETED,
            content_candidates=[
                ContentCandidateResponse(
                    content_id="content_banner_fake_001",
                    content_option_id="banner_fake_option_001",
                    segment_id="seg_repeat_hotel_no_booking",
                    channel=ContentChannel.ONSITE_BANNER,
                    title="Book this weekend's rooms",
                    body="Compare refundable summer offers before rooms run out.",
                    cta="View hotel deals",
                    image_prompt="bright modern hotel room, summer travel banner",
                    image_url=None,
                    landing_url="https://demo-stay.example.com/summer",
                    status=ContentCandidateStatus.DRAFT,
                )
            ],
        )


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self._connection = connection

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: dict[str, object] | None = None) -> None:
        self._connection.executed.append((query, params))

    def fetchone(self) -> dict[str, object] | None:
        return self._connection.fetchone_result

    def fetchall(self) -> list[dict[str, object]]:
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        fetchone_result: dict[str, object] | None | object = DEFAULT_FETCHONE_RESULT,
    ) -> None:
        self.fetchone_result = (
            {"ok": True}
            if fetchone_result is DEFAULT_FETCHONE_RESULT
            else fetchone_result
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
