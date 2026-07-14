from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import load_settings
from tests.config_env import required_env_values
from app.generation.router import get_generation_service
from app.generation.schemas import (
    GenerationAcceptedResponse,
    GenerationRequest,
    GenerationStatus,
)
from app.main import create_app


FORBIDDEN_API_REQUESTS = (
    ("POST", "/decision/v1/segments/query-preview"),
    ("POST", "/decision/v1/segments"),
    ("POST", "/decision/v1/chatkit/session"),
    ("POST", "/decision/v1/chatkit/actions"),
    ("POST", "/decision/v1/promotion-runs/prun_banner_001_loop_1/segment-match"),
    ("GET", "/decision/v1/promotion-runs/prun_banner_001_loop_1/active-contents"),
)


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


def make_client() -> TestClient:
    app = create_app(settings=load_settings(valid_env()))
    app.dependency_overrides[get_generation_service] = (
        lambda: FakeGenerationSubmissionService()
    )
    return TestClient(app)


def test_forbidden_dashboard_chatkit_and_hot_path_apis_are_absent() -> None:
    client = make_client()

    for method, path in FORBIDDEN_API_REQUESTS:
        response = client.request(method, path, json={})

        assert response.status_code in {404, 405}, f"{method} {path} is exposed"


def test_generation_submission_api_returns_only_a_durable_receipt() -> None:
    client = make_client()

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
        headers={"Idempotency-Key": "forbidden-api-boundary-001"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "generation_id": "generation_banner_001_receipt",
        "promotion_id": "promo_banner_001",
        "status": "requested",
    }
    assert "content_candidates" not in response.json()


class FakeGenerationSubmissionService:
    def submit(
        self,
        request: GenerationRequest,
        *,
        idempotency_key: str,
    ) -> GenerationAcceptedResponse:
        assert idempotency_key == "forbidden-api-boundary-001"
        return GenerationAcceptedResponse(
            generation_id="generation_banner_001_receipt",
            promotion_id=request.promotion_id,
            status=GenerationStatus.REQUESTED,
        )
