from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.generation.router import get_generation_service
from app.generation.service import GenerationService
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


def make_client() -> TestClient:
    app = create_app(settings=load_settings(valid_env()))
    app.dependency_overrides[get_generation_service] = lambda: GenerationService()
    return TestClient(app)


def test_forbidden_dashboard_chatkit_and_hot_path_apis_are_absent() -> None:
    client = make_client()

    for method, path in FORBIDDEN_API_REQUESTS:
        response = client.request(method, path, json={})

        assert response.status_code in {404, 405}, f"{method} {path} is exposed"
