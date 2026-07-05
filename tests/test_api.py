from __future__ import annotations

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


def make_client() -> TestClient:
    return TestClient(create_app(settings=load_settings(valid_env())))


def test_health_returns_200() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "decision-api",
        "env": "test",
    }


def test_internal_health_requires_loop_ad_internal_key() -> None:
    response = make_client().get("/internal/health")

    assert response.status_code == 401


def test_internal_health_rejects_wrong_loop_ad_internal_key() -> None:
    response = make_client().get(
        "/internal/health",
        headers={"X-Loop-Ad-Internal-Key": "wrong"},
    )

    assert response.status_code == 401


def test_internal_health_accepts_loop_ad_internal_key() -> None:
    env = valid_env()
    client = TestClient(create_app(settings=load_settings(env)))

    response = client.get(
        "/internal/health",
        headers={"X-Loop-Ad-Internal-Key": env["LOOPAD_INTERNAL_API_KEY"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "decision-api",
    }
