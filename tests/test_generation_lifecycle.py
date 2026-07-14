from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app import main as main_module


def runtime_settings(*, env: str) -> Settings:
    return Settings(
        env=env,
        service_id="decision-api",
        port=8080,
        internal_api_key="test-key",
        aurora_host="localhost",
        aurora_port=5432,
        aurora_database="loopad",
        aurora_username="loopad",
        aurora_password="secret",
        clickhouse_url="http://localhost:8123",
        clickhouse_database="loopad",
        clickhouse_username="loopad",
        clickhouse_password="secret",
        data_storage_bucket="loopad-test",
        genai_assets_base_prefix="genai",
        brand_context_base_prefix="brand-context",
        openai_api_key="openai-test",
        gemini_api_key="gemini-test",
        genai_assets_public_base_url="https://assets.example.test",
        openai_content_model="gpt-test",
        gemini_image_model="gemini-test",
        generation_worker_max_concurrency=2,
        generation_poll_interval_seconds=1,
        generation_idle_poll_interval_seconds=30,
        generation_lease_seconds=180,
        generation_heartbeat_seconds=30,
        generation_max_retries=3,
        generation_retry_backoff_seconds=(60, 300, 900),
        generation_provider_timeout_seconds=30,
        generation_db_operation_timeout_seconds=5,
        generation_shutdown_grace_seconds=20,
    )


class FakeCoordinator:
    instances: list["FakeCoordinator"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.stopped = True


def test_non_test_app_owns_coordinator_lifecycle(monkeypatch) -> None:
    FakeCoordinator.instances.clear()
    monkeypatch.setattr(main_module, "GenerationCoordinator", FakeCoordinator)
    app = main_module.create_app(settings=runtime_settings(env="dev"))

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        coordinator = FakeCoordinator.instances[0]
        assert coordinator.started is True
        assert app.state.generation_coordinator is coordinator

    assert coordinator.stopped is True
    assert app.state.generation_coordinator is None


def test_test_app_does_not_start_background_workers(monkeypatch) -> None:
    FakeCoordinator.instances.clear()
    monkeypatch.setattr(main_module, "GenerationCoordinator", FakeCoordinator)
    app = main_module.create_app(settings=runtime_settings(env="test"))

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert FakeCoordinator.instances == []
