from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from threading import Barrier, Thread
import time

import pytest
from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.dependencies import checkout_postgres_connection, get_postgres_pool
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


def test_health_does_not_create_postgres_pool(monkeypatch) -> None:
    calls: list[object] = []

    def fake_create_postgres_pool(_settings):
        calls.append(object())
        return RecordingPool(object())

    monkeypatch.setattr(
        "app.dependencies.create_postgres_pool",
        fake_create_postgres_pool,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert calls == []


def test_postgres_pool_is_created_once_for_concurrent_first_requests(
    monkeypatch,
) -> None:
    pools: list[RecordingPool] = []
    app = create_app(settings=load_settings(valid_env()))
    request = SimpleNamespace(app=app)
    barrier = Barrier(5)
    results: list[RecordingPool] = []

    def fake_create_postgres_pool(_settings):
        time.sleep(0.01)
        pool = RecordingPool(object())
        pools.append(pool)
        return pool

    monkeypatch.setattr(
        "app.dependencies.create_postgres_pool",
        fake_create_postgres_pool,
    )

    def worker() -> None:
        barrier.wait()
        results.append(get_postgres_pool(request))

    threads = [Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(pools) == 1
    assert results == [pools[0]] * 5


def test_postgres_checkout_returns_connection_after_exception(monkeypatch) -> None:
    pool = RecordingPool(object())
    app = create_app(settings=load_settings(valid_env()))
    request = SimpleNamespace(app=app)

    monkeypatch.setattr(
        "app.dependencies.create_postgres_pool",
        lambda _settings: pool,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with checkout_postgres_connection(request):
            raise RuntimeError("boom")

    assert pool.checkout_count == 1
    assert pool.return_count == 1


class RecordingPool:
    def __init__(self, connection: object) -> None:
        self.connection_object = connection
        self.checkout_count = 0
        self.return_count = 0
        self.close_count = 0

    @contextmanager
    def connection(self) -> Iterator[object]:
        self.checkout_count += 1
        try:
            yield self.connection_object
        finally:
            self.return_count += 1

    def close(self) -> None:
        self.close_count += 1
