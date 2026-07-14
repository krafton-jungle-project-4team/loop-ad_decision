from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from tests.config_env import required_env_values
from app.logging import configure_logging, log, log_context_scope
from app.main import create_app


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


def test_request_logging_middleware_emits_structured_completion_log(capsys) -> None:
    client = TestClient(create_app(settings=load_settings(valid_env())))

    response = client.get(
        "/health?project_id=project_1",
        headers={"X-Request-Id": "request-1"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "request-1"
    records = _stderr_records(capsys)
    completion = _last_event(records, "http_request_completed")
    assert completion["service"] == "decision-api"
    assert completion["environment"] == "test"
    assert completion["requestId"] == "request-1"
    assert completion["method"] == "GET"
    assert completion["path"] == "/health"
    assert completion["projectId"] == "project_1"
    assert completion["statusCode"] == 200
    assert completion["outcome"] == "success"
    assert isinstance(completion["durationMs"], int)


def test_log_context_scope_logs_failed_with_active_context(capsys) -> None:
    configure_logging(load_settings(valid_env()))

    @log_context_scope
    def fail() -> None:
        log.assign_context({"projectId": "project_1"})
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fail()

    failed = _last_event(_stderr_records(capsys), "failed")
    assert failed["operation"].endswith("fail")
    assert failed["projectId"] == "project_1"
    assert failed["err"] == {"type": "RuntimeError", "message": "boom", "cause": None}
    assert isinstance(failed["durationMs"], int)


def _stderr_records(capsys) -> list[dict[str, object]]:
    captured = capsys.readouterr()
    return [json.loads(line) for line in captured.err.splitlines() if line.strip()]


def _last_event(
    records: list[dict[str, object]],
    event: str,
) -> dict[str, object]:
    return [record for record in records if record.get("event") == event][-1]
