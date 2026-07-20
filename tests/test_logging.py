from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from tests.config_env import required_env_values
from app.logging import LOGGER_NAME, configure_logging, log, log_context_scope
from app.main import create_app


EVENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
SECRET_FIELD_PATTERN = re.compile(
    r"^(?:api_?key|authorization|cookie|password|redirect_?token|refresh_?token|sdk_?key|session_?token|write_?key)$",
    re.IGNORECASE,
)


@pytest.fixture(autouse=True)
def reset_test_logger() -> None:
    yield
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())


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


def test_logger_recursively_redacts_secret_fields(capsys) -> None:
    configure_logging(load_settings(valid_env()))
    secret = "must-not-appear"

    log.info(
        "secret_redaction_test",
        {
            "authorization": secret,
            "nested": {
                "openai_api_key": secret,
                "items": [
                    {"redirect_token": secret},
                    {"write_key": secret},
                ],
            },
            "safe_value": "visible",
        },
    )

    record = _last_event(_stderr_records(capsys), "secret_redaction_test")
    assert secret not in json.dumps(record)
    assert record["safeValue"] == "visible"


def test_application_log_calls_follow_event_and_error_field_contract() -> None:
    application_root = Path(__file__).resolve().parents[1] / "app"
    violations: list[str] = []

    for path in application_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_log_call(node):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                violations.append(f"{path}:{node.lineno}: event must be a string literal")
                continue
            event = node.args[0].value
            if not isinstance(event, str) or EVENT_NAME_PATTERN.fullmatch(event) is None:
                violations.append(f"{path}:{node.lineno}: invalid event {event!r}")
            if len(node.args) > 1 and isinstance(node.args[1], ast.Dict):
                keys = {
                    key.value
                    for key in node.args[1].keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if "error" in keys:
                    violations.append(
                        f"{path}:{node.lineno}: exceptions must use the err field"
                    )
                for key in keys:
                    if SECRET_FIELD_PATTERN.fullmatch(key):
                        violations.append(
                            f"{path}:{node.lineno}: secret fields must not be logged"
                        )

    assert violations == []


def _is_log_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"debug", "error", "info", "warn"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "log"
    )


def _stderr_records(capsys) -> list[dict[str, object]]:
    captured = capsys.readouterr()
    return [json.loads(line) for line in captured.err.splitlines() if line.strip()]


def _last_event(
    records: list[dict[str, object]],
    event: str,
) -> dict[str, object]:
    return [record for record in records if record.get("event") == event][-1]
