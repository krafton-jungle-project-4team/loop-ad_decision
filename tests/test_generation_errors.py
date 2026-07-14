from __future__ import annotations

import urllib.error

import pytest

from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
    classify_generation_error,
    retry_backoff_seconds,
)


class ProviderHttpError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"provider secret response for HTTP {status_code}")


class OperationalError(RuntimeError):
    pass


def wrapped_error(cause: BaseException) -> RuntimeError:
    error = RuntimeError("outer secret-token-value")
    error.__cause__ = cause
    return error


def test_classify_generation_error_preserves_explicit_retryable_type() -> None:
    error = RetryableGenerationError(
        code="S3 Temporary Failure",
        safe_message="Temporary artifact storage failure.",
    )

    info = classify_generation_error(error)

    assert info.code == "s3_temporary_failure"
    assert info.message == "Temporary artifact storage failure."
    assert info.retryable is True


def test_classify_generation_error_preserves_explicit_permanent_type() -> None:
    error = PermanentGenerationError(
        code="guardrail_failed",
        safe_message="Generated content did not satisfy required policy.",
    )

    info = classify_generation_error(wrapped_error(error))

    assert info.code == "guardrail_failed"
    assert info.retryable is False


@pytest.mark.parametrize(
    ("status_code", "expected_code", "expected_retryable"),
    [
        (408, "provider_timeout", True),
        (429, "provider_rate_limited", True),
        (500, "provider_server_error", True),
        (503, "provider_server_error", True),
        (400, "provider_request_rejected", False),
        (401, "provider_auth_error", False),
        (403, "provider_auth_error", False),
        (404, "provider_request_rejected", False),
    ],
)
def test_classify_generation_error_uses_http_status(
    status_code: int,
    expected_code: str,
    expected_retryable: bool,
) -> None:
    info = classify_generation_error(wrapped_error(ProviderHttpError(status_code)))

    assert info.code == expected_code
    assert info.retryable is expected_retryable
    assert info.status_code == status_code
    assert "secret" not in info.message


def test_classify_generation_error_walks_timeout_cause() -> None:
    info = classify_generation_error(
        wrapped_error(TimeoutError("secret provider timeout detail"))
    )

    assert info.code == "provider_timeout"
    assert info.retryable is True
    assert "secret" not in info.message


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("secret network endpoint"),
        urllib.error.URLError("secret DNS failure"),
        OperationalError("secret database endpoint"),
    ],
)
def test_classify_generation_error_walks_network_cause(error: BaseException) -> None:
    info = classify_generation_error(wrapped_error(error))

    assert info.code == "provider_network_error"
    assert info.retryable is True
    assert "secret" not in info.message


def test_classify_generation_error_treats_validation_as_permanent_and_safe() -> None:
    info = classify_generation_error(
        ValueError("required input contained secret-token-value")
    )

    assert info.code == "generation_invalid_input"
    assert info.retryable is False
    assert "secret-token-value" not in info.message


def test_classify_generation_error_defaults_unknown_failure_to_permanent() -> None:
    info = classify_generation_error(RuntimeError("secret provider response body"))

    assert info.code == "generation_failed"
    assert info.message == "Generation failed."
    assert info.retryable is False


def test_retry_backoff_seconds_uses_one_based_retry_count() -> None:
    schedule = (60, 300, 900)

    assert retry_backoff_seconds(1, schedule) == 60
    assert retry_backoff_seconds(2, schedule) == 300
    assert retry_backoff_seconds(3, schedule) == 900


def test_retry_backoff_seconds_accepts_injectable_additive_jitter() -> None:
    observed: list[float] = []

    def jitter(base_seconds: float) -> float:
        observed.append(base_seconds)
        return 7.5

    delay = retry_backoff_seconds(2, (60, 300, 900), jitter=jitter)

    assert observed == [300.0]
    assert delay == 307.5


@pytest.mark.parametrize("retry_count", [0, 4])
def test_retry_backoff_seconds_rejects_uncovered_retry_count(
    retry_count: int,
) -> None:
    with pytest.raises(ValueError):
        retry_backoff_seconds(retry_count, (60, 300, 900))
