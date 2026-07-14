from __future__ import annotations

import errno
import math
import re
import socket
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass


_SAFE_CODE_PATTERN = re.compile(r"[^a-z0-9_]+")
_MAX_SAFE_CODE_LENGTH = 100
_MAX_SAFE_MESSAGE_LENGTH = 500

_TIMEOUT_EXCEPTION_NAMES = frozenset(
    {
        "ConnectTimeout",
        "DeadlineExceeded",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "WriteTimeout",
    }
)
_NETWORK_EXCEPTION_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectionClosed",
        "EndpointConnectionError",
        "InterfaceError",
        "NetworkError",
        "OperationalError",
        "ProtocolError",
        "RemoteProtocolError",
    }
)
_NETWORK_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)


@dataclass(frozen=True, slots=True)
class GenerationErrorInfo:
    code: str
    message: str
    retryable: bool
    status_code: int | None = None


class GenerationError(RuntimeError):
    """An explicitly classified error whose message is safe to persist."""

    def __init__(
        self,
        *,
        code: str,
        safe_message: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        self.info = GenerationErrorInfo(
            code=_safe_code(code),
            message=_safe_message(safe_message),
            retryable=retryable,
            status_code=_normalise_http_status(status_code),
        )
        super().__init__(self.info.message)

    @property
    def code(self) -> str:
        return self.info.code

    @property
    def safe_message(self) -> str:
        return self.info.message

    @property
    def retryable(self) -> bool:
        return self.info.retryable

    @property
    def status_code(self) -> int | None:
        return self.info.status_code


class RetryableGenerationError(GenerationError):
    def __init__(
        self,
        *,
        code: str = "generation_retryable_error",
        safe_message: str = "Generation failed temporarily.",
        status_code: int | None = None,
    ) -> None:
        super().__init__(
            code=code,
            safe_message=safe_message,
            retryable=True,
            status_code=status_code,
        )


class PermanentGenerationError(GenerationError):
    def __init__(
        self,
        *,
        code: str = "generation_permanent_error",
        safe_message: str = "Generation request cannot be completed.",
        status_code: int | None = None,
    ) -> None:
        super().__init__(
            code=code,
            safe_message=safe_message,
            retryable=False,
            status_code=status_code,
        )


def classify_generation_error(error: BaseException) -> GenerationErrorInfo:
    """Classify provider failures without persisting raw exception text or bodies."""

    chain = tuple(_exception_chain(error))

    for item in chain:
        if isinstance(item, GenerationError):
            return item.info

    for item in chain:
        status_code = _http_status(item)
        if status_code is not None:
            return _http_error_info(status_code)

    if any(_is_timeout_error(item) for item in chain):
        return GenerationErrorInfo(
            code="provider_timeout",
            message="Provider request timed out.",
            retryable=True,
        )

    if any(_is_network_error(item) for item in chain):
        return GenerationErrorInfo(
            code="provider_network_error",
            message="Provider network request failed temporarily.",
            retryable=True,
        )

    if any(isinstance(item, CancelledError) for item in chain):
        return GenerationErrorInfo(
            code="generation_cancelled",
            message="Generation was cancelled before completion.",
            retryable=False,
        )

    if any(isinstance(item, (KeyError, TypeError, ValueError)) for item in chain):
        return GenerationErrorInfo(
            code="generation_invalid_input",
            message="Generation input or provider response was invalid.",
            retryable=False,
        )

    return GenerationErrorInfo(
        code="generation_failed",
        message="Generation failed.",
        retryable=False,
    )


def retry_backoff_seconds(
    retry_count: int,
    schedule: Sequence[int | float],
    *,
    jitter: Callable[[float], float] | None = None,
) -> float:
    """Return delay for a one-based retry count with optional additive jitter."""

    if retry_count < 1:
        raise ValueError("retry_count must be at least 1")
    if retry_count > len(schedule):
        raise ValueError("retry schedule does not cover retry_count")

    base_seconds = float(schedule[retry_count - 1])
    if not math.isfinite(base_seconds) or base_seconds <= 0:
        raise ValueError("retry schedule values must be positive finite numbers")

    jitter_seconds = 0.0 if jitter is None else float(jitter(base_seconds))
    if not math.isfinite(jitter_seconds):
        raise ValueError("retry jitter must be finite")
    return max(0.0, base_seconds + jitter_seconds)


def _exception_chain(error: BaseException) -> Sequence[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _http_error_info(status_code: int) -> GenerationErrorInfo:
    if status_code == 408:
        return GenerationErrorInfo(
            code="provider_timeout",
            message="Provider request timed out.",
            retryable=True,
            status_code=status_code,
        )
    if status_code == 429:
        return GenerationErrorInfo(
            code="provider_rate_limited",
            message="Provider rate limit was reached.",
            retryable=True,
            status_code=status_code,
        )
    if 500 <= status_code <= 599:
        return GenerationErrorInfo(
            code="provider_server_error",
            message="Provider service is temporarily unavailable.",
            retryable=True,
            status_code=status_code,
        )
    if status_code in {401, 403}:
        return GenerationErrorInfo(
            code="provider_auth_error",
            message="Provider authentication or authorization failed.",
            retryable=False,
            status_code=status_code,
        )
    if 400 <= status_code <= 499:
        return GenerationErrorInfo(
            code="provider_request_rejected",
            message="Provider rejected the generation request.",
            retryable=False,
            status_code=status_code,
        )
    return GenerationErrorInfo(
        code="provider_http_error",
        message="Provider returned an unexpected HTTP response.",
        retryable=False,
        status_code=status_code,
    )


def _http_status(error: BaseException) -> int | None:
    for attribute_name in ("status_code", "status", "code"):
        status_code = _normalise_http_status(_safe_getattr(error, attribute_name))
        if status_code is not None:
            return status_code

    response = _safe_getattr(error, "response")
    if isinstance(response, Mapping):
        response_metadata = response.get("ResponseMetadata")
        if isinstance(response_metadata, Mapping):
            status_code = _normalise_http_status(
                response_metadata.get("HTTPStatusCode")
            )
            if status_code is not None:
                return status_code
        status_code = _normalise_http_status(response.get("status_code"))
        if status_code is not None:
            return status_code
    elif response is not None:
        for attribute_name in ("status_code", "status"):
            status_code = _normalise_http_status(
                _safe_getattr(response, attribute_name)
            )
            if status_code is not None:
                return status_code
    return None


def _normalise_http_status(value: object) -> int | None:
    if isinstance(value, bool) or value is None or callable(value):
        return None
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return None
    return status_code if 100 <= status_code <= 599 else None


def _is_timeout_error(error: BaseException) -> bool:
    return isinstance(error, TimeoutError) or type(error).__name__ in _TIMEOUT_EXCEPTION_NAMES


def _is_network_error(error: BaseException) -> bool:
    if isinstance(error, (ConnectionError, socket.gaierror, urllib.error.URLError)):
        return True
    if type(error).__name__ in _NETWORK_EXCEPTION_NAMES:
        return True
    return isinstance(error, OSError) and error.errno in _NETWORK_ERRNOS


def _safe_code(value: str) -> str:
    normalised = _SAFE_CODE_PATTERN.sub("_", str(value).strip().lower()).strip("_")
    return (normalised or "generation_error")[:_MAX_SAFE_CODE_LENGTH]


def _safe_message(value: str) -> str:
    normalised = " ".join(str(value).split())
    return (normalised or "Generation failed.")[:_MAX_SAFE_MESSAGE_LENGTH]


def _safe_getattr(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except Exception:
        return None
