from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import os
import platform
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from importlib import metadata
from typing import Any, ParamSpec, TypeVar

from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from app.config import Settings


LOGGER_NAME = "loopad"
REQUEST_ID_HEADER = "X-Request-Id"
_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "loopad_log_context",
    default={},
)

P = ParamSpec("P")
R = TypeVar("R")


class ContextLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def assign_context(self, fields: Mapping[str, Any]) -> None:
        context = dict(_LOG_CONTEXT.get())
        _apply_context_fields(context, fields)
        _LOG_CONTEXT.set(context)

    def debug(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._emit(logging.DEBUG, event, payload)

    def info(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._emit(logging.INFO, event, payload)

    def warn(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._emit(logging.WARNING, event, payload)

    def error(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._emit(logging.ERROR, event, payload)

    def _emit(
        self,
        level: int,
        event: str,
        payload: Mapping[str, Any] | None,
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        record = {
            **_LOG_CONTEXT.get(),
            "event": event,
            **_remove_none_values(dict(payload or {})),
        }
        self._logger.log(level, "", extra={"loopad_record": record})


class JsonLogFormatter(logging.Formatter):
    def __init__(self, *, settings: Settings) -> None:
        super().__init__()
        self._base = _remove_none_values(
            {
                "service": settings.service_id,
                "environment": settings.env,
                "version": _package_version(),
                "region": os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION"),
                "runtime": os.environ.get("AWS_EXECUTION_ENV") or _runtime_name(),
            }
        )

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "loopad_record", {})
        if not isinstance(payload, Mapping):
            payload = {"event": str(record.getMessage())}
        body = {
            "timestamp": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            **self._base,
            **dict(payload),
        }
        if record.exc_info and "err" not in body:
            body["err"] = record.exc_info[1] or record.getMessage()
        return json.dumps(_to_jsonable(body), ensure_ascii=False, separators=(",", ":"))


log = ContextLogger(logging.getLogger(LOGGER_NAME))


def configure_logging(settings: Settings) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter(settings=settings))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def assign_context(fields: Mapping[str, Any]) -> None:
    log.assign_context(fields)


def duration_ms(started_at: float) -> int:
    return max(0, round((time.perf_counter() - started_at) * 1000))


def now_ms() -> float:
    return time.perf_counter()


def log_context_scope(func: Callable[P, R]) -> Callable[P, R]:
    operation = _operation_name(func)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            started_at = now_ms()
            token = _push_context({"operation": operation})
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                log.error("failed", {"err": exc, "durationMs": duration_ms(started_at)})
                raise
            finally:
                _LOG_CONTEXT.reset(token)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started_at = now_ms()
        token = _push_context({"operation": operation})
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            log.error("failed", {"err": exc, "durationMs": duration_ms(started_at)})
            raise
        finally:
            _LOG_CONTEXT.reset(token)

    return wrapper


async def request_logging_middleware(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    started_at = now_ms()
    request_id = _request_id(request)
    token = _push_context(
        {
            "requestId": request_id,
            "method": request.method,
            "path": request.url.path,
            **_request_context_fields(request),
        }
    )
    try:
        response = await call_next(request)
    except Exception:
        log.error("http_request_completed", {"statusCode": 500, "outcome": "error", "durationMs": duration_ms(started_at)})
        raise
    else:
        response.headers[REQUEST_ID_HEADER] = request_id
        status_code = response.status_code
        payload = {
            "statusCode": status_code,
            "outcome": "success" if status_code < 400 else "error",
            "durationMs": duration_ms(started_at),
        }
        if status_code >= 500:
            log.error("http_request_completed", payload)
        else:
            log.info("http_request_completed", payload)
        return response
    finally:
        _LOG_CONTEXT.reset(token)


def _push_context(fields: Mapping[str, Any]) -> contextvars.Token[dict[str, Any]]:
    context = dict(_LOG_CONTEXT.get())
    _apply_context_fields(context, fields)
    return _LOG_CONTEXT.set(context)


def _apply_context_fields(context: dict[str, Any], fields: Mapping[str, Any]) -> None:
    for key, value in fields.items():
        if value is None:
            context.pop(key, None)
        else:
            context[key] = value


def _remove_none_values(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def _request_id(request: Request) -> str:
    raw_value = request.headers.get(REQUEST_ID_HEADER)
    if raw_value and raw_value.strip():
        return raw_value.strip()
    return uuid.uuid4().hex


def _request_context_fields(request: Request) -> dict[str, str]:
    fields: dict[str, str] = {}
    for source_name, context_name in _CONTEXT_FIELD_NAMES.items():
        value = request.query_params.get(source_name)
        if value and value.strip():
            fields[context_name] = value.strip()
    return fields


def _operation_name(func: Callable[..., Any]) -> str:
    module = func.__module__
    qualified_name = func.__qualname__.replace("<locals>.", "")
    return f"{module}.{qualified_name}"


def _package_version() -> str:
    try:
        return metadata.version("loop-ad-decision-api")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _runtime_name() -> str:
    return f"python-{platform.python_version()}"


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
            "cause": _to_jsonable(value.__cause__) if value.__cause__ else None,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump())
    if isinstance(value, Mapping):
        return {
            _snake_to_camel(str(key)): _to_jsonable(item)
            for key, item in value.items()
            if item is not None and not _is_secret_key(str(key))
        }
    if isinstance(value, bytes):
        return {"byteLength": len(value)}
    if isinstance(value, list | tuple | set | frozenset):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    if len(parts) == 1:
        return value
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _is_secret_key(value: str) -> bool:
    normalized = value.lower().replace("_", "").replace("-", "")
    return normalized in _SECRET_KEYS or any(
        token in normalized for token in _SECRET_TOKENS
    )


_CONTEXT_FIELD_NAMES = {
    "ad_experiment_id": "adExperimentId",
    "analysis_id": "analysisId",
    "campaign_id": "campaignId",
    "content_id": "contentId",
    "content_option_id": "contentOptionId",
    "generation_id": "generationId",
    "project_id": "projectId",
    "promotion_id": "promotionId",
    "promotion_run_id": "promotionRunId",
    "redirect_id": "redirectId",
    "segment_id": "segmentId",
    "thread_id": "threadId",
    "user_id": "userId",
}

_SECRET_KEYS = {
    "authorization",
    "cookie",
    "setcookie",
    "apikey",
    "password",
    "sessiontoken",
    "refreshtoken",
}
_SECRET_TOKENS = ("secret", "token", "credential")
