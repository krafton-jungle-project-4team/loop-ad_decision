from __future__ import annotations

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

import structlog
from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from app.config import Settings


LOGGER_NAME = "loopad"
LOG_LEVEL = logging.DEBUG
REQUEST_ID_HEADER = "X-Request-Id"

P = ParamSpec("P")
R = TypeVar("R")


class ContextLogger:
    def assign_context(self, fields: Mapping[str, Any]) -> None:
        context = structlog.contextvars.get_contextvars()
        _apply_context_fields(context, fields)
        _replace_context(context)

    def debug(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._logger.debug(event, **_event_fields(payload))

    def info(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._logger.info(event, **_event_fields(payload))

    def warn(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._logger.warning(event, **_event_fields(payload))

    def error(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._logger.error(event, **_event_fields(payload))

    @property
    def _logger(self) -> Any:
        return structlog.get_logger(LOGGER_NAME)


log = ContextLogger()


def configure_logging(settings: Settings) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            _event_processor(settings),
            structlog.processors.JSONRenderer(serializer=_json_dumps),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


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
            previous_context = _push_context({"operation": operation})
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                log.error("failed", {"err": exc, "durationMs": duration_ms(started_at)})
                raise
            finally:
                _replace_context(previous_context)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started_at = now_ms()
        previous_context = _push_context({"operation": operation})
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            log.error("failed", {"err": exc, "durationMs": duration_ms(started_at)})
            raise
        finally:
            _replace_context(previous_context)

    return wrapper


async def request_logging_middleware(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    started_at = now_ms()
    request_id = _request_id(request)
    previous_context = _replace_context(
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
        _replace_context(previous_context)


def _push_context(fields: Mapping[str, Any]) -> dict[str, Any]:
    previous_context = structlog.contextvars.get_contextvars()
    context = dict(previous_context)
    _apply_context_fields(context, fields)
    _replace_context(context)
    return previous_context


def _replace_context(fields: Mapping[str, Any]) -> dict[str, Any]:
    previous_context = structlog.contextvars.get_contextvars()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**_remove_none_values(dict(fields)))
    return previous_context


def _apply_context_fields(context: dict[str, Any], fields: Mapping[str, Any]) -> None:
    for key, value in fields.items():
        if value is None:
            context.pop(key, None)
        else:
            context[key] = value


def _remove_none_values(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def _event_fields(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return _remove_none_values(dict(payload or {}))


def _event_processor(settings: Settings) -> Callable[[Any, str, dict[str, Any]], dict[str, Any]]:
    base_fields = _base_fields(settings)

    def processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        body = {
            "timestamp": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": _level_name(method_name),
            **base_fields,
            **event_dict,
        }
        return _to_jsonable(_remove_none_values(body))

    return processor


def _base_fields(settings: Settings) -> dict[str, Any]:
    return _remove_none_values(
        {
            "service": settings.service_id,
            "environment": settings.env,
            "version": _package_version(),
            "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
            "runtime": os.environ.get("AWS_EXECUTION_ENV") or _runtime_name(),
        }
    )


def _level_name(method_name: str) -> str:
    if method_name == "warn":
        return "warning"
    return method_name


def _json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), **kwargs)


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
            if item is not None
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
