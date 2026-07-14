from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


DECISION_SERVICE_ID = "decision-api"
DEFAULT_GENERATION_RETRY_BACKOFF_SECONDS = (60, 300, 900)
DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL = "https://gen-ai.asset.dev.loop-ad.org"
DEFAULT_GENAI_SOURCE_MANIFEST_PREFIX = "genai-source/"


class SettingsError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid.

    The settings loader uses raise RuntimeError-compatible failures before serving
    traffic so missing secrets or malformed env values fail fast at startup.
    """


@dataclass(frozen=True, slots=True)
class Settings:
    env: str
    service_id: str
    port: int
    internal_api_key: str
    aurora_host: str
    aurora_port: int
    aurora_database: str
    aurora_username: str
    aurora_password: str
    clickhouse_url: str
    clickhouse_database: str
    clickhouse_username: str
    clickhouse_password: str
    data_storage_bucket: str
    genai_assets_base_prefix: str
    openai_api_key: str
    gemini_api_key: str
    genai_assets_public_base_url: str = DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL
    genai_source_manifest_prefix: str = DEFAULT_GENAI_SOURCE_MANIFEST_PREFIX
    openai_content_model: str | None = None
    gemini_image_model: str | None = None
    segment_performance_model_path: str | None = None
    generation_worker_max_concurrency: int = 2
    generation_poll_interval_seconds: int = 1
    generation_idle_poll_interval_seconds: int = 30
    generation_lease_seconds: int = 180
    generation_heartbeat_seconds: int = 30
    generation_max_retries: int = 3
    generation_retry_backoff_seconds: tuple[int, ...] = (
        DEFAULT_GENERATION_RETRY_BACKOFF_SECONDS
    )
    generation_provider_timeout_seconds: int = 30
    generation_db_operation_timeout_seconds: int = 5
    generation_shutdown_grace_seconds: int = 20


REQUIRED_ENV_NAMES = (
    "LOOPAD_ENV",
    "LOOPAD_SERVICE_ID",
    "PORT",
    "LOOPAD_INTERNAL_API_KEY",
    "LOOPAD_AURORA_HOST",
    "LOOPAD_AURORA_PORT",
    "LOOPAD_AURORA_DATABASE",
    "LOOPAD_AURORA_USERNAME",
    "LOOPAD_AURORA_PASSWORD",
    "LOOPAD_CLICKHOUSE_URL",
    "LOOPAD_CLICKHOUSE_DATABASE",
    "LOOPAD_CLICKHOUSE_USERNAME",
    "LOOPAD_CLICKHOUSE_PASSWORD",
    "LOOPAD_DATA_STORAGE_BUCKET",
    "LOOPAD_GENAI_ASSETS_BASE_PREFIX",
    "LOOPAD_OPENAI_API_KEY",
    "LOOPAD_GEMINI_API_KEY",
)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    if environ is None:
        load_local_dotenv()
    source = environ if environ is not None else os.environ
    missing = [name for name in REQUIRED_ENV_NAMES if not _read_required(source, name)]
    if missing:
        raise SettingsError("missing required env: " + ", ".join(missing))

    service_id = _read_required(source, "LOOPAD_SERVICE_ID")
    if service_id != DECISION_SERVICE_ID:
        raise SettingsError(
            f"LOOPAD_SERVICE_ID must be {DECISION_SERVICE_ID!r}, got {service_id!r}"
        )

    settings = Settings(
        env=_read_required(source, "LOOPAD_ENV"),
        service_id=service_id,
        port=_read_positive_int(source, "PORT"),
        internal_api_key=_read_required(source, "LOOPAD_INTERNAL_API_KEY"),
        aurora_host=_read_required(source, "LOOPAD_AURORA_HOST"),
        aurora_port=_read_positive_int(source, "LOOPAD_AURORA_PORT"),
        aurora_database=_read_required(source, "LOOPAD_AURORA_DATABASE"),
        aurora_username=_read_required(source, "LOOPAD_AURORA_USERNAME"),
        aurora_password=_read_required(source, "LOOPAD_AURORA_PASSWORD"),
        clickhouse_url=_read_required(source, "LOOPAD_CLICKHOUSE_URL"),
        clickhouse_database=_read_required(source, "LOOPAD_CLICKHOUSE_DATABASE"),
        clickhouse_username=_read_required(source, "LOOPAD_CLICKHOUSE_USERNAME"),
        clickhouse_password=_read_required(source, "LOOPAD_CLICKHOUSE_PASSWORD"),
        data_storage_bucket=_read_required(source, "LOOPAD_DATA_STORAGE_BUCKET"),
        genai_assets_base_prefix=_read_required(
            source,
            "LOOPAD_GENAI_ASSETS_BASE_PREFIX",
        ),
        openai_api_key=_read_required(source, "LOOPAD_OPENAI_API_KEY"),
        gemini_api_key=_read_required(source, "LOOPAD_GEMINI_API_KEY"),
        genai_assets_public_base_url=(
            _read_optional(source, "LOOPAD_GENAI_ASSETS_PUBLIC_BASE_URL")
            or DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL
        ),
        genai_source_manifest_prefix=(
            _read_optional(source, "LOOPAD_GENAI_SOURCE_MANIFEST_PREFIX")
            or DEFAULT_GENAI_SOURCE_MANIFEST_PREFIX
        ),
        openai_content_model=_read_optional(source, "LOOPAD_OPENAI_CONTENT_MODEL"),
        gemini_image_model=_read_optional(source, "LOOPAD_GEMINI_IMAGE_MODEL"),
        segment_performance_model_path=_read_optional(
            source,
            "LOOPAD_SEGMENT_PERFORMANCE_MODEL_PATH",
        ),
        generation_worker_max_concurrency=_read_optional_positive_int(
            source,
            "GENERATION_WORKER_MAX_CONCURRENCY",
            default=2,
        ),
        generation_poll_interval_seconds=_read_optional_positive_int(
            source,
            "GENERATION_POLL_INTERVAL_SECONDS",
            default=1,
        ),
        generation_idle_poll_interval_seconds=_read_optional_positive_int(
            source,
            "GENERATION_IDLE_POLL_INTERVAL_SECONDS",
            default=30,
        ),
        generation_lease_seconds=_read_optional_positive_int(
            source,
            "GENERATION_LEASE_SECONDS",
            default=180,
        ),
        generation_heartbeat_seconds=_read_optional_positive_int(
            source,
            "GENERATION_HEARTBEAT_SECONDS",
            default=30,
        ),
        generation_max_retries=_read_optional_non_negative_int(
            source,
            "GENERATION_MAX_RETRIES",
            default=3,
        ),
        generation_retry_backoff_seconds=_read_optional_positive_int_tuple(
            source,
            "GENERATION_RETRY_BACKOFF_SECONDS",
            default=DEFAULT_GENERATION_RETRY_BACKOFF_SECONDS,
        ),
        generation_provider_timeout_seconds=_read_optional_positive_int(
            source,
            "GENERATION_PROVIDER_TIMEOUT_SECONDS",
            default=30,
        ),
        generation_db_operation_timeout_seconds=_read_optional_positive_int(
            source,
            "GENERATION_DB_OPERATION_TIMEOUT_SECONDS",
            default=5,
        ),
        generation_shutdown_grace_seconds=_read_optional_positive_int(
            source,
            "GENERATION_SHUTDOWN_GRACE_SECONDS",
            default=20,
        ),
    )
    _validate_generation_settings(settings)
    return settings


def load_local_dotenv() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _read_required(source: Mapping[str, str], name: str) -> str:
    return str(source.get(name, "")).strip()


def _read_optional(source: Mapping[str, str], name: str) -> str | None:
    value = str(source.get(name, "")).strip()
    return value or None


def _read_positive_int(source: Mapping[str, str], name: str) -> int:
    raw_value = _read_required(source, name)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be a positive integer")
    return value


def _read_optional_positive_int(
    source: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    raw_value = _read_optional(source, name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be a positive integer")
    return value


def _read_optional_non_negative_int(
    source: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    raw_value = _read_optional(source, name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise SettingsError(f"{name} must be a non-negative integer")
    return value


def _read_optional_positive_int_tuple(
    source: Mapping[str, str],
    name: str,
    *,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    raw_value = _read_optional(source, name)
    if raw_value is None:
        return default
    raw_items = raw_value.split(",")
    try:
        values = tuple(int(item.strip()) for item in raw_items)
    except ValueError as exc:
        raise SettingsError(
            f"{name} must be a comma-separated list of positive integers"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise SettingsError(
            f"{name} must be a comma-separated list of positive integers"
        )
    return values


def _validate_generation_settings(settings: Settings) -> None:
    public_prefix = settings.genai_assets_base_prefix.strip("/")
    source_prefix = settings.genai_source_manifest_prefix.strip("/")
    if (
        not public_prefix
        or not source_prefix
        or source_prefix == public_prefix
        or source_prefix.startswith(f"{public_prefix}/")
    ):
        raise SettingsError(
            "LOOPAD_GENAI_SOURCE_MANIFEST_PREFIX must be outside the public "
            "LOOPAD_GENAI_ASSETS_BASE_PREFIX"
        )
    if settings.generation_heartbeat_seconds >= settings.generation_lease_seconds:
        raise SettingsError(
            "GENERATION_HEARTBEAT_SECONDS must be less than "
            "GENERATION_LEASE_SECONDS"
        )
    if len(settings.generation_retry_backoff_seconds) < settings.generation_max_retries:
        raise SettingsError(
            "GENERATION_RETRY_BACKOFF_SECONDS must provide at least "
            "GENERATION_MAX_RETRIES entries"
        )
    heartbeat_budget = (
        settings.generation_heartbeat_seconds
        + 2
        * (settings.generation_worker_max_concurrency + 1)
        * settings.generation_db_operation_timeout_seconds
    )
    if heartbeat_budget >= settings.generation_lease_seconds:
        raise SettingsError(
            "GENERATION_HEARTBEAT_SECONDS plus coordinator DB timeout budget "
            "must be less than GENERATION_LEASE_SECONDS"
        )
