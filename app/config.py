from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


DECISION_SERVICE_ID = "decision-api"
GENAI_ASSETS_PUBLIC_BASE_URL = "https://gen-ai.asset.dev.loop-ad.org"
GENAI_SOURCE_MANIFEST_PREFIX = "genai-source/"
OPENAI_CONTENT_MODEL = "gpt-4o-mini"
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
GENERATION_WORKER_MAX_CONCURRENCY = 2
GENERATION_POLL_INTERVAL_SECONDS = 1
GENERATION_IDLE_POLL_INTERVAL_SECONDS = 30
GENERATION_LEASE_SECONDS = 180
GENERATION_HEARTBEAT_SECONDS = 30
GENERATION_MAX_RETRIES = 3
GENERATION_RETRY_BACKOFF_SECONDS = (60, 300, 900)
GENERATION_PROVIDER_TIMEOUT_SECONDS = 30
GENERATION_DB_OPERATION_TIMEOUT_SECONDS = 5
GENERATION_SHUTDOWN_GRACE_SECONDS = 20


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
    genai_assets_public_base_url: str = GENAI_ASSETS_PUBLIC_BASE_URL
    openai_content_model: str = OPENAI_CONTENT_MODEL
    gemini_image_model: str = GEMINI_IMAGE_MODEL
    generation_worker_max_concurrency: int = GENERATION_WORKER_MAX_CONCURRENCY
    generation_poll_interval_seconds: int = GENERATION_POLL_INTERVAL_SECONDS
    generation_idle_poll_interval_seconds: int = GENERATION_IDLE_POLL_INTERVAL_SECONDS
    generation_lease_seconds: int = GENERATION_LEASE_SECONDS
    generation_heartbeat_seconds: int = GENERATION_HEARTBEAT_SECONDS
    generation_max_retries: int = GENERATION_MAX_RETRIES
    generation_retry_backoff_seconds: tuple[int, ...] = (
        GENERATION_RETRY_BACKOFF_SECONDS
    )
    generation_provider_timeout_seconds: int = GENERATION_PROVIDER_TIMEOUT_SECONDS
    generation_db_operation_timeout_seconds: int = (
        GENERATION_DB_OPERATION_TIMEOUT_SECONDS
    )
    generation_shutdown_grace_seconds: int = GENERATION_SHUTDOWN_GRACE_SECONDS
    segment_performance_model_path: str | None = None


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
        segment_performance_model_path=_read_optional(
            source,
            "LOOPAD_SEGMENT_PERFORMANCE_MODEL_PATH",
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


def _validate_generation_settings(settings: Settings) -> None:
    public_prefix = settings.genai_assets_base_prefix.strip("/")
    source_prefix = GENAI_SOURCE_MANIFEST_PREFIX.strip("/")
    if (
        not public_prefix
        or not source_prefix
        or source_prefix == public_prefix
        or source_prefix.startswith(f"{public_prefix}/")
    ):
        raise SettingsError(
            "GENAI_SOURCE_MANIFEST_PREFIX must be outside the public "
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
