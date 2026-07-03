from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


DECISION_SERVICE_ID = "decision-api"


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
    openai_content_model: str | None = None


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

    return Settings(
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
        openai_content_model=_read_optional(source, "LOOPAD_OPENAI_CONTENT_MODEL"),
    )


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
