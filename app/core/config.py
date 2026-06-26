from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    loopad_env: str = Field(alias="LOOPAD_ENV")
    loopad_service_id: str = Field(alias="LOOPAD_SERVICE_ID")
    loopad_runtime: str = Field(alias="LOOPAD_RUNTIME")
    port: int = Field(alias="PORT")

    loopad_aurora_host: str = Field(alias="LOOPAD_AURORA_HOST")
    loopad_aurora_port: int = Field(alias="LOOPAD_AURORA_PORT")
    loopad_aurora_database: str = Field(alias="LOOPAD_AURORA_DATABASE")
    loopad_aurora_username: str = Field(alias="LOOPAD_AURORA_USERNAME")
    loopad_aurora_password: SecretStr = Field(alias="LOOPAD_AURORA_PASSWORD")

    loopad_clickhouse_url: str = Field(alias="LOOPAD_CLICKHOUSE_URL")
    loopad_clickhouse_username: str = Field(alias="LOOPAD_CLICKHOUSE_USERNAME")

    loopad_data_storage_bucket: str = Field(alias="LOOPAD_DATA_STORAGE_BUCKET")
    loopad_genai_generated_assets_prefix: str = Field(
        alias="LOOPAD_GENAI_GENERATED_ASSETS_PREFIX",
    )
    loopad_openai_api_key: SecretStr = Field(alias="LOOPAD_OPENAI_API_KEY")

    loopad_postgres_auto_create_tables: bool = Field(
        alias="LOOPAD_POSTGRES_AUTO_CREATE_TABLES",
    )
    loopad_analysis_worker_poll_interval_seconds: float = Field(
        alias="LOOPAD_ANALYSIS_WORKER_POLL_INTERVAL_SECONDS",
    )

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    @field_validator(
        "loopad_env",
        "loopad_service_id",
        "loopad_runtime",
        "loopad_aurora_host",
        "loopad_aurora_database",
        "loopad_aurora_username",
        "loopad_clickhouse_url",
        "loopad_clickhouse_username",
        "loopad_data_storage_bucket",
        "loopad_genai_generated_assets_prefix",
    )
    @classmethod
    def require_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("loopad_service_id")
    @classmethod
    def validate_service_id(cls, value: str) -> str:
        if value != "decision-api":
            raise ValueError("LOOPAD_SERVICE_ID must be decision-api")
        return value

    @field_validator("loopad_runtime")
    @classmethod
    def validate_runtime(cls, value: str) -> str:
        if value != "go":
            raise ValueError("LOOPAD_RUNTIME must be go")
        return value

    @field_validator("loopad_clickhouse_url")
    @classmethod
    def validate_clickhouse_url(cls, value: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None or parsed.port is None:
            raise ValueError("LOOPAD_CLICKHOUSE_URL must be http(s)://host:port")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
