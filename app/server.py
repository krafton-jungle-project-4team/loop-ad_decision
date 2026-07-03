import os

from dotenv import load_dotenv
import uvicorn


REQUIRED_ENV_VARS: tuple[str, ...] = (
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


def validate_required_env_vars() -> None:
    missing_vars = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing_vars:
        names = ", ".join(missing_vars)
        raise RuntimeError(f"Missing required environment variables: {names}")


def get_port() -> int:
    port = os.environ.get("PORT")
    if port is None:
        raise RuntimeError("PORT environment variable is required")

    try:
        return int(port)
    except ValueError as exc:
        raise RuntimeError("PORT environment variable must be an integer") from exc


def main() -> None:
    load_dotenv()
    validate_required_env_vars()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=get_port(),
    )


if __name__ == "__main__":
    main()
