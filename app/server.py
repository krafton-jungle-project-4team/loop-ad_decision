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


def main() -> None:
    load_dotenv()
    missing_vars = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing_vars:
        names = ", ".join(missing_vars)
        raise RuntimeError(f"Missing required environment variables: {names}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ["PORT"]),
    )


if __name__ == "__main__":
    main()

