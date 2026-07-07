from __future__ import annotations

from app.config import load_settings
from app.generation.adapters import ImageArtifact
from app.generation.image_tasks import (
    ImageGenerationJob,
    run_image_generation_jobs,
)


def valid_env() -> dict[str, str]:
    return {
        "LOOPAD_ENV": "dev",
        "LOOPAD_SERVICE_ID": "decision-api",
        "PORT": "8080",
        "LOOPAD_INTERNAL_API_KEY": "internal-key",
        "LOOPAD_AURORA_HOST": "localhost",
        "LOOPAD_AURORA_PORT": "15432",
        "LOOPAD_AURORA_DATABASE": "loopad",
        "LOOPAD_AURORA_USERNAME": "loopad",
        "LOOPAD_AURORA_PASSWORD": "loopad",
        "LOOPAD_CLICKHOUSE_URL": "http://localhost:18123",
        "LOOPAD_CLICKHOUSE_DATABASE": "loopad",
        "LOOPAD_CLICKHOUSE_USERNAME": "loopad",
        "LOOPAD_CLICKHOUSE_PASSWORD": "loopad",
        "LOOPAD_DATA_STORAGE_BUCKET": "loop-ad-dev-data-storage",
        "LOOPAD_GENAI_ASSETS_BASE_PREFIX": "genai/",
        "LOOPAD_OPENAI_API_KEY": "openai-key",
        "LOOPAD_GEMINI_API_KEY": "gemini-key",
    }


def test_run_image_generation_jobs_updates_candidate_image_url() -> None:
    connection = FakeConnection()
    image_client = FakeImageClient()
    asset_storage = FakeAssetStorage()

    run_image_generation_jobs(
        settings=load_settings(valid_env()),
        jobs=[
            ImageGenerationJob(
                content_id="content_banner_repeat_hotel_001",
                image_prompt="bright hotel suite banner",
            )
        ],
        image_client=image_client,
        asset_storage=asset_storage,
        connection_factory=lambda _settings: connection,
    )

    assert image_client.prompts == ["bright hotel suite banner"]
    assert asset_storage.saved_content_ids == ["content_banner_repeat_hotel_001"]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    query, params = connection.executed[0]
    assert "UPDATE content_candidates" in query
    assert params == {
        "content_id": "content_banner_repeat_hotel_001",
        "image_url": (
            "https://gen-ai.asset.dev.loop-ad.org/generated/"
            "content_banner_repeat_hotel_001.png"
        ),
    }


class FakeImageClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        self.prompts.append(image_prompt)
        return ImageArtifact(data=b"image-bytes", content_type="image/png")


class FakeAssetStorage:
    def __init__(self) -> None:
        self.saved_content_ids: list[str] = []

    def store_image(self, *, content_id: str, image: ImageArtifact) -> str:
        del image
        self.saved_content_ids.append(content_id)
        return f"https://gen-ai.asset.dev.loop-ad.org/generated/{content_id}.png"


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: dict[str, object] | None = None) -> None:
        self._connection.executed.append((query, params))

    def fetchone(self) -> dict[str, object]:
        return {"content_id": "content_banner_repeat_hotel_001"}


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, object] | None]] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self, *, row_factory: object = None) -> FakeCursor:
        del row_factory
        return FakeCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1
