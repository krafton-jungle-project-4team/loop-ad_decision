from __future__ import annotations

import threading

from app.config import load_settings
from app.generation.adapters import ImageArtifact
from app.generation.artifacts import ArtifactIdentity, StoredAsset
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
        "LOOPAD_GENAI_SOURCE_MANIFEST_PREFIX": "genai-source/",
        "LOOPAD_OPENAI_API_KEY": "openai-key",
        "LOOPAD_GEMINI_API_KEY": "gemini-key",
    }


def identity(content_id: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        generation_id="generation_banner_001",
        content_id=content_id,
    )


def test_run_image_generation_jobs_updates_candidate_image_url() -> None:
    connection = FakeConnection()
    image_client = FakeImageClient()
    asset_storage = FakeAssetStorage()

    run_image_generation_jobs(
        settings=load_settings(valid_env()),
        jobs=[
            ImageGenerationJob(
                identity=identity("content_banner_repeat_hotel_001"),
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
    assert "%(image_url)s::text" in query
    assert params == {
        "content_id": "content_banner_repeat_hotel_001",
        "image_url": (
            "https://gen-ai.asset.dev.loop-ad.org/hotel-client-a/"
            "promo_banner_001/generation_banner_001/"
            "content_banner_repeat_hotel_001/image.png"
        ),
    }


def test_run_image_generation_jobs_records_image_generation_failure() -> None:
    connection = FakeConnection()

    run_image_generation_jobs(
        settings=load_settings(valid_env()),
        jobs=[
            ImageGenerationJob(
                identity=identity("content_banner_repeat_hotel_001"),
                image_prompt="bright hotel suite banner",
            )
        ],
        image_client=FailingImageClient(),
        asset_storage=FakeAssetStorage(),
        connection_factory=lambda _settings: connection,
    )

    assert connection.rollback_count == 1
    assert connection.commit_count == 1
    assert connection.close_count == 1
    query, params = connection.executed[0]
    assert "UPDATE content_candidates" in query
    assert "%(error_code)s::text" in query
    assert params == {
        "content_id": "content_banner_repeat_hotel_001",
        "error_code": "image_generation_failed",
    }


def test_run_image_generation_jobs_processes_multiple_jobs_in_parallel() -> None:
    connections: list[FakeConnection] = []
    connection_lock = threading.Lock()
    image_client = BarrierImageClient(job_count=3)
    asset_storage = FakeAssetStorage()

    def connection_factory(_settings: object) -> FakeConnection:
        connection = FakeConnection()
        with connection_lock:
            connections.append(connection)
        return connection

    run_image_generation_jobs(
        settings=load_settings(valid_env()),
        jobs=[
            ImageGenerationJob(
                identity=identity(f"content_banner_repeat_hotel_00{index}"),
                image_prompt=f"bright hotel suite banner {index}",
            )
            for index in range(1, 4)
        ],
        image_client=image_client,
        asset_storage=asset_storage,
        connection_factory=connection_factory,
    )

    assert sorted(image_client.prompts) == [
        "bright hotel suite banner 1",
        "bright hotel suite banner 2",
        "bright hotel suite banner 3",
    ]
    assert sorted(asset_storage.saved_content_ids) == [
        "content_banner_repeat_hotel_001",
        "content_banner_repeat_hotel_002",
        "content_banner_repeat_hotel_003",
    ]
    assert len(connections) == 3
    assert sum(connection.commit_count for connection in connections) == 3
    assert sum(connection.rollback_count for connection in connections) == 0
    assert sum(connection.close_count for connection in connections) == 3


class FakeImageClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        with self._lock:
            self.prompts.append(image_prompt)
        return ImageArtifact(data=b"image-bytes", content_type="image/png")


class BarrierImageClient(FakeImageClient):
    def __init__(self, *, job_count: int) -> None:
        super().__init__()
        self._barrier = threading.Barrier(job_count)

    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        with self._lock:
            self.prompts.append(image_prompt)
        self._barrier.wait(timeout=1)
        return ImageArtifact(data=b"image-bytes", content_type="image/png")


class FailingImageClient:
    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        del image_prompt
        raise RuntimeError("image provider unavailable")


class FakeAssetStorage:
    def __init__(self) -> None:
        self.saved_content_ids: list[str] = []
        self._lock = threading.Lock()

    def store_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        image: ImageArtifact,
    ) -> StoredAsset:
        del image_prompt_sha256
        with self._lock:
            self.saved_content_ids.append(identity.content_id)
        public_url = (
            "https://gen-ai.asset.dev.loop-ad.org/hotel-client-a/"
            "promo_banner_001/generation_banner_001/"
            f"{identity.content_id}/image.png"
        )
        return StoredAsset(
            storage_key=(
                "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
                f"{identity.content_id}/image.png"
            ),
            public_url=public_url,
            sha256="0" * 64,
            bytes=len(image.data),
            content_type=image.content_type,
        )


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
