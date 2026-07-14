from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings
from app.db import create_postgres_connection
from app.generation.adapters import (
    DEFAULT_GEMINI_IMAGE_MODEL,
    GeminiImageClient,
    ImageClient,
    ImageArtifact,
    S3AssetStorage,
)
from app.generation.artifacts import (
    ArtifactIdentity,
    StoredAsset,
    image_prompt_sha256,
)
from app.generation.repositories import ContentCandidateRepository
from app.logging import duration_ms, log, log_context_scope, now_ms


@dataclass(frozen=True)
class ImageGenerationJob:
    identity: ArtifactIdentity
    image_prompt: str

    @property
    def content_id(self) -> str:
        return self.identity.content_id


class ImageGenerationJobCollector:
    def __init__(self) -> None:
        self._jobs: list[ImageGenerationJob] = []

    @property
    def jobs(self) -> tuple[ImageGenerationJob, ...]:
        return tuple(self._jobs)

    def enqueue(self, job: ImageGenerationJob) -> None:
        self._jobs.append(job)


class AssetStorageClient(Protocol):
    def store_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        image: ImageArtifact,
    ) -> StoredAsset:
        ...


ConnectionFactory = Callable[[Settings], Any]
MAX_IMAGE_GENERATION_WORKERS = 3


@log_context_scope
def run_image_generation_jobs(
    *,
    settings: Settings,
    jobs: Sequence[ImageGenerationJob],
    image_client: ImageClient | None = None,
    asset_storage: AssetStorageClient | None = None,
    connection_factory: ConnectionFactory = create_postgres_connection,
) -> None:
    if not jobs:
        return

    started_at = now_ms()
    log.assign_context({"jobType": "deferred_image_generation"})
    worker_count = _image_generation_worker_count(len(jobs))
    log.info("started", {"jobCount": len(jobs), "workerCount": worker_count})

    succeeded_job_count = 0
    failed_job_count = 0
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="loop-ad-image-worker",
    ) as executor:
        futures = [
            executor.submit(
                _run_image_generation_job_with_connection,
                settings=settings,
                job=job,
                image_client=image_client,
                asset_storage=asset_storage,
                connection_factory=connection_factory,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            try:
                if future.result():
                    succeeded_job_count += 1
                else:
                    failed_job_count += 1
            except Exception as exc:
                failed_job_count += 1
                log.error("image_generation_worker_failed", {"err": exc})

    log.info(
        "completed",
        {
            "jobCount": len(jobs),
            "succeededJobCount": succeeded_job_count,
            "failedJobCount": failed_job_count,
            "durationMs": duration_ms(started_at),
        },
    )


def _run_image_generation_job_with_connection(
    *,
    settings: Settings,
    job: ImageGenerationJob,
    image_client: ImageClient | None,
    asset_storage: AssetStorageClient | None,
    connection_factory: ConnectionFactory,
) -> bool:
    log.assign_context(
        {"jobType": "deferred_image_generation", "contentId": job.content_id}
    )
    try:
        connection = connection_factory(settings)
    except Exception as exc:
        log.error("image_generation_connection_failed", {"err": exc})
        return False

    try:
        repository = ContentCandidateRepository(connection)
        resolved_image_client = image_client or GeminiImageClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_image_model or DEFAULT_GEMINI_IMAGE_MODEL,
        )
        resolved_asset_storage = asset_storage or S3AssetStorage(
            bucket_name=settings.data_storage_bucket,
            base_prefix=settings.genai_assets_base_prefix,
            public_base_url=settings.genai_assets_public_base_url,
        )
        return _run_single_image_generation_job(
            job=job,
            repository=repository,
            image_client=resolved_image_client,
            asset_storage=resolved_asset_storage,
            connection=connection,
        )
    finally:
        connection.close()


def _run_single_image_generation_job(
    *,
    job: ImageGenerationJob,
    repository: ContentCandidateRepository,
    image_client: ImageClient,
    asset_storage: AssetStorageClient,
    connection: Any,
) -> bool:
    job_started_at = now_ms()
    log.assign_context({"contentId": job.content_id})
    try:
        image = image_client.generate_image(image_prompt=job.image_prompt)
        stored_image = asset_storage.store_image(
            identity=job.identity,
            image_prompt_sha256=image_prompt_sha256(job.image_prompt),
            image=image,
        )
        repository.update_image_url(
            content_id=job.content_id,
            image_url=stored_image.public_url,
        )
        connection.commit()
        log.info(
            "image_generation_completed",
            {
                "imageUrl": stored_image.public_url,
                "durationMs": duration_ms(job_started_at),
            },
        )
        return True
    except Exception as exc:
        connection.rollback()
        log.warn(
            "image_generation_failed",
            {"err": exc, "durationMs": duration_ms(job_started_at)},
        )
        try:
            repository.mark_image_generation_failed(
                content_id=job.content_id,
                error_code=_safe_image_generation_error_code(exc),
            )
            connection.commit()
            log.info("image_generation_failure_recorded")
        except Exception as record_exc:
            connection.rollback()
            log.error("image_generation_failure_record_failed", {"err": record_exc})
        return False


def _safe_image_generation_error_code(exc: Exception) -> str:
    del exc
    return "image_generation_failed"


def _image_generation_worker_count(job_count: int) -> int:
    return max(1, min(job_count, MAX_IMAGE_GENERATION_WORKERS))
