from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
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
from app.generation.repositories import ContentCandidateRepository
from app.logging import log, log_context_scope, now_ms, duration_ms


@dataclass(frozen=True)
class ImageGenerationJob:
    content_id: str
    image_prompt: str


class ImageGenerationJobCollector:
    def __init__(self) -> None:
        self._jobs: list[ImageGenerationJob] = []

    @property
    def jobs(self) -> tuple[ImageGenerationJob, ...]:
        return tuple(self._jobs)

    def enqueue(self, job: ImageGenerationJob) -> None:
        self._jobs.append(job)


class AssetStorageClient(Protocol):
    def store_image(self, *, content_id: str, image: ImageArtifact) -> str:
        ...


ConnectionFactory = Callable[[Settings], Any]


def dispatch_image_generation_jobs(
    *,
    settings: Settings,
    jobs: Sequence[ImageGenerationJob],
) -> None:
    job_list = list(jobs)
    if not job_list:
        return

    thread = threading.Thread(
        target=run_image_generation_jobs,
        kwargs={"settings": settings, "jobs": job_list},
        name="loop-ad-image-generation",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        log.error("image_generation_dispatch_failed", {"jobCount": len(job_list)})
    else:
        log.info("image_generation_dispatched", {"jobCount": len(job_list)})


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
    log.info("started", {"jobCount": len(jobs)})
    try:
        connection = connection_factory(settings)
    except Exception as exc:
        log.error("image_generation_connection_failed", {"err": exc})
        return

    try:
        repository = ContentCandidateRepository(connection)
        image_client = image_client or GeminiImageClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_image_model or DEFAULT_GEMINI_IMAGE_MODEL,
        )
        asset_storage = asset_storage or S3AssetStorage(
            bucket_name=settings.data_storage_bucket,
            base_prefix=settings.genai_assets_base_prefix,
        )

        for job in jobs:
            _run_single_image_generation_job(
                job=job,
                repository=repository,
                image_client=image_client,
                asset_storage=asset_storage,
                connection=connection,
            )
        log.info("completed", {"jobCount": len(jobs), "durationMs": duration_ms(started_at)})
    finally:
        connection.close()


def _run_single_image_generation_job(
    *,
    job: ImageGenerationJob,
    repository: ContentCandidateRepository,
    image_client: ImageClient,
    asset_storage: AssetStorageClient,
    connection: Any,
) -> None:
    job_started_at = now_ms()
    log.assign_context({"contentId": job.content_id})
    try:
        image = image_client.generate_image(image_prompt=job.image_prompt)
        image_url = asset_storage.store_image(content_id=job.content_id, image=image)
        repository.update_image_url(content_id=job.content_id, image_url=image_url)
        connection.commit()
        log.info("image_generation_completed", {"imageUrl": image_url, "durationMs": duration_ms(job_started_at)})
    except Exception as exc:
        connection.rollback()
        log.warn("image_generation_failed", {"err": exc, "durationMs": duration_ms(job_started_at)})
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


def _safe_image_generation_error_code(exc: Exception) -> str:
    del exc
    return "image_generation_failed"
