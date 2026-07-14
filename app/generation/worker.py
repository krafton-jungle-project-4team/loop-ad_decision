from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import UUID

from app.config import Settings
from app.db import create_postgres_connection
from app.generation.adapters import (
    build_external_content_generator,
    build_s3_creative_artifact_publisher,
)
from app.generation.errors import (
    PermanentGenerationError,
    classify_generation_error,
    retry_backoff_seconds,
)
from app.generation.repositories import (
    ContentCandidateRecord,
    ContentCandidateRepository,
    GenerationRunRepository,
)
from app.generation.service import DurableGenerationResult, GenerationService
from app.generation.submission import prompt_inputs_from_snapshot
from app.logging import log


class DurableGenerationExecutor(Protocol):
    def execute_durable(
        self,
        *,
        generation_id: str,
        prompt_inputs: list[Any],
    ) -> DurableGenerationResult:
        ...


ConnectionFactory = Callable[[Settings], Any]
ServiceFactory = Callable[[], DurableGenerationExecutor]
Clock = Callable[[], datetime]
Jitter = Callable[[float], float]


class GenerationFenceRejected(RuntimeError):
    """The active attempt no longer owns the durable run or was not ready."""


class GenerationJobProcessor:
    """Execute one claimed run and persist its terminal/retry transition.

    Provider and S3 work deliberately runs without an open database
    transaction. Candidate UPSERTs and the strict completion gate share one
    short fenced transaction.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        connection_factory: ConnectionFactory = create_postgres_connection,
        generation_service_factory: ServiceFactory | None = None,
        clock: Clock | None = None,
        jitter: Jitter | None = None,
    ) -> None:
        self._settings = settings
        self._connection_factory = connection_factory
        self._generation_service_factory = (
            generation_service_factory or self._build_generation_service
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._jitter = jitter or _default_retry_jitter

    def process(
        self,
        claimed_row: Mapping[str, Any],
        *,
        worker_id: str,
        lease_token: UUID,
    ) -> None:
        generation_id = _required_text(claimed_row, "generation_id")
        try:
            input_snapshot = claimed_row.get("input_json")
            if not isinstance(input_snapshot, Mapping):
                raise ValueError("generation input_json must be an object")
            prompt_inputs = prompt_inputs_from_snapshot(input_snapshot)
            result = self._generation_service_factory().execute_durable(
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
            )
            self._persist_success(
                result,
                worker_id=worker_id,
                lease_token=lease_token,
            )
        except Exception as exc:
            self._record_failure(
                generation_id=generation_id,
                retry_count=_retry_count(claimed_row),
                worker_id=worker_id,
                lease_token=lease_token,
                error=exc,
            )

    def _persist_success(
        self,
        result: DurableGenerationResult,
        *,
        worker_id: str,
        lease_token: UUID,
    ) -> None:
        connection = self._connection_factory(self._settings)
        try:
            candidate_repository = ContentCandidateRepository(connection)
            for candidate in result.content_candidates:
                persisted = candidate_repository.upsert_fenced(
                    candidate,
                    worker_id=worker_id,
                    lease_token=lease_token,
                )
                if persisted is None:
                    raise GenerationFenceRejected(
                        "candidate write was rejected by the active lease fence"
                    )

            completed = GenerationRunRepository(connection).complete_if_ready_fenced(
                generation_id=result.generation_id,
                worker_id=worker_id,
                lease_token=lease_token,
                output_json=result.output_json,
                generation_report_json=result.generation_report_json,
            )
            if completed is None:
                raise GenerationFenceRejected(
                    "generation completion readiness or lease fence was rejected"
                )
            connection.commit()
            log.info(
                "generation_job_completed",
                {
                    "generationId": result.generation_id,
                    "workerId": worker_id,
                    "candidateCount": len(result.content_candidates),
                },
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _record_failure(
        self,
        *,
        generation_id: str,
        retry_count: int,
        worker_id: str,
        lease_token: UUID,
        error: Exception,
    ) -> None:
        if isinstance(error, GenerationFenceRejected):
            error = PermanentGenerationError(
                code="generation_fence_rejected",
                safe_message="Generation result was rejected by its lease or readiness fence.",
            )
        error_info = classify_generation_error(error)
        should_retry = (
            error_info.retryable
            and retry_count < self._settings.generation_max_retries
        )

        connection = self._connection_factory(self._settings)
        try:
            repository = GenerationRunRepository(connection)
            if should_retry:
                next_retry_count = retry_count + 1
                delay_seconds = retry_backoff_seconds(
                    next_retry_count,
                    self._settings.generation_retry_backoff_seconds,
                    jitter=self._jitter,
                )
                transitioned = repository.schedule_retry_fenced(
                    generation_id=generation_id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    next_retry_at=self._clock() + timedelta(seconds=delay_seconds),
                    error_code=error_info.code,
                    error_message=error_info.message,
                )
                transition = "retry_scheduled"
            else:
                transitioned = repository.mark_failed_fenced(
                    generation_id=generation_id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    error_code=error_info.code,
                    error_message=error_info.message,
                )
                transition = "failed"
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        log.warn(
            "generation_job_failed",
            {
                "generationId": generation_id,
                "workerId": worker_id,
                "retryCount": retry_count,
                "errorCode": error_info.code,
                "transition": transition if transitioned else "fence_rejected",
            },
        )

    def _build_generation_service(self) -> GenerationService:
        return GenerationService(
            content_generator=build_external_content_generator(
                self._settings,
                generate_images=True,
            ),
            artifact_publisher=build_s3_creative_artifact_publisher(self._settings),
        )


def _required_text(value: Mapping[str, Any], key: str) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise ValueError(f"claimed generation row is missing {key}")
    return text


def _retry_count(value: Mapping[str, Any]) -> int:
    try:
        count = int(value.get("retry_count") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("generation retry_count must be an integer") from exc
    if count < 0:
        raise ValueError("generation retry_count must not be negative")
    return count


def _default_retry_jitter(base_seconds: float) -> float:
    return random.uniform(0.0, min(5.0, base_seconds * 0.1))
