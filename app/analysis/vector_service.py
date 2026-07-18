from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.analysis.repositories import SegmentVectorRecord, UserBehaviorVectorRecord
from app.logging import log, log_context_scope, now_ms, duration_ms


VECTOR_DIM = 64
DEFAULT_VECTOR_VERSION = "v1"
DECISION_ANALYSIS_VECTOR_SOURCE = "decision_analysis"
BEHAVIOR_QUERY_VECTOR_SOURCE = "behavior_query"
FIXTURE_VECTOR_SOURCE = "fixture"


class SegmentVectorDataUnavailableError(Exception):
    pass


class SegmentVectorConflictError(Exception):
    def __init__(self, reason: str, *, segment_id: str) -> None:
        super().__init__(reason)
        self.segment_id = segment_id
        self.reason = reason

    def to_detail(self) -> dict[str, str]:
        return {
            "code": "segment_vector_semantic_conflict",
            "segment_id": self.segment_id,
            "reason": self.reason,
        }


class SegmentVectorStore(Protocol):
    def get_by_segment_snapshot(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_id: str,
        vector_version: str,
    ) -> SegmentVectorRecord | None:
        ...

    def get_latest_by_segment(
        self,
        *,
        project_id: str,
        promotion_id: str,
        segment_id: str,
        vector_version: str,
    ) -> SegmentVectorRecord | None:
        ...

    def save(self, vector: SegmentVectorRecord) -> None:
        ...


class UserBehaviorVectorReader(Protocol):
    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: Sequence[str],
        vector_version: str = DEFAULT_VECTOR_VERSION,
    ) -> list[UserBehaviorVectorRecord]:
        ...


@dataclass(frozen=True)
class SegmentVectorBuildRequest:
    project_id: str
    promotion_id: str
    analysis_id: str
    segment_id: str
    candidate_user_ids: Sequence[str] = ()
    vector_version: str = DEFAULT_VECTOR_VERSION
    query_vector: Sequence[float] | None = None


@dataclass(frozen=True)
class SegmentVectorBuildResult:
    segment_id: str
    segment_vector_id: str
    vector_values: list[float]
    source: str


class SegmentVectorService:
    def __init__(
        self,
        *,
        segment_vector_repository: SegmentVectorStore,
        user_behavior_vector_repository: UserBehaviorVectorReader,
    ) -> None:
        self._segment_vector_repository = segment_vector_repository
        self._user_behavior_vector_repository = user_behavior_vector_repository

    @log_context_scope
    def prepare_segment_vector(
        self,
        request: SegmentVectorBuildRequest,
    ) -> SegmentVectorBuildResult:
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": request.project_id,
                "promotionId": request.promotion_id,
                "analysisId": request.analysis_id,
                "segmentId": request.segment_id,
            }
        )
        log.info(
            "started",
            {
                "candidateUserCount": len(request.candidate_user_ids),
                "vectorVersion": request.vector_version,
            },
        )
        existing = self._segment_vector_repository.get_by_segment_snapshot(
            project_id=request.project_id,
            promotion_id=request.promotion_id,
            analysis_id=request.analysis_id,
            segment_id=request.segment_id,
            vector_version=request.vector_version,
        )
        if existing is not None:
            if existing.source == FIXTURE_VECTOR_SOURCE:
                raise SegmentVectorDataUnavailableError(
                    "segment vector data unavailable for existing snapshot"
                )
            _validate_vector(existing.vector_values, existing.vector_dim)
            if request.query_vector is not None:
                expected = _l2_normalize(request.query_vector)
                if (
                    existing.source != BEHAVIOR_QUERY_VECTOR_SOURCE
                    or not _vectors_identical(existing.vector_values, expected)
                ):
                    raise SegmentVectorConflictError(
                        "behavior query vector conflicts with the stored analysis snapshot",
                        segment_id=request.segment_id,
                    )
            response = SegmentVectorBuildResult(
                segment_id=existing.segment_id,
                segment_vector_id=existing.segment_vector_id,
                vector_values=[float(value) for value in existing.vector_values],
                source=existing.source,
            )
            log.assign_context({"segmentVectorId": response.segment_vector_id})
            log.info("segment_vector_reused", {"source": response.source})
            log.info(
                "completed",
                {
                    "source": response.source,
                    "durationMs": duration_ms(started_at),
                },
            )
            return response

        reusable = (
            None
            if request.query_vector is not None
            else self._segment_vector_repository.get_latest_by_segment(
                project_id=request.project_id,
                promotion_id=request.promotion_id,
                segment_id=request.segment_id,
                vector_version=request.vector_version,
            )
        )
        if reusable is not None and reusable.source != FIXTURE_VECTOR_SOURCE:
            _validate_vector(reusable.vector_values, reusable.vector_dim)
            source = reusable.source
            normalized_values = [float(value) for value in reusable.vector_values]
            log.info("segment_vector_source_reused", {"segmentVectorId": reusable.segment_vector_id, "source": source})
        else:
            if reusable is not None:
                log.warn("segment_vector_fixture_reuse_rejected", {"segmentVectorId": reusable.segment_vector_id})
            if request.query_vector is not None:
                source = BEHAVIOR_QUERY_VECTOR_SOURCE
                normalized_values = _l2_normalize(request.query_vector)
            else:
                candidate_user_ids = _dedupe(request.candidate_user_ids)
                if not candidate_user_ids:
                    raise SegmentVectorDataUnavailableError(
                        "segment vector data unavailable: candidate user ids are required"
                    )
                user_vectors = self._user_behavior_vector_repository.list_by_user_ids(
                    project_id=request.project_id,
                    user_ids=candidate_user_ids,
                    vector_version=request.vector_version,
                )

                if not user_vectors:
                    raise SegmentVectorDataUnavailableError(
                        "segment vector data unavailable: user behavior vectors are required"
                    )

                source = DECISION_ANALYSIS_VECTOR_SOURCE
                vector_values = _mean_user_vectors(user_vectors)
                log.info("user_vectors_loaded", {"userVectorCount": len(user_vectors)})
                normalized_values = _l2_normalize(vector_values)

        segment_vector_id = _segment_vector_id(
            analysis_id=request.analysis_id,
            segment_id=request.segment_id,
            vector_version=request.vector_version,
        )
        record = SegmentVectorRecord(
            segment_vector_id=segment_vector_id,
            project_id=request.project_id,
            promotion_id=request.promotion_id,
            promotion_run_id=None,
            analysis_id=request.analysis_id,
            segment_id=request.segment_id,
            vector_dim=VECTOR_DIM,
            vector_values=normalized_values,
            vector_version=request.vector_version,
            source=source,
        )
        self._segment_vector_repository.save(record)
        response = SegmentVectorBuildResult(
            segment_id=request.segment_id,
            segment_vector_id=segment_vector_id,
            vector_values=normalized_values,
            source=source,
        )
        log.assign_context({"segmentVectorId": response.segment_vector_id})
        log.info("segment_vector_created", {"source": source})
        log.info(
            "completed",
            {
                "source": response.source,
                "durationMs": duration_ms(started_at),
            },
        )
        return response


def _mean_user_vectors(
    user_vectors: Sequence[UserBehaviorVectorRecord],
) -> list[float]:
    totals = [0.0] * VECTOR_DIM
    for user_vector in user_vectors:
        _validate_vector(user_vector.vector_values, user_vector.vector_dim)
        for index, value in enumerate(user_vector.vector_values):
            totals[index] += float(value)
    return [value / len(user_vectors) for value in totals]


def _validate_vector(vector_values: Sequence[float], vector_dim: int) -> None:
    if vector_dim != VECTOR_DIM:
        raise ValueError("segment vector_dim must be 64")
    if len(vector_values) != VECTOR_DIM:
        raise ValueError("segment vector_values must contain 64 values")


def _l2_normalize(vector_values: Sequence[float]) -> list[float]:
    _validate_vector(vector_values, len(vector_values))
    norm = math.sqrt(sum(float(value) * float(value) for value in vector_values))
    if norm == 0:
        raise ValueError("segment vector must not be a zero vector")
    return [float(value) / norm for value in vector_values]


def _vectors_identical(
    stored: Sequence[float],
    expected: Sequence[float],
) -> bool:
    if len(stored) != len(expected):
        return False
    # PostgreSQL arrays may round-trip through Float32 storage. This tolerance
    # accepts representation noise only, not a semantic query change.
    return all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-7)
        for left, right in zip(stored, expected)
    )


def _segment_vector_id(
    *,
    analysis_id: str,
    segment_id: str,
    vector_version: str,
) -> str:
    digest = hashlib.sha1(  # noqa: S324 - stable non-security identifier.
        f"{analysis_id}:{segment_id}:{vector_version}".encode("utf-8")
    ).hexdigest()[:10]
    prefix = "segvec_"
    suffix = f"_{vector_version}_{digest}"
    segment_part = _safe_identifier_part(segment_id)
    segment_part = segment_part[: 100 - len(prefix) - len(suffix)]
    return f"{prefix}{segment_part}{suffix}"


def _safe_identifier_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in value
    )


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
