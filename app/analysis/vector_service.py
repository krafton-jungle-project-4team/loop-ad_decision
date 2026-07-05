from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.analysis.repositories import SegmentVectorRecord, UserBehaviorVectorRecord


VECTOR_DIM = 64
DEFAULT_VECTOR_VERSION = "v1"


class SegmentVectorStore(Protocol):
    def get_by_segment(
        self,
        *,
        project_id: str,
        promotion_id: str,
        segment_id: str,
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

    def prepare_segment_vector(
        self,
        request: SegmentVectorBuildRequest,
    ) -> SegmentVectorBuildResult:
        existing = self._segment_vector_repository.get_by_segment(
            project_id=request.project_id,
            promotion_id=request.promotion_id,
            segment_id=request.segment_id,
        )
        if existing is not None:
            _validate_existing_segment_vector(existing)
            return SegmentVectorBuildResult(
                segment_id=existing.segment_id,
                segment_vector_id=existing.segment_vector_id,
                vector_values=[float(value) for value in existing.vector_values],
                source=existing.source,
            )

        candidate_user_ids = _dedupe(request.candidate_user_ids)
        user_vectors = (
            self._user_behavior_vector_repository.list_by_user_ids(
                project_id=request.project_id,
                user_ids=candidate_user_ids,
                vector_version=request.vector_version,
            )
            if candidate_user_ids
            else []
        )

        source = "decision_analysis"
        if user_vectors:
            vector_values = _mean_user_vectors(user_vectors)
        else:
            source = "fixture"
            vector_values = _fixture_vector(request.segment_id)

        try:
            normalized_values = _l2_normalize(vector_values)
        except ValueError:
            source = "fixture"
            normalized_values = _l2_normalize(_fixture_vector(request.segment_id))

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
        return SegmentVectorBuildResult(
            segment_id=request.segment_id,
            segment_vector_id=segment_vector_id,
            vector_values=normalized_values,
            source=source,
        )


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
    numeric_values = [float(value) for value in vector_values]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("segment vector_values must be finite")
    norm = math.sqrt(sum(value * value for value in numeric_values))
    if norm == 0:
        raise ValueError("segment vector must not be a zero vector")
    return [value / norm for value in numeric_values]


def _validate_existing_segment_vector(vector: SegmentVectorRecord) -> None:
    _validate_reusable_vector_values(
        vector.vector_values,
        vector.vector_dim,
        field_name="segment vector_values",
    )
    if vector.embedding is None:
        raise ValueError("segment embedding is required")
    _validate_reusable_vector_values(
        _parse_vector_values(vector.embedding, field_name="segment embedding"),
        vector.vector_dim,
        field_name="segment embedding",
    )


def _validate_reusable_vector_values(
    vector_values: Sequence[float],
    vector_dim: int,
    *,
    field_name: str,
) -> None:
    if vector_dim != VECTOR_DIM:
        raise ValueError(f"{field_name} vector_dim must be 64")
    if len(vector_values) != VECTOR_DIM:
        raise ValueError(f"{field_name} must contain 64 values")
    numeric_values = [float(value) for value in vector_values]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(f"{field_name} must be finite")
    if math.sqrt(sum(value * value for value in numeric_values)) == 0:
        raise ValueError(f"{field_name} must not be a zero vector")


def _parse_vector_values(value: object, *, field_name: str) -> list[float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, SequenceABC) or isinstance(parsed, (bytes, str)):
        raise ValueError(f"{field_name} must be an array")
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numbers") from exc


def _fixture_vector(segment_id: str) -> list[float]:
    values: list[float] = []
    for index in range(VECTOR_DIM):
        digest = hashlib.sha256(f"{segment_id}:{index}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 2001
        values.append((bucket - 1000) / 1000.0)
    return values


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
