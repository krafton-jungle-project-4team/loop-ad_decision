from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence


VECTOR_DIM = 64
SIMILARITY_THRESHOLD = 0.65
FALLBACK_SEGMENT_ID = "seg_existing_all"


class SegmentMatchValidationError(Exception):
    pass


@dataclass(frozen=True)
class UserVector:
    user_id: str
    vector_dim: int
    vector_values: Sequence[float]


@dataclass(frozen=True)
class SegmentVector:
    segment_id: str
    vector_dim: int
    vector_values: Sequence[float]


@dataclass(frozen=True)
class MatchResult:
    segment_id: str
    similarity_score: float | None
    fallback: bool


class SegmentMatcher(Protocol):
    """Match eligible users to one segment.

    Future ANN implementations should hide behind this interface.
    Exact matching iterates users and scores all segments. ANN matching will
    likely search by segment, so a user may appear in multiple top-N result
    sets and must be deduplicated by best similarity before returning results.
    Candidate A is ClickHouse HNSW, which needs schema/team agreement and is
    currently a riskier beta dependency. Candidate B is in-memory faiss, which
    avoids schema changes but makes index build, memory, and freshness app
    responsibilities. Distance-to-similarity conversion and DB score clamping
    stay common outside the matcher.
    """

    def match(
        self,
        eligible_users: Sequence[UserVector],
        segment_vectors: Sequence[SegmentVector],
    ) -> dict[str, MatchResult]:
        ...


class ExactCosineMatcher:
    def __init__(
        self,
        *,
        threshold: float = SIMILARITY_THRESHOLD,
        fallback_segment_id: str = FALLBACK_SEGMENT_ID,
        vector_dim: int = VECTOR_DIM,
    ) -> None:
        self._threshold = threshold
        self._fallback_segment_id = fallback_segment_id
        self._vector_dim = vector_dim

    def match(
        self,
        eligible_users: Sequence[UserVector],
        segment_vectors: Sequence[SegmentVector],
    ) -> dict[str, MatchResult]:
        normalized_segments = [
            (
                segment.segment_id,
                _normalize_segment_vector(segment, self._vector_dim),
            )
            for segment in sorted(segment_vectors, key=lambda item: item.segment_id)
        ]
        if not normalized_segments:
            raise SegmentMatchValidationError(
                "at least one non-fallback segment vector is required"
            )

        results: dict[str, MatchResult] = {}
        for user in eligible_users:
            user_vector = _normalize_user_vector(user, self._vector_dim)
            if user_vector is None:
                results[user.user_id] = MatchResult(
                    segment_id=self._fallback_segment_id,
                    similarity_score=None,
                    fallback=True,
                )
                continue

            best_segment_id: str | None = None
            best_score: float | None = None
            for segment_id, segment_vector in normalized_segments:
                score = _dot(user_vector, segment_vector)
                if best_score is None or score > best_score:
                    best_segment_id = segment_id
                    best_score = score

            if best_segment_id is None or best_score is None:
                results[user.user_id] = MatchResult(
                    segment_id=self._fallback_segment_id,
                    similarity_score=None,
                    fallback=True,
                )
                continue

            fallback = best_score < self._threshold
            results[user.user_id] = MatchResult(
                segment_id=self._fallback_segment_id if fallback else best_segment_id,
                similarity_score=best_score,
                fallback=fallback,
            )
        return results


def _normalize_segment_vector(
    segment: SegmentVector,
    vector_dim: int,
) -> list[float]:
    try:
        return _normalize_values(segment.vector_values, segment.vector_dim, vector_dim)
    except ValueError as exc:
        raise SegmentMatchValidationError(
            f"invalid segment vector: {segment.segment_id}"
        ) from exc


def _normalize_user_vector(user: UserVector, vector_dim: int) -> list[float] | None:
    try:
        return _normalize_values(user.vector_values, user.vector_dim, vector_dim)
    except ValueError:
        return None


def _normalize_values(
    values: Sequence[float],
    declared_dim: int,
    vector_dim: int,
) -> list[float]:
    if declared_dim != vector_dim:
        raise ValueError("vector_dim must be 64")
    if len(values) != vector_dim:
        raise ValueError("vector_values must contain 64 values")

    numeric_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("vector_values must be finite")

    norm = math.sqrt(sum(value * value for value in numeric_values))
    if norm == 0:
        raise ValueError("vector must not be zero")
    return [value / norm for value in numeric_values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
