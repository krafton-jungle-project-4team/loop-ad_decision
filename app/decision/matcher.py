from __future__ import annotations

import json
import math
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Sequence


VECTOR_DIM = 64
SIMILARITY_THRESHOLD = 0.65
FALLBACK_SEGMENT_ID = "seg_existing_all"
ANN_CANDIDATE_LIMIT = 50
ANN_QUERY_USER_BATCH_SIZE = 256
HNSW_EF_SEARCH = 100
HNSW_MAX_SCAN_TUPLES = 20000

FALLBACK_REASON_BELOW_THRESHOLD = "below_threshold"
FALLBACK_REASON_NO_CANDIDATE = "no_candidate"
FALLBACK_REASON_INVALID_USER_VECTOR = "invalid_user_vector"


class SegmentMatchValidationError(Exception):
    pass


@dataclass(frozen=True)
class UserVector:
    user_id: str
    vector_dim: int
    vector_values: Sequence[float]


@dataclass(frozen=True)
class SegmentVector:
    segment_vector_id: str
    segment_id: str
    vector_dim: int
    embedding_values: Sequence[float]


@dataclass(frozen=True)
class MatchResult:
    segment_id: str
    similarity_score: float | None
    fallback: bool
    fallback_reason: str | None = None


class SegmentCandidateReranker:
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

    def normalize_user_vector(self, user: UserVector) -> list[float] | None:
        try:
            return normalize_values(user.vector_values, user.vector_dim, self._vector_dim)
        except ValueError:
            return None

    def rerank(
        self,
        *,
        normalized_user_vector: Sequence[float],
        candidates: Sequence[SegmentVector],
    ) -> MatchResult:
        if not candidates:
            return MatchResult(
                segment_id=self._fallback_segment_id,
                similarity_score=None,
                fallback=True,
                fallback_reason=FALLBACK_REASON_NO_CANDIDATE,
            )

        scored_candidates: list[tuple[float, str]] = []
        for candidate in candidates:
            segment_vector = _normalize_segment_vector(candidate, self._vector_dim)
            scored_candidates.append(
                (_dot(normalized_user_vector, segment_vector), candidate.segment_id)
            )

        best_score, best_segment_id = sorted(
            scored_candidates,
            key=lambda item: (-item[0], item[1]),
        )[0]
        if best_score < self._threshold:
            return MatchResult(
                segment_id=self._fallback_segment_id,
                similarity_score=best_score,
                fallback=True,
                fallback_reason=FALLBACK_REASON_BELOW_THRESHOLD,
            )
        return MatchResult(
            segment_id=best_segment_id,
            similarity_score=best_score,
            fallback=False,
            fallback_reason=None,
        )


def invalid_user_vector_result() -> MatchResult:
    return MatchResult(
        segment_id=FALLBACK_SEGMENT_ID,
        similarity_score=None,
        fallback=True,
        fallback_reason=FALLBACK_REASON_INVALID_USER_VECTOR,
    )


def parse_vector_values(value: Any) -> list[float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, SequenceABC) or isinstance(parsed, (bytes, str)):
        raise ValueError("vector_values must be an array")
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError("vector_values must contain numbers") from exc


def normalize_values(
    values: Sequence[float],
    declared_dim: int,
    vector_dim: int = VECTOR_DIM,
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


def _normalize_segment_vector(
    segment: SegmentVector,
    vector_dim: int,
) -> list[float]:
    try:
        return normalize_values(segment.embedding_values, segment.vector_dim, vector_dim)
    except ValueError as exc:
        raise SegmentMatchValidationError(
            f"invalid segment embedding: {segment.segment_id}"
        ) from exc


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
