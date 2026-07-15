from __future__ import annotations

import pytest

from app.decision.matcher import (
    FALLBACK_REASON_BELOW_THRESHOLD,
    FALLBACK_REASON_INVALID_USER_VECTOR,
    FALLBACK_REASON_NO_CANDIDATE,
    FALLBACK_SEGMENT_ID,
    SegmentCandidateReranker,
    SegmentMatchValidationError,
    SegmentVector,
    UserVector,
    invalid_user_vector_result,
)


def unit_vector(index: int, value: float = 1.0) -> list[float]:
    vector = [0.0] * 64
    vector[index] = value
    return vector


def segment_vector(segment_id: str, values: list[float]) -> SegmentVector:
    return SegmentVector(
        segment_vector_id=f"segvec_{segment_id}_v1",
        segment_id=segment_id,
        vector_dim=64,
        embedding_values=values,
    )


def test_segment_candidate_reranker_selects_best_exact_score() -> None:
    reranker = SegmentCandidateReranker()
    user_vector = reranker.normalize_user_vector(
        UserVector("user_001", 64, unit_vector(0))
    )
    assert user_vector is not None

    result = reranker.rerank(
        normalized_user_vector=user_vector,
        candidates=[
            segment_vector("seg_mobile_user", unit_vector(1)),
            segment_vector("seg_family_trip", unit_vector(0)),
        ],
    )

    assert result.segment_id == "seg_family_trip"
    assert result.fallback is False
    assert result.fallback_reason is None
    assert result.similarity_score == pytest.approx(1.0)


def test_segment_candidate_reranker_falls_back_below_threshold() -> None:
    reranker = SegmentCandidateReranker(threshold=0.65)
    user_vector = reranker.normalize_user_vector(
        UserVector("user_001", 64, unit_vector(0))
    )
    assert user_vector is not None

    result = reranker.rerank(
        normalized_user_vector=user_vector,
        candidates=[segment_vector("seg_mobile_user", unit_vector(1))],
    )

    assert result.segment_id == FALLBACK_SEGMENT_ID
    assert result.fallback is True
    assert result.fallback_reason == FALLBACK_REASON_BELOW_THRESHOLD
    assert result.similarity_score == pytest.approx(0.0)


def test_segment_candidate_reranker_assigns_zero_score_with_default_threshold() -> None:
    reranker = SegmentCandidateReranker()
    user_vector = reranker.normalize_user_vector(
        UserVector("user_001", 64, unit_vector(0))
    )
    assert user_vector is not None

    result = reranker.rerank(
        normalized_user_vector=user_vector,
        candidates=[segment_vector("seg_mobile_user", unit_vector(1))],
    )

    assert result.segment_id == "seg_mobile_user"
    assert result.fallback is False
    assert result.fallback_reason is None
    assert result.similarity_score == pytest.approx(0.0)


def test_segment_candidate_reranker_falls_back_for_negative_score() -> None:
    reranker = SegmentCandidateReranker()
    user_vector = reranker.normalize_user_vector(
        UserVector("user_001", 64, unit_vector(0))
    )
    assert user_vector is not None

    result = reranker.rerank(
        normalized_user_vector=user_vector,
        candidates=[segment_vector("seg_mobile_user", unit_vector(0, -1.0))],
    )

    assert result.segment_id == FALLBACK_SEGMENT_ID
    assert result.fallback is True
    assert result.fallback_reason == FALLBACK_REASON_BELOW_THRESHOLD
    assert result.similarity_score == pytest.approx(-1.0)


def test_segment_candidate_reranker_falls_back_for_no_candidates() -> None:
    reranker = SegmentCandidateReranker()
    user_vector = reranker.normalize_user_vector(
        UserVector("user_001", 64, unit_vector(0))
    )
    assert user_vector is not None

    result = reranker.rerank(
        normalized_user_vector=user_vector,
        candidates=[],
    )

    assert result.segment_id == FALLBACK_SEGMENT_ID
    assert result.fallback is True
    assert result.fallback_reason == FALLBACK_REASON_NO_CANDIDATE
    assert result.similarity_score is None


def test_segment_candidate_reranker_detects_invalid_user_vector() -> None:
    reranker = SegmentCandidateReranker()

    normalized = reranker.normalize_user_vector(
        UserVector("user_001", 64, [0.0] * 64)
    )
    result = invalid_user_vector_result()

    assert normalized is None
    assert result.segment_id == FALLBACK_SEGMENT_ID
    assert result.fallback is True
    assert result.fallback_reason == FALLBACK_REASON_INVALID_USER_VECTOR


def test_segment_candidate_reranker_rejects_invalid_segment_embedding() -> None:
    reranker = SegmentCandidateReranker()
    user_vector = reranker.normalize_user_vector(
        UserVector("user_001", 64, unit_vector(0))
    )
    assert user_vector is not None

    with pytest.raises(SegmentMatchValidationError, match="seg_family_trip"):
        reranker.rerank(
            normalized_user_vector=user_vector,
            candidates=[segment_vector("seg_family_trip", [0.0] * 64)],
        )


def test_segment_candidate_reranker_uses_exact_score_tie_break() -> None:
    reranker = SegmentCandidateReranker()
    user_vector = reranker.normalize_user_vector(
        UserVector("user_001", 64, unit_vector(0))
    )
    assert user_vector is not None

    result = reranker.rerank(
        normalized_user_vector=user_vector,
        candidates=[
            segment_vector("seg_zeta", unit_vector(0)),
            segment_vector("seg_alpha", unit_vector(0)),
        ],
    )

    assert result.segment_id == "seg_alpha"
