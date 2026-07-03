from __future__ import annotations

import pytest

from app.decision.matcher import (
    FALLBACK_SEGMENT_ID,
    ExactCosineMatcher,
    SegmentMatchValidationError,
    SegmentVector,
    UserVector,
)


def unit_vector(index: int, value: float = 1.0) -> list[float]:
    vector = [0.0] * 64
    vector[index] = value
    return vector


def test_exact_cosine_matcher_selects_best_segment() -> None:
    matcher = ExactCosineMatcher()

    results = matcher.match(
        eligible_users=[
            UserVector(
                user_id="user_001",
                vector_dim=64,
                vector_values=unit_vector(0),
            )
        ],
        segment_vectors=[
            SegmentVector(
                segment_id="seg_mobile_user",
                vector_dim=64,
                vector_values=unit_vector(1),
            ),
            SegmentVector(
                segment_id="seg_family_trip",
                vector_dim=64,
                vector_values=unit_vector(0),
            ),
        ],
    )

    assert results["user_001"].segment_id == "seg_family_trip"
    assert results["user_001"].fallback is False
    assert results["user_001"].similarity_score == pytest.approx(1.0)


def test_exact_cosine_matcher_falls_back_below_threshold() -> None:
    matcher = ExactCosineMatcher()

    results = matcher.match(
        eligible_users=[
            UserVector(
                user_id="user_001",
                vector_dim=64,
                vector_values=unit_vector(0),
            )
        ],
        segment_vectors=[
            SegmentVector(
                segment_id="seg_mobile_user",
                vector_dim=64,
                vector_values=unit_vector(1),
            )
        ],
    )

    assert results["user_001"].segment_id == FALLBACK_SEGMENT_ID
    assert results["user_001"].fallback is True
    assert results["user_001"].similarity_score == pytest.approx(0.0)


def test_exact_cosine_matcher_falls_back_for_invalid_user_vector() -> None:
    matcher = ExactCosineMatcher()

    results = matcher.match(
        eligible_users=[
            UserVector(
                user_id="user_001",
                vector_dim=64,
                vector_values=[0.0] * 64,
            )
        ],
        segment_vectors=[
            SegmentVector(
                segment_id="seg_family_trip",
                vector_dim=64,
                vector_values=unit_vector(0),
            )
        ],
    )

    assert results["user_001"].segment_id == FALLBACK_SEGMENT_ID
    assert results["user_001"].fallback is True
    assert results["user_001"].similarity_score is None


def test_exact_cosine_matcher_rejects_invalid_segment_vector() -> None:
    matcher = ExactCosineMatcher()

    with pytest.raises(SegmentMatchValidationError, match="seg_family_trip"):
        matcher.match(
            eligible_users=[
                UserVector(
                    user_id="user_001",
                    vector_dim=64,
                    vector_values=unit_vector(0),
                )
            ],
            segment_vectors=[
                SegmentVector(
                    segment_id="seg_family_trip",
                    vector_dim=64,
                    vector_values=[0.0] * 64,
                )
            ],
        )


def test_exact_cosine_matcher_uses_segment_id_tie_break() -> None:
    matcher = ExactCosineMatcher()

    results = matcher.match(
        eligible_users=[
            UserVector(
                user_id="user_001",
                vector_dim=64,
                vector_values=unit_vector(0),
            )
        ],
        segment_vectors=[
            SegmentVector(
                segment_id="seg_zeta",
                vector_dim=64,
                vector_values=unit_vector(0),
            ),
            SegmentVector(
                segment_id="seg_alpha",
                vector_dim=64,
                vector_values=unit_vector(0),
            ),
        ],
    )

    assert results["user_001"].segment_id == "seg_alpha"
