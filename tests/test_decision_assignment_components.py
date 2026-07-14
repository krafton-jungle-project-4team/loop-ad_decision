from __future__ import annotations

import pytest

from app.decision.assignment_service import (
    AssignmentPageMatcher,
    ExactAssignmentPageMatcher,
)
from app.decision.matcher import (
    FALLBACK_REASON_BELOW_THRESHOLD,
    FALLBACK_REASON_INVALID_USER_VECTOR,
    FALLBACK_SEGMENT_ID,
    SegmentCandidateReranker,
    SegmentVector,
    UserVector,
)
from app.decision.repositories import SegmentVectorRecord


def test_exact_matcher_breaks_equal_score_ties_by_lexical_segment_id() -> None:
    matcher = ExactAssignmentPageMatcher(reranker=SegmentCandidateReranker())
    segments = (
        segment_vector("seg_zulu", vector(0)),
        segment_vector("seg_alpha", vector(0)),
    )

    result = match_page(
        matcher,
        users=(user_vector("user_tie", vector(0)),),
        segment_vectors=segments,
    )

    assert result.matches["user_tie"].segment_id == "seg_alpha"
    assert result.matches["user_tie"].fallback is False
    assert result.exact_reranked_pair_count == 2
    assert result.ann_candidate_count == 0
    assert result.ann_query_user_count == 0
    assert result.ann_underfilled_user_count == 0
    assert result.exact_rescue_user_count == 0


def test_exact_matcher_falls_back_below_similarity_threshold() -> None:
    matcher = ExactAssignmentPageMatcher(reranker=SegmentCandidateReranker())

    result = match_page(
        matcher,
        users=(user_vector("user_below_threshold", vector(0)),),
        segment_vectors=(segment_vector("seg_orthogonal", vector(1)),),
    )

    match = result.matches["user_below_threshold"]
    assert match.segment_id == FALLBACK_SEGMENT_ID
    assert match.similarity_score == pytest.approx(0.0)
    assert match.fallback is True
    assert match.fallback_reason == FALLBACK_REASON_BELOW_THRESHOLD
    assert result.fallback_count == 1
    assert result.below_threshold_fallback_count == 1
    assert result.exact_reranked_pair_count == 1


def test_exact_matcher_marks_invalid_user_without_counting_exact_pairs() -> None:
    matcher = ExactAssignmentPageMatcher(reranker=SegmentCandidateReranker())

    result = match_page(
        matcher,
        users=(user_vector("user_invalid", [0.0] * 64),),
        segment_vectors=(segment_vector("seg_valid", vector(0)),),
    )

    match = result.matches["user_invalid"]
    assert match.segment_id == FALLBACK_SEGMENT_ID
    assert match.similarity_score is None
    assert match.fallback is True
    assert match.fallback_reason == FALLBACK_REASON_INVALID_USER_VECTOR
    assert result.invalid_user_vector_fallback_count == 1
    assert result.exact_reranked_pair_count == 0
    assert result.ann_query_user_count == 0


def test_ann_duplicate_candidate_triggers_full_exact_rescue() -> None:
    corpus = (
        segment_vector("seg_alpha", vector(0)),
        segment_vector("seg_best", vector(1)),
    )
    duplicate = segment_vector_record("seg_alpha", vector(0))
    repository = StubSegmentVectorRepository([duplicate, duplicate])
    matcher = AssignmentPageMatcher(
        segment_vector_repository=repository,
        reranker=SegmentCandidateReranker(),
    )

    result = match_page(
        matcher,
        users=(user_vector("user_target", vector(1)),),
        segment_vectors=corpus,
    )

    assert result.matches["user_target"].segment_id == "seg_best"
    assert result.matches["user_target"].fallback is False
    assert result.ann_candidate_count == 2
    assert result.ann_query_user_count == 1
    assert result.ann_underfilled_user_count == 1
    assert result.exact_rescue_user_count == 1
    assert result.exact_reranked_pair_count == len(corpus)
    assert repository.configure_ann_search_count == 1


@pytest.mark.parametrize("invalid_kind", ["foreign_vector_id", "mismatched_segment_id"])
def test_ann_foreign_or_mismatched_candidate_triggers_full_exact_rescue(
    invalid_kind: str,
) -> None:
    corpus = (
        segment_vector("seg_alpha", vector(0)),
        segment_vector("seg_best", vector(1)),
    )
    valid_candidate = segment_vector_record("seg_alpha", vector(0))
    if invalid_kind == "foreign_vector_id":
        invalid_candidate = segment_vector_record(
            "seg_foreign",
            vector(1),
            segment_vector_id="segvec_foreign_v1",
        )
    else:
        invalid_candidate = segment_vector_record(
            "seg_wrong",
            vector(1),
            segment_vector_id="segvec_seg_best_v1",
        )
    repository = StubSegmentVectorRepository(
        [valid_candidate, invalid_candidate]
    )
    matcher = AssignmentPageMatcher(
        segment_vector_repository=repository,
        reranker=SegmentCandidateReranker(),
    )

    result = match_page(
        matcher,
        users=(user_vector("user_target", vector(1)),),
        segment_vectors=corpus,
    )

    assert result.matches["user_target"].segment_id == "seg_best"
    assert result.matches["user_target"].fallback is False
    assert result.ann_candidate_count == 2
    assert result.ann_query_user_count == 1
    assert result.ann_underfilled_user_count == 1
    assert result.exact_rescue_user_count == 1
    assert result.exact_reranked_pair_count == len(corpus)


@pytest.mark.parametrize(
    "malformed_values",
    (
        [0.0] * 64,
        [1.0] * 63,
        [float("nan")] + [0.0] * 63,
    ),
)
def test_ann_malformed_candidate_triggers_full_exact_rescue(
    malformed_values: list[float],
) -> None:
    corpus = (
        segment_vector("seg_alpha", vector(0)),
        segment_vector("seg_best", vector(1)),
    )
    repository = StubSegmentVectorRepository(
        [
            segment_vector_record("seg_alpha", malformed_values),
            segment_vector_record("seg_best", vector(1)),
        ]
    )
    matcher = AssignmentPageMatcher(
        segment_vector_repository=repository,
        reranker=SegmentCandidateReranker(),
    )

    result = match_page(
        matcher,
        users=(user_vector("user_target", vector(1)),),
        segment_vectors=corpus,
    )

    assert result.matches["user_target"].segment_id == "seg_best"
    assert result.ann_candidate_count == 2
    assert result.ann_underfilled_user_count == 1
    assert result.exact_rescue_user_count == 1
    assert result.exact_reranked_pair_count == len(corpus)


class StubSegmentVectorRepository:
    def __init__(self, candidates: list[SegmentVectorRecord]) -> None:
        self._candidates = candidates
        self.configure_ann_search_count = 0

    def configure_ann_search(self) -> None:
        self.configure_ann_search_count += 1

    def list_ann_candidates_for_users(
        self,
        **kwargs: object,
    ) -> dict[str, list[SegmentVectorRecord]]:
        return {
            str(user_id): list(self._candidates)
            for user_id in kwargs["user_ids"]
        }


def match_page(
    matcher: AssignmentPageMatcher | ExactAssignmentPageMatcher,
    *,
    users: tuple[UserVector, ...],
    segment_vectors: tuple[SegmentVector, ...],
):
    return matcher.match_page(
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        vector_version="v1",
        users=users,
        segment_vectors=segment_vectors,
    )


def vector(index: int, value: float = 1.0) -> list[float]:
    values = [0.0] * 64
    values[index] = value
    return values


def user_vector(user_id: str, values: list[float]) -> UserVector:
    return UserVector(
        user_id=user_id,
        vector_dim=64,
        vector_values=values,
    )


def segment_vector(segment_id: str, values: list[float]) -> SegmentVector:
    return SegmentVector(
        segment_vector_id=f"segvec_{segment_id}_v1",
        segment_id=segment_id,
        vector_dim=64,
        embedding_values=values,
    )


def segment_vector_record(
    segment_id: str,
    values: list[float],
    *,
    segment_vector_id: str | None = None,
) -> SegmentVectorRecord:
    return SegmentVectorRecord(
        segment_vector_id=segment_vector_id or f"segvec_{segment_id}_v1",
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        promotion_run_id=None,
        analysis_id="analysis_banner_001",
        segment_id=segment_id,
        vector_dim=64,
        vector_values=values,
        vector_version="v1",
        source="decision_analysis",
        embedding=values,
    )
