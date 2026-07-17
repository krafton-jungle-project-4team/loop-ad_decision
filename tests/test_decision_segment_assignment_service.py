from __future__ import annotations

from decimal import Decimal

import pytest

from app.decision.assignment_service import (
    ASSIGNMENT_PAGE_SIZE,
    SegmentAssignmentRunNotFoundError,
    SegmentAssignmentService,
    SegmentAssignmentValidationError,
)
from app.decision.audience_snapshots import TargetAudienceResolution
from app.decision.matcher import (
    FALLBACK_REASON_BELOW_THRESHOLD,
    FALLBACK_REASON_INVALID_USER_VECTOR,
    FALLBACK_REASON_NO_CANDIDATE,
    SegmentCandidateReranker,
)
from app.decision.repositories import (
    AdExperimentRecord,
    PromotionRunRecord,
    SegmentVectorRecord,
    UserBehaviorVectorRecord,
    UserSegmentAssignmentInsertRecord,
    UserSegmentAssignmentWrite,
)
from app.decision.schemas import (
    AssignmentSource,
    Channel,
    GoalBasis,
    GoalMetric,
    SegmentAssignmentBuildRequest,
)


DEFAULT_RUN = object()
AnnCandidates = list[SegmentVectorRecord] | dict[str, list[SegmentVectorRecord]] | None


def vector(index: int, value: float = 1.0) -> list[float]:
    values = [0.0] * 64
    values[index] = value
    return values


def user_vector_records(count: int) -> list[UserBehaviorVectorRecord]:
    return [
        user_vector_record(f"user_{index:06d}", vector(0))
        for index in range(count)
    ]


def test_assignment_service_preserves_fallback_for_existing_runs() -> None:
    service, repos = make_service(
        user_vectors=[
            user_vector_record("user_family", vector(0)),
            user_vector_record("user_fallback", vector(1)),
        ],
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family", "user_fallback"]),
    )

    assert response.assignment_count == 2
    assert response.page_count == 1
    assert response.processed_user_count == 2
    assert response.insert_conflict_count == 0
    assert response.segment_assignment_counts == {
        "seg_existing_all": 1,
        "seg_family_trip": 1,
    }
    assert response.matching_mode == "pgvector_hnsw_rerank"
    assert response.ann_candidate_limit == 50
    assert response.ann_candidate_count == 2
    assert response.exact_reranked_pair_count == 2
    assert response.batch_has_fallback is True
    assert response.fallback_count == 1
    assert response.fallback_rate == 0.5
    assert response.fallback_reason_counts == {
        "below_threshold": 1,
        "no_candidate": 0,
        "invalid_user_vector": 0,
    }
    assert response.below_threshold_fallback_count == 1
    assert response.no_candidate_fallback_count == 0
    assert response.invalid_user_vector_fallback_count == 0
    assert response.ann_underfilled_user_count == 0
    assert response.ann_applied is True
    assert response.ann_not_applied_reason is None
    assert response.similarity_score_buckets == {
        "not_available": 0,
        "lt_0_00": 0,
        "0_00_to_0_50": 1,
        "0_50_to_0_65": 0,
        "0_65_to_0_80": 0,
        "0_80_to_0_90": 0,
        "gte_0_90": 1,
    }
    assert response.model_dump()["insufficient_segment_count"] == 0
    assert response.completion_scope == "current_request"
    assert response.assignment_mode == "explicit_user_ids"
    assert response.input_stability == "not_snapshotted"
    assignments_by_user = {
        assignment.user_id: assignment for assignment in repos.assignments.inserted
    }
    regular = assignments_by_user["user_family"]
    fallback = assignments_by_user["user_fallback"]
    assert regular.segment_id == "seg_family_trip"
    assert regular.ad_experiment_id == "adexp_seg_family_trip"
    assert regular.content_id == "content_seg_family_trip"
    assert regular.content_option_id == "option_seg_family_trip"
    assert regular.assignment_source == AssignmentSource.DECISION_BATCH.value
    assert regular.fallback is False
    assert fallback.segment_id == "seg_existing_all"
    assert fallback.ad_experiment_id == "adexp_seg_existing_all"
    assert fallback.content_id == "content_seg_existing_all"
    assert fallback.content_option_id == "option_seg_existing_all"
    assert fallback.assignment_source == AssignmentSource.FALLBACK.value
    assert fallback.fallback is True
    assert fallback.fallback_reason == FALLBACK_REASON_BELOW_THRESHOLD
    assert repos.segment_vectors.configure_ann_search_count == 1
    assert len(repos.segment_vectors.ann_calls) == 1


def test_assignment_service_requires_snapshot_repository_at_construction() -> None:
    _, repos = make_service()

    with pytest.raises(TypeError, match="audience_snapshot_repository"):
        SegmentAssignmentService(
            promotion_run_repository=repos.runs,
            ad_experiment_repository=repos.ad_experiments,
            segment_vector_repository=repos.segment_vectors,
            user_behavior_vector_repository=repos.user_vectors,
            user_segment_assignment_repository=repos.assignments,
            reranker=SegmentCandidateReranker(),
        )


def test_assignment_service_leaves_below_threshold_user_unassigned_without_fallback() -> None:
    service, repos = make_service(
        experiments=[
            ad_experiment_record(segment_id="seg_family_trip"),
        ],
        user_vectors=[
            user_vector_record("user_fallback", vector(1)),
        ],
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_fallback"]),
    )

    assert response.assignment_count == 0
    assert response.fallback_count == 0
    assert response.batch_has_fallback is False
    assert response.unassigned_count == 1
    assert response.below_threshold_unassigned_count == 1
    assert response.no_candidate_unassigned_count == 0
    assert response.invalid_user_vector_unassigned_count == 0
    assert response.unassigned_reason_counts == {
        "below_threshold": 1,
        "no_candidate": 0,
        "invalid_user_vector": 0,
    }
    assert response.similarity_score_buckets["0_00_to_0_50"] == 1
    assert repos.assignments.inserted == []


def test_assignment_service_falls_back_for_no_ann_candidate() -> None:
    service, repos = make_service(ann_candidates=[])

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
    )

    assert response.fallback_count == 1
    assert response.no_candidate_fallback_count == 1
    assert response.fallback_reason_counts["no_candidate"] == 1
    assert response.similarity_score_buckets["not_available"] == 1
    assert response.ann_underfilled_user_count == 1
    assert repos.assignments.inserted[0].fallback_reason == FALLBACK_REASON_NO_CANDIDATE


def test_assignment_service_falls_back_for_invalid_user_vector() -> None:
    service, repos = make_service(
        user_vectors=[user_vector_record("user_bad", [0.0] * 64)]
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_bad"]),
    )

    assert response.fallback_count == 1
    assert response.invalid_user_vector_fallback_count == 1
    assert response.ann_candidate_count == 0
    assert response.ann_applied is False
    assert response.ann_not_applied_reason == "no_valid_user_vectors"
    assert response.similarity_score_buckets["not_available"] == 1
    assert repos.segment_vectors.configure_ann_search_count == 0
    assert repos.segment_vectors.ann_calls == []
    assert (
        repos.assignments.inserted[0].fallback_reason
        == FALLBACK_REASON_INVALID_USER_VECTOR
    )


def test_assignment_service_marks_ann_applied_when_any_page_runs_ann(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.decision.assignment_service.ASSIGNMENT_PAGE_SIZE",
        1,
    )
    service, _repos = make_service(
        user_vectors=[
            user_vector_record("user_000001", [0.0] * 64),
            user_vector_record("user_000002", vector(0)),
        ]
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(),
    )

    assert response.page_count == 2
    assert response.processed_user_count == 2
    assert response.ann_applied is True
    assert response.ann_not_applied_reason is None
    assert response.invalid_user_vector_fallback_count == 1


def test_assignment_service_skips_existing_assignments() -> None:
    service, repos = make_service(existing_user_ids={"user_family"})

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
    )

    assert response.assignment_count == 0
    assert response.page_count == 1
    assert response.processed_user_count == 1
    assert response.insert_conflict_count == 0
    assert response.skipped_existing_count == 1
    assert response.fallback_rate is None
    assert response.fallback_reason_counts == {
        "below_threshold": 0,
        "no_candidate": 0,
        "invalid_user_vector": 0,
    }
    assert all(count == 0 for count in response.similarity_score_buckets.values())
    assert response.ann_applied is False
    assert response.ann_not_applied_reason == "no_users_to_match"
    assert repos.segment_vectors.configure_ann_search_count == 0
    assert repos.segment_vectors.ann_calls == []
    assert repos.assignments.inserted == []


def test_assignment_service_counts_insert_conflicts_separately() -> None:
    service, _repos = make_service(
        user_vectors=[
            user_vector_record("user_family", vector(0)),
            user_vector_record("user_fallback", vector(1)),
        ],
        insert_conflict_user_ids={"user_fallback"},
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(
            user_ids=["user_family", "user_fallback"]
        ),
    )

    assert response.processed_user_count == 2
    assert response.skipped_existing_count == 0
    assert response.assignment_count == 1
    assert response.insert_conflict_count == 1
    assert response.processed_user_count == (
        response.skipped_existing_count
        + response.assignment_count
        + response.insert_conflict_count
    )
    assert response.ann_candidate_count == 2
    assert response.segment_assignment_counts == {"seg_family_trip": 1}
    assert response.fallback_count == 0
    assert response.batch_has_fallback is False
    assert response.fallback_rate == 0.0
    assert response.similarity_score_buckets["gte_0_90"] == 1


def test_assignment_service_splits_valid_users_into_batch_chunks() -> None:
    user_vectors = [
        user_vector_record(f"user_{index:03d}", vector(0))
        for index in range(257)
    ]
    service, repos = make_service(user_vectors=user_vectors)

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(
            user_ids=[user.user_id for user in user_vectors]
        ),
    )

    assert response.assignment_count == 257
    assert repos.segment_vectors.configure_ann_search_count == 1
    assert len(repos.segment_vectors.ann_calls) == 2
    assert len(repos.segment_vectors.ann_calls[0]["user_ids"]) == 256
    assert len(repos.segment_vectors.ann_calls[1]["user_ids"]) == 1


def test_assignment_service_falls_back_for_valid_users_without_candidates() -> None:
    service, repos = make_service(
        user_vectors=[
            user_vector_record("user_family", vector(0)),
            user_vector_record("user_missing_candidate", vector(0)),
        ],
        ann_candidates={
            "user_family": [segment_vector_record("seg_family_trip", vector(0))],
            "user_missing_candidate": [],
        },
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(
            user_ids=["user_family", "user_missing_candidate"]
        ),
    )

    assert response.assignment_count == 2
    assert response.fallback_count == 1
    assert response.no_candidate_fallback_count == 1
    assert response.ann_underfilled_user_count == 1
    assert repos.assignments.inserted[1].fallback_reason == FALLBACK_REASON_NO_CANDIDATE


def test_assignment_service_deduplicates_users_before_batch_ann() -> None:
    service, repos = make_service(
        user_vectors=[user_vector_record("user_family", vector(0))],
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family", "user_family"]),
    )

    assert response.assignment_count == 1
    assert [assignment.user_id for assignment in repos.assignments.inserted] == [
        "user_family"
    ]
    assert repos.segment_vectors.ann_calls[0]["user_ids"] == ["user_family"]


def test_assignment_service_converts_batch_ann_contract_errors() -> None:
    service, repos = make_service(ann_error=ValueError("bad batch contract"))

    with pytest.raises(SegmentAssignmentValidationError, match="bad batch contract"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_counts_underfilled_only_against_corpus_size() -> None:
    service, _repos = make_service(
        experiments=[
            ad_experiment_record(segment_id="seg_family_trip"),
            ad_experiment_record(segment_id="seg_mobile_user"),
            ad_experiment_record(segment_id="seg_existing_all"),
        ],
        segment_vectors=[
            segment_vector_record("seg_family_trip", vector(0)),
            segment_vector_record("seg_mobile_user", vector(1)),
        ],
        ann_candidates=[
            segment_vector_record("seg_family_trip", vector(0)),
        ],
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
    )

    assert response.ann_candidate_count == 1
    assert response.ann_underfilled_user_count == 1


def test_assignment_service_rejects_missing_segment_vector_without_writes() -> None:
    service, repos = make_service(segment_vectors=[])

    with pytest.raises(SegmentAssignmentValidationError, match="segment vector"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_rejects_duplicate_segment_vector_without_writes() -> None:
    service, repos = make_service(
        segment_vectors=[
            segment_vector_record("seg_family_trip", vector(0)),
            segment_vector_record("seg_family_trip", vector(0)),
        ]
    )

    with pytest.raises(SegmentAssignmentValidationError, match="segment vector"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_rejects_invalid_segment_embedding_without_writes() -> None:
    service, repos = make_service(
        segment_vectors=[
            segment_vector_record("seg_family_trip", [0.0] * 64),
        ],
    )

    with pytest.raises(SegmentAssignmentValidationError, match="invalid segment embedding"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_rejects_fixture_segment_vector_without_writes() -> None:
    service, repos = make_service(
        segment_vectors=[
            segment_vector_record("seg_family_trip", vector(0), source="fixture"),
        ],
    )

    with pytest.raises(SegmentAssignmentValidationError, match="fixture segment vector"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_does_not_mark_status_from_assignment_volume() -> None:
    service, repos = make_service(
        run=promotion_run_record(min_sample_size=1_000),
        user_vectors=[user_vector_record("user_family", vector(0))],
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(),
    )

    assert response.assignment_count == 1
    assert response.model_dump()["insufficient_segment_count"] == 0
    assert [experiment.status for experiment in repos.ad_experiments.experiments] == [
        "planned",
        "planned",
    ]


def test_assignment_service_uses_project_limit_when_user_ids_omitted() -> None:
    service, repos = make_service(user_vectors=[])

    service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(eligible_user_limit=50),
    )

    assert repos.user_vectors.project_calls == [
        ("hotel-client-a", "v1", 50, None, None)
    ]
    assert repos.user_vectors.user_id_calls == []


def test_assignment_service_uses_page_size_for_unlimited_project_scope() -> None:
    service, repos = make_service(user_vectors=[])

    service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(),
    )

    assert repos.user_vectors.project_calls == [
        (
            "hotel-client-a",
            "v1",
            ASSIGNMENT_PAGE_SIZE,
            None,
            None,
        )
    ]


@pytest.mark.parametrize(
    ("user_count", "expected_page_sizes", "expected_after_user_ids"),
    [
        (0, [10_000], [None]),
        (1, [10_000], [None]),
        (10_000, [10_000, 10_000], [None, "user_009999"]),
        (10_001, [10_000, 10_000], [None, "user_009999"]),
        (
            25_001,
            [10_000, 10_000, 10_000],
            [None, "user_009999", "user_019999"],
        ),
    ],
)
def test_assignment_service_scans_all_project_vector_pages(
    user_count: int,
    expected_page_sizes: list[int],
    expected_after_user_ids: list[str | None],
) -> None:
    service, repos = make_service(user_vectors=user_vector_records(user_count))

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(),
    )

    assert response.assignment_count == user_count
    assert response.page_count == (user_count + ASSIGNMENT_PAGE_SIZE - 1) // (
        ASSIGNMENT_PAGE_SIZE
    )
    assert response.processed_user_count == user_count
    assert response.skipped_existing_count == 0
    assert response.insert_conflict_count == 0
    assert response.assignment_mode == "live_keyset"
    assert [call[2] for call in repos.user_vectors.project_calls] == (
        expected_page_sizes
    )
    assert [call[4] for call in repos.user_vectors.project_calls] == (
        expected_after_user_ids
    )
    assigned_at_values = {
        assignment.assigned_at for assignment in repos.assignments.inserted
    }
    assert len(assigned_at_values) <= 1

    if user_count == 25_001:
        first_assignments = list(repos.assignments.inserted)
        second = service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(),
        )

        assert second.assignment_count == 0
        assert second.page_count == 3
        assert second.processed_user_count == 25_001
        assert second.skipped_existing_count == 25_001
        assert second.insert_conflict_count == 0
        assert second.ann_candidate_count == 0
        assert second.exact_reranked_pair_count == 0
        assert second.fallback_count == 0
        assert second.fallback_rate is None
        assert second.ann_applied is False
        assert second.ann_not_applied_reason == "no_users_to_match"
        assert second.fallback_reason_counts == {
            "below_threshold": 0,
            "no_candidate": 0,
            "invalid_user_vector": 0,
        }
        assert all(
            count == 0 for count in second.similarity_score_buckets.values()
        )
        assert second.batch_has_fallback is False
        assert repos.assignments.inserted == first_assignments


@pytest.mark.parametrize(
    ("total_limit", "expected_page_sizes", "expected_after_user_ids"),
    [
        (5_000, [5_000], [None]),
        (15_000, [10_000, 5_000], [None, "user_009999"]),
    ],
)
def test_assignment_service_applies_total_limit_across_project_pages(
    total_limit: int,
    expected_page_sizes: list[int],
    expected_after_user_ids: list[str | None],
) -> None:
    service, repos = make_service(user_vectors=user_vector_records(25_001))

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(eligible_user_limit=total_limit),
    )

    assert response.assignment_count == total_limit
    assert [call[2] for call in repos.user_vectors.project_calls] == (
        expected_page_sizes
    )
    assert [call[4] for call in repos.user_vectors.project_calls] == (
        expected_after_user_ids
    )


def test_assignment_service_chunks_sorted_unique_explicit_user_ids() -> None:
    records = user_vector_records(10_001)
    requested_user_ids = [record.user_id for record in reversed(records)]
    requested_user_ids.extend(["user_000000", "user_010000"])
    service, repos = make_service(user_vectors=records)

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=requested_user_ids),
    )

    assert response.assignment_count == 10_001
    assert response.page_count == 2
    assert response.processed_user_count == 10_001
    assert response.assignment_mode == "explicit_user_ids"
    assert [len(call[1]) for call in repos.user_vectors.user_id_calls] == [10_000, 1]
    assert repos.user_vectors.user_id_calls[0][1][0] == "user_000000"
    assert repos.user_vectors.user_id_calls[0][1][-1] == "user_009999"
    assert repos.user_vectors.user_id_calls[1][1] == ("user_010000",)


def test_assignment_service_excludes_empty_explicit_user_id_chunks_from_pages() -> None:
    service, _repos = make_service(user_vectors=[])

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["missing_user"]),
    )

    assert response.page_count == 0
    assert response.processed_user_count == 0
    assert response.assignment_count == 0
    assert response.skipped_existing_count == 0
    assert response.insert_conflict_count == 0
    assert response.ann_applied is False
    assert response.ann_not_applied_reason == "no_users_to_match"
    assert response.assignment_mode == "explicit_user_ids"


def test_assignment_service_applies_total_limit_to_explicit_user_ids() -> None:
    records = user_vector_records(10)
    service, repos = make_service(user_vectors=records)

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(
            user_ids=[record.user_id for record in reversed(records)],
            eligible_user_limit=5,
        ),
    )

    assert response.assignment_count == 5
    assert repos.user_vectors.user_id_calls[0][1] == tuple(
        f"user_{index:06d}" for index in range(5)
    )


def test_assignment_service_rejects_duplicate_user_ids_in_vector_page() -> None:
    duplicate = user_vector_record("user_000001", vector(0))
    service, repos = make_service(
        user_vectors=[],
        project_pages=[[duplicate, duplicate]],
    )

    with pytest.raises(SegmentAssignmentValidationError, match="duplicate user_id"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_rejects_unordered_vector_page() -> None:
    service, repos = make_service(
        user_vectors=[],
        project_pages=[
            [
                user_vector_record("user_000002", vector(0)),
                user_vector_record("user_000001", vector(0)),
            ]
        ],
    )

    with pytest.raises(SegmentAssignmentValidationError, match="ordered"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_rejects_non_increasing_page_cursor(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.decision.assignment_service.ASSIGNMENT_PAGE_SIZE",
        2,
    )
    service, repos = make_service(
        user_vectors=[],
        project_pages=[
            [
                user_vector_record("user_000001", vector(0)),
                user_vector_record("user_000002", vector(0)),
            ],
            [
                user_vector_record("user_000002", vector(0)),
                user_vector_record("user_000003", vector(0)),
            ],
        ],
    )

    with pytest.raises(SegmentAssignmentValidationError, match="monotonically"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(),
        )

    assert len(repos.assignments.inserted) == 2


def test_assignment_service_recall_skips_existing_without_diagnostics() -> None:
    service, repos = make_service(
        user_vectors=[
            user_vector_record("user_family", vector(0)),
            user_vector_record("user_fallback", vector(1)),
        ]
    )
    request = SegmentAssignmentBuildRequest()

    first = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=request,
    )
    first_assignments = list(repos.assignments.inserted)
    second = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=request,
    )

    assert first.assignment_count == 2
    assert first.skipped_existing_count == 0
    assert second.assignment_count == 0
    assert second.skipped_existing_count == 2
    assert second.ann_candidate_count == 0
    assert second.exact_reranked_pair_count == 0
    assert second.fallback_count == 0
    assert second.batch_has_fallback is False
    assert repos.assignments.inserted == first_assignments


def test_assignment_service_uses_audience_scope_vector_version_when_request_omits_it() -> None:
    run = promotion_run_record(
        goal_snapshot_json={
            "min_sample_size": 1,
            "audience_scope": {
                "vector_version": "v2",
                "selection_policy": {"limit": 10},
            },
        }
    )
    service, repos = make_service(run=run)

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(),
    )

    assert response.vector_version == "v2"
    assert repos.user_vectors.project_calls == [
        ("hotel-client-a", "v2", 10, None, None)
    ]
    assert repos.segment_vectors.ann_calls[0]["vector_version"] == "v2"


def test_assignment_service_rejects_conflicting_scope_and_request_vector_versions() -> None:
    run = promotion_run_record(
        goal_snapshot_json={
            "min_sample_size": 1,
            "audience_scope": {"vector_version": "v2"},
        }
    )
    service, repos = make_service(run=run)

    with pytest.raises(SegmentAssignmentValidationError, match="vector_version"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(vector_version="v1"),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_rejects_audience_scope_with_user_ids_for_mvp() -> None:
    run = promotion_run_record(
        goal_snapshot_json={
            "min_sample_size": 1,
            "audience_scope": {"selection_policy": {"limit": 10}},
        }
    )
    service, repos = make_service(run=run)

    with pytest.raises(SegmentAssignmentValidationError, match="user_ids"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_rejects_project_id_filter() -> None:
    run = promotion_run_record(
        goal_snapshot_json={
            "min_sample_size": 1,
            "audience_scope": {"filters": {"project_id": "other-project"}},
        }
    )
    service, repos = make_service(run=run)

    with pytest.raises(SegmentAssignmentValidationError, match="project_id"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(),
        )

    assert repos.assignments.inserted == []


@pytest.mark.parametrize(
    "audience_scope",
    [
        {"base": "promotion_run_eligible_users"},
        {"selection_policy": {"ordering": "stable_hash"}},
        {"filters": {"country": "KR"}},
        {"filters": {"has_valid_vector": False}},
    ],
)
def test_assignment_service_rejects_unsupported_audience_scope_values(
    audience_scope: dict[str, object],
) -> None:
    run = promotion_run_record(
        goal_snapshot_json={
            "min_sample_size": 1,
            "audience_scope": audience_scope,
        }
    )
    service, repos = make_service(run=run)

    with pytest.raises(SegmentAssignmentValidationError):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_applies_scope_source_filter_and_min_limit() -> None:
    run = promotion_run_record(
        goal_snapshot_json={
            "min_sample_size": 1,
            "audience_scope": {
                "filters": {"source": "booking_profile"},
                "selection_policy": {"limit": 20, "ordering": "user_id_asc"},
            },
        }
    )
    service, repos = make_service(run=run, user_vectors=[])

    service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(eligible_user_limit=5),
    )

    assert repos.user_vectors.project_calls == [
        ("hotel-client-a", "v1", 5, "booking_profile", None)
    ]


def test_assignment_service_preserves_scope_filters_across_pages(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.decision.assignment_service.ASSIGNMENT_PAGE_SIZE",
        2,
    )
    run = promotion_run_record(
        goal_snapshot_json={
            "min_sample_size": 1,
            "audience_scope": {
                "vector_version": "v2",
                "filters": {"source": "booking_profile"},
                "selection_policy": {"ordering": "user_id_asc"},
            },
        }
    )
    service, repos = make_service(run=run, user_vectors=user_vector_records(3))

    service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(),
    )

    assert repos.user_vectors.project_calls == [
        ("hotel-client-a", "v2", 2, "booking_profile", None),
        ("hotel-client-a", "v2", 2, "booking_profile", "user_000001"),
    ]


def test_assignment_service_does_not_require_min_sample_snapshot() -> None:
    run = promotion_run_record(goal_snapshot_json={})
    service, repos = make_service(run=run)

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
    )

    assert response.assignment_count == 1
    assert response.model_dump()["insufficient_segment_count"] == 0
    assert len(repos.assignments.inserted) == 1


def test_assignment_service_raises_not_found_for_missing_run() -> None:
    service, _repos = make_service(run=None)

    with pytest.raises(SegmentAssignmentRunNotFoundError):
        service.build_assignments(
            promotion_run_id="missing_run",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )


class FakePromotionRunRepository:
    def __init__(self, run: PromotionRunRecord | None) -> None:
        self.run = run

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        return self.run


class FakeLegacyAudienceSnapshotRepository:
    def resolve_target_contract(
        self,
        *,
        analysis_id: str,
        segment_ids: list[str],
    ) -> TargetAudienceResolution:
        return TargetAudienceResolution(
            analysis_id=analysis_id,
            segment_ids=tuple(sorted(set(segment_ids))),
            contract="legacy",
        )


class FakeAdExperimentRepository:
    def __init__(self, experiments: list[AdExperimentRecord]) -> None:
        self.experiments = experiments

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        return self.experiments

class FakeSegmentVectorRepository:
    def __init__(
        self,
        vectors: list[SegmentVectorRecord],
        ann_candidates: AnnCandidates,
        ann_error: Exception | None,
    ) -> None:
        self.vectors = vectors
        self.ann_candidates = ann_candidates
        self.ann_error = ann_error
        self.configure_ann_search_count = 0
        self.ann_calls: list[dict[str, object]] = []

    def list_for_run_segments(self, **_kwargs: object) -> list[SegmentVectorRecord]:
        return self.vectors

    def configure_ann_search(self) -> None:
        self.configure_ann_search_count += 1

    def list_ann_candidates(self, **kwargs: object) -> list[SegmentVectorRecord]:
        self.ann_calls.append(dict(kwargs))
        return self.vectors if self.ann_candidates is None else self.ann_candidates

    def list_ann_candidates_for_users(
        self,
        **kwargs: object,
    ) -> dict[str, list[SegmentVectorRecord]]:
        self.ann_calls.append(dict(kwargs))
        if self.ann_error is not None:
            raise self.ann_error

        user_ids = list(kwargs["user_ids"])
        if isinstance(self.ann_candidates, dict):
            return {
                str(user_id): self.ann_candidates.get(str(user_id), [])
                for user_id in user_ids
            }
        candidates = self.vectors if self.ann_candidates is None else self.ann_candidates
        return {str(user_id): list(candidates) for user_id in user_ids}


class FakeUserBehaviorVectorRepository:
    def __init__(
        self,
        vectors: list[UserBehaviorVectorRecord],
        project_pages: list[list[UserBehaviorVectorRecord]] | None = None,
    ) -> None:
        self.vectors = vectors
        self.project_pages = project_pages
        self.user_id_calls: list[tuple[str, tuple[str, ...], str, str | None]] = []
        self.project_calls: list[
            tuple[str, str, int, str | None, str | None]
        ] = []

    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: list[str],
        vector_version: str,
        source: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        self.user_id_calls.append((project_id, tuple(user_ids), vector_version, source))
        requested_user_ids = set(user_ids)
        return sorted(
            [
                record
                for record in self.vectors
                if record.user_id in requested_user_ids
            ],
            key=lambda record: record.user_id,
        )

    def list_for_project(
        self,
        *,
        project_id: str,
        vector_version: str,
        limit: int,
        source: str | None = None,
        after_user_id: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        self.project_calls.append(
            (project_id, vector_version, limit, source, after_user_id)
        )
        if self.project_pages is not None:
            call_index = len(self.project_calls) - 1
            if call_index >= len(self.project_pages):
                return []
            return self.project_pages[call_index]
        records = sorted(self.vectors, key=lambda record: record.user_id)
        if after_user_id is not None:
            records = [
                record for record in records if record.user_id > after_user_id
            ]
        return records[:limit]


class FakeUserSegmentAssignmentRepository:
    def __init__(
        self,
        existing_user_ids: set[str],
        insert_conflict_user_ids: set[str],
    ) -> None:
        self.existing_user_ids = existing_user_ids
        self.insert_conflict_user_ids = insert_conflict_user_ids
        self.inserted: list[UserSegmentAssignmentWrite] = []

    def list_existing_user_ids(
        self,
        *,
        promotion_run_id: str,
        user_ids: list[str],
    ) -> set[str]:
        return self.existing_user_ids.intersection(user_ids)

    def insert_many(
        self,
        assignments: list[UserSegmentAssignmentWrite],
    ) -> list[UserSegmentAssignmentInsertRecord]:
        inserted = [
            assignment
            for assignment in assignments
            if assignment.user_id not in self.insert_conflict_user_ids
        ]
        self.inserted.extend(inserted)
        self.existing_user_ids.update(
            assignment.user_id for assignment in assignments
        )
        return [
            UserSegmentAssignmentInsertRecord(
                user_id=assignment.user_id,
                segment_id=assignment.segment_id,
                fallback=assignment.fallback,
                fallback_reason=assignment.fallback_reason,
                similarity_score=assignment.similarity_score,
            )
            for assignment in inserted
        ]


class FakeRepositoryBundle:
    def __init__(
        self,
        *,
        run: PromotionRunRecord | None,
        experiments: list[AdExperimentRecord],
        segment_vectors: list[SegmentVectorRecord],
        ann_candidates: AnnCandidates,
        ann_error: Exception | None,
        user_vectors: list[UserBehaviorVectorRecord],
        project_pages: list[list[UserBehaviorVectorRecord]] | None,
        existing_user_ids: set[str],
        insert_conflict_user_ids: set[str],
    ) -> None:
        self.runs = FakePromotionRunRepository(run)
        self.ad_experiments = FakeAdExperimentRepository(experiments)
        self.segment_vectors = FakeSegmentVectorRepository(
            segment_vectors,
            ann_candidates,
            ann_error,
        )
        self.user_vectors = FakeUserBehaviorVectorRepository(
            user_vectors,
            project_pages,
        )
        self.assignments = FakeUserSegmentAssignmentRepository(
            existing_user_ids,
            insert_conflict_user_ids,
        )
        self.audience_snapshots = FakeLegacyAudienceSnapshotRepository()


def make_service(
    *,
    run: PromotionRunRecord | None | object = DEFAULT_RUN,
    experiments: list[AdExperimentRecord] | None = None,
    segment_vectors: list[SegmentVectorRecord] | None = None,
    ann_candidates: AnnCandidates = None,
    ann_error: Exception | None = None,
    user_vectors: list[UserBehaviorVectorRecord] | None = None,
    project_pages: list[list[UserBehaviorVectorRecord]] | None = None,
    existing_user_ids: set[str] | None = None,
    insert_conflict_user_ids: set[str] | None = None,
) -> tuple[SegmentAssignmentService, FakeRepositoryBundle]:
    repos = FakeRepositoryBundle(
        run=promotion_run_record() if run is DEFAULT_RUN else run,
        experiments=experiments
        if experiments is not None
        else [
            ad_experiment_record(segment_id="seg_family_trip"),
            ad_experiment_record(segment_id="seg_existing_all"),
        ],
        segment_vectors=segment_vectors
        if segment_vectors is not None
        else [
            segment_vector_record("seg_family_trip", vector(0)),
        ],
        ann_candidates=ann_candidates,
        ann_error=ann_error,
        user_vectors=user_vectors
        if user_vectors is not None
        else [
            user_vector_record("user_family", vector(0)),
        ],
        project_pages=project_pages,
        existing_user_ids=existing_user_ids or set(),
        insert_conflict_user_ids=insert_conflict_user_ids or set(),
    )
    service = SegmentAssignmentService(
        promotion_run_repository=repos.runs,
        ad_experiment_repository=repos.ad_experiments,
        segment_vector_repository=repos.segment_vectors,
        user_behavior_vector_repository=repos.user_vectors,
        user_segment_assignment_repository=repos.assignments,
        reranker=SegmentCandidateReranker(),
        audience_snapshot_repository=repos.audience_snapshots,
    )
    return service, repos


def promotion_run_record(
    *,
    min_sample_size: int = 1,
    goal_snapshot_json: dict[str, object] | None = None,
) -> PromotionRunRecord:
    return PromotionRunRecord(
        promotion_run_id="prun_banner_001_loop_1",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        loop_count=1,
        status="planned",
        goal_snapshot_json=goal_snapshot_json
        if goal_snapshot_json is not None
        else {"min_sample_size": min_sample_size},
        segment_scope_json=["seg_family_trip"],
        segment_scope_fingerprint="a" * 64,
    )


def ad_experiment_record(segment_id: str) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=f"adexp_{segment_id}",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        segment_id=segment_id,
        segment_name=segment_id.replace("_", " "),
        content_id=f"content_{segment_id}",
        content_option_id=f"option_{segment_id}",
        channel=Channel.ONSITE_BANNER.value,
        loop_count=1,
        status="planned",
        goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        goal_target_value=Decimal("0.030000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
    )


def segment_vector_record(
    segment_id: str,
    values: list[float],
    *,
    source: str = "decision_analysis",
) -> SegmentVectorRecord:
    return SegmentVectorRecord(
        segment_vector_id=f"segvec_{segment_id}_v1",
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        promotion_run_id=None,
        analysis_id="analysis_banner_001",
        segment_id=segment_id,
        vector_dim=64,
        vector_values=values,
        vector_version="v1",
        source=source,
        embedding=values,
    )


def user_vector_record(user_id: str, values: list[float]) -> UserBehaviorVectorRecord:
    return UserBehaviorVectorRecord(
        project_id="hotel-client-a",
        user_id=user_id,
        vector_dim=64,
        vector_values=values,
        vector_version="v1",
        source="batch_profile",
    )
