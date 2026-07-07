from __future__ import annotations

from decimal import Decimal

import pytest

from app.decision.assignment_service import (
    DEFAULT_ASSIGNMENT_ELIGIBLE_USER_LIMIT,
    SegmentAssignmentRunNotFoundError,
    SegmentAssignmentService,
    SegmentAssignmentValidationError,
)
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
    UserSegmentAssignmentWrite,
)
from app.decision.schemas import (
    AdExperimentStatus,
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


def test_assignment_service_builds_ann_reranked_and_fallback_assignments() -> None:
    service, repos = make_service(
        user_vectors=[
            user_vector_record("user_family", vector(0)),
            user_vector_record("user_fallback", vector(1)),
        ],
        segment_counts={"seg_family_trip": 1},
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family", "user_fallback"]),
    )

    assert response.assignment_count == 2
    assert response.matching_mode == "pgvector_hnsw_rerank"
    assert response.ann_candidate_limit == 50
    assert response.ann_candidate_count == 2
    assert response.exact_reranked_pair_count == 2
    assert response.batch_has_fallback is True
    assert response.fallback_count == 1
    assert response.below_threshold_fallback_count == 1
    assert response.no_candidate_fallback_count == 0
    assert response.invalid_user_vector_fallback_count == 0
    assert response.ann_underfilled_user_count == 0
    assert response.insufficient_segment_count == 0
    assert [assignment.user_id for assignment in repos.assignments.inserted] == [
        "user_family",
        "user_fallback",
    ]
    regular, fallback = repos.assignments.inserted
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


def test_assignment_service_requires_fallback_experiment_before_writes() -> None:
    service, repos = make_service(
        experiments=[
            ad_experiment_record(segment_id="seg_family_trip"),
        ],
        user_vectors=[
            user_vector_record("user_fallback", vector(1)),
        ],
    )

    with pytest.raises(SegmentAssignmentValidationError, match="fallback"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_fallback"]),
        )

    assert repos.assignments.inserted == []


def test_assignment_service_falls_back_for_no_ann_candidate() -> None:
    service, repos = make_service(ann_candidates=[])

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
    )

    assert response.fallback_count == 1
    assert response.no_candidate_fallback_count == 1
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
    assert repos.segment_vectors.configure_ann_search_count == 0
    assert repos.segment_vectors.ann_calls == []
    assert (
        repos.assignments.inserted[0].fallback_reason
        == FALLBACK_REASON_INVALID_USER_VECTOR
    )


def test_assignment_service_skips_existing_assignments() -> None:
    service, repos = make_service(existing_user_ids={"user_family"})

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
    )

    assert response.assignment_count == 0
    assert response.skipped_existing_count == 1
    assert repos.segment_vectors.configure_ann_search_count == 0
    assert repos.segment_vectors.ann_calls == []
    assert repos.assignments.inserted == []


def test_assignment_service_splits_valid_users_into_batch_chunks() -> None:
    user_vectors = [
        user_vector_record(f"user_{index:03d}", vector(0))
        for index in range(257)
    ]
    service, repos = make_service(
        user_vectors=user_vectors,
        segment_counts={"seg_family_trip": 257},
    )

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
        user_vectors=[
            user_vector_record("user_family", vector(0)),
            user_vector_record("user_family", vector(1)),
        ],
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


def test_assignment_service_marks_insufficient_segments_from_final_counts() -> None:
    service, repos = make_service(
        run=promotion_run_record(min_sample_size=2),
        user_vectors=[],
        segment_counts={
            "seg_family_trip": 1,
            "seg_existing_all": 0,
        },
    )

    response = service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(eligible_user_limit=10),
    )

    assert response.insufficient_segment_count == 1
    assert repos.ad_experiments.status_updates == [
        ("adexp_seg_family_trip", AdExperimentStatus.INSUFFICIENT_DATA.value)
    ]
    assert repos.target_segments.status_updates == [
        (
            "analysis_banner_001",
            "seg_family_trip",
            AdExperimentStatus.INSUFFICIENT_DATA.value,
        )
    ]


def test_assignment_service_uses_project_limit_when_user_ids_omitted() -> None:
    service, repos = make_service(user_vectors=[])

    service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(eligible_user_limit=50),
    )

    assert repos.user_vectors.project_calls == [("hotel-client-a", "v1", 50, None)]
    assert repos.user_vectors.user_id_calls == []


def test_assignment_service_uses_default_limit_for_implicit_project_scope() -> None:
    service, repos = make_service(user_vectors=[])

    service.build_assignments(
        promotion_run_id="prun_banner_001_loop_1",
        request=SegmentAssignmentBuildRequest(),
    )

    assert repos.user_vectors.project_calls == [
        (
            "hotel-client-a",
            "v1",
            DEFAULT_ASSIGNMENT_ELIGIBLE_USER_LIMIT,
            None,
        )
    ]


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
    assert repos.user_vectors.project_calls == [("hotel-client-a", "v2", 10, None)]
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
        ("hotel-client-a", "v1", 5, "booking_profile")
    ]


def test_assignment_service_requires_min_sample_snapshot() -> None:
    run = promotion_run_record(goal_snapshot_json={})
    service, repos = make_service(run=run)

    with pytest.raises(SegmentAssignmentValidationError, match="min_sample_size"):
        service.build_assignments(
            promotion_run_id="prun_banner_001_loop_1",
            request=SegmentAssignmentBuildRequest(user_ids=["user_family"]),
        )

    assert repos.assignments.inserted == []


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


class FakeAdExperimentRepository:
    def __init__(self, experiments: list[AdExperimentRecord]) -> None:
        self.experiments = experiments
        self.status_updates: list[tuple[str, str]] = []

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        return self.experiments

    def update_status(self, *, ad_experiment_id: str, status: str) -> None:
        self.status_updates.append((ad_experiment_id, status))


class FakePromotionTargetSegmentRepository:
    def __init__(self) -> None:
        self.status_updates: list[tuple[str, str, str]] = []

    def update_status(
        self,
        *,
        analysis_id: str,
        segment_id: str,
        status: str,
    ) -> None:
        self.status_updates.append((analysis_id, segment_id, status))


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
    def __init__(self, vectors: list[UserBehaviorVectorRecord]) -> None:
        self.vectors = vectors
        self.user_id_calls: list[tuple[str, tuple[str, ...], str, str | None]] = []
        self.project_calls: list[tuple[str, str, int, str | None]] = []

    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: list[str],
        vector_version: str,
        source: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        self.user_id_calls.append((project_id, tuple(user_ids), vector_version, source))
        return self.vectors

    def list_for_project(
        self,
        *,
        project_id: str,
        vector_version: str,
        limit: int,
        source: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        self.project_calls.append((project_id, vector_version, limit, source))
        return self.vectors


class FakeUserSegmentAssignmentRepository:
    def __init__(self, counts: dict[str, int], existing_user_ids: set[str]) -> None:
        self.counts = counts
        self.existing_user_ids = existing_user_ids
        self.inserted: list[UserSegmentAssignmentWrite] = []

    def list_existing_user_ids(
        self,
        *,
        promotion_run_id: str,
        user_ids: list[str],
    ) -> set[str]:
        return self.existing_user_ids.intersection(user_ids)

    def insert_many(self, assignments: list[UserSegmentAssignmentWrite]) -> int:
        self.inserted.extend(assignments)
        return len(assignments)

    def count_by_run_segments(
        self,
        *,
        promotion_run_id: str,
        segment_ids: list[str],
    ) -> dict[str, int]:
        return {
            segment_id: self.counts.get(segment_id, 0)
            for segment_id in segment_ids
        }


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
        segment_counts: dict[str, int],
        existing_user_ids: set[str],
    ) -> None:
        self.runs = FakePromotionRunRepository(run)
        self.ad_experiments = FakeAdExperimentRepository(experiments)
        self.target_segments = FakePromotionTargetSegmentRepository()
        self.segment_vectors = FakeSegmentVectorRepository(
            segment_vectors,
            ann_candidates,
            ann_error,
        )
        self.user_vectors = FakeUserBehaviorVectorRepository(user_vectors)
        self.assignments = FakeUserSegmentAssignmentRepository(
            segment_counts,
            existing_user_ids,
        )


def make_service(
    *,
    run: PromotionRunRecord | None | object = DEFAULT_RUN,
    experiments: list[AdExperimentRecord] | None = None,
    segment_vectors: list[SegmentVectorRecord] | None = None,
    ann_candidates: AnnCandidates = None,
    ann_error: Exception | None = None,
    user_vectors: list[UserBehaviorVectorRecord] | None = None,
    segment_counts: dict[str, int] | None = None,
    existing_user_ids: set[str] | None = None,
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
        segment_counts=segment_counts or {"seg_family_trip": 1},
        existing_user_ids=existing_user_ids or set(),
    )
    service = SegmentAssignmentService(
        promotion_run_repository=repos.runs,
        ad_experiment_repository=repos.ad_experiments,
        promotion_target_segment_repository=repos.target_segments,
        segment_vector_repository=repos.segment_vectors,
        user_behavior_vector_repository=repos.user_vectors,
        user_segment_assignment_repository=repos.assignments,
        reranker=SegmentCandidateReranker(),
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
        source="decision_analysis",
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
