from __future__ import annotations

from decimal import Decimal

import pytest

from app.decision.assignment_service import (
    SegmentAssignmentRunNotFoundError,
    SegmentAssignmentService,
    SegmentAssignmentValidationError,
)
from app.decision.matcher import ExactCosineMatcher
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


def vector(index: int, value: float = 1.0) -> list[float]:
    values = [0.0] * 64
    values[index] = value
    return values


def test_assignment_service_builds_exact_and_fallback_assignments() -> None:
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
    assert response.fallback_count == 1
    assert response.insufficient_segment_count == 0
    assert [assignment.user_id for assignment in repos.assignments.inserted] == [
        "user_family",
        "user_fallback",
    ]
    regular, fallback = repos.assignments.inserted
    assert regular.segment_id == "seg_family_trip"
    assert regular.assignment_source == AssignmentSource.DECISION_BATCH.value
    assert regular.fallback is False
    assert fallback.segment_id == "seg_existing_all"
    assert fallback.assignment_source == AssignmentSource.FALLBACK.value
    assert fallback.fallback is True


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


def test_assignment_service_rejects_invalid_segment_vector_without_writes() -> None:
    service, repos = make_service(
        segment_vectors=[
            segment_vector_record("seg_family_trip", [0.0] * 64),
        ],
    )

    with pytest.raises(SegmentAssignmentValidationError, match="invalid segment vector"):
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

    assert repos.user_vectors.project_calls == [("hotel-client-a", "v1", 50)]
    assert repos.user_vectors.user_id_calls == []


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
    def __init__(self, vectors: list[SegmentVectorRecord]) -> None:
        self.vectors = vectors

    def list_for_run_segments(self, **_kwargs: object) -> list[SegmentVectorRecord]:
        return self.vectors


class FakeUserBehaviorVectorRepository:
    def __init__(self, vectors: list[UserBehaviorVectorRecord]) -> None:
        self.vectors = vectors
        self.user_id_calls: list[tuple[str, tuple[str, ...], str]] = []
        self.project_calls: list[tuple[str, str, int]] = []

    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: list[str],
        vector_version: str,
    ) -> list[UserBehaviorVectorRecord]:
        self.user_id_calls.append((project_id, tuple(user_ids), vector_version))
        return self.vectors

    def list_for_project(
        self,
        *,
        project_id: str,
        vector_version: str,
        limit: int,
    ) -> list[UserBehaviorVectorRecord]:
        self.project_calls.append((project_id, vector_version, limit))
        return self.vectors


class FakeUserSegmentAssignmentRepository:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.inserted: list[UserSegmentAssignmentWrite] = []

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
        user_vectors: list[UserBehaviorVectorRecord],
        segment_counts: dict[str, int],
    ) -> None:
        self.runs = FakePromotionRunRepository(run)
        self.ad_experiments = FakeAdExperimentRepository(experiments)
        self.target_segments = FakePromotionTargetSegmentRepository()
        self.segment_vectors = FakeSegmentVectorRepository(segment_vectors)
        self.user_vectors = FakeUserBehaviorVectorRepository(user_vectors)
        self.assignments = FakeUserSegmentAssignmentRepository(segment_counts)


def make_service(
    *,
    run: PromotionRunRecord | None | object = DEFAULT_RUN,
    experiments: list[AdExperimentRecord] | None = None,
    segment_vectors: list[SegmentVectorRecord] | None = None,
    user_vectors: list[UserBehaviorVectorRecord] | None = None,
    segment_counts: dict[str, int] | None = None,
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
        user_vectors=user_vectors
        if user_vectors is not None
        else [
            user_vector_record("user_family", vector(0)),
        ],
        segment_counts=segment_counts or {"seg_family_trip": 1},
    )
    service = SegmentAssignmentService(
        promotion_run_repository=repos.runs,
        ad_experiment_repository=repos.ad_experiments,
        promotion_target_segment_repository=repos.target_segments,
        segment_vector_repository=repos.segment_vectors,
        user_behavior_vector_repository=repos.user_vectors,
        user_segment_assignment_repository=repos.assignments,
        matcher=ExactCosineMatcher(),
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
