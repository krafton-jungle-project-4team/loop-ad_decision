from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.decision.assignment_service import SegmentAssignmentService
from app.decision.audience_snapshots import (
    AudienceSnapshotBinding,
    AudienceSnapshotMember,
    AudienceSnapshotSet,
    TargetAudienceResolution,
)
from app.decision.experiment_assignment_repository import (
    AdExperimentUnitRecord,
    SegmentAssignmentExecutionRecord,
)
from app.decision.experiment_design import (
    ExperimentDesignConflictError,
    RandomizedHoldoutAudienceTooSmallError,
    RandomizedHoldoutConfigurationError,
)
from app.decision.matcher import SegmentCandidateReranker
from app.decision.outcome_spec import build_frozen_outcome_spec
from app.decision.repositories import (
    AdExperimentRecord,
    PromotionRunRecord,
    UserSegmentAssignmentInsertRecord,
)
from app.decision.schemas import SegmentAssignmentBuildRequest


ASSIGNED_AT = datetime(2026, 7, 21, 12, tzinfo=UTC)
CUTOFF = datetime(2026, 7, 21, 11, tzinfo=UTC)


def test_default_all_treatment_records_full_population_and_serves_every_user() -> None:
    service, repos = make_service(member_counts=(2, 3), randomization_salt=None)

    response = service.build_assignments(
        promotion_run_id="run",
        request=SegmentAssignmentBuildRequest(),
    )

    assert response.experiment_design is not None
    assert response.experiment_design.mode == "all_treatment"
    assert response.processed_user_count == 5
    assert response.assignment_count == 5
    assert len(repos.executions.units) == 5
    assert {unit.arm for unit in repos.executions.units} == {"treatment"}
    assert len(repos.assignments.inserted) == 5
    assert all(
        assignment.segment_assignment_execution_id
        == response.segment_assignment_execution_id
        for assignment in repos.assignments.inserted
    )
    assert all(unit.assigned_at == ASSIGNED_AT for unit in repos.executions.units)
    assert all(
        unit.outcome_window_start == unit.assigned_at
        for unit in repos.executions.units
    )
    assert repos.executions.finalized_execution_ids == [
        response.segment_assignment_execution_id
    ]


def test_randomized_holdout_records_control_but_only_serves_treatment() -> None:
    service, repos = make_service(member_counts=(4, 4), randomization_salt="secret")

    response = service.build_assignments(
        promotion_run_id="run",
        request=SegmentAssignmentBuildRequest.model_validate(
            {
                "experiment_design": {
                    "mode": "randomized_holdout",
                    "treatment_ratio": 0.5,
                    "outcome_window_days": 30,
                }
            }
        ),
    )

    assert response.processed_user_count == 8
    assert response.assignment_count == 4
    assert sum(unit.arm == "treatment" for unit in repos.executions.units) == 4
    assert sum(unit.arm == "control" for unit in repos.executions.units) == 4
    assert len(repos.assignments.inserted) == 4
    assert {
        assignment.user_id for assignment in repos.assignments.inserted
    } == {
        unit.user_id for unit in repos.executions.units if unit.arm == "treatment"
    }
    assert all(
        result.treatment_count == result.control_count == 2
        for result in response.allocation_results
    )
    assert repos.executions.finalized_execution_ids == [
        response.segment_assignment_execution_id
    ]


def test_randomized_holdout_rejects_small_experiment_before_writes() -> None:
    service, repos = make_service(member_counts=(2, 1), randomization_salt="secret")

    with pytest.raises(RandomizedHoldoutAudienceTooSmallError):
        service.build_assignments(
            promotion_run_id="run",
            request=holdout_request(),
        )

    assert repos.executions.inserted_execution is None
    assert repos.executions.units == []
    assert repos.assignments.inserted == []


def test_randomized_holdout_requires_salt_before_writes() -> None:
    service, repos = make_service(member_counts=(2,), randomization_salt=None)

    with pytest.raises(RandomizedHoldoutConfigurationError):
        service.build_assignments(
            promotion_run_id="run",
            request=holdout_request(),
        )

    assert repos.executions.inserted_execution is None
    assert repos.executions.units == []
    assert repos.assignments.inserted == []


def test_existing_run_rejects_different_experiment_design() -> None:
    service, repos = make_service(member_counts=(2,), randomization_salt="secret")
    service.build_assignments(
        promotion_run_id="run",
        request=SegmentAssignmentBuildRequest(),
    )

    with pytest.raises(ExperimentDesignConflictError):
        service.build_assignments(
            promotion_run_id="run",
            request=holdout_request(),
        )


def test_same_request_reuses_existing_execution_and_arm_results() -> None:
    service, repos = make_service(member_counts=(4,), randomization_salt="secret")
    request = holdout_request()

    first = service.build_assignments(promotion_run_id="run", request=request)
    first_units = list(repos.executions.units)
    first_assignment_count = len(repos.assignments.inserted)
    second = service.build_assignments(promotion_run_id="run", request=request)

    assert second.segment_assignment_execution_id == (
        first.segment_assignment_execution_id
    )
    assert repos.executions.units == first_units
    assert len(repos.assignments.inserted) == first_assignment_count
    assert second.assignment_count == first.assignment_count
    assert repos.executions.finalized_execution_ids == [
        first.segment_assignment_execution_id
    ]


def holdout_request() -> SegmentAssignmentBuildRequest:
    return SegmentAssignmentBuildRequest.model_validate(
        {
            "experiment_design": {
                "mode": "randomized_holdout",
                "treatment_ratio": 0.5,
                "outcome_window_days": 30,
            }
        }
    )


class _RunRepository:
    def __init__(self, run: PromotionRunRecord) -> None:
        self.run = run
        self.lock_count = 0

    def get_by_id(self, promotion_run_id: str):
        return self.run if promotion_run_id == self.run.promotion_run_id else None

    def get_by_id_for_update(self, promotion_run_id: str):
        self.lock_count += 1
        return self.get_by_id(promotion_run_id)


class _ExperimentRepository:
    def __init__(self, experiments: list[AdExperimentRecord]) -> None:
        self.experiments = experiments

    def list_by_run(self, promotion_run_id: str):
        return list(self.experiments)


class _AudienceRepository:
    def __init__(
        self,
        bindings: tuple[AudienceSnapshotBinding, ...],
        members: list[AudienceSnapshotMember],
    ) -> None:
        self.bindings = bindings
        self.members = members
        self.consumed = False

    def resolve_run_contract(self, **_kwargs):
        return TargetAudienceResolution(
            analysis_id="analysis",
            segment_ids=tuple(binding.segment_id for binding in self.bindings),
            contract="segment_audience.v1",
        )

    def require_run_binding_set(self, **_kwargs):
        return AudienceSnapshotSet(
            analysis_id="analysis",
            segment_ids=tuple(binding.segment_id for binding in self.bindings),
            vector_version="hotel_behavior.v2",
            member_count=len(self.members),
            snapshot_ids=tuple(
                binding.audience_snapshot_id for binding in self.bindings
            ),
            bindings=self.bindings,
        )

    def list_run_members(self, *, after_user_id, limit, **_kwargs):
        page = [
            member
            for member in self.members
            if after_user_id is None or member.user_id > after_user_id
        ]
        return page[:limit]

    def consume_run_members(self, **_kwargs):
        self.consumed = True


class _AssignmentRepository:
    def __init__(self) -> None:
        self.inserted = []

    def insert_many(self, assignments):
        self.inserted.extend(assignments)
        return [
            UserSegmentAssignmentInsertRecord(
                user_id=assignment.user_id,
                segment_id=assignment.segment_id,
                fallback=assignment.fallback,
                fallback_reason=assignment.fallback_reason,
                similarity_score=assignment.similarity_score,
            )
            for assignment in assignments
        ]

    def list_existing_user_ids(self, **_kwargs):
        return set()


class _ExecutionRepository:
    def __init__(self) -> None:
        self.inserted_execution = None
        self.units = []
        self.finalized_execution_ids = []

    def database_clock(self):
        return ASSIGNED_AT

    def list_uplift_ready_executions(self, _promotion_run_id):
        return [self.inserted_execution] if self.inserted_execution else []

    def insert_execution(self, execution):
        self.inserted_execution = SegmentAssignmentExecutionRecord(
            segment_assignment_execution_id=(
                execution.segment_assignment_execution_id
            ),
            promotion_run_id=execution.promotion_run_id,
            request_fingerprint=execution.request_fingerprint,
            input_fingerprint=execution.input_fingerprint,
            matcher_strategy=execution.matcher_strategy,
            matcher_version=execution.matcher_version,
            vector_version=execution.vector_version,
            source_cutoff_at=execution.source_cutoff_at,
            input_manifest_json=execution.input_manifest_json,
        )
        return self.inserted_execution

    def insert_units(self, units):
        self.units.extend(
            AdExperimentUnitRecord(
                experiment_unit_id=unit.experiment_unit_id,
                project_id=unit.project_id,
                promotion_run_id=unit.promotion_run_id,
                ad_experiment_id=unit.ad_experiment_id,
                segment_id=unit.segment_id,
                audience_snapshot_id=unit.audience_snapshot_id,
                vector_generation_id=unit.vector_generation_id,
                segment_assignment_execution_id=(
                    unit.segment_assignment_execution_id
                ),
                user_id=unit.user_id,
                arm=unit.arm,
                treatment_probability=unit.treatment_probability,
                assigned_at=unit.assigned_at,
                outcome_window_start=unit.outcome_window_start,
                outcome_window_end=unit.outcome_window_end,
            )
            for unit in units
        )

    def finalize_execution(self, execution_id):
        self.finalized_execution_ids.append(execution_id)

    def list_units_by_execution(self, execution_id):
        return [
            unit
            for unit in self.units
            if unit.segment_assignment_execution_id == execution_id
        ]


class _Repos:
    def __init__(self, assignments, executions):
        self.assignments = assignments
        self.executions = executions


def make_service(
    *,
    member_counts: tuple[int, ...],
    randomization_salt: str | None,
):
    outcome_spec, outcome_hash = build_frozen_outcome_spec(
        goal_metric="booking_conversion_rate",
        target_segment_rules=[],
    )
    run = PromotionRunRecord(
        promotion_run_id="run",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        analysis_id="analysis",
        generation_id="content_generation",
        loop_count=1,
        status="planned",
        goal_snapshot_json={
            "goal_metric": "booking_conversion_rate",
            "outcome_spec": outcome_spec,
            "outcome_spec_hash": outcome_hash,
        },
        segment_scope_json=tuple(
            f"segment_{index}" for index in range(len(member_counts))
        ),
        segment_scope_fingerprint="scope",
    )
    experiments = [
        AdExperimentRecord(
            ad_experiment_id=f"experiment_{index}",
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
            promotion_run_id="run",
            analysis_id="analysis",
            generation_id="content_generation",
            segment_id=f"segment_{index}",
            segment_name=f"Segment {index}",
            content_id=f"content_{index}",
            content_option_id=f"option_{index}",
            channel="email",
            loop_count=1,
            status="approved",
            goal_metric="booking_conversion_rate",
            goal_target_value=Decimal("0.1"),
            goal_basis="all_segments",
        )
        for index in range(len(member_counts))
    ]
    bindings = tuple(
        AudienceSnapshotBinding(
            segment_id=f"segment_{index}",
            audience_snapshot_id=f"snapshot_{index}",
            vector_generation_id=f"vector_generation_{index}",
            vector_version="hotel_behavior.v2",
            source_cutoff=CUTOFF,
            generation_window_end=CUTOFF,
            generation_source_revision_cutoff=CUTOFF,
            member_count=count,
        )
        for index, count in enumerate(member_counts)
    )
    members = [
        AudienceSnapshotMember(
            user_id=f"user_{segment_index}_{user_index:03d}",
            segment_id=f"segment_{segment_index}",
            behavior_fit_score=Decimal("0.8"),
        )
        for segment_index, count in enumerate(member_counts)
        for user_index in range(count)
    ]
    assignments = _AssignmentRepository()
    executions = _ExecutionRepository()
    service = SegmentAssignmentService(
        promotion_run_repository=_RunRepository(run),
        ad_experiment_repository=_ExperimentRepository(experiments),
        segment_vector_repository=None,
        user_behavior_vector_repository=None,
        user_segment_assignment_repository=assignments,
        reranker=SegmentCandidateReranker(),
        audience_snapshot_repository=_AudienceRepository(bindings, members),
        experiment_assignment_repository=executions,
        randomization_salt=randomization_salt,
    )
    return service, _Repos(assignments, executions)
