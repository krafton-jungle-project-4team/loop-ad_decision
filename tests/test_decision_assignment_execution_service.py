from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.decision.assignment_provenance import build_input_fingerprint
from app.decision.assignment_selector import (
    EXACT_MATCHER_STRATEGY,
    AssignmentMatcherSelector,
)
from app.decision.assignment_service import (
    MATCHER_VERSION,
    AssignmentAudienceScope,
    AssignmentBuildInput,
    AssignmentExperimentSet,
    AssignmentResultWriter,
    SegmentAssignmentService,
    SegmentAssignmentValidationError,
)
from app.decision.matcher import (
    FALLBACK_REASON_BELOW_THRESHOLD,
    FALLBACK_SEGMENT_ID,
    SegmentVector,
)
from app.decision.repositories import (
    AdExperimentRecord,
    PromotionRunRecord,
    UserBehaviorVectorRecord,
    UserSegmentAssignmentInsertRecord,
    UserSegmentAssignmentRunAggregateRecord,
    UserSegmentAssignmentWrite,
)
from app.decision.schemas import SegmentAssignmentBuildRequest


RUN_ID = "prun_execution_test_loop_1"
SOURCE_CUTOFF = datetime(2026, 7, 14, 1, 2, 3, 456789, tzinfo=UTC)


def test_first_call_creates_provisional_linked_rows_and_exact_final_manifest() -> None:
    service, loader, assignments, executions, ann_matcher = _make_harness()

    response = service.build_assignments(
        promotion_run_id=RUN_ID,
        request=SegmentAssignmentBuildRequest(
            user_ids=["user-1"],
            expires_in_days=7,
        ),
    )

    assert response.matching_mode == EXACT_MATCHER_STRATEGY
    assert response.assignment_count == 1
    assert response.ann_applied is False
    assert response.ann_not_applied_reason == "matcher_selected_exact"
    assert loader.source_cutoff_call_count == 1
    assert loader.page_read_count == 1
    assert ann_matcher.call_count == 0

    assert len(executions.provisional_calls) == 1
    assert len(executions.finalize_calls) == 1
    provisional = executions.provisional_calls[0]
    execution_id = provisional["segment_assignment_execution_id"]
    assert provisional["matcher_strategy"] == "provisional"
    assert assignments.inserted_writes[0].segment_assignment_execution_id == execution_id

    finalized = executions.executions_by_fingerprint[
        provisional["request_fingerprint"]
    ]
    assert finalized["matcher_strategy"] == EXACT_MATCHER_STRATEGY
    assert finalized["matcher_version"] == MATCHER_VERSION
    assert finalized["source_cutoff_at"] == SOURCE_CUTOFF
    manifest = finalized["input_manifest_json"]
    assert manifest["canonical_input"]["source_table"] == (
        "user_behavior_vector_revisions"
    )
    assert manifest["matcher"]["strategy"] == EXACT_MATCHER_STRATEGY
    assert manifest["matcher"]["matcher_version"] == MATCHER_VERSION
    assert manifest["result_summary"]["newly_linked_count"] == 1
    assert manifest["result_summary"]["reused_existing_count"] == 0
    assert executions.count_linked_assignments(
        promotion_run_id=RUN_ID,
        segment_assignment_execution_id=execution_id,
    ) == 1


def test_same_logical_retry_replays_stable_response_without_source_reads() -> None:
    service, loader, assignments, executions, _ann_matcher = _make_harness()
    request = SegmentAssignmentBuildRequest(
        user_ids=["user-1"],
        expires_in_days=7,
    )
    first = service.build_assignments(promotion_run_id=RUN_ID, request=request)
    cutoff_calls = loader.source_cutoff_call_count
    page_reads = loader.page_read_count
    inserted_count = len(assignments.inserted_writes)

    replayed = service.build_assignments(
        promotion_run_id=RUN_ID,
        request=SegmentAssignmentBuildRequest(
            user_ids=["user-1", "user-1"],
            expires_in_days=7,
        ),
    )

    assert replayed.model_dump(mode="json") == first.model_dump(mode="json")
    assert loader.source_cutoff_call_count == cutoff_calls
    assert loader.page_read_count == page_reads
    assert len(assignments.inserted_writes) == inserted_count
    assert len(executions.provisional_calls) == 1
    assert len(executions.finalize_calls) == 1


def test_expires_in_days_change_creates_a_distinct_execution() -> None:
    service, loader, assignments, executions, _ann_matcher = _make_harness()

    first = service.build_assignments(
        promotion_run_id=RUN_ID,
        request=SegmentAssignmentBuildRequest(
            user_ids=["user-1"],
            expires_in_days=7,
        ),
    )
    second = service.build_assignments(
        promotion_run_id=RUN_ID,
        request=SegmentAssignmentBuildRequest(
            user_ids=["user-1"],
            expires_in_days=8,
        ),
    )

    assert len(executions.executions_by_fingerprint) == 2
    assert len({call["request_fingerprint"] for call in executions.provisional_calls}) == 2
    assert len(executions.provisional_calls) == 2
    assert loader.source_cutoff_call_count == 2
    assert loader.page_read_count == 2
    assert len(assignments.inserted_writes) == 1
    assert first.assignment_count == second.assignment_count == 1
    second_execution = executions.executions_by_fingerprint[
        executions.provisional_calls[1]["request_fingerprint"]
    ]
    assert second_execution["input_manifest_json"]["result_summary"][
        "newly_linked_count"
    ] == 0
    assert second_execution["input_manifest_json"]["result_summary"][
        "reused_existing_count"
    ] == 1


def test_legacy_null_assignment_is_reused_untouched_with_fallback_signal() -> None:
    legacy = UserSegmentAssignmentInsertRecord(
        user_id="user-1",
        segment_id=FALLBACK_SEGMENT_ID,
        fallback=True,
        fallback_reason=FALLBACK_REASON_BELOW_THRESHOLD,
        similarity_score=Decimal("0.100000"),
        segment_assignment_execution_id=None,
    )
    service, _loader, assignments, executions, _ann_matcher = _make_harness(
        existing_records=[legacy]
    )

    response = service.build_assignments(
        promotion_run_id=RUN_ID,
        request=SegmentAssignmentBuildRequest(
            user_ids=["user-1"],
            expires_in_days=7,
        ),
    )

    assert response.assignment_count == 1
    assert response.fallback_count == 1
    assert response.batch_has_fallback is True
    assert response.fallback_rate == 1.0
    assert response.skipped_existing_count == 1
    assert assignments.rows["user-1"] is legacy
    assert assignments.rows["user-1"].segment_assignment_execution_id is None
    assert assignments.inserted_writes == []
    execution = next(iter(executions.executions_by_fingerprint.values()))
    summary = execution["input_manifest_json"]["result_summary"]
    assert summary["newly_linked_count"] == 0
    assert summary["reused_existing_count"] == 1
    assert summary["fallback_count"] == 1


def test_final_execution_rejects_linked_count_mismatch() -> None:
    service, _loader, _assignments, _executions, _ann_matcher = _make_harness(
        linked_count_override=0
    )

    with pytest.raises(
        SegmentAssignmentValidationError,
        match="linked assignment count",
    ):
        service.build_assignments(
            promotion_run_id=RUN_ID,
            request=SegmentAssignmentBuildRequest(
                user_ids=["user-1"],
                expires_in_days=7,
            ),
        )


def test_concurrent_insert_loser_replays_winner_without_reading_pages() -> None:
    winner_service, _winner_loader, _winner_assignments, winner_repo, _ann = (
        _make_harness()
    )
    request = SegmentAssignmentBuildRequest(
        user_ids=["user-1"],
        expires_in_days=7,
    )
    winner_response = winner_service.build_assignments(
        promotion_run_id=RUN_ID,
        request=request,
    )
    winner_execution = next(iter(winner_repo.executions_by_fingerprint.values()))

    loser_cutoff = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)
    loser_service, loser_loader, _assignments, loser_repo, loser_ann = _make_harness(
        source_cutoff=loser_cutoff,
        concurrent_winner=winner_execution,
        linked_count_override=1,
    )

    replayed = loser_service.build_assignments(
        promotion_run_id=RUN_ID,
        request=request,
    )

    assert replayed.model_dump(mode="json") == winner_response.model_dump(mode="json")
    assert loser_loader.source_cutoff_call_count == 1
    assert loser_loader.page_read_count == 0
    assert loser_ann.call_count == 0
    assert len(loser_repo.provisional_calls) == 1
    assert loser_repo.provisional_calls[0]["source_cutoff_at"] == loser_cutoff
    assert winner_execution["source_cutoff_at"] == SOURCE_CUTOFF
    assert loser_repo.finalize_calls == []
    assert loser_repo.last_returned_execution is winner_execution


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("canonical_input", "source_table", "other_table", "input selection"),
        (
            "canonical_input",
            "selection_version",
            "unknown.v2",
            "input selection",
        ),
        ("canonical_input", "vector_version", "v2", "vector version"),
        ("canonical_input", "vector_source", "other", "vector source"),
        ("canonical_input", "selection_mode", "live_keyset", "selection mode"),
        ("canonical_input", "user_count", 2, "user count"),
        ("canonical_input", "dimension", 63, "vector dimension"),
        ("result_summary", "assignment_mode", "live_keyset", "assignment mode"),
    ],
)
def test_replay_rejects_cross_field_incoherent_final_manifest(
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    service, _loader, _assignments, executions, _ann_matcher = _make_harness()
    request = SegmentAssignmentBuildRequest(
        user_ids=["user-1"],
        expires_in_days=7,
    )
    service.build_assignments(promotion_run_id=RUN_ID, request=request)
    execution = next(iter(executions.executions_by_fingerprint.values()))
    execution["input_manifest_json"][section][key] = value
    execution["input_fingerprint"] = build_input_fingerprint(
        execution["input_manifest_json"]
    )

    with pytest.raises(SegmentAssignmentValidationError, match=message):
        service.build_assignments(promotion_run_id=RUN_ID, request=request)


class _FakeInputLoader:
    page_size = 10_000

    def __init__(
        self,
        *,
        build_input: AssignmentBuildInput,
        pages: Sequence[Sequence[UserBehaviorVectorRecord]],
        source_cutoff: datetime,
    ) -> None:
        self.build_input = build_input
        self.pages = tuple(tuple(page) for page in pages)
        self.source_cutoff = source_cutoff
        self.source_cutoff_call_count = 0
        self.page_read_count = 0
        self.page_cutoffs: list[datetime] = []

    def load(
        self,
        *,
        promotion_run_id: str,
        request: SegmentAssignmentBuildRequest,
    ) -> AssignmentBuildInput:
        del request
        assert promotion_run_id == self.build_input.run.promotion_run_id
        return self.build_input

    def get_source_cutoff(self) -> datetime:
        self.source_cutoff_call_count += 1
        return self.source_cutoff

    def iter_user_record_pages(
        self,
        build_input: AssignmentBuildInput,
        *,
        source_cutoff_at: datetime,
    ) -> Iterator[list[UserBehaviorVectorRecord]]:
        assert build_input is self.build_input
        assert source_cutoff_at == self.source_cutoff
        self.page_read_count += 1
        self.page_cutoffs.append(source_cutoff_at)
        for page in self.pages:
            yield list(page)


class _NeverAnnMatcher:
    def __init__(self) -> None:
        self.call_count = 0

    def match_page(self, **_kwargs: object) -> Any:
        self.call_count += 1
        raise AssertionError("production exact-only policy must not call ANN")


class _FakeAssignmentRepository:
    def __init__(
        self,
        existing_records: Sequence[UserSegmentAssignmentInsertRecord] = (),
    ) -> None:
        self.rows = {record.user_id: record for record in existing_records}
        self.inserted_writes: list[UserSegmentAssignmentWrite] = []

    def list_existing_assignments(
        self,
        *,
        promotion_run_id: str,
        user_ids: Sequence[str],
    ) -> list[UserSegmentAssignmentInsertRecord]:
        assert promotion_run_id == RUN_ID
        return [self.rows[user_id] for user_id in sorted(set(user_ids)) if user_id in self.rows]

    def insert_many(
        self,
        assignments: Sequence[UserSegmentAssignmentWrite],
    ) -> list[UserSegmentAssignmentInsertRecord]:
        inserted: list[UserSegmentAssignmentInsertRecord] = []
        for assignment in assignments:
            if assignment.user_id in self.rows:
                continue
            self.inserted_writes.append(assignment)
            record = UserSegmentAssignmentInsertRecord(
                user_id=assignment.user_id,
                segment_id=assignment.segment_id,
                fallback=assignment.fallback,
                fallback_reason=assignment.fallback_reason,
                similarity_score=assignment.similarity_score,
                segment_assignment_execution_id=(
                    assignment.segment_assignment_execution_id
                ),
            )
            self.rows[assignment.user_id] = record
            inserted.append(record)
        return inserted

    def summarize_run(
        self,
        promotion_run_id: str,
    ) -> UserSegmentAssignmentRunAggregateRecord:
        assert promotion_run_id == RUN_ID
        return UserSegmentAssignmentRunAggregateRecord(
            assignment_count=len(self.rows),
            fallback_count=sum(record.fallback for record in self.rows.values()),
        )


class _FakeExecutionRepository:
    def __init__(
        self,
        *,
        assignment_repository: _FakeAssignmentRepository,
        concurrent_winner: Mapping[str, Any] | None = None,
        linked_count_override: int | None = None,
    ) -> None:
        self.assignment_repository = assignment_repository
        self.concurrent_winner = concurrent_winner
        self.linked_count_override = linked_count_override
        self.concurrent_insert_attempted = False
        self.executions_by_fingerprint: dict[str, dict[str, Any]] = {}
        self.provisional_calls: list[dict[str, Any]] = []
        self.finalize_calls: list[dict[str, Any]] = []
        self.last_returned_execution: Mapping[str, Any] | None = None

    def get_by_request(
        self,
        *,
        promotion_run_id: str,
        request_fingerprint: str,
    ) -> Mapping[str, Any] | None:
        assert promotion_run_id == RUN_ID
        if self.concurrent_winner is not None:
            if not self.concurrent_insert_attempted:
                return None
            assert self.concurrent_winner["request_fingerprint"] == request_fingerprint
            self.last_returned_execution = self.concurrent_winner
            return self.concurrent_winner
        execution = self.executions_by_fingerprint.get(request_fingerprint)
        self.last_returned_execution = execution
        return execution

    def insert_provisional(self, **values: Any) -> Mapping[str, Any] | None:
        call = dict(values)
        self.provisional_calls.append(call)
        if self.concurrent_winner is not None:
            self.concurrent_insert_attempted = True
            return None
        execution = {**call, "created_at": datetime.now(UTC)}
        self.executions_by_fingerprint[call["request_fingerprint"]] = execution
        return execution

    def finalize(self, **values: Any) -> Mapping[str, Any]:
        call = dict(values)
        self.finalize_calls.append(call)
        execution = self.executions_by_fingerprint[call["request_fingerprint"]]
        assert execution["segment_assignment_execution_id"] == call[
            "segment_assignment_execution_id"
        ]
        execution.update(call)
        return execution

    def count_linked_assignments(
        self,
        *,
        promotion_run_id: str,
        segment_assignment_execution_id: str,
    ) -> int:
        assert promotion_run_id == RUN_ID
        if self.linked_count_override is not None:
            return self.linked_count_override
        return sum(
            record.segment_assignment_execution_id
            == segment_assignment_execution_id
            for record in self.assignment_repository.rows.values()
        )


def _make_harness(
    *,
    existing_records: Sequence[UserSegmentAssignmentInsertRecord] = (),
    source_cutoff: datetime = SOURCE_CUTOFF,
    concurrent_winner: Mapping[str, Any] | None = None,
    linked_count_override: int | None = None,
) -> tuple[
    SegmentAssignmentService,
    _FakeInputLoader,
    _FakeAssignmentRepository,
    _FakeExecutionRepository,
    _NeverAnnMatcher,
]:
    assignments = _FakeAssignmentRepository(existing_records)
    executions = _FakeExecutionRepository(
        assignment_repository=assignments,
        concurrent_winner=concurrent_winner,
        linked_count_override=linked_count_override,
    )
    input_loader = _FakeInputLoader(
        build_input=_build_input(),
        pages=[[_user_record()]],
        source_cutoff=source_cutoff,
    )
    ann_matcher = _NeverAnnMatcher()
    service = SegmentAssignmentService(
        input_loader=input_loader,
        page_matcher=ann_matcher,
        result_writer=AssignmentResultWriter(
            user_segment_assignment_repository=assignments
        ),
        matcher_selector=AssignmentMatcherSelector(),
        execution_repository=executions,
    )
    return service, input_loader, assignments, executions, ann_matcher


def _build_input() -> AssignmentBuildInput:
    return AssignmentBuildInput(
        run=PromotionRunRecord(
            promotion_run_id=RUN_ID,
            project_id="project-1",
            campaign_id="campaign-1",
            promotion_id="promotion-1",
            analysis_id="analysis-1",
            generation_id="generation-1",
            loop_count=1,
            status="planned",
            goal_snapshot_json={"audience_scope": {"base": "user_behavior_vectors"}},
            segment_scope_json=["segment-1"],
            segment_scope_fingerprint="a" * 64,
        ),
        audience_scope=AssignmentAudienceScope(
            effective_vector_version="v1",
            effective_limit=None,
            source="batch_profile",
            user_ids=["user-1"],
        ),
        experiments=AssignmentExperimentSet(
            non_fallback=[_experiment("segment-1")],
            fallback=_experiment(FALLBACK_SEGMENT_ID),
        ),
        segment_vectors=(
            SegmentVector(
                segment_vector_id="segment-vector-1",
                segment_id="segment-1",
                vector_dim=64,
                embedding_values=_vector(),
            ),
        ),
    )


def _experiment(segment_id: str) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=f"experiment-{segment_id}",
        project_id="project-1",
        campaign_id="campaign-1",
        promotion_id="promotion-1",
        promotion_run_id=RUN_ID,
        analysis_id="analysis-1",
        generation_id="generation-1",
        segment_id=segment_id,
        segment_name=segment_id,
        content_id=f"content-{segment_id}",
        content_option_id=f"option-{segment_id}",
        channel="onsite_banner",
        loop_count=1,
        status="planned",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.030000"),
        goal_basis="all_segments",
    )


def _user_record() -> UserBehaviorVectorRecord:
    observed_at = datetime(2026, 7, 13, tzinfo=UTC)
    return UserBehaviorVectorRecord(
        project_id="project-1",
        user_id="user-1",
        vector_dim=64,
        vector_values=_vector(),
        vector_version="v1",
        source="batch_profile",
        window_start=observed_at,
        window_end=observed_at,
        updated_at=observed_at,
        vector_row_id="vector-row-1",
    )


def _vector() -> list[float]:
    return [1.0] + [0.0] * 63
