from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWrite,
    ContentCandidateRecord,
    GenerationRunRecord,
    NextLoopPreparationRecord,
    PromotionAnalysisRecord,
    PromotionEvaluationRecord,
    PromotionRecord,
    PromotionRunRecord,
    PromotionRunWrite,
    PromotionTargetSegmentRecord,
)
from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.schemas import (
    AdExperimentStatus,
    Channel,
    GoalBasis,
    GoalMetric,
    PromotionEvaluationStatus,
    PromotionRunStatus,
    RunCreateRequest,
)
from app.decision.service import (
    PromotionRunService,
    RunConflictError,
    RunValidationError,
    build_bounded_decision_id,
    build_segment_scope_fingerprint,
)


DEFAULT_LATEST = object()


def test_run_service_uses_latest_completed_analysis_and_generation() -> None:
    service, repos = make_service()

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(),
    )

    assert repos.analyses.latest_calls == ["promo_banner_001"]
    assert repos.generations.latest_calls == ["promo_banner_001"]
    assert repos.target_segments.calls == ["analysis_banner_001"]
    assert repos.target_segments.approved_calls == []
    assert response.analysis_id == "analysis_banner_001"
    assert response.generation_id == "generation_banner_001"
    assert response.status == PromotionRunStatus.PLANNED
    assert len(repos.runs.inserted) == 1
    assert repos.runs.inserted[0].goal_snapshot_json == {
        "source": "promotions",
        "promotion_id": "promo_banner_001",
        "channel": Channel.ONSITE_BANNER.value,
        "goal_metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
        "goal_target_value": "0.030000",
        "goal_basis": GoalBasis.ALL_SEGMENTS.value,
        "min_sample_size": 1000,
        "max_loop_count": 3,
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
    }
    assert repos.runs.inserted[0].segment_scope_json == ("seg_family_trip",)
    assert repos.runs.inserted[0].segment_scope_fingerprint == (
        build_segment_scope_fingerprint(["seg_family_trip"])
    )


def test_run_service_rejects_generation_for_different_analysis_without_writes() -> None:
    service, repos = make_service(
        generation=generation_record(analysis_id="analysis_other_001"),
    )

    with pytest.raises(RunValidationError, match="selected promotion analysis"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(
                analysis_id="analysis_banner_001",
                generation_id="generation_banner_001",
            ),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


@pytest.mark.parametrize(
    ("missing_dependency", "message"),
    [
        ("analysis", "completed promotion analysis"),
        ("generation", "completed generation run"),
    ],
)
def test_run_service_requires_latest_completed_dependencies(
    missing_dependency: str,
    message: str,
) -> None:
    service, repos = make_service(
        latest_analysis=None
        if missing_dependency == "analysis"
        else DEFAULT_LATEST,
        latest_generation=None
        if missing_dependency == "generation"
        else DEFAULT_LATEST,
    )

    with pytest.raises(RunValidationError, match=message):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_rejects_duplicate_promotion_loop_without_writes() -> None:
    service, repos = make_service(run_exists=True)

    with pytest.raises(RunConflictError, match="promotion_id and loop_count"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(loop_count=2),
        )

    assert repos.runs.exists_calls == [("promo_banner_001", 2)]
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


@pytest.mark.parametrize(
    "candidate_count",
    [0, 2],
)
def test_run_service_requires_exactly_one_content_candidate_per_segment(
    candidate_count: int,
) -> None:
    candidates = [
        content_candidate_record(content_id=f"content_family_{index:03d}")
        for index in range(candidate_count)
    ]
    service, repos = make_service(candidates=candidates)

    with pytest.raises(RunValidationError, match="exactly one approved or active"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_rejects_content_candidate_context_mismatch() -> None:
    service, repos = make_service(
        candidates=[
            content_candidate_record(project_id="other-project"),
        ],
    )

    with pytest.raises(RunValidationError, match="project_id"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_creates_run_and_fallback_ad_experiment() -> None:
    target_segments = [
        target_segment_record(
            segment_id="seg_family_trip",
            segment_name="Family hotel trip",
        ),
        target_segment_record(
            segment_id="seg_mobile_user",
            segment_name="Mobile hotel users",
        ),
    ]
    service, repos = make_service(
        target_segments=target_segments,
        candidates=[
            content_candidate_record(
                segment_id="seg_family_trip",
                content_id="content_family_001",
                content_option_id="family_option_a",
            ),
            content_candidate_record(
                segment_id="seg_mobile_user",
                content_id="content_mobile_001",
                content_option_id="mobile_option_a",
            ),
        ],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            analysis_id="analysis_banner_001",
            generation_id="generation_banner_001",
            loop_count=3,
        ),
    )

    assert len(repos.runs.inserted) == 1
    run = repos.runs.inserted[0]
    assert run.loop_count == 3
    assert run.status == PromotionRunStatus.PLANNED.value
    assert len(repos.ad_experiments.inserted_batches) == 1
    experiments = repos.ad_experiments.inserted_batches[0]
    assert [experiment.segment_id for experiment in experiments] == [
        "seg_family_trip",
        "seg_mobile_user",
        FALLBACK_SEGMENT_ID,
    ]
    assert all(experiment.loop_count == run.loop_count for experiment in experiments)
    assert all(
        experiment.status == AdExperimentStatus.PLANNED.value
        for experiment in experiments
    )
    assert experiments[0].content_id == "content_family_001"
    assert experiments[1].content_option_id == "mobile_option_a"
    assert experiments[2].content_id == "content_family_001"
    assert experiments[2].content_option_id == "family_option_a"
    assert response.promotion_run_id == run.promotion_run_id
    assert [item.segment_id for item in response.ad_experiments] == [
        "seg_family_trip",
        "seg_mobile_user",
        FALLBACK_SEGMENT_ID,
    ]


def test_run_service_creates_experiments_only_for_requested_approved_segments() -> None:
    target_segments = [
        target_segment_record(
            segment_id="seg_family_trip",
            segment_name="Family hotel trip",
            status="approved",
        ),
        target_segment_record(
            segment_id="seg_mobile_user",
            segment_name="Mobile hotel users",
            status="approved",
        ),
    ]
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_mobile_user"]),
        target_segments=target_segments,
        candidates=[
            content_candidate_record(
                segment_id="seg_family_trip",
                content_id="content_family_001",
                content_option_id="family_option_a",
            ),
            content_candidate_record(
                segment_id="seg_mobile_user",
                content_id="content_mobile_001",
                content_option_id="mobile_option_a",
            ),
        ],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            analysis_id="analysis_banner_001",
            generation_id="generation_banner_001",
            segment_ids=["seg_mobile_user"],
        ),
    )

    assert repos.target_segments.calls == []
    assert repos.target_segments.approved_calls == [
        ("analysis_banner_001", ["seg_mobile_user"])
    ]
    assert [experiment.segment_id for experiment in response.ad_experiments] == [
        "seg_mobile_user",
        FALLBACK_SEGMENT_ID,
    ]


def test_run_service_rejects_requested_segment_ids_that_are_not_approved_without_writes() -> None:
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_family_trip"]),
        target_segments=[target_segment_record(status="planned")],
    )

    with pytest.raises(RunValidationError, match="segment_ids"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(segment_ids=["seg_family_trip"]),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_rejects_generation_snapshot_mismatch_without_writes() -> None:
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_family_trip"]),
        target_segments=[
            target_segment_record(
                segment_id="seg_mobile_user",
                status="approved",
            )
        ],
    )

    with pytest.raises(RunValidationError, match="generation target_segment_ids snapshot"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(segment_ids=["seg_mobile_user"]),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_uses_dedicated_fallback_content_when_available() -> None:
    service, repos = make_service(
        target_segments=[
            target_segment_record(
                segment_id="seg_family_trip",
                segment_name="Family hotel trip",
            ),
        ],
        candidates=[
            content_candidate_record(
                segment_id="seg_family_trip",
                content_id="content_family_001",
                content_option_id="family_option_a",
            ),
            content_candidate_record(
                segment_id=FALLBACK_SEGMENT_ID,
                content_id="content_existing_all_001",
                content_option_id="existing_all_option_a",
            ),
        ],
    )

    service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(),
    )

    experiments = repos.ad_experiments.inserted_batches[0]
    assert [experiment.segment_id for experiment in experiments] == [
        "seg_family_trip",
        FALLBACK_SEGMENT_ID,
    ]
    assert experiments[1].content_id == "content_existing_all_001"
    assert experiments[1].content_option_id == "existing_all_option_a"


def test_run_service_creates_next_loop_run_from_approved_focus_candidate() -> None:
    analysis_id = "analysis_banner_001_loop_2"
    generation_id = "generation_banner_001_loop_2"
    service, repos = make_service(
        analysis=analysis_record(analysis_id=analysis_id),
        generation=generation_record(
            generation_id=generation_id,
            analysis_id=analysis_id,
        ),
        target_segments=[
            target_segment_record(
                analysis_id=analysis_id,
                segment_id="seg_luxury",
                segment_name="Luxury hotel users",
            ),
        ],
        candidates=[
            content_candidate_record(
                analysis_id=analysis_id,
                generation_id=generation_id,
                segment_id="seg_luxury",
                content_id="content_luxury_approved_001",
                content_option_id="luxury_option_a",
                status="approved",
            ),
        ],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            analysis_id=analysis_id,
            generation_id=generation_id,
            loop_count=2,
        ),
    )

    assert response.analysis_id == analysis_id
    assert response.generation_id == generation_id
    assert response.loop_count == 2
    assert len(repos.runs.inserted) == 1
    assert repos.runs.inserted[0].loop_count == 2
    experiments = repos.ad_experiments.inserted_batches[0]
    assert [experiment.segment_id for experiment in experiments] == [
        "seg_luxury",
        FALLBACK_SEGMENT_ID,
    ]
    assert experiments[0].content_id == "content_luxury_approved_001"
    assert experiments[1].content_id == "content_luxury_approved_001"


def source_promotion_run_record() -> PromotionRunRecord:
    return PromotionRunRecord(
        promotion_run_id="prun_source_loop_1",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        loop_count=1,
        status=PromotionRunStatus.GOAL_NOT_MET.value,
        goal_snapshot_json={},
        segment_scope_json=("seg_family_trip", "seg_mobile_user"),
        segment_scope_fingerprint=build_segment_scope_fingerprint(
            ["seg_family_trip", "seg_mobile_user"]
        ),
    )


def canonical_promotion_run_record() -> PromotionRunRecord:
    return PromotionRunRecord(
        promotion_run_id=build_bounded_decision_id(
            "prun",
            "promo_banner_001",
            "loop_2",
        ),
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_loop_2",
        generation_id="generation_banner_loop_2",
        loop_count=2,
        status=PromotionRunStatus.PLANNED.value,
        goal_snapshot_json={},
        segment_scope_json=("seg_family_trip", "seg_mobile_user"),
        segment_scope_fingerprint=build_segment_scope_fingerprint(
            ["seg_family_trip", "seg_mobile_user"]
        ),
    )


def preparation_record(
    *,
    status: str = "awaiting_content_approval",
    activated_promotion_run_id: str | None = None,
) -> NextLoopPreparationRecord:
    now = datetime(2026, 7, 13, tzinfo=UTC)
    return NextLoopPreparationRecord(
        next_loop_preparation_id="prep_loop_2",
        source_promotion_run_id="prun_source_loop_1",
        analysis_id="analysis_banner_loop_2",
        generation_id="generation_banner_loop_2",
        attempt_no=1,
        failed_segment_ids_json=("seg_family_trip", "seg_mobile_user"),
        failed_ad_experiment_ids_json=(
            "adexp_source_family",
            "adexp_source_mobile",
        ),
        source_evaluation_ids_json=(
            "eval_source_family",
            "eval_source_mobile",
        ),
        status=status,  # type: ignore[arg-type]
        activated_promotion_run_id=activated_promotion_run_id,
        created_at=now,
        updated_at=now,
    )


def activation_request() -> RunCreateRequest:
    return RunCreateRequest(
        analysis_id="analysis_banner_loop_2",
        generation_id="generation_banner_loop_2",
        segment_ids=["seg_family_trip", "seg_mobile_user"],
        loop_count=2,
        next_loop_preparation_id="prep_loop_2",
    )


def activation_candidates() -> list[ContentCandidateRecord]:
    return [
        content_candidate_record(
            analysis_id="analysis_banner_loop_2",
            generation_id="generation_banner_loop_2",
            segment_id="seg_family_trip",
            content_id="content_family_loop_2",
            status="approved",
        ),
        content_candidate_record(
            analysis_id="analysis_banner_loop_2",
            generation_id="generation_banner_loop_2",
            segment_id="seg_mobile_user",
            content_id="content_mobile_loop_2",
            content_option_id="mobile_option_a",
            status="active",
        ),
    ]


def source_ad_experiments() -> list[AdExperimentRecord]:
    return [
        ad_experiment_record(
            ad_experiment_id="adexp_source_family",
            promotion_run_id="prun_source_loop_1",
            analysis_id="analysis_banner_001",
            generation_id="generation_banner_001",
            segment_id="seg_family_trip",
            loop_count=1,
        ),
        ad_experiment_record(
            ad_experiment_id="adexp_source_mobile",
            promotion_run_id="prun_source_loop_1",
            analysis_id="analysis_banner_001",
            generation_id="generation_banner_001",
            segment_id="seg_mobile_user",
            loop_count=1,
        ),
    ]


def activation_evaluations() -> list[PromotionEvaluationRecord]:
    return [
        promotion_evaluation_record(
            evaluation_id="eval_source_family",
            ad_experiment_id="adexp_source_family",
            segment_id="seg_family_trip",
        ),
        promotion_evaluation_record(
            evaluation_id="eval_source_mobile",
            ad_experiment_id="adexp_source_mobile",
            segment_id="seg_mobile_user",
        ),
    ]


def canonical_ad_experiments(
    run: PromotionRunRecord,
) -> list[AdExperimentRecord]:
    return [
        ad_experiment_record(
            ad_experiment_id="adexp_child_family",
            promotion_run_id=run.promotion_run_id,
            analysis_id=run.analysis_id,
            generation_id=run.generation_id,
            segment_id="seg_family_trip",
            loop_count=run.loop_count,
            parent_ad_experiment_id="adexp_source_family",
            source_evaluation_id="eval_source_family",
        ),
        ad_experiment_record(
            ad_experiment_id="adexp_child_mobile",
            promotion_run_id=run.promotion_run_id,
            analysis_id=run.analysis_id,
            generation_id=run.generation_id,
            segment_id="seg_mobile_user",
            loop_count=run.loop_count,
            parent_ad_experiment_id="adexp_source_mobile",
            source_evaluation_id="eval_source_mobile",
        ),
        ad_experiment_record(
            ad_experiment_id="adexp_child_fallback",
            promotion_run_id=run.promotion_run_id,
            analysis_id=run.analysis_id,
            generation_id=run.generation_id,
            segment_id=FALLBACK_SEGMENT_ID,
            loop_count=run.loop_count,
        ),
    ]


def make_preparation_activation_service(
    *,
    candidates: list[ContentCandidateRecord] | None = None,
    source_run: PromotionRunRecord | None = None,
    source_experiments: list[AdExperimentRecord] | None = None,
    evaluations: list[PromotionEvaluationRecord] | None = None,
    preparation: NextLoopPreparationRecord | None = None,
    canonical_run: PromotionRunRecord | None = None,
    canonical_experiments: list[AdExperimentRecord] | None = None,
    manual_activation_enabled: bool = True,
) -> tuple[PromotionRunService, FakeRepositoryBundle]:
    resolved_source_run = source_run or source_promotion_run_record()
    runs = [resolved_source_run]
    if canonical_run is not None:
        runs.append(canonical_run)
    experiments = (
        source_ad_experiments()
        if source_experiments is None
        else source_experiments
    )
    if canonical_experiments is not None:
        experiments += canonical_experiments
    return make_service(
        analysis=analysis_record(analysis_id="analysis_banner_loop_2"),
        generation=generation_record(
            generation_id="generation_banner_loop_2",
            analysis_id="analysis_banner_loop_2",
            target_segment_ids=["seg_family_trip", "seg_mobile_user"],
        ),
        target_segments=[
            target_segment_record(
                analysis_id="analysis_banner_loop_2",
                segment_id="seg_family_trip",
                status="approved",
            ),
            target_segment_record(
                analysis_id="analysis_banner_loop_2",
                segment_id="seg_mobile_user",
                status="approved",
            ),
        ],
        candidates=activation_candidates() if candidates is None else candidates,
        promotion_runs=runs,
        ad_experiments=experiments,
        evaluations=(
            activation_evaluations() if evaluations is None else evaluations
        ),
        preparation=preparation or preparation_record(),
        manual_activation_enabled=manual_activation_enabled,
    )


def test_preparation_activation_is_disabled_by_default_without_writes() -> None:
    service, repos = make_preparation_activation_service(
        manual_activation_enabled=False,
    )

    with pytest.raises(RunConflictError, match="activation is disabled"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=activation_request(),
        )

    assert repos.preparations.locked_calls == []
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_preparation_activation_persists_segment_lineage_and_nullable_fallback() -> None:
    candidates = activation_candidates() + [
        content_candidate_record(
            analysis_id="analysis_banner_loop_2",
            generation_id="generation_banner_loop_2",
            segment_id="seg_family_trip",
            content_id="content_family_draft",
            status="draft",
        ),
        content_candidate_record(
            analysis_id="analysis_banner_loop_2",
            generation_id="generation_banner_loop_2",
            segment_id="seg_mobile_user",
            content_id="content_mobile_rejected",
            status="rejected",
        ),
    ]
    service, repos = make_preparation_activation_service(candidates=candidates)

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=activation_request(),
    )

    assert response.loop_count == 2
    experiments = repos.ad_experiments.inserted_batches[0]
    lineage = {
        experiment.segment_id: (
            experiment.parent_ad_experiment_id,
            experiment.source_evaluation_id,
        )
        for experiment in experiments
    }
    assert lineage == {
        "seg_family_trip": ("adexp_source_family", "eval_source_family"),
        "seg_mobile_user": ("adexp_source_mobile", "eval_source_mobile"),
        FALLBACK_SEGMENT_ID: (None, None),
    }
    assert repos.preparations.activated_calls == [
        ("prep_loop_2", response.promotion_run_id)
    ]


@pytest.mark.parametrize(
    "case",
    ["zero", "partial", "unexpected", "duplicate", "cross-generation"],
)
def test_preparation_activation_rejects_invalid_generation_candidate_scope(
    case: str,
) -> None:
    candidates = activation_candidates()
    if case == "zero":
        candidates = []
    elif case == "partial":
        candidates = candidates[:1]
    elif case == "unexpected":
        candidates.append(
            content_candidate_record(
                analysis_id="analysis_banner_loop_2",
                generation_id="generation_banner_loop_2",
                segment_id="seg_unexpected",
                content_id="content_unexpected",
            )
        )
    elif case == "duplicate":
        candidates.append(
            content_candidate_record(
                analysis_id="analysis_banner_loop_2",
                generation_id="generation_banner_loop_2",
                segment_id="seg_family_trip",
                content_id="content_family_duplicate",
            )
        )
    else:
        candidates = [
            replace(candidate, generation_id="generation_other")
            for candidate in candidates
        ]
    service, repos = make_preparation_activation_service(candidates=candidates)

    with pytest.raises(RunValidationError):
        service.create_run(
            promotion_id="promo_banner_001",
            request=activation_request(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []
    assert repos.preparations.activated_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "other-project"),
        ("campaign_id", "other-campaign"),
        ("promotion_id", "other-promotion"),
        ("promotion_run_id", "other-run"),
        ("segment_id", "seg_mobile_user"),
        ("ad_experiment_id", "adexp_source_mobile"),
        ("ad_experiment_id", None),
        ("status", PromotionEvaluationStatus.GOAL_MET.value),
        ("evaluation_id", "eval_stale_family"),
    ],
)
def test_preparation_activation_rejects_invalid_segment_evaluation_lineage(
    field: str,
    value: str | None,
) -> None:
    evaluations = activation_evaluations()
    evaluations[0] = replace(evaluations[0], **{field: value})
    service, repos = make_preparation_activation_service(evaluations=evaluations)

    with pytest.raises(RunValidationError):
        service.create_run(
            promotion_id="promo_banner_001",
            request=activation_request(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []
    assert repos.preparations.activated_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "other-project"),
        ("campaign_id", "other-campaign"),
        ("promotion_id", "other-promotion"),
        ("promotion_run_id", "other-run"),
        ("analysis_id", "other-analysis"),
        ("generation_id", "other-generation"),
        ("segment_id", "seg_mobile_user"),
        ("loop_count", 7),
    ],
)
def test_preparation_activation_rejects_invalid_parent_experiment_context(
    field: str,
    value: str | int,
) -> None:
    experiments = source_ad_experiments()
    experiments[0] = replace(experiments[0], **{field: value})
    service, repos = make_preparation_activation_service(
        source_experiments=experiments,
    )

    with pytest.raises(RunValidationError):
        service.create_run(
            promotion_id="promo_banner_001",
            request=activation_request(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []
    assert repos.preparations.activated_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("analysis_id", "other-analysis"),
        ("generation_id", "other-generation"),
        ("loop_count", 3),
        ("segment_ids", ["seg_family_trip"]),
    ],
)
def test_preparation_activation_rejects_request_context_mismatch(
    field: str,
    value: str | int | list[str],
) -> None:
    service, repos = make_preparation_activation_service()
    request = activation_request().model_copy(update={field: value})

    with pytest.raises(RunValidationError):
        service.create_run(
            promotion_id="promo_banner_001",
            request=request,
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []
    assert repos.preparations.activated_calls == []


def test_preparation_activation_rejects_rejected_preparation_without_writes() -> None:
    service, repos = make_preparation_activation_service(
        preparation=preparation_record(status="rejected"),
    )

    with pytest.raises(RunValidationError, match="rejected"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=activation_request(),
        )

    assert repos.promotions.calls == []
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_preparation_activation_rejects_foreign_source_run() -> None:
    source_run = replace(source_promotion_run_record(), project_id="other-project")
    service, repos = make_preparation_activation_service(source_run=source_run)

    with pytest.raises(RunValidationError, match="source promotion run"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=activation_request(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_activated_preparation_retry_returns_canonical_before_candidate_or_evaluation_reads() -> None:
    canonical_run = canonical_promotion_run_record()
    preparation = preparation_record(
        status="activated",
        activated_promotion_run_id=canonical_run.promotion_run_id,
    )
    service, repos = make_preparation_activation_service(
        preparation=preparation,
        canonical_run=canonical_run,
        canonical_experiments=canonical_ad_experiments(canonical_run),
        candidates=[],
        evaluations=[],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=activation_request(),
    )

    assert response.promotion_run_id == canonical_run.promotion_run_id
    assert repos.contents.calls == []
    assert repos.evaluations.list_calls == []
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []
    assert repos.preparations.activated_calls == []


def test_activated_preparation_with_missing_canonical_run_is_conflict_without_writes() -> None:
    preparation = preparation_record(
        status="activated",
        activated_promotion_run_id="prun_missing",
    )
    service, repos = make_preparation_activation_service(
        preparation=preparation,
    )

    with pytest.raises(RunConflictError, match="canonical promotion run"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=activation_request(),
        )

    assert repos.contents.calls == []
    assert repos.evaluations.list_calls == []
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_bounded_decision_id_is_stable_and_under_contract_length() -> None:
    long_promotion_id = "promo_" + ("very_long_hotel_campaign_" * 10)

    first = build_bounded_decision_id("prun", long_promotion_id, "loop_12")
    second = build_bounded_decision_id("prun", long_promotion_id, "loop_12")

    assert first == second
    assert first.startswith("prun_")
    assert len(first) <= 100


def test_segment_scope_fingerprint_is_stable_and_excludes_fallback() -> None:
    first = build_segment_scope_fingerprint(
        ["seg_mobile_user", FALLBACK_SEGMENT_ID, "seg_family_trip"]
    )
    second = build_segment_scope_fingerprint(
        ["seg_family_trip", "seg_mobile_user"]
    )

    assert first == second
    assert len(first) == 64


class FakePromotionRepository:
    def __init__(self, promotion: PromotionRecord | None) -> None:
        self.promotion = promotion
        self.calls: list[str] = []

    def get_by_id(self, promotion_id: str) -> PromotionRecord | None:
        self.calls.append(promotion_id)
        return self.promotion


class FakePromotionAnalysisRepository:
    def __init__(
        self,
        *,
        analysis: PromotionAnalysisRecord | None,
        latest: PromotionAnalysisRecord | None,
    ) -> None:
        self.analysis = analysis
        self.latest = latest
        self.get_calls: list[str] = []
        self.latest_calls: list[str] = []

    def get_by_id(self, analysis_id: str) -> PromotionAnalysisRecord | None:
        self.get_calls.append(analysis_id)
        return self.analysis

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> PromotionAnalysisRecord | None:
        self.latest_calls.append(promotion_id)
        return self.latest


class FakeGenerationRunRepository:
    def __init__(
        self,
        *,
        generation: GenerationRunRecord | None,
        latest: GenerationRunRecord | None,
    ) -> None:
        self.generation = generation
        self.latest = latest
        self.get_calls: list[str] = []
        self.latest_calls: list[str] = []

    def get_by_id(self, generation_id: str) -> GenerationRunRecord | None:
        self.get_calls.append(generation_id)
        return self.generation

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> GenerationRunRecord | None:
        self.latest_calls.append(promotion_id)
        return self.latest


class FakePromotionTargetSegmentRepository:
    def __init__(self, segments: list[PromotionTargetSegmentRecord]) -> None:
        self.segments = segments
        self.calls: list[str] = []
        self.approved_calls: list[tuple[str, list[str]]] = []

    def list_for_analysis(
        self,
        analysis_id: str,
    ) -> list[PromotionTargetSegmentRecord]:
        self.calls.append(analysis_id)
        return self.segments

    def list_approved_for_analysis(
        self,
        analysis_id: str,
        segment_ids: list[str],
    ) -> list[PromotionTargetSegmentRecord]:
        self.approved_calls.append((analysis_id, list(segment_ids)))
        requested_ids = set(segment_ids)
        return [
            segment
            for segment in self.segments
            if segment.segment_id in requested_ids and segment.status == "approved"
        ]


class FakeContentCandidateRepository:
    def __init__(self, candidates: list[ContentCandidateRecord]) -> None:
        self.candidates = candidates
        self.calls: list[str] = []

    def list_approved_or_active_for_generation(
        self,
        generation_id: str,
    ) -> list[ContentCandidateRecord]:
        self.calls.append(generation_id)
        return [
            candidate
            for candidate in self.candidates
            if candidate.generation_id == generation_id
            and candidate.status in {"approved", "active"}
        ]


class FakePromotionRunRepository:
    def __init__(
        self,
        *,
        exists: bool,
        records: list[PromotionRunRecord] | None = None,
    ) -> None:
        self.exists = exists
        self.records = {
            record.promotion_run_id: record for record in records or []
        }
        self.get_calls: list[str] = []
        self.exists_calls: list[tuple[str, int]] = []
        self.inserted: list[PromotionRunWrite] = []

    def insert(self, run: PromotionRunWrite) -> None:
        self.inserted.append(run)

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        self.get_calls.append(promotion_run_id)
        return self.records.get(promotion_run_id)

    def exists_for_promotion_loop(self, *, promotion_id: str, loop_count: int) -> bool:
        self.exists_calls.append((promotion_id, loop_count))
        return self.exists


class FakeAdExperimentRepository:
    def __init__(
        self,
        existing_segments: set[str] | None = None,
        records: list[AdExperimentRecord] | None = None,
    ) -> None:
        self.existing_segments = existing_segments or set()
        self.records = list(records or [])
        self.list_calls: list[str] = []
        self.exists_calls: list[tuple[str, str]] = []
        self.inserted_batches: list[list[AdExperimentWrite]] = []

    def insert_many(self, experiments: list[AdExperimentWrite]) -> None:
        self.inserted_batches.append(list(experiments))

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        self.list_calls.append(promotion_run_id)
        return [
            record
            for record in self.records
            if record.promotion_run_id == promotion_run_id
        ]

    def exists_for_run_segment(
        self,
        *,
        promotion_run_id: str,
        segment_id: str,
    ) -> bool:
        self.exists_calls.append((promotion_run_id, segment_id))
        return segment_id in self.existing_segments


class FakePromotionEvaluationRepository:
    def __init__(
        self,
        records: list[PromotionEvaluationRecord] | None = None,
    ) -> None:
        self.records = list(records or [])
        self.list_calls: list[str] = []

    def list_latest_by_run_ad_experiments(
        self,
        promotion_run_id: str,
    ) -> list[PromotionEvaluationRecord]:
        self.list_calls.append(promotion_run_id)
        return [
            record
            for record in self.records
            if record.promotion_run_id == promotion_run_id
        ]


class FakeNextLoopPreparationRepository:
    def __init__(
        self,
        record: NextLoopPreparationRecord | None = None,
    ) -> None:
        self.record = record
        self.locked_calls: list[str] = []
        self.activated_calls: list[tuple[str, str]] = []

    def get_by_id_for_update(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        self.locked_calls.append(next_loop_preparation_id)
        if (
            self.record is None
            or self.record.next_loop_preparation_id != next_loop_preparation_id
        ):
            return None
        return self.record

    def mark_activated(
        self,
        *,
        next_loop_preparation_id: str,
        activated_promotion_run_id: str,
    ) -> NextLoopPreparationRecord | None:
        self.activated_calls.append(
            (next_loop_preparation_id, activated_promotion_run_id)
        )
        if (
            self.record is None
            or self.record.next_loop_preparation_id != next_loop_preparation_id
            or self.record.status != "awaiting_content_approval"
        ):
            return None
        self.record = NextLoopPreparationRecord(
            **{
                **self.record.__dict__,
                "status": "activated",
                "activated_promotion_run_id": activated_promotion_run_id,
            }
        )
        return self.record


class FakeRepositoryBundle:
    def __init__(
        self,
        *,
        promotion: PromotionRecord | None = None,
        analysis: PromotionAnalysisRecord | None = None,
        latest_analysis: PromotionAnalysisRecord | None | object = DEFAULT_LATEST,
        generation: GenerationRunRecord | None = None,
        latest_generation: GenerationRunRecord | None | object = DEFAULT_LATEST,
        target_segments: list[PromotionTargetSegmentRecord] | None = None,
        candidates: list[ContentCandidateRecord] | None = None,
        run_exists: bool = False,
        promotion_runs: list[PromotionRunRecord] | None = None,
        ad_experiments: list[AdExperimentRecord] | None = None,
        evaluations: list[PromotionEvaluationRecord] | None = None,
        preparation: NextLoopPreparationRecord | None = None,
    ) -> None:
        resolved_analysis = analysis if analysis is not None else analysis_record()
        resolved_generation = (
            generation if generation is not None else generation_record()
        )
        self.promotions = FakePromotionRepository(promotion or promotion_record())
        self.analyses = FakePromotionAnalysisRepository(
            analysis=resolved_analysis,
            latest=(
                resolved_analysis
                if latest_analysis is DEFAULT_LATEST
                else latest_analysis
            ),
        )
        self.generations = FakeGenerationRunRepository(
            generation=resolved_generation,
            latest=(
                resolved_generation
                if latest_generation is DEFAULT_LATEST
                else latest_generation
            ),
        )
        self.target_segments = FakePromotionTargetSegmentRepository(
            target_segments
            if target_segments is not None
            else [
                target_segment_record(),
            ],
        )
        self.contents = FakeContentCandidateRepository(
            candidates
            if candidates is not None
            else [
                content_candidate_record(),
            ],
        )
        self.runs = FakePromotionRunRepository(
            exists=run_exists,
            records=promotion_runs,
        )
        self.ad_experiments = FakeAdExperimentRepository(
            records=ad_experiments,
        )
        self.evaluations = FakePromotionEvaluationRepository(evaluations)
        self.preparations = FakeNextLoopPreparationRepository(preparation)


def make_service(
    *,
    promotion: PromotionRecord | None = None,
    analysis: PromotionAnalysisRecord | None = None,
    latest_analysis: PromotionAnalysisRecord | None | object = DEFAULT_LATEST,
    generation: GenerationRunRecord | None = None,
    latest_generation: GenerationRunRecord | None | object = DEFAULT_LATEST,
    target_segments: list[PromotionTargetSegmentRecord] | None = None,
    candidates: list[ContentCandidateRecord] | None = None,
    run_exists: bool = False,
    promotion_runs: list[PromotionRunRecord] | None = None,
    ad_experiments: list[AdExperimentRecord] | None = None,
    evaluations: list[PromotionEvaluationRecord] | None = None,
    preparation: NextLoopPreparationRecord | None = None,
    manual_activation_enabled: bool = False,
) -> tuple[PromotionRunService, FakeRepositoryBundle]:
    repos = FakeRepositoryBundle(
        promotion=promotion,
        analysis=analysis,
        latest_analysis=latest_analysis,
        generation=generation,
        latest_generation=latest_generation,
        target_segments=target_segments,
        candidates=candidates,
        run_exists=run_exists,
        promotion_runs=promotion_runs,
        ad_experiments=ad_experiments,
        evaluations=evaluations,
        preparation=preparation,
    )
    return (
        PromotionRunService(
            promotion_repository=repos.promotions,
            promotion_analysis_repository=repos.analyses,
            promotion_target_segment_repository=repos.target_segments,
            generation_run_repository=repos.generations,
            content_candidate_repository=repos.contents,
            promotion_run_repository=repos.runs,
            ad_experiment_repository=repos.ad_experiments,
            promotion_evaluation_repository=repos.evaluations,
            next_loop_preparation_repository=repos.preparations,
            manual_activation_enabled=manual_activation_enabled,
        ),
        repos,
    )


def promotion_record() -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel=Channel.ONSITE_BANNER.value,
        goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        goal_target_value=Decimal("0.030000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
        min_sample_size=1000,
        max_loop_count=3,
    )


def analysis_record(
    *,
    analysis_id: str = "analysis_banner_001",
    status: str = "completed",
) -> PromotionAnalysisRecord:
    return PromotionAnalysisRecord(
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        focus_segment_ids_json=None,
        operator_instruction=None,
        input_snapshot_json={"promotion_id": "promo_banner_001"},
        profile_summary_json={"selected_segment_count": 1},
        output_json={"target_segment_count": 1},
        status=status,
    )


def generation_record(
    *,
    generation_id: str = "generation_banner_001",
    analysis_id: str = "analysis_banner_001",
    target_segment_ids: list[str] | None = None,
    status: str = "completed",
) -> GenerationRunRecord:
    input_json = {"analysis_id": analysis_id}
    if target_segment_ids is not None:
        input_json["target_segment_ids"] = target_segment_ids
    return GenerationRunRecord(
        generation_id=generation_id,
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        content_option_count=2,
        operator_instruction=None,
        input_json=input_json,
        output_json={"content_count": 1},
        generation_report_json={"status": "completed"},
        status=status,
    )


def target_segment_record(
    *,
    analysis_id: str = "analysis_banner_001",
    segment_id: str = "seg_family_trip",
    segment_name: str = "Family hotel trip",
    status: str = "planned",
) -> PromotionTargetSegmentRecord:
    return PromotionTargetSegmentRecord(
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        segment_id=segment_id,
        segment_name=segment_name,
        segment_vector_id=f"segvec_{segment_id}_v1",
        rule_json={"segment_id": segment_id},
        profile_json={"segment_id": segment_id},
        content_brief_json={"message_direction": "Highlight hotel benefits."},
        data_evidence_json={"event_count": 120},
        estimated_size=1200,
        priority="high",
        status=status,
    )


def content_candidate_record(
    *,
    content_id: str = "content_family_001",
    content_option_id: str = "family_option_a",
    analysis_id: str = "analysis_banner_001",
    generation_id: str = "generation_banner_001",
    segment_id: str = "seg_family_trip",
    project_id: str = "hotel-client-a",
    status: str = "approved",
) -> ContentCandidateRecord:
    return ContentCandidateRecord(
        content_id=content_id,
        content_option_id=content_option_id,
        generation_id=generation_id,
        analysis_id=analysis_id,
        project_id=project_id,
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        segment_id=segment_id,
        channel=Channel.ONSITE_BANNER.value,
        status=status,
    )


def ad_experiment_record(
    *,
    ad_experiment_id: str,
    promotion_run_id: str,
    analysis_id: str,
    generation_id: str,
    segment_id: str,
    loop_count: int,
    parent_ad_experiment_id: str | None = None,
    source_evaluation_id: str | None = None,
) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=ad_experiment_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id=promotion_run_id,
        analysis_id=analysis_id,
        generation_id=generation_id,
        segment_id=segment_id,
        segment_name=segment_id,
        content_id=f"content_{segment_id}",
        content_option_id=f"option_{segment_id}",
        channel=Channel.ONSITE_BANNER.value,
        loop_count=loop_count,
        status=AdExperimentStatus.PLANNED.value,
        goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        goal_target_value=Decimal("0.030000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
        parent_ad_experiment_id=parent_ad_experiment_id,
        source_evaluation_id=source_evaluation_id,
    )


def promotion_evaluation_record(
    *,
    evaluation_id: str,
    ad_experiment_id: str | None,
    segment_id: str | None,
) -> PromotionEvaluationRecord:
    return PromotionEvaluationRecord(
        evaluation_id=evaluation_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_source_loop_1",
        ad_experiment_id=ad_experiment_id,
        segment_id=segment_id,
        content_id="content_source",
        content_option_id="option_source",
        metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        target_value=Decimal("0.030000"),
        actual_value=Decimal("0.010000"),
        numerator_count=10,
        denominator_count=1000,
        sample_size=1000,
        basis="individual",
        status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
        feedback=None,
        next_loop_required=True,
        result_json={},
    )
