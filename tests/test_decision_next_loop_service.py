from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Sequence

import pytest

from app.decision.next_loop_service import (
    NextLoopAnalysisResult,
    NextLoopConflictError,
    NextLoopGenerationResult,
    NextLoopNotFoundError,
    NextLoopService,
    NextLoopValidationError,
)
from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.repositories import (
    AdExperimentRecord,
    GenerationRunRecord,
    NextLoopPreparationRecord,
    NextLoopPreparationWrite,
    PromotionEvaluationRecord,
    PromotionRecord,
    PromotionRunRecord,
)
from app.decision.schemas import (
    AdExperimentCreateResponse,
    AdExperimentStatus,
    Channel,
    ContentApprovalMode,
    NextLoopPreparationStatus,
    NextLoopRequest,
    PromotionEvaluationStatus,
    PromotionRunStatus,
    RunCreateRequest,
    RunCreateResponse,
)


def test_next_loop_prepares_focus_generation_for_goal_not_met_segments_only() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_luxury"],
            failed_ad_experiment_ids=["adexp_luxury_001"],
            operator_instruction="Emphasize breakfast benefits.",
        ),
    )

    assert "status" not in response.model_dump()
    assert response.previous_promotion_run_id == "prun_banner_001_loop_1"
    assert response.next_promotion_run_id == "prun_banner_001_loop_2"
    assert response.loop_count == 2
    assert response.segment_ids == ["seg_luxury"]
    assert response.next_analysis_id == "analysis_next_001"
    assert response.next_generation_id == "generation_next_001"
    assert [experiment.segment_id for experiment in response.next_ad_experiments] == [
        "seg_luxury"
    ]
    assert repos.analysis_gateway.calls == [
        (
            "hotel-client-a",
            "camp_summer_2026",
            "promo_banner_001",
            ("seg_luxury",),
            2,
            "prun_banner_001_loop_1",
            ("adexp_luxury_001",),
            "Emphasize breakfast benefits.",
            "approved",
        )
    ]
    assert repos.run_creator.calls == [
        (
            "promo_banner_001",
            "analysis_next_001",
            "generation_next_001",
            ("seg_luxury",),
            2,
        )
    ]
    assert repos.generation_gateway.calls == [
        (
            "hotel-client-a",
            "camp_summer_2026",
            "promo_banner_001",
            "analysis_next_001",
            ("seg_luxury",),
            2,
            "prun_banner_001_loop_1",
            "generation_banner_001",
            "Emphasize breakfast benefits.",
        )
    ]
    assert repos.preparations.calls == []


def test_next_loop_allows_created_fallback_ad_experiment() -> None:
    repos = FakeNextLoopRepos(
        run_creator=FakeRunCreator(
            created_segment_ids=["seg_luxury", FALLBACK_SEGMENT_ID],
        )
    )
    service = make_service(repos)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_luxury"],
            failed_ad_experiment_ids=["adexp_luxury_001"],
        ),
    )

    assert [experiment.segment_id for experiment in response.next_ad_experiments] == [
        "seg_luxury",
        FALLBACK_SEGMENT_ID,
    ]


def test_next_loop_noops_when_failed_ids_are_empty() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(),
    )

    assert "status" not in response.model_dump()
    assert response.next_promotion_run_id is None
    assert response.segment_ids == ["seg_family_trip", "seg_luxury"]
    assert response.next_ad_experiments == []
    assert repos.analysis_gateway.calls == []
    assert repos.generation_gateway.calls == []


@pytest.mark.parametrize(
    "status",
    [
        PromotionEvaluationStatus.GOAL_MET.value,
        PromotionEvaluationStatus.INSUFFICIENT_DATA.value,
        PromotionEvaluationStatus.PARTIAL_GOAL_MET.value,
    ],
)
def test_next_loop_allows_only_goal_not_met_evaluations(status: str) -> None:
    repos = FakeNextLoopRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=status,
            )
        ]
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="only goal_not_met"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_rejects_failed_ids_outside_previous_run() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="failed_segment_ids"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_missing"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_rejects_failed_segment_experiment_mismatch() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="must match"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_family_trip"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )


def test_next_loop_rejects_max_loop_count_exceeded() -> None:
    repos = FakeNextLoopRepos(
        run=promotion_run_record(loop_count=2),
        promotion=promotion_record(max_loop_count=2),
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="max_loop_count"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_2",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )


def test_next_loop_rejects_failed_segment_outside_stored_run_scope() -> None:
    repos = FakeNextLoopRepos(
        run=promotion_run_record(segment_scope_json=["seg_family_trip"]),
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="previous promotion_run scope"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )


def test_next_loop_rejects_gateway_segments_outside_failed_set() -> None:
    repos = FakeNextLoopRepos(
        analysis_gateway=FakeAnalysisGateway(target_segment_ids=["seg_luxury", "seg_spa"])
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="analysis result"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )

    assert repos.generation_gateway.calls == []


def test_next_loop_rejects_generation_segments_outside_failed_set() -> None:
    repos = FakeNextLoopRepos(
        generation_gateway=FakeGenerationGateway(
            generated_segment_ids=["seg_luxury", "seg_spa"],
        )
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="generation result"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_rejects_generation_that_is_not_completed() -> None:
    repos = FakeNextLoopRepos(
        generation_gateway=FakeGenerationGateway(status="failed"),
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="must be completed"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )


def test_next_loop_rejects_created_ad_experiments_outside_failed_set() -> None:
    repos = FakeNextLoopRepos(
        run_creator=FakeRunCreator(created_segment_ids=["seg_spa"]),
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="created ad_experiments result"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_prepares_focus_generation_for_multiple_failed_segments() -> None:
    repos = FakeNextLoopRepos(
        experiments=default_experiments(),
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
        ],
        analysis_gateway=FakeAnalysisGateway(
            target_segment_ids=["seg_family_trip", "seg_luxury"],
        ),
        generation_gateway=FakeGenerationGateway(
            generated_segment_ids=["seg_family_trip", "seg_luxury"],
        ),
        run_creator=FakeRunCreator(
            created_segment_ids=["seg_family_trip", "seg_luxury"],
        ),
    )
    service = make_service(repos)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_family_trip", "seg_luxury"],
            failed_ad_experiment_ids=[
                "adexp_family_trip_001",
                "adexp_luxury_001",
            ],
        ),
    )

    assert "status" not in response.model_dump()
    assert response.next_promotion_run_id == "prun_banner_001_loop_2"
    assert response.segment_ids == ["seg_family_trip", "seg_luxury"]
    assert {
        experiment.segment_id for experiment in response.next_ad_experiments
    } == {"seg_family_trip", "seg_luxury"}
    assert response.next_analysis_id == "analysis_next_001"
    assert response.next_generation_id == "generation_next_001"
    assert response.loop_count == 2


def test_next_loop_rejects_missing_previous_run() -> None:
    repos = FakeNextLoopRepos(run=None)
    service = make_service(repos)

    with pytest.raises(NextLoopNotFoundError):
        service.create_next_loop(
            promotion_run_id="missing",
            request=NextLoopRequest(),
        )


def test_manual_next_loop_switch_off_rejects_before_manual_dependencies() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    with pytest.raises(NextLoopConflictError, match="disabled"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.calls == []
    assert repos.analysis_gateway.calls == []
    assert repos.generation_gateway.manual_calls == []
    assert repos.run_creator.calls == []


def test_manual_next_loop_rejects_segment_outside_source_scope() -> None:
    repos = FakeNextLoopRepos(
        run=promotion_run_record(segment_scope_json=["seg_family_trip"]),
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="previous promotion_run scope"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.calls == []
    assert repos.analysis_gateway.calls == []
    assert repos.generation_gateway.manual_calls == []
    assert repos.run_creator.calls == []


def test_manual_next_loop_stores_multi_candidate_preparation_without_run() -> None:
    repos = FakeNextLoopRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
        ],
        analysis_gateway=FakeAnalysisGateway(
            target_segment_ids=["seg_family_trip", "seg_luxury"]
        ),
        generation_gateway=FakeGenerationGateway(
            generated_segment_ids=["seg_family_trip", "seg_luxury"]
        ),
        candidates=manual_candidates(["seg_family_trip", "seg_luxury"]),
    )
    service = make_service(repos, manual_enabled=True)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_luxury", "seg_family_trip", "seg_luxury"],
            failed_ad_experiment_ids=[
                "adexp_luxury_001",
                "adexp_family_trip_001",
                "adexp_luxury_001",
            ],
            operator_instruction="  Emphasize   breakfast benefits.  ",
            content_approval_mode=ContentApprovalMode.MANUAL,
        ),
    )

    assert response.status.value == "awaiting_content_approval"
    assert response.content_approval_required is True
    assert response.next_promotion_run_id is None
    assert response.segment_ids == ["seg_family_trip", "seg_luxury"]
    assert response.next_ad_experiments == []
    assert response.pending_content_ids == [
        "content_seg_family_trip_1",
        "content_seg_family_trip_2",
        "content_seg_luxury_1",
        "content_seg_luxury_2",
    ]
    assert repos.run_creator.calls == []
    inserted = repos.preparations.inserted[0]
    assert inserted.failed_segment_ids_json == ("seg_family_trip", "seg_luxury")
    assert inserted.failed_ad_experiment_ids_json == (
        "adexp_family_trip_001",
        "adexp_luxury_001",
    )
    assert inserted.attempt_no == 1
    assert inserted.next_loop_preparation_id.startswith("nlprep_")
    assert repos.analysis_gateway.calls[0][-1] == "planned"
    assert repos.generation_gateway.manual_calls[0][6] == 1


def test_manual_next_loop_reuses_active_preparation_for_same_set_intent() -> None:
    active = preparation_record()
    repos = FakeNextLoopRepos(
        active_preparation=active,
        generation=generation_record(operator_instruction="Emphasize breakfast"),
        candidates=manual_candidates(["seg_luxury"]),
    )
    service = make_service(repos, manual_enabled=True)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_luxury", "seg_luxury"],
            failed_ad_experiment_ids=["adexp_luxury_001", "adexp_luxury_001"],
            operator_instruction=" Emphasize   breakfast ",
            content_approval_mode=ContentApprovalMode.MANUAL,
        ),
    )

    assert response.next_loop_preparation_id == active.next_loop_preparation_id
    assert response.next_generation_id == active.generation_id
    assert response.pending_content_ids == [
        "content_seg_luxury_1",
        "content_seg_luxury_2",
    ]
    assert repos.preparations.inserted == []
    assert repos.analysis_gateway.calls == []
    assert repos.generation_gateway.manual_calls == []
    assert repos.run_creator.calls == []
    assert repos.candidates.calls == [
        ("list_for_update", "generation_next_001")
    ]


def test_manual_next_loop_reuses_rejected_and_remaining_draft_candidates() -> None:
    candidates = manual_candidates(["seg_luxury"])
    candidates[0]["status"] = "rejected"
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        generation=generation_record(),
        candidates=candidates,
    )
    service = make_service(repos, manual_enabled=True)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=manual_request(),
    )

    assert response.next_loop_preparation_id == "nlprep_existing_001"
    assert response.next_generation_id == "generation_next_001"
    assert response.pending_content_ids == ["content_seg_luxury_2"]
    assert repos.preparations.inserted == []
    assert repos.generation_gateway.manual_calls == []
    assert repos.candidates.calls == [
        ("list_for_update", "generation_next_001")
    ]


def test_manual_next_loop_keeps_approved_active_only_for_activation() -> None:
    candidates = manual_candidates(["seg_luxury"])
    candidates[0]["status"] = "approved"
    candidates[1]["status"] = "active"
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        generation=generation_record(),
        candidates=candidates,
    )
    service = make_service(repos, manual_enabled=True)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=manual_request(),
    )

    assert response.next_loop_preparation_id == "nlprep_existing_001"
    assert response.pending_content_ids == []
    assert response.status is NextLoopPreparationStatus.AWAITING_CONTENT_APPROVAL
    assert repos.preparations.inserted == []
    assert repos.generation_gateway.manual_calls == []


@pytest.mark.parametrize(
    ("operator_instruction", "expected_instruction"),
    [
        (None, "Emphasize breakfast"),
        ("   ", "Emphasize breakfast"),
        ("  Emphasize   spa access  ", "Emphasize spa access"),
    ],
)
def test_manual_next_loop_regenerates_full_scope_when_one_segment_is_exhausted(
    operator_instruction: str | None,
    expected_instruction: str,
) -> None:
    old_candidates = manual_candidates(["seg_family_trip", "seg_luxury"])
    for candidate in old_candidates:
        if candidate["segment_id"] == "seg_luxury":
            candidate["status"] = "rejected"
    old_candidates[0]["status"] = "approved"
    replacement_candidates = manual_candidates(
        ["seg_family_trip", "seg_luxury"],
        generation_id="generation_next_002",
        id_suffix="_attempt_2",
    )
    repos = FakeNextLoopRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
        ],
        active_preparation=preparation_record(
            segment_ids=("seg_family_trip", "seg_luxury"),
            ad_experiment_ids=(
                "adexp_family_trip_001",
                "adexp_luxury_001",
            ),
            evaluation_ids=(
                "eval_adexp_family_trip_001",
                "eval_adexp_luxury_001",
            ),
        ),
        next_attempt_no=2,
        generation=generation_record(
            operator_instruction="  Emphasize   breakfast ",
            segment_ids=("seg_family_trip", "seg_luxury"),
        ),
        generation_gateway=FakeGenerationGateway(
            generated_segment_ids=["seg_family_trip", "seg_luxury"]
        ),
        candidates=[*old_candidates, *replacement_candidates],
    )
    service = make_service(repos, manual_enabled=True)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_luxury", "seg_family_trip"],
            failed_ad_experiment_ids=[
                "adexp_luxury_001",
                "adexp_family_trip_001",
            ],
            operator_instruction=operator_instruction,
            content_approval_mode=ContentApprovalMode.MANUAL,
        ),
    )

    assert response.next_analysis_id == "analysis_next_001"
    assert response.next_generation_id == "generation_next_002"
    assert response.pending_content_ids == [
        "content_seg_family_trip_attempt_2_1",
        "content_seg_family_trip_attempt_2_2",
        "content_seg_luxury_attempt_2_1",
        "content_seg_luxury_attempt_2_2",
    ]
    assert repos.analysis_gateway.calls == []
    assert repos.generation_gateway.manual_calls == [
        (
            "hotel-client-a",
            "camp_summer_2026",
            "promo_banner_001",
            "analysis_next_001",
            ("seg_family_trip", "seg_luxury"),
            2,
            2,
            "prun_banner_001_loop_1",
            "generation_banner_001",
            expected_instruction,
        )
    ]
    assert [record.status for record in repos.preparations.history] == [
        "rejected",
        "awaiting_content_approval",
    ]
    assert repos.preparations.history[0].generation_id == "generation_next_001"
    assert repos.preparations.history[1].generation_id == "generation_next_002"
    assert repos.preparations.history[1].attempt_no == 2
    assert repos.preparations.history[1].failed_segment_ids_json == (
        "seg_family_trip",
        "seg_luxury",
    )


def test_manual_next_loop_retry_after_regeneration_reuses_attempt_two() -> None:
    old_candidates = manual_candidates(["seg_luxury"])
    for candidate in old_candidates:
        candidate["status"] = "rejected"
    replacement_candidates = manual_candidates(
        ["seg_luxury"],
        generation_id="generation_next_002",
        id_suffix="_attempt_2",
    )
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        next_attempt_no=2,
        generation=generation_record(operator_instruction="Breakfast"),
        candidates=[*old_candidates, *replacement_candidates],
    )
    service = make_service(repos, manual_enabled=True)
    request = NextLoopRequest(
        failed_segment_ids=["seg_luxury"],
        failed_ad_experiment_ids=["adexp_luxury_001"],
        operator_instruction="Breakfast",
        content_approval_mode=ContentApprovalMode.MANUAL,
    )

    first = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=request,
    )
    repos.generation_runs.generation = generation_record(
        operator_instruction="Breakfast",
        generation_id="generation_next_002",
        attempt_no=2,
    )
    second = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=request,
    )

    assert second == first
    assert first.next_generation_id == "generation_next_002"
    assert len(repos.generation_gateway.manual_calls) == 1
    assert len(repos.preparations.inserted) == 1


def test_manual_next_loop_rejects_non_continuous_replacement_attempt() -> None:
    candidates = manual_candidates(["seg_luxury"])
    for candidate in candidates:
        candidate["status"] = "rejected"
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        next_attempt_no=3,
        generation=generation_record(),
        candidates=candidates,
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopConflictError, match="not continuous"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.generation_gateway.manual_calls == []


def test_manual_next_loop_rejects_reused_replacement_candidate_ids() -> None:
    old_candidates = manual_candidates(["seg_luxury"])
    for candidate in old_candidates:
        candidate["status"] = "rejected"
    replacement_candidates = manual_candidates(
        ["seg_luxury"],
        generation_id="generation_next_002",
    )
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        next_attempt_no=2,
        generation=generation_record(),
        candidates=[*old_candidates, *replacement_candidates],
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="new content_id"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.inserted == []


def test_manual_next_loop_rejects_reused_replacement_generation_id() -> None:
    candidates = manual_candidates(["seg_luxury"])
    for candidate in candidates:
        candidate["status"] = "rejected"
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        next_attempt_no=2,
        generation=generation_record(),
        generation_gateway=FakeGenerationGateway(
            manual_generation_id="generation_next_001"
        ),
        candidates=candidates,
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="new generation_id"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.inserted == []


def test_manual_next_loop_rejects_reused_replacement_content_option_ids() -> None:
    old_candidates = manual_candidates(["seg_luxury"])
    for candidate in old_candidates:
        candidate["status"] = "rejected"
    replacement_candidates = manual_candidates(
        ["seg_luxury"],
        generation_id="generation_next_002",
        id_suffix="_attempt_2",
    )
    for option_index, candidate in enumerate(replacement_candidates, start=1):
        candidate["content_option_id"] = f"option_{option_index}"
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        next_attempt_no=2,
        generation=generation_record(),
        candidates=[*old_candidates, *replacement_candidates],
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="new content_option_id"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.inserted == []


@pytest.mark.parametrize("status", ["queued", "unknown"])
def test_manual_next_loop_rejects_unknown_candidate_status(status: str) -> None:
    candidates = manual_candidates(["seg_luxury"])
    candidates[0]["status"] = status
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        generation=generation_record(),
        candidates=candidates,
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="status is invalid"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.inserted == []


def test_manual_next_loop_rejects_missing_expected_segment_candidates() -> None:
    repos = FakeNextLoopRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
        ],
        active_preparation=preparation_record(
            segment_ids=("seg_family_trip", "seg_luxury"),
            ad_experiment_ids=(
                "adexp_family_trip_001",
                "adexp_luxury_001",
            ),
            evaluation_ids=(
                "eval_adexp_family_trip_001",
                "eval_adexp_luxury_001",
            ),
        ),
        generation=generation_record(
            segment_ids=("seg_family_trip", "seg_luxury")
        ),
        candidates=manual_candidates(["seg_family_trip"]),
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="cover every failed segment"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_family_trip", "seg_luxury"],
                failed_ad_experiment_ids=[
                    "adexp_family_trip_001",
                    "adexp_luxury_001",
                ],
                content_approval_mode=ContentApprovalMode.MANUAL,
            ),
        )

    assert repos.preparations.inserted == []


def test_manual_next_loop_never_reuses_rejected_preparation() -> None:
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(status="rejected"),
        generation=generation_record(),
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopConflictError, match="no longer awaiting"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.candidates.calls == []


def test_manual_next_loop_rejects_active_preparation_with_different_intent() -> None:
    repos = FakeNextLoopRepos(
        active_preparation=preparation_record(),
        generation=generation_record(operator_instruction="Breakfast"),
        candidates=manual_candidates(["seg_luxury"]),
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopConflictError, match="different intent"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
                operator_instruction="Pool benefit",
                content_approval_mode=ContentApprovalMode.MANUAL,
            ),
        )

    assert repos.run_creator.calls == []


def test_manual_next_loop_rejects_non_draft_generation_candidates() -> None:
    candidates = manual_candidates(["seg_luxury"])
    candidates[0]["status"] = "approved"
    candidates[1]["status"] = "rejected"
    repos = FakeNextLoopRepos(candidates=candidates)
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="draft content"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.inserted == []
    assert repos.run_creator.calls == []


def test_manual_next_loop_rejects_stale_or_aggregate_evaluation() -> None:
    aggregate = replace(
        default_evaluations()[1],
        evaluation_id="eval_aggregate",
        ad_experiment_id=None,
        segment_id=None,
    )
    repos = FakeNextLoopRepos(evaluations=[aggregate])
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="latest goal_not_met"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )


@pytest.mark.parametrize(
    "status",
    [
        PromotionEvaluationStatus.GOAL_MET.value,
        PromotionEvaluationStatus.INSUFFICIENT_DATA.value,
    ],
)
def test_manual_next_loop_rejects_latest_non_failure_evaluation(
    status: str,
) -> None:
    repos = FakeNextLoopRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=status,
            )
        ]
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="only goal_not_met"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.calls == []
    assert repos.analysis_gateway.calls == []


def test_manual_next_loop_rejects_evaluation_segment_mismatch() -> None:
    repos = FakeNextLoopRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            )
        ]
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="must match the source"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )


def test_manual_next_loop_rejects_candidate_provenance_mismatch() -> None:
    candidates = manual_candidates(["seg_luxury"])
    candidates[0]["analysis_id"] = "analysis_other"
    repos = FakeNextLoopRepos(candidates=candidates)
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="do not match the generation"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.inserted == []
    assert repos.run_creator.calls == []


def test_manual_next_loop_rejects_failed_generation_without_side_effects() -> None:
    repos = FakeNextLoopRepos(
        generation_gateway=FakeGenerationGateway(status="failed"),
    )
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopValidationError, match="must be completed"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.preparations.inserted == []
    assert repos.run_creator.calls == []


def test_manual_next_loop_does_not_create_replacement_attempt() -> None:
    repos = FakeNextLoopRepos(next_attempt_no=2)
    service = make_service(repos, manual_enabled=True)

    with pytest.raises(NextLoopConflictError, match="replacement generation"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=manual_request(),
        )

    assert repos.analysis_gateway.calls == []
    assert repos.generation_gateway.manual_calls == []


def make_service(
    repos: "FakeNextLoopRepos",
    *,
    manual_enabled: bool = False,
) -> NextLoopService:
    return NextLoopService(
        promotion_repository=repos.promotions,
        promotion_run_repository=repos.runs,
        ad_experiment_repository=repos.experiments,
        promotion_evaluation_repository=repos.evaluations,
        next_loop_preparation_repository=repos.preparations,
        generation_run_repository=repos.generation_runs,
        content_candidate_repository=repos.candidates,
        analysis_gateway=repos.analysis_gateway,
        generation_gateway=repos.generation_gateway,
        run_creator=repos.run_creator,
        manual_prepare_enabled=manual_enabled,
    )


class FakeNextLoopRepos:
    def __init__(
        self,
        *,
        run: PromotionRunRecord | None = None,
        promotion: PromotionRecord | None = None,
        experiments: list[AdExperimentRecord] | None = None,
        evaluations: list[PromotionEvaluationRecord] | None = None,
        analysis_gateway: "FakeAnalysisGateway" | None = None,
        generation_gateway: "FakeGenerationGateway" | None = None,
        run_creator: "FakeRunCreator" | None = None,
        active_preparation: NextLoopPreparationRecord | None = None,
        next_attempt_no: int = 1,
        generation: GenerationRunRecord | None = None,
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        self.promotions = FakePromotionRepository(promotion or promotion_record())
        self.runs = FakePromotionRunRepository(
            run if run is not None else promotion_run_record(),
        )
        self.experiments = FakeAdExperimentRepository(
            experiments if experiments is not None else default_experiments()
        )
        self.evaluations = FakePromotionEvaluationRepository(
            evaluations if evaluations is not None else default_evaluations()
        )
        self.preparations = FakeNextLoopPreparationRepository(
            active=active_preparation,
            next_attempt_no=next_attempt_no,
        )
        self.generation_runs = FakeGenerationRunRepository(
            generation or generation_record()
        )
        self.candidates = FakeContentCandidateRepository(
            candidates if candidates is not None else manual_candidates(["seg_luxury"])
        )
        self.analysis_gateway = analysis_gateway or FakeAnalysisGateway()
        self.generation_gateway = generation_gateway or FakeGenerationGateway()
        self.run_creator = run_creator or FakeRunCreator()


class FakePromotionRepository:
    def __init__(self, promotion: PromotionRecord | None) -> None:
        self.promotion = promotion

    def get_by_id(self, promotion_id: str) -> PromotionRecord | None:
        if self.promotion is None or self.promotion.promotion_id != promotion_id:
            return None
        return self.promotion


class FakePromotionRunRepository:
    def __init__(
        self,
        run: PromotionRunRecord | None,
    ) -> None:
        self.run = run

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        if self.run is None or self.run.promotion_run_id != promotion_run_id:
            return None
        return self.run

    def insert_if_absent(self, _run: object) -> bool:
        raise AssertionError("insert should be called through run creator")

    def update_status(self, *, promotion_run_id: str, status: str) -> None:
        _ = promotion_run_id, status


class FakeAdExperimentRepository:
    def __init__(self, experiments: list[AdExperimentRecord]) -> None:
        self.experiments = experiments

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        return [
            experiment
            for experiment in self.experiments
            if experiment.promotion_run_id == promotion_run_id
        ]

    def get_by_id(self, ad_experiment_id: str) -> AdExperimentRecord | None:
        return next(
            (
                experiment
                for experiment in self.experiments
                if experiment.ad_experiment_id == ad_experiment_id
            ),
            None,
        )

    def insert_many(self, _experiments: Sequence[object]) -> None:
        raise AssertionError("insert_many should be called through run creator")

    def exists_for_run_segment(
        self,
        *,
        promotion_run_id: str,
        segment_id: str,
    ) -> bool:
        _ = promotion_run_id, segment_id
        return False

    def update_status(self, *, ad_experiment_id: str, status: str) -> None:
        _ = ad_experiment_id, status


class FakePromotionEvaluationRepository:
    def __init__(self, evaluations: list[PromotionEvaluationRecord]) -> None:
        self.evaluations = evaluations

    def list_latest_by_run_ad_experiments(
        self,
        promotion_run_id: str,
    ) -> list[PromotionEvaluationRecord]:
        return [
            evaluation
            for evaluation in self.evaluations
            if evaluation.promotion_run_id == promotion_run_id
        ]

    def insert(self, _evaluation: object) -> None:
        raise AssertionError("B6 should not insert evaluations")


class FakeNextLoopPreparationRepository:
    def __init__(
        self,
        *,
        active: NextLoopPreparationRecord | None,
        next_attempt_no: int,
    ) -> None:
        self.active = active
        self.next_attempt_no = next_attempt_no
        self.calls: list[tuple[str, str]] = []
        self.inserted: list[NextLoopPreparationWrite] = []
        self.history: list[NextLoopPreparationRecord] = (
            [active] if active is not None else []
        )

    def get_active_by_source_run(
        self,
        source_promotion_run_id: str,
    ) -> NextLoopPreparationRecord | None:
        self.calls.append(("get_active", source_promotion_run_id))
        return self.active

    def get_next_attempt_no(self, source_promotion_run_id: str) -> int:
        self.calls.append(("get_next_attempt", source_promotion_run_id))
        return self.next_attempt_no

    def insert(self, write: NextLoopPreparationWrite) -> NextLoopPreparationRecord:
        self.calls.append(("insert", write.next_loop_preparation_id))
        self.inserted.append(write)
        self.active = NextLoopPreparationRecord(
            next_loop_preparation_id=write.next_loop_preparation_id,
            source_promotion_run_id=write.source_promotion_run_id,
            analysis_id=write.analysis_id,
            generation_id=write.generation_id,
            attempt_no=write.attempt_no,
            failed_segment_ids_json=write.failed_segment_ids_json,
            failed_ad_experiment_ids_json=write.failed_ad_experiment_ids_json,
            source_evaluation_ids_json=write.source_evaluation_ids_json,
            status="awaiting_content_approval",
            activated_promotion_run_id=None,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self.history.append(self.active)
        return self.active

    def get_by_id(self, _next_loop_preparation_id: str) -> None:
        return None

    def mark_rejected(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        self.calls.append(("mark_rejected", next_loop_preparation_id))
        if (
            self.active is None
            or self.active.next_loop_preparation_id != next_loop_preparation_id
            or self.active.status != "awaiting_content_approval"
        ):
            return None
        rejected = replace(
            self.active,
            status="rejected",
            updated_at=datetime.now(UTC),
        )
        self.history[-1] = rejected
        self.active = None
        return rejected

    def mark_activated(self, **_kwargs: str) -> None:
        return None


class FakeGenerationRunRepository:
    def __init__(self, generation: GenerationRunRecord) -> None:
        self.generation = generation

    def get_by_id(self, generation_id: str) -> GenerationRunRecord | None:
        if self.generation.generation_id == generation_id:
            return self.generation
        return None


class FakeContentCandidateRepository:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, str]] = []

    def list_by_generation(self, generation_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list", generation_id))
        return self._for_generation(generation_id)

    def list_by_generation_for_update(
        self,
        generation_id: str,
    ) -> list[dict[str, Any]]:
        self.calls.append(("list_for_update", generation_id))
        return self._for_generation(generation_id)

    def _for_generation(self, generation_id: str) -> list[dict[str, Any]]:
        return [
            candidate
            for candidate in self.candidates
            if candidate["generation_id"] == generation_id
        ]


class FakeAnalysisGateway:
    def __init__(self, target_segment_ids: list[str] | None = None) -> None:
        self.target_segment_ids = target_segment_ids or ["seg_luxury"]
        self.calls: list[
            tuple[
                str,
                str,
                str,
                tuple[str, ...],
                int,
                str,
                tuple[str, ...],
                str | None,
                str,
            ]
        ] = []

    def start_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        focus_segment_ids: Sequence[str],
        loop_count: int,
        source_promotion_run_id: str,
        source_failed_ad_experiment_ids: Sequence[str],
        operator_instruction: str | None,
        target_status: str,
    ) -> NextLoopAnalysisResult:
        self.calls.append(
            (
                project_id,
                campaign_id,
                promotion_id,
                tuple(focus_segment_ids),
                loop_count,
                source_promotion_run_id,
                tuple(source_failed_ad_experiment_ids),
                operator_instruction,
                target_status,
            )
        )
        return NextLoopAnalysisResult(
            analysis_id="analysis_next_001",
            target_segment_ids=self.target_segment_ids,
        )


class FakeGenerationGateway:
    def __init__(
        self,
        generated_segment_ids: list[str] | None = None,
        *,
        status: str = "completed",
        manual_generation_id: str | None = None,
    ) -> None:
        self.generated_segment_ids = generated_segment_ids or ["seg_luxury"]
        self.status = status
        self.manual_generation_id = manual_generation_id
        self.calls: list[
            tuple[str, str, str, str, tuple[str, ...], int, str, str, str | None]
        ] = []
        self.manual_calls: list[tuple[Any, ...]] = []

    def start_generation(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        analysis_id: str,
        focus_segment_ids: Sequence[str],
        loop_count: int,
        source_promotion_run_id: str,
        source_generation_id: str,
        operator_instruction: str | None,
    ) -> NextLoopGenerationResult:
        self.calls.append(
            (
                project_id,
                campaign_id,
                promotion_id,
                analysis_id,
                tuple(focus_segment_ids),
                loop_count,
                source_promotion_run_id,
                source_generation_id,
                operator_instruction,
            )
        )
        return NextLoopGenerationResult(
            generation_id="generation_next_001",
            generated_segment_ids=self.generated_segment_ids,
            status=self.status,
        )

    def start_manual_generation(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        analysis_id: str,
        focus_segment_ids: Sequence[str],
        loop_count: int,
        attempt_no: int,
        source_promotion_run_id: str,
        source_generation_id: str,
        operator_instruction: str | None,
    ) -> NextLoopGenerationResult:
        self.manual_calls.append(
            (
                project_id,
                campaign_id,
                promotion_id,
                analysis_id,
                tuple(focus_segment_ids),
                loop_count,
                attempt_no,
                source_promotion_run_id,
                source_generation_id,
                operator_instruction,
            )
        )
        return NextLoopGenerationResult(
            generation_id=(
                self.manual_generation_id
                or f"generation_next_{attempt_no:03d}"
            ),
            generated_segment_ids=self.generated_segment_ids,
            status=self.status,
        )


class FakeRunCreator:
    def __init__(self, created_segment_ids: list[str] | None = None) -> None:
        self.created_segment_ids = created_segment_ids or ["seg_luxury"]
        self.calls: list[
            tuple[str, str | None, str | None, tuple[str, ...], int]
        ] = []

    def create_run(
        self,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        self.calls.append(
            (
                promotion_id,
                request.analysis_id,
                request.generation_id,
                tuple(request.segment_ids or []),
                request.loop_count,
            )
        )
        return RunCreateResponse(
            promotion_run_id=f"prun_banner_001_loop_{request.loop_count}",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id=promotion_id,
            analysis_id=request.analysis_id or "analysis_next_001",
            generation_id=request.generation_id or "generation_next_001",
            loop_count=request.loop_count,
            status=PromotionRunStatus.PLANNED,
            goal_snapshot_json={
                "goal_metric": "booking_conversion_rate",
                "goal_target_value": "0.300000",
                "goal_basis": "all_segments",
                "min_sample_size": 10,
            },
            segment_ids=list(request.segment_ids or []),
            ad_experiments=[
                AdExperimentCreateResponse(
                    ad_experiment_id=f"adexp_{segment_id}_loop_{request.loop_count}",
                    segment_id=segment_id,
                    segment_name=f"Segment {segment_id}",
                    content_id=f"content_{segment_id}_next",
                    content_option_id=f"option_{segment_id}_next",
                    channel=Channel.ONSITE_BANNER,
                    loop_count=request.loop_count,
                    status=AdExperimentStatus.PLANNED,
                    is_fallback=segment_id == FALLBACK_SEGMENT_ID,
                )
                for segment_id in self.created_segment_ids
            ],
        )


def promotion_record(*, max_loop_count: int = 3) -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel=Channel.ONSITE_BANNER.value,
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.300000"),
        goal_basis="all_segments",
        min_sample_size=10,
        max_loop_count=max_loop_count,
    )


def promotion_run_record(
    *,
    loop_count: int = 1,
    segment_scope_json: list[str] | None = None,
) -> PromotionRunRecord:
    return PromotionRunRecord(
        promotion_run_id=f"prun_banner_001_loop_{loop_count}",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id=f"analysis_banner_00{loop_count}",
        generation_id=f"generation_banner_00{loop_count}",
        loop_count=loop_count,
        status=PromotionRunStatus.PARTIAL_GOAL_MET.value,
        goal_snapshot_json={
            "goal_metric": "booking_conversion_rate",
            "goal_target_value": "0.300000",
            "goal_basis": "all_segments",
            "min_sample_size": 10,
        },
        segment_scope_json=(
            segment_scope_json
            if segment_scope_json is not None
            else ["seg_family_trip", "seg_luxury"]
        ),
        segment_scope_fingerprint="a" * 64,
    )


def default_experiments() -> list[AdExperimentRecord]:
    return [
        ad_experiment_record(
            ad_experiment_id="adexp_family_trip_001",
            segment_id="seg_family_trip",
        ),
        ad_experiment_record(
            ad_experiment_id="adexp_luxury_001",
            segment_id="seg_luxury",
        ),
    ]


def ad_experiment_record(
    *,
    ad_experiment_id: str,
    segment_id: str,
) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=ad_experiment_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        segment_id=segment_id,
        segment_name=f"Segment {segment_id}",
        content_id=f"content_{segment_id}_001",
        content_option_id=f"option_{segment_id}_001",
        channel=Channel.ONSITE_BANNER.value,
        loop_count=1,
        status=AdExperimentStatus.GOAL_NOT_MET.value,
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.300000"),
        goal_basis="all_segments",
    )


def default_evaluations() -> list[PromotionEvaluationRecord]:
    return [
        evaluation_record(
            ad_experiment_id="adexp_family_trip_001",
            segment_id="seg_family_trip",
            status=PromotionEvaluationStatus.GOAL_MET.value,
        ),
        evaluation_record(
            ad_experiment_id="adexp_luxury_001",
            segment_id="seg_luxury",
            status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
        ),
    ]


def evaluation_record(
    *,
    ad_experiment_id: str,
    segment_id: str,
    status: str,
) -> PromotionEvaluationRecord:
    return PromotionEvaluationRecord(
        evaluation_id=f"eval_{ad_experiment_id}",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        ad_experiment_id=ad_experiment_id,
        segment_id=segment_id,
        content_id=f"content_{segment_id}_001",
        content_option_id=f"option_{segment_id}_001",
        metric="booking_conversion_rate",
        target_value=Decimal("0.300000"),
        actual_value=Decimal("0.100000"),
        numerator_count=1,
        denominator_count=10,
        sample_size=10,
        basis="all_segments",
        status=status,
        feedback=None,
        next_loop_required=status == PromotionEvaluationStatus.GOAL_NOT_MET.value,
        result_json={"status_reason": status},
    )


def manual_request() -> NextLoopRequest:
    return NextLoopRequest(
        failed_segment_ids=["seg_luxury"],
        failed_ad_experiment_ids=["adexp_luxury_001"],
        content_approval_mode=ContentApprovalMode.MANUAL,
    )


def preparation_record(
    *,
    generation_id: str = "generation_next_001",
    attempt_no: int = 1,
    segment_ids: Sequence[str] = ("seg_luxury",),
    ad_experiment_ids: Sequence[str] = ("adexp_luxury_001",),
    evaluation_ids: Sequence[str] = ("eval_adexp_luxury_001",),
    status: str = "awaiting_content_approval",
) -> NextLoopPreparationRecord:
    now = datetime.now(UTC)
    return NextLoopPreparationRecord(
        next_loop_preparation_id="nlprep_existing_001",
        source_promotion_run_id="prun_banner_001_loop_1",
        analysis_id="analysis_next_001",
        generation_id=generation_id,
        attempt_no=attempt_no,
        failed_segment_ids_json=tuple(segment_ids),
        failed_ad_experiment_ids_json=tuple(ad_experiment_ids),
        source_evaluation_ids_json=tuple(evaluation_ids),
        status=status,
        activated_promotion_run_id=None,
        created_at=now,
        updated_at=now,
    )


def generation_record(
    *,
    operator_instruction: str | None = None,
    generation_id: str = "generation_next_001",
    analysis_id: str = "analysis_next_001",
    segment_ids: Sequence[str] = ("seg_luxury",),
    attempt_no: int = 1,
) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_id=generation_id,
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        content_option_count=3,
        operator_instruction=operator_instruction,
        input_json={
            "next_loop": {
                "loop_count": 2,
                "source_promotion_run_id": "prun_banner_001_loop_1",
                "source_generation_id": "generation_banner_001",
                "focus_segment_ids": list(segment_ids),
                "attempt_no": attempt_no,
            }
        },
        output_json={},
        generation_report_json={},
        status="completed",
    )


def manual_candidates(
    segment_ids: Sequence[str],
    *,
    generation_id: str = "generation_next_001",
    analysis_id: str = "analysis_next_001",
    id_suffix: str = "",
) -> list[dict[str, Any]]:
    return [
        {
            "content_id": f"content_{segment_id}{id_suffix}_{option}",
            "content_option_id": f"option{id_suffix}_{option}",
            "generation_id": generation_id,
            "analysis_id": analysis_id,
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "segment_id": segment_id,
            "status": "draft",
        }
        for segment_id in segment_ids
        for option in (1, 2)
    ]
