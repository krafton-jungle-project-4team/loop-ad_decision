from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWriter,
    PromotionEvaluationRecord,
    PromotionEvaluationWriter,
    PromotionReader,
    PromotionRunRecord,
    PromotionRunWriter,
)
from app.decision.schemas import (
    NextLoopRequest,
    NextLoopResponse,
    PromotionEvaluationStatus,
    RunCreateRequest,
    RunCreateResponse,
)


class NextLoopNotFoundError(Exception):
    pass


class NextLoopValidationError(Exception):
    pass


class NextLoopConflictError(Exception):
    pass


@dataclass(frozen=True)
class NextLoopAnalysisResult:
    analysis_id: str
    target_segment_ids: Sequence[str]


@dataclass(frozen=True)
class NextLoopGenerationResult:
    generation_id: str
    generated_segment_ids: Sequence[str]


class NextLoopAnalysisGateway(Protocol):
    def start_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        focus_segment_ids: Sequence[str],
        operator_instruction: str | None,
    ) -> NextLoopAnalysisResult:
        ...


class NextLoopGenerationGateway(Protocol):
    def start_generation(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        analysis_id: str,
        focus_segment_ids: Sequence[str],
        operator_instruction: str | None,
    ) -> NextLoopGenerationResult:
        ...


class PromotionRunCreator(Protocol):
    def create_run(
        self,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        ...


class UnavailableNextLoopAnalysisGateway:
    def start_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        focus_segment_ids: Sequence[str],
        operator_instruction: str | None,
    ) -> NextLoopAnalysisResult:
        _ = (
            project_id,
            campaign_id,
            promotion_id,
            focus_segment_ids,
            operator_instruction,
        )
        raise NextLoopValidationError(
            "next-loop analysis integration is not configured; Owner 1 "
            "focus_segment_ids support is required"
        )


class UnavailableNextLoopGenerationGateway:
    def start_generation(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        analysis_id: str,
        focus_segment_ids: Sequence[str],
        operator_instruction: str | None,
    ) -> NextLoopGenerationResult:
        _ = (
            project_id,
            campaign_id,
            promotion_id,
            analysis_id,
            focus_segment_ids,
            operator_instruction,
        )
        raise NextLoopValidationError(
            "next-loop generation integration is not configured; Owner 3 "
            "promotion_target_segments based generation is required"
        )


class NextLoopService:
    def __init__(
        self,
        *,
        promotion_repository: PromotionReader,
        promotion_run_repository: PromotionRunWriter,
        ad_experiment_repository: AdExperimentWriter,
        promotion_evaluation_repository: PromotionEvaluationWriter,
        promotion_run_creator: PromotionRunCreator,
        analysis_gateway: NextLoopAnalysisGateway,
        generation_gateway: NextLoopGenerationGateway,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._promotion_evaluation_repository = promotion_evaluation_repository
        self._promotion_run_creator = promotion_run_creator
        self._analysis_gateway = analysis_gateway
        self._generation_gateway = generation_gateway

    def create_next_loop(
        self,
        *,
        promotion_run_id: str,
        request: NextLoopRequest,
    ) -> NextLoopResponse:
        previous_run = self._get_previous_run(promotion_run_id)
        failed_segment_ids = _deduplicate_or_raise(
            request.failed_segment_ids,
            "failed_segment_ids",
        )
        failed_ad_experiment_ids = _deduplicate_or_raise(
            request.failed_ad_experiment_ids,
            "failed_ad_experiment_ids",
        )
        if not failed_segment_ids and not failed_ad_experiment_ids:
            return _no_op_response(previous_run)

        promotion = self._promotion_repository.get_by_id(previous_run.promotion_id)
        if promotion is None:
            raise NextLoopValidationError(
                "promotion for previous promotion_run was not found"
            )
        next_loop_count = previous_run.loop_count + 1
        if next_loop_count > promotion.max_loop_count:
            raise NextLoopValidationError("promotion max_loop_count exceeded")
        if self._promotion_run_repository.exists_for_promotion_loop(
            promotion_id=previous_run.promotion_id,
            loop_count=next_loop_count,
        ):
            raise NextLoopConflictError(
                "promotion_run already exists for next loop_count"
            )

        experiments = self._ad_experiment_repository.list_by_run(
            previous_run.promotion_run_id,
        )
        if not experiments:
            raise NextLoopValidationError(
                "previous promotion_run must have ad experiments"
            )
        _validate_failed_ids(
            failed_segment_ids=failed_segment_ids,
            failed_ad_experiment_ids=failed_ad_experiment_ids,
            experiments=experiments,
            evaluations=(
                self._promotion_evaluation_repository.list_latest_by_run_ad_experiments(
                    previous_run.promotion_run_id
                )
            ),
        )

        analysis_result = self._analysis_gateway.start_analysis(
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            focus_segment_ids=failed_segment_ids,
            operator_instruction=request.operator_instruction,
        )
        _validate_gateway_segments(
            label="analysis",
            expected_segment_ids=failed_segment_ids,
            actual_segment_ids=analysis_result.target_segment_ids,
        )
        generation_result = self._generation_gateway.start_generation(
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            analysis_id=analysis_result.analysis_id,
            focus_segment_ids=failed_segment_ids,
            operator_instruction=request.operator_instruction,
        )
        _validate_gateway_segments(
            label="generation",
            expected_segment_ids=failed_segment_ids,
            actual_segment_ids=generation_result.generated_segment_ids,
        )

        created_run = self._promotion_run_creator.create_run(
            promotion_id=previous_run.promotion_id,
            request=RunCreateRequest(
                analysis_id=analysis_result.analysis_id,
                generation_id=generation_result.generation_id,
                loop_count=next_loop_count,
            ),
        )
        _validate_gateway_segments(
            label="created ad_experiments",
            expected_segment_ids=failed_segment_ids,
            actual_segment_ids=[
                experiment.segment_id for experiment in created_run.ad_experiments
            ],
        )

        return NextLoopResponse(
            previous_promotion_run_id=previous_run.promotion_run_id,
            next_promotion_run_id=created_run.promotion_run_id,
            promotion_id=previous_run.promotion_id,
            loop_count=created_run.loop_count,
            next_analysis_id=analysis_result.analysis_id,
            next_generation_id=generation_result.generation_id,
            next_ad_experiments=created_run.ad_experiments,
            status="created",
        )

    def _get_previous_run(self, promotion_run_id: str) -> PromotionRunRecord:
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            raise NextLoopNotFoundError(f"promotion_run not found: {promotion_run_id}")
        return run


def _no_op_response(run: PromotionRunRecord) -> NextLoopResponse:
    return NextLoopResponse(
        previous_promotion_run_id=run.promotion_run_id,
        next_promotion_run_id=None,
        promotion_id=run.promotion_id,
        loop_count=run.loop_count,
        next_analysis_id=None,
        next_generation_id=None,
        next_ad_experiments=[],
        status="no_op",
    )


def _deduplicate_or_raise(values: Sequence[str], field_name: str) -> list[str]:
    cleaned = [str(value).strip() for value in values]
    if any(not value for value in cleaned):
        raise NextLoopValidationError(f"{field_name} must not contain empty values")
    if len(set(cleaned)) != len(cleaned):
        raise NextLoopValidationError(f"{field_name} must not contain duplicates")
    return cleaned


def _validate_failed_ids(
    *,
    failed_segment_ids: Sequence[str],
    failed_ad_experiment_ids: Sequence[str],
    experiments: Sequence[AdExperimentRecord],
    evaluations: Sequence[PromotionEvaluationRecord],
) -> list[AdExperimentRecord]:
    if not failed_segment_ids or not failed_ad_experiment_ids:
        raise NextLoopValidationError(
            "failed_segment_ids and failed_ad_experiment_ids must be provided together"
        )

    experiments_by_id = {
        experiment.ad_experiment_id: experiment for experiment in experiments
    }
    experiments_by_segment = {experiment.segment_id: experiment for experiment in experiments}
    missing_ad_experiment_ids = [
        ad_experiment_id
        for ad_experiment_id in failed_ad_experiment_ids
        if ad_experiment_id not in experiments_by_id
    ]
    if missing_ad_experiment_ids:
        raise NextLoopValidationError(
            "failed_ad_experiment_ids must belong to the previous promotion_run"
        )
    missing_segment_ids = [
        segment_id
        for segment_id in failed_segment_ids
        if segment_id not in experiments_by_segment
    ]
    if missing_segment_ids:
        raise NextLoopValidationError(
            "failed_segment_ids must belong to the previous promotion_run"
        )

    selected_experiments = [
        experiments_by_id[ad_experiment_id]
        for ad_experiment_id in failed_ad_experiment_ids
    ]
    selected_segment_ids = [experiment.segment_id for experiment in selected_experiments]
    if set(selected_segment_ids) != set(failed_segment_ids):
        raise NextLoopValidationError(
            "failed_segment_ids must match failed_ad_experiment_ids"
        )

    latest_by_experiment_id = {
        str(evaluation.ad_experiment_id): evaluation
        for evaluation in evaluations
        if evaluation.ad_experiment_id is not None
    }
    for experiment in selected_experiments:
        evaluation = latest_by_experiment_id.get(experiment.ad_experiment_id)
        if evaluation is None:
            raise NextLoopValidationError(
                "latest goal_not_met evaluation is required for each failed ad_experiment"
            )
        if evaluation.status != PromotionEvaluationStatus.GOAL_NOT_MET.value:
            raise NextLoopValidationError(
                "only goal_not_met ad_experiments can enter next-loop"
            )
    return selected_experiments


def _validate_gateway_segments(
    *,
    label: str,
    expected_segment_ids: Sequence[str],
    actual_segment_ids: Sequence[str],
) -> None:
    expected = set(expected_segment_ids)
    actual = set(actual_segment_ids)
    if len(actual_segment_ids) != len(actual):
        raise NextLoopValidationError(f"{label} returned duplicate segment ids")
    if actual != expected:
        raise NextLoopValidationError(
            f"{label} result must contain only failed segment ids"
        )
