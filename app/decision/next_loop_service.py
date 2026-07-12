from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from app.analysis.service import (
    NextLoopFocusAnalysisRequest,
    PromotionAnalysisService,
    PromotionNotFoundError as AnalysisPromotionNotFoundError,
    SegmentSelectionError,
)
from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWriter,
    GenerationRunReader,
    NextLoopPreparationConflictError,
    NextLoopPreparationRecord,
    NextLoopPreparationWrite,
    NextLoopPreparationWriter,
    PromotionEvaluationRecord,
    PromotionEvaluationWriter,
    PromotionReader,
    PromotionRunRecord,
    PromotionRunWriter,
)
from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.schemas import (
    ContentApprovalMode,
    NextLoopPreparationStatus,
    NextLoopRequest,
    NextLoopResponse,
    PromotionEvaluationStatus,
    RunCreateRequest,
    RunCreateResponse,
)
from app.decision.service import (
    PromotionNotFoundError as RunPromotionNotFoundError,
    RunConflictError,
    RunValidationError,
)
from app.generation.schemas import ContentCandidateStatus
from app.generation.service import GenerationService, NextLoopFocusGenerationRequest
from app.logging import duration_ms, log, log_context_scope, now_ms

COMPLETED_STATUS = "completed"
MANUAL_CONTENT_OPTION_COUNT = 3


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
    status: str = COMPLETED_STATUS


class NextLoopAnalysisGateway(Protocol):
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
        loop_count: int,
        source_promotion_run_id: str,
        source_generation_id: str,
        operator_instruction: str | None,
    ) -> NextLoopGenerationResult:
        ...

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
        ...


class NextLoopContentCandidateReader(Protocol):
    def list_by_generation(self, generation_id: str) -> list[Mapping[str, Any]]:
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
        loop_count: int,
        source_promotion_run_id: str,
        source_failed_ad_experiment_ids: Sequence[str],
        operator_instruction: str | None,
    ) -> NextLoopAnalysisResult:
        _ = (
            project_id,
            campaign_id,
            promotion_id,
            focus_segment_ids,
            loop_count,
            source_promotion_run_id,
            source_failed_ad_experiment_ids,
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
        loop_count: int,
        source_promotion_run_id: str,
        source_generation_id: str,
        operator_instruction: str | None,
    ) -> NextLoopGenerationResult:
        _ = (
            project_id,
            campaign_id,
            promotion_id,
            analysis_id,
            focus_segment_ids,
            loop_count,
            source_promotion_run_id,
            source_generation_id,
            operator_instruction,
        )
        raise NextLoopValidationError(
            "next-loop generation integration is not configured; Owner 3 "
            "promotion_target_segments based generation is required"
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
        _ = (
            project_id,
            campaign_id,
            promotion_id,
            analysis_id,
            focus_segment_ids,
            loop_count,
            attempt_no,
            source_promotion_run_id,
            source_generation_id,
            operator_instruction,
        )
        raise NextLoopValidationError(
            "manual next-loop generation integration is not configured; Owner 3 "
            "multi-candidate draft focus generation is required"
        )


class ServiceNextLoopAnalysisGateway:
    def __init__(self, analysis_service: PromotionAnalysisService) -> None:
        self._analysis_service = analysis_service

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
    ) -> NextLoopAnalysisResult:
        try:
            result = self._analysis_service.analyze_focus(
                NextLoopFocusAnalysisRequest(
                    project_id=project_id,
                    campaign_id=campaign_id,
                    promotion_id=promotion_id,
                    focus_segment_ids=focus_segment_ids,
                    loop_count=loop_count,
                    source_promotion_run_id=source_promotion_run_id,
                    source_failed_ad_experiment_ids=source_failed_ad_experiment_ids,
                    operator_instruction=operator_instruction,
                )
            )
        except (AnalysisPromotionNotFoundError, SegmentSelectionError) as exc:
            raise NextLoopValidationError(str(exc)) from exc

        return NextLoopAnalysisResult(
            analysis_id=result.analysis.analysis_id,
            target_segment_ids=[
                target_segment.segment_id for target_segment in result.target_segments
            ],
        )


class ServiceNextLoopGenerationGateway:
    def __init__(self, generation_service: GenerationService) -> None:
        self._generation_service = generation_service

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
        try:
            result = self._generation_service.generate_focus(
                NextLoopFocusGenerationRequest(
                    project_id=project_id,
                    campaign_id=campaign_id,
                    promotion_id=promotion_id,
                    analysis_id=analysis_id,
                    focus_segment_ids=focus_segment_ids,
                    loop_count=loop_count,
                    source_promotion_run_id=source_promotion_run_id,
                    source_generation_id=source_generation_id,
                    operator_instruction=operator_instruction,
                )
            )
        except ValueError as exc:
            raise NextLoopValidationError(str(exc)) from exc

        return NextLoopGenerationResult(
            generation_id=result.generation_id,
            generated_segment_ids=result.generated_segment_ids,
            status=result.status.value,
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
        try:
            result = self._generation_service.generate_focus(
                NextLoopFocusGenerationRequest(
                    project_id=project_id,
                    campaign_id=campaign_id,
                    promotion_id=promotion_id,
                    analysis_id=analysis_id,
                    focus_segment_ids=focus_segment_ids,
                    loop_count=loop_count,
                    content_option_count=MANUAL_CONTENT_OPTION_COUNT,
                    attempt_no=attempt_no,
                    source_promotion_run_id=source_promotion_run_id,
                    source_generation_id=source_generation_id,
                    operator_instruction=operator_instruction,
                    candidate_status=ContentCandidateStatus.DRAFT,
                )
            )
        except ValueError as exc:
            raise NextLoopValidationError(str(exc)) from exc

        return NextLoopGenerationResult(
            generation_id=result.generation_id,
            generated_segment_ids=result.generated_segment_ids,
            status=result.status.value,
        )


class NextLoopService:
    def __init__(
        self,
        *,
        promotion_repository: PromotionReader,
        promotion_run_repository: PromotionRunWriter,
        ad_experiment_repository: AdExperimentWriter,
        promotion_evaluation_repository: PromotionEvaluationWriter,
        next_loop_preparation_repository: NextLoopPreparationWriter,
        generation_run_repository: GenerationRunReader,
        content_candidate_repository: NextLoopContentCandidateReader,
        analysis_gateway: NextLoopAnalysisGateway,
        generation_gateway: NextLoopGenerationGateway,
        run_creator: PromotionRunCreator,
        manual_prepare_enabled: bool = False,
        partial_segment_scope_enabled: bool = False,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._promotion_evaluation_repository = promotion_evaluation_repository
        self._next_loop_preparation_repository = next_loop_preparation_repository
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository
        self._analysis_gateway = analysis_gateway
        self._generation_gateway = generation_gateway
        self._run_creator = run_creator
        self._manual_prepare_enabled = manual_prepare_enabled
        self._partial_segment_scope_enabled = partial_segment_scope_enabled

    @log_context_scope
    def create_next_loop(
        self,
        *,
        promotion_run_id: str,
        request: NextLoopRequest,
    ) -> NextLoopResponse:
        if request.content_approval_mode is ContentApprovalMode.MANUAL:
            if not self._manual_prepare_enabled:
                raise NextLoopConflictError(
                    "manual next-loop preparation is disabled"
                )
            if not self._partial_segment_scope_enabled:
                raise NextLoopConflictError(
                    "partial segment scope is disabled until Dashboard scope lineage is ready"
                )
            return self._prepare_manual_next_loop(
                promotion_run_id=promotion_run_id,
                request=request,
            )
        return self._create_automatic_next_loop(
            promotion_run_id=promotion_run_id,
            request=request,
        )

    def _create_automatic_next_loop(
        self,
        *,
        promotion_run_id: str,
        request: NextLoopRequest,
    ) -> NextLoopResponse:
        started_at = now_ms()
        log.assign_context({"promotionRunId": promotion_run_id})
        log.info("started", {"promotionRunId": promotion_run_id, "request": request})
        previous_run = self._get_previous_run(promotion_run_id)
        log.assign_context(
            {
                "projectId": previous_run.project_id,
                "campaignId": previous_run.campaign_id,
                "promotionId": previous_run.promotion_id,
                "analysisId": previous_run.analysis_id,
                "generationId": previous_run.generation_id,
            }
        )
        log.info("promotion_run_loaded", {"promotionRun": previous_run})
        failed_segment_ids = sorted(
            _deduplicate_or_raise(
                request.failed_segment_ids,
                "failed_segment_ids",
            )
        )
        failed_ad_experiment_ids = _deduplicate_or_raise(
            request.failed_ad_experiment_ids,
            "failed_ad_experiment_ids",
        )
        if not failed_segment_ids and not failed_ad_experiment_ids:
            response = _no_op_response(previous_run)
            log.info("next_loop_skipped", {"reason": "no_failed_targets"})
            log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
            return response

        if not self._partial_segment_scope_enabled:
            raise NextLoopConflictError(
                "partial segment scope is disabled until Dashboard scope lineage is ready"
            )

        promotion = self._promotion_repository.get_by_id(previous_run.promotion_id)
        if promotion is None:
            log.warn("promotion_not_found", {"promotionId": previous_run.promotion_id})
            raise NextLoopValidationError(
                "promotion for previous promotion_run was not found"
            )
        next_loop_count = previous_run.loop_count + 1
        if next_loop_count > promotion.max_loop_count:
            log.warn("promotion_loop_limit_exceeded", {"loopCount": next_loop_count, "maxLoopCount": promotion.max_loop_count})
            raise NextLoopValidationError("promotion max_loop_count exceeded")
        experiments = self._ad_experiment_repository.list_by_run(
            previous_run.promotion_run_id,
        )
        if not experiments:
            log.warn("ad_experiments_empty", {"promotionRunId": previous_run.promotion_run_id})
            raise NextLoopValidationError(
                "previous promotion_run must have ad experiments"
            )
        log.info("ad_experiments_loaded", {"adExperimentCount": len(experiments)})
        if not set(failed_segment_ids).issubset(set(previous_run.segment_scope_json)):
            raise NextLoopValidationError(
                "failed_segment_ids must stay within the previous promotion_run scope"
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
            loop_count=next_loop_count,
            source_promotion_run_id=previous_run.promotion_run_id,
            source_failed_ad_experiment_ids=failed_ad_experiment_ids,
            operator_instruction=request.operator_instruction,
        )
        log.assign_context({"analysisId": analysis_result.analysis_id})
        log.info("next_loop_analysis_created", {"analysis": analysis_result})
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
            loop_count=next_loop_count,
            source_promotion_run_id=previous_run.promotion_run_id,
            source_generation_id=previous_run.generation_id,
            operator_instruction=request.operator_instruction,
        )
        log.assign_context({"generationId": generation_result.generation_id})
        log.info("next_loop_generation_created", {"generation": generation_result})
        _validate_generation_completed(generation_result)
        _validate_gateway_segments(
            label="generation",
            expected_segment_ids=failed_segment_ids,
            actual_segment_ids=generation_result.generated_segment_ids,
        )

        try:
            run_result = self._run_creator.create_run(
                promotion_id=previous_run.promotion_id,
                request=RunCreateRequest(
                    analysis_id=analysis_result.analysis_id,
                    generation_id=generation_result.generation_id,
                    segment_ids=failed_segment_ids,
                    loop_count=next_loop_count,
                ),
            )
        except (RunPromotionNotFoundError, RunValidationError) as exc:
            log.warn("next_loop_run_invalid", {"err": exc})
            raise NextLoopValidationError(str(exc)) from exc
        except RunConflictError as exc:
            log.warn("next_loop_run_conflict", {"err": exc})
            raise NextLoopConflictError(str(exc)) from exc

        _validate_gateway_segments(
            label="created ad_experiments",
            expected_segment_ids=failed_segment_ids,
            actual_segment_ids=[
                experiment.segment_id
                for experiment in run_result.ad_experiments
                if experiment.segment_id != FALLBACK_SEGMENT_ID
            ],
        )

        response = NextLoopResponse(
            previous_promotion_run_id=previous_run.promotion_run_id,
            next_promotion_run_id=run_result.promotion_run_id,
            promotion_id=run_result.promotion_id,
            loop_count=run_result.loop_count,
            next_analysis_id=run_result.analysis_id,
            next_generation_id=run_result.generation_id,
            next_ad_experiments=run_result.ad_experiments,
        )
        log.assign_context({"nextPromotionRunId": response.next_promotion_run_id})
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response

    def _prepare_manual_next_loop(
        self,
        *,
        promotion_run_id: str,
        request: NextLoopRequest,
    ) -> NextLoopResponse:
        failed_segment_ids = _normalize_manual_id_set(
            request.failed_segment_ids,
            "failed_segment_ids",
        )
        failed_ad_experiment_ids = _normalize_manual_id_set(
            request.failed_ad_experiment_ids,
            "failed_ad_experiment_ids",
        )
        if not failed_segment_ids or not failed_ad_experiment_ids:
            raise NextLoopValidationError(
                "manual next-loop requires failed_segment_ids and "
                "failed_ad_experiment_ids"
            )

        previous_run = self._get_previous_run(promotion_run_id)
        if not set(failed_segment_ids).issubset(
            set(previous_run.segment_scope_json)
        ):
            raise NextLoopValidationError(
                "failed_segment_ids must stay within the previous promotion_run scope"
            )
        promotion = self._promotion_repository.get_by_id(previous_run.promotion_id)
        if promotion is None:
            raise NextLoopValidationError(
                "promotion for previous promotion_run was not found"
            )
        next_loop_count = previous_run.loop_count + 1
        if next_loop_count > promotion.max_loop_count:
            raise NextLoopValidationError("promotion max_loop_count exceeded")
        experiments = self._ad_experiment_repository.list_by_run(
            previous_run.promotion_run_id
        )
        if not experiments:
            raise NextLoopValidationError(
                "previous promotion_run must have ad experiments"
            )
        selected_sources = _validate_failed_ids(
            failed_segment_ids=failed_segment_ids,
            failed_ad_experiment_ids=failed_ad_experiment_ids,
            experiments=experiments,
            evaluations=(
                self._promotion_evaluation_repository.list_latest_by_run_ad_experiments(
                    previous_run.promotion_run_id
                )
            ),
        )
        source_evaluation_ids = sorted(
            evaluation.evaluation_id for _experiment, evaluation in selected_sources
        )

        active = self._next_loop_preparation_repository.get_active_by_source_run(
            previous_run.promotion_run_id
        )
        if active is not None:
            return self._reuse_active_preparation(
                preparation=active,
                previous_run=previous_run,
                next_loop_count=next_loop_count,
                failed_segment_ids=failed_segment_ids,
                failed_ad_experiment_ids=failed_ad_experiment_ids,
                source_evaluation_ids=source_evaluation_ids,
                operator_instruction=request.operator_instruction,
            )

        attempt_no = self._next_loop_preparation_repository.get_next_attempt_no(
            previous_run.promotion_run_id
        )
        if attempt_no != 1:
            raise NextLoopConflictError(
                "next-loop replacement generation is not supported"
            )

        analysis_result = self._analysis_gateway.start_analysis(
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            focus_segment_ids=failed_segment_ids,
            loop_count=next_loop_count,
            source_promotion_run_id=previous_run.promotion_run_id,
            source_failed_ad_experiment_ids=failed_ad_experiment_ids,
            operator_instruction=request.operator_instruction,
        )
        _validate_gateway_segments(
            label="analysis",
            expected_segment_ids=failed_segment_ids,
            actual_segment_ids=analysis_result.target_segment_ids,
        )
        generation_result = self._generation_gateway.start_manual_generation(
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            analysis_id=analysis_result.analysis_id,
            focus_segment_ids=failed_segment_ids,
            loop_count=next_loop_count,
            attempt_no=attempt_no,
            source_promotion_run_id=previous_run.promotion_run_id,
            source_generation_id=previous_run.generation_id,
            operator_instruction=request.operator_instruction,
        )
        _validate_generation_completed(generation_result)
        _validate_gateway_segments(
            label="generation",
            expected_segment_ids=failed_segment_ids,
            actual_segment_ids=generation_result.generated_segment_ids,
        )
        pending_content_ids = _validate_generation_candidates(
            candidates=self._content_candidate_repository.list_by_generation(
                generation_result.generation_id
            ),
            generation_id=generation_result.generation_id,
            analysis_id=analysis_result.analysis_id,
            promotion_id=previous_run.promotion_id,
            expected_segment_ids=failed_segment_ids,
            require_draft_per_segment=True,
        )
        preparation_write = NextLoopPreparationWrite(
            next_loop_preparation_id=_next_loop_preparation_id(
                previous_run.promotion_run_id,
                attempt_no,
            ),
            source_promotion_run_id=previous_run.promotion_run_id,
            analysis_id=analysis_result.analysis_id,
            generation_id=generation_result.generation_id,
            attempt_no=attempt_no,
            failed_segment_ids_json=tuple(failed_segment_ids),
            failed_ad_experiment_ids_json=tuple(failed_ad_experiment_ids),
            source_evaluation_ids_json=tuple(source_evaluation_ids),
        )
        try:
            preparation = self._next_loop_preparation_repository.insert(
                preparation_write
            )
        except NextLoopPreparationConflictError as exc:
            raise NextLoopConflictError(
                "next-loop preparation already exists; retry the request"
            ) from exc
        return _manual_preparation_response(
            preparation=preparation,
            previous_run=previous_run,
            loop_count=next_loop_count,
            pending_content_ids=pending_content_ids,
        )

    def _reuse_active_preparation(
        self,
        *,
        preparation: NextLoopPreparationRecord,
        previous_run: PromotionRunRecord,
        next_loop_count: int,
        failed_segment_ids: Sequence[str],
        failed_ad_experiment_ids: Sequence[str],
        source_evaluation_ids: Sequence[str],
        operator_instruction: str | None,
    ) -> NextLoopResponse:
        if (
            set(preparation.failed_segment_ids_json) != set(failed_segment_ids)
            or set(preparation.failed_ad_experiment_ids_json)
            != set(failed_ad_experiment_ids)
            or set(preparation.source_evaluation_ids_json)
            != set(source_evaluation_ids)
        ):
            raise NextLoopConflictError(
                "active next-loop preparation has a different intent"
            )
        generation = self._generation_run_repository.get_by_id(
            preparation.generation_id
        )
        if generation is None:
            raise NextLoopValidationError(
                "active next-loop preparation generation was not found"
            )
        next_loop_context = generation.input_json.get("next_loop")
        if not isinstance(next_loop_context, Mapping):
            raise NextLoopValidationError(
                "active next-loop preparation generation context is invalid"
            )
        if (
            generation.analysis_id != preparation.analysis_id
            or generation.promotion_id != previous_run.promotion_id
            or int(next_loop_context.get("loop_count", 0)) != next_loop_count
            or str(next_loop_context.get("source_promotion_run_id", ""))
            != previous_run.promotion_run_id
            or _normalize_instruction(generation.operator_instruction)
            != _normalize_instruction(operator_instruction)
        ):
            raise NextLoopConflictError(
                "active next-loop preparation has a different intent"
            )
        pending_content_ids = _validate_generation_candidates(
            candidates=self._content_candidate_repository.list_by_generation(
                preparation.generation_id
            ),
            generation_id=preparation.generation_id,
            analysis_id=preparation.analysis_id,
            promotion_id=previous_run.promotion_id,
            expected_segment_ids=failed_segment_ids,
            require_draft_per_segment=False,
        )
        return _manual_preparation_response(
            preparation=preparation,
            previous_run=previous_run,
            loop_count=next_loop_count,
            pending_content_ids=pending_content_ids,
        )

    def _get_previous_run(self, promotion_run_id: str) -> PromotionRunRecord:
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            log.warn("promotion_run_not_found", {"promotionRunId": promotion_run_id})
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
    )


def _deduplicate_or_raise(values: Sequence[str], field_name: str) -> list[str]:
    cleaned = [str(value).strip() for value in values]
    if any(not value for value in cleaned):
        log.warn("next_loop_request_invalid", {"fieldName": field_name})
        raise NextLoopValidationError(f"{field_name} must not contain empty values")
    if len(set(cleaned)) != len(cleaned):
        log.warn("next_loop_request_conflict", {"fieldName": field_name})
        raise NextLoopValidationError(f"{field_name} must not contain duplicates")
    return cleaned


def _normalize_manual_id_set(
    values: Sequence[str],
    field_name: str,
) -> list[str]:
    cleaned = [str(value).strip() for value in values]
    if any(not value for value in cleaned):
        raise NextLoopValidationError(f"{field_name} must not contain empty values")
    return sorted(set(cleaned))


def _validate_failed_ids(
    *,
    failed_segment_ids: Sequence[str],
    failed_ad_experiment_ids: Sequence[str],
    experiments: Sequence[AdExperimentRecord],
    evaluations: Sequence[PromotionEvaluationRecord],
) -> list[tuple[AdExperimentRecord, PromotionEvaluationRecord]]:
    if not failed_segment_ids or not failed_ad_experiment_ids:
        log.warn("failed_target_mismatch", {"failedSegmentCount": len(failed_segment_ids), "failedAdExperimentCount": len(failed_ad_experiment_ids)})
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
        log.warn("failed_ad_experiments_invalid", {"missingAdExperimentIds": missing_ad_experiment_ids})
        raise NextLoopValidationError(
            "failed_ad_experiment_ids must belong to the previous promotion_run"
        )
    missing_segment_ids = [
        segment_id
        for segment_id in failed_segment_ids
        if segment_id not in experiments_by_segment
    ]
    if missing_segment_ids:
        log.warn("failed_segments_invalid", {"missingSegmentIds": missing_segment_ids})
        raise NextLoopValidationError(
            "failed_segment_ids must belong to the previous promotion_run"
        )

    selected_experiments = [
        experiments_by_id[ad_experiment_id]
        for ad_experiment_id in failed_ad_experiment_ids
    ]
    selected_segment_ids = [experiment.segment_id for experiment in selected_experiments]
    if set(selected_segment_ids) != set(failed_segment_ids):
        log.warn("failed_target_mismatch", {"failedSegmentIds": failed_segment_ids, "selectedSegmentIds": selected_segment_ids})
        raise NextLoopValidationError(
            "failed_segment_ids must match failed_ad_experiment_ids"
        )

    latest_by_experiment_id = {
        str(evaluation.ad_experiment_id): evaluation
        for evaluation in evaluations
        if evaluation.ad_experiment_id is not None
    }
    selected_sources: list[tuple[AdExperimentRecord, PromotionEvaluationRecord]] = []
    for experiment in selected_experiments:
        evaluation = latest_by_experiment_id.get(experiment.ad_experiment_id)
        if evaluation is None:
            log.warn("ad_experiment_evaluation_not_found", {"adExperimentId": experiment.ad_experiment_id})
            raise NextLoopValidationError(
                "latest goal_not_met evaluation is required for each failed ad_experiment"
            )
        if evaluation.status != PromotionEvaluationStatus.GOAL_NOT_MET.value:
            log.warn("ad_experiment_evaluation_invalid", {"adExperimentId": experiment.ad_experiment_id, "status": evaluation.status})
            raise NextLoopValidationError(
                "only goal_not_met ad_experiments can enter next-loop"
            )
        if (
            evaluation.promotion_run_id != experiment.promotion_run_id
            or evaluation.promotion_id != experiment.promotion_id
            or evaluation.segment_id != experiment.segment_id
        ):
            raise NextLoopValidationError(
                "latest evaluation must match the source ad_experiment and segment"
            )
        selected_sources.append((experiment, evaluation))
    return selected_sources


def _validate_generation_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    generation_id: str,
    analysis_id: str,
    promotion_id: str,
    expected_segment_ids: Sequence[str],
    require_draft_per_segment: bool,
) -> list[str]:
    if not candidates:
        raise NextLoopValidationError(
            "manual next-loop generation must create content candidates"
        )
    expected = set(expected_segment_ids)
    covered: set[str] = set()
    draft_segments: set[str] = set()
    pending: list[tuple[str, str, str]] = []
    for candidate in candidates:
        candidate_generation_id = str(candidate.get("generation_id", ""))
        candidate_analysis_id = str(candidate.get("analysis_id", ""))
        candidate_promotion_id = str(candidate.get("promotion_id", ""))
        segment_id = str(candidate.get("segment_id", ""))
        content_id = str(candidate.get("content_id", ""))
        content_option_id = str(candidate.get("content_option_id", ""))
        status = str(candidate.get("status", ""))
        if (
            candidate_generation_id != generation_id
            or candidate_analysis_id != analysis_id
            or candidate_promotion_id != promotion_id
            or segment_id not in expected
            or not content_id
            or not content_option_id
        ):
            raise NextLoopValidationError(
                "manual next-loop content candidates do not match the generation"
            )
        covered.add(segment_id)
        if status == "draft":
            draft_segments.add(segment_id)
            pending.append((segment_id, content_option_id, content_id))
    if covered != expected:
        raise NextLoopValidationError(
            "manual next-loop content candidates must cover every failed segment"
        )
    if require_draft_per_segment and draft_segments != expected:
        raise NextLoopValidationError(
            "manual next-loop generation must create draft content for every segment"
        )
    return [content_id for _segment, _option, content_id in sorted(pending)]


def _next_loop_preparation_id(source_promotion_run_id: str, attempt_no: int) -> str:
    digest = hashlib.sha256(
        f"{source_promotion_run_id}:{attempt_no}".encode("utf-8")
    ).hexdigest()[:32]
    return f"nlprep_{digest}"


def _normalize_instruction(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _manual_preparation_response(
    *,
    preparation: NextLoopPreparationRecord,
    previous_run: PromotionRunRecord,
    loop_count: int,
    pending_content_ids: Sequence[str],
) -> NextLoopResponse:
    return NextLoopResponse(
        status=NextLoopPreparationStatus.AWAITING_CONTENT_APPROVAL,
        content_approval_required=True,
        next_loop_preparation_id=preparation.next_loop_preparation_id,
        previous_promotion_run_id=previous_run.promotion_run_id,
        next_promotion_run_id=None,
        promotion_id=previous_run.promotion_id,
        loop_count=loop_count,
        next_analysis_id=preparation.analysis_id,
        next_generation_id=preparation.generation_id,
        pending_content_ids=list(pending_content_ids),
        next_ad_experiments=[],
    )


def _validate_gateway_segments(
    *,
    label: str,
    expected_segment_ids: Sequence[str],
    actual_segment_ids: Sequence[str],
) -> None:
    expected = set(expected_segment_ids)
    actual = set(actual_segment_ids)
    if len(actual_segment_ids) != len(actual):
        log.warn("gateway_segments_conflict", {"label": label, "actualSegmentIds": actual_segment_ids})
        raise NextLoopValidationError(f"{label} returned duplicate segment ids")
    if actual != expected:
        log.warn(
            "gateway_segments_mismatch",
            {
                "label": label,
                "expectedSegmentIds": expected_segment_ids,
                "actualSegmentIds": actual_segment_ids,
            },
        )
        raise NextLoopValidationError(
            f"{label} result must contain only failed segment ids"
        )


def _validate_generation_completed(result: NextLoopGenerationResult) -> None:
    if result.status != COMPLETED_STATUS:
        log.warn("next_loop_generation_invalid", {"status": result.status})
        raise NextLoopValidationError(
            "next-loop generation result must be completed before run creation"
        )
