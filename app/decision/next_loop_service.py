from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from app.analysis.service import (
    NextLoopFocusAnalysisRequest,
    PromotionAnalysisService,
    PromotionNotFoundError as AnalysisPromotionNotFoundError,
    SegmentSelectionError,
    TargetSegmentStatus,
)
from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWriter,
    GenerationRunReader,
    NextLoopGenerationAttemptRecord,
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
FAILED_STATUS = "failed"
MANUAL_CONTENT_OPTION_COUNT = 3
SELECTABLE_CANDIDATE_STATUSES = frozenset({"draft", "approved", "active"})
EXHAUSTED_CANDIDATE_STATUSES = frozenset({"rejected", "archived"})
KNOWN_CANDIDATE_STATUSES = (
    SELECTABLE_CANDIDATE_STATUSES | EXHAUSTED_CANDIDATE_STATUSES
)


class NextLoopNotFoundError(Exception):
    pass


class NextLoopValidationError(Exception):
    pass


class NextLoopGenerationFailedError(NextLoopValidationError):
    """A failed generation attempt whose diagnostics are ready to commit."""


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
    failure_persisted: bool = False


@dataclass(frozen=True)
class GenerationCandidateSnapshot:
    pending_content_ids: tuple[str, ...]
    selectable_segment_ids: frozenset[str]
    exhausted_segment_ids: frozenset[str]
    content_ids: frozenset[str]
    content_option_keys: frozenset[tuple[str, str]]


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
        attempt_no: int,
        operator_instruction: str | None,
        target_status: TargetSegmentStatus,
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
        attempt_no: int,
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

    def list_by_generation_for_update(
        self,
        generation_id: str,
    ) -> list[Mapping[str, Any]]:
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
        attempt_no: int,
        operator_instruction: str | None,
        target_status: TargetSegmentStatus,
    ) -> NextLoopAnalysisResult:
        _ = (
            project_id,
            campaign_id,
            promotion_id,
            focus_segment_ids,
            loop_count,
            source_promotion_run_id,
            source_failed_ad_experiment_ids,
            attempt_no,
            operator_instruction,
            target_status,
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
        attempt_no: int,
        operator_instruction: str | None,
        target_status: TargetSegmentStatus,
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
                    attempt_no=attempt_no,
                    operator_instruction=operator_instruction,
                ),
                target_status=target_status,
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
                    attempt_no=(attempt_no if attempt_no > 1 else None),
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
            failure_persisted=result.status.value == FAILED_STATUS,
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
            failure_persisted=result.status.value == FAILED_STATUS,
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
        attempt_no = _next_loop_generation_attempt_no(
            attempts=self._generation_run_repository.list_next_loop_generation_attempts(
                previous_run.promotion_run_id
            ),
            previous_run=previous_run,
            loop_count=next_loop_count,
            focus_segment_ids=failed_segment_ids,
            failed_ad_experiment_ids=failed_ad_experiment_ids,
            operator_instruction=request.operator_instruction,
            candidate_status=ContentCandidateStatus.APPROVED,
            content_option_count=1,
        )

        analysis_result = self._analysis_gateway.start_analysis(
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            focus_segment_ids=failed_segment_ids,
            loop_count=next_loop_count,
            source_promotion_run_id=previous_run.promotion_run_id,
            source_failed_ad_experiment_ids=failed_ad_experiment_ids,
            attempt_no=attempt_no,
            operator_instruction=request.operator_instruction,
            target_status="approved",
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
            attempt_no=attempt_no,
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
            segment_ids=run_result.segment_ids,
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

        next_preparation_attempt_no = (
            self._next_loop_preparation_repository.get_next_attempt_no(
                previous_run.promotion_run_id
            )
        )
        if next_preparation_attempt_no != 1:
            raise NextLoopConflictError(
                "next-loop replacement generation is not supported"
            )
        attempt_no = max(
            next_preparation_attempt_no,
            _next_loop_generation_attempt_no(
                attempts=self._generation_run_repository.list_next_loop_generation_attempts(
                    previous_run.promotion_run_id
                ),
                previous_run=previous_run,
                loop_count=next_loop_count,
                focus_segment_ids=failed_segment_ids,
                failed_ad_experiment_ids=failed_ad_experiment_ids,
                operator_instruction=request.operator_instruction,
                candidate_status=ContentCandidateStatus.DRAFT,
                content_option_count=MANUAL_CONTENT_OPTION_COUNT,
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
            attempt_no=attempt_no,
            operator_instruction=request.operator_instruction,
            target_status="planned",
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
        candidate_snapshot = _inspect_generation_candidates(
            candidates=self._content_candidate_repository.list_by_generation(
                generation_result.generation_id
            ),
            generation_id=generation_result.generation_id,
            analysis_id=analysis_result.analysis_id,
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
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
            pending_content_ids=candidate_snapshot.pending_content_ids,
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
        if preparation.status != "awaiting_content_approval":
            raise NextLoopConflictError(
                "next-loop preparation is no longer awaiting content approval"
            )
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
        expected_segment_ids = tuple(sorted(preparation.failed_segment_ids_json))
        if (
            generation.analysis_id != preparation.analysis_id
            or generation.promotion_id != previous_run.promotion_id
            or generation.project_id != previous_run.project_id
            or generation.campaign_id != previous_run.campaign_id
            or generation.status != COMPLETED_STATUS
            or _context_positive_int(next_loop_context.get("loop_count"))
            != next_loop_count
            or _context_positive_int(next_loop_context.get("attempt_no"))
            != preparation.attempt_no
            or str(next_loop_context.get("source_promotion_run_id", ""))
            != previous_run.promotion_run_id
            or str(next_loop_context.get("source_generation_id", ""))
            != previous_run.generation_id
            or _context_id_set(next_loop_context.get("focus_segment_ids"))
            != frozenset(expected_segment_ids)
        ):
            raise NextLoopConflictError(
                "active next-loop preparation has a different intent"
            )
        candidate_snapshot = _inspect_generation_candidates(
            candidates=self._content_candidate_repository.list_by_generation_for_update(
                preparation.generation_id
            ),
            generation_id=preparation.generation_id,
            analysis_id=preparation.analysis_id,
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            expected_segment_ids=expected_segment_ids,
            require_draft_per_segment=False,
        )
        if not candidate_snapshot.exhausted_segment_ids:
            if _normalize_instruction(generation.operator_instruction) != (
                _normalize_instruction(operator_instruction)
            ):
                raise NextLoopConflictError(
                    "active next-loop preparation has a different intent"
                )
            return _manual_preparation_response(
                preparation=preparation,
                previous_run=previous_run,
                loop_count=next_loop_count,
                pending_content_ids=candidate_snapshot.pending_content_ids,
            )

        effective_instruction = (
            _normalize_instruction(operator_instruction)
            or _normalize_instruction(generation.operator_instruction)
        )
        next_preparation_attempt_no = (
            self._next_loop_preparation_repository.get_next_attempt_no(
                previous_run.promotion_run_id
            )
        )
        if next_preparation_attempt_no != preparation.attempt_no + 1:
            raise NextLoopConflictError(
                "next-loop preparation attempt sequence is not continuous"
            )
        attempt_no = max(
            next_preparation_attempt_no,
            _next_loop_generation_attempt_no(
                attempts=self._generation_run_repository.list_next_loop_generation_attempts(
                    previous_run.promotion_run_id
                ),
                previous_run=previous_run,
                loop_count=next_loop_count,
                focus_segment_ids=expected_segment_ids,
                failed_ad_experiment_ids=(
                    preparation.failed_ad_experiment_ids_json
                ),
                operator_instruction=effective_instruction,
                candidate_status=ContentCandidateStatus.DRAFT,
                content_option_count=MANUAL_CONTENT_OPTION_COUNT,
                allowed_completed_generation_id=preparation.generation_id,
                operator_validation_after_attempt_no=preparation.attempt_no,
            ),
        )

        generation_result = self._generation_gateway.start_manual_generation(
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            analysis_id=preparation.analysis_id,
            focus_segment_ids=expected_segment_ids,
            loop_count=next_loop_count,
            attempt_no=attempt_no,
            source_promotion_run_id=previous_run.promotion_run_id,
            source_generation_id=previous_run.generation_id,
            operator_instruction=effective_instruction,
        )
        _validate_generation_completed(generation_result)
        _validate_gateway_segments(
            label="generation",
            expected_segment_ids=expected_segment_ids,
            actual_segment_ids=generation_result.generated_segment_ids,
        )
        if generation_result.generation_id == preparation.generation_id:
            raise NextLoopValidationError(
                "replacement generation must use a new generation_id"
            )

        replacement_snapshot = _inspect_generation_candidates(
            candidates=self._content_candidate_repository.list_by_generation(
                generation_result.generation_id
            ),
            generation_id=generation_result.generation_id,
            analysis_id=preparation.analysis_id,
            project_id=previous_run.project_id,
            campaign_id=previous_run.campaign_id,
            promotion_id=previous_run.promotion_id,
            expected_segment_ids=expected_segment_ids,
            require_draft_per_segment=True,
        )
        if candidate_snapshot.content_ids & replacement_snapshot.content_ids:
            raise NextLoopValidationError(
                "replacement generation must use new content_id values"
            )
        if (
            candidate_snapshot.content_option_keys
            & replacement_snapshot.content_option_keys
        ):
            raise NextLoopValidationError(
                "replacement generation must use new content_option_id values"
            )

        rejected = self._next_loop_preparation_repository.mark_rejected(
            preparation.next_loop_preparation_id
        )
        if (
            rejected is None
            or rejected.next_loop_preparation_id
            != preparation.next_loop_preparation_id
            or rejected.status != "rejected"
        ):
            raise NextLoopConflictError(
                "next-loop preparation changed while regeneration was requested"
            )

        preparation_write = NextLoopPreparationWrite(
            next_loop_preparation_id=_next_loop_preparation_id(
                previous_run.promotion_run_id,
                attempt_no,
            ),
            source_promotion_run_id=previous_run.promotion_run_id,
            analysis_id=preparation.analysis_id,
            generation_id=generation_result.generation_id,
            attempt_no=attempt_no,
            failed_segment_ids_json=tuple(expected_segment_ids),
            failed_ad_experiment_ids_json=tuple(
                sorted(preparation.failed_ad_experiment_ids_json)
            ),
            source_evaluation_ids_json=tuple(
                sorted(preparation.source_evaluation_ids_json)
            ),
        )
        try:
            replacement = self._next_loop_preparation_repository.insert(
                preparation_write
            )
        except NextLoopPreparationConflictError as exc:
            raise NextLoopConflictError(
                "next-loop preparation already exists; retry the request"
            ) from exc
        return _manual_preparation_response(
            preparation=replacement,
            previous_run=previous_run,
            loop_count=next_loop_count,
            pending_content_ids=replacement_snapshot.pending_content_ids,
        )

    def _get_previous_run(self, promotion_run_id: str) -> PromotionRunRecord:
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            log.warn("promotion_run_not_found", {"promotionRunId": promotion_run_id})
            raise NextLoopNotFoundError(f"promotion_run not found: {promotion_run_id}")
        return run


def _next_loop_generation_attempt_no(
    *,
    attempts: Sequence[NextLoopGenerationAttemptRecord],
    previous_run: PromotionRunRecord,
    loop_count: int,
    focus_segment_ids: Sequence[str],
    failed_ad_experiment_ids: Sequence[str],
    operator_instruction: str | None,
    candidate_status: ContentCandidateStatus,
    content_option_count: int,
    allowed_completed_generation_id: str | None = None,
    operator_validation_after_attempt_no: int = 0,
) -> int:
    expected_focus_ids = frozenset(focus_segment_ids)
    expected_failed_ad_experiment_ids = frozenset(failed_ad_experiment_ids)
    expected_instruction = _normalize_instruction(operator_instruction)
    seen_attempt_numbers: set[int] = set()
    highest_attempt_no = 0

    for attempt in attempts:
        next_loop_context = attempt.input_json.get("next_loop")
        analysis_snapshot = attempt.analysis_input_snapshot_json
        analysis_next_loop_context = (
            analysis_snapshot.get("next_loop")
            if isinstance(analysis_snapshot, Mapping)
            else None
        )
        raw_attempt_no = (
            next_loop_context.get("attempt_no")
            if isinstance(next_loop_context, Mapping)
            else None
        )
        attempt_no = (
            1
            if raw_attempt_no is None
            else _context_positive_int(raw_attempt_no)
        )
        raw_analysis_attempt_no = (
            analysis_next_loop_context.get("attempt_no")
            if isinstance(analysis_next_loop_context, Mapping)
            else None
        )
        analysis_attempt_no = (
            1
            if raw_analysis_attempt_no is None
            else _context_positive_int(raw_analysis_attempt_no)
        )
        has_matching_preparation = (
            attempt.preparation_analysis_id == attempt.analysis_id
            and attempt.preparation_attempt_no == attempt_no
        )
        is_rejected_preparation_history = (
            candidate_status is ContentCandidateStatus.DRAFT
            and has_matching_preparation
            and attempt.preparation_status == "rejected"
        )
        is_allowed_active_preparation = (
            candidate_status is ContentCandidateStatus.DRAFT
            and has_matching_preparation
            and attempt.generation_id == allowed_completed_generation_id
            and attempt.preparation_status == "awaiting_content_approval"
        )
        is_allowed_completed = (
            attempt.status == COMPLETED_STATUS
            and (
                is_rejected_preparation_history
                or is_allowed_active_preparation
            )
        )

        intent_matches = (
            isinstance(next_loop_context, Mapping)
            and isinstance(analysis_snapshot, Mapping)
            and isinstance(analysis_next_loop_context, Mapping)
            and attempt.project_id == previous_run.project_id
            and attempt.campaign_id == previous_run.campaign_id
            and attempt.promotion_id == previous_run.promotion_id
            and attempt.content_option_count == content_option_count
            and _context_positive_int(next_loop_context.get("loop_count"))
            == loop_count
            and str(next_loop_context.get("source_promotion_run_id", ""))
            == previous_run.promotion_run_id
            and str(next_loop_context.get("source_generation_id", ""))
            == previous_run.generation_id
            and _context_id_set(next_loop_context.get("focus_segment_ids"))
            == expected_focus_ids
            and _context_positive_int(
                next_loop_context.get("content_option_count")
            )
            == content_option_count
            and str(next_loop_context.get("candidate_status", ""))
            == candidate_status.value
            and _context_positive_int(
                analysis_next_loop_context.get("loop_count")
            )
            == loop_count
            and str(
                analysis_next_loop_context.get(
                    "source_promotion_run_id",
                    "",
                )
            )
            == previous_run.promotion_run_id
            and _context_id_set(
                analysis_next_loop_context.get(
                    "source_failed_ad_experiment_ids"
                )
            )
            == expected_failed_ad_experiment_ids
            and _context_id_set(analysis_snapshot.get("focus_segment_ids"))
            == expected_focus_ids
            and attempt_no is not None
            and analysis_attempt_no is not None
            and analysis_attempt_no <= attempt_no
            and (
                is_allowed_completed
                or attempt_no <= operator_validation_after_attempt_no
                or _normalize_instruction(attempt.operator_instruction)
                == expected_instruction
            )
        )
        if not intent_matches:
            raise NextLoopConflictError(
                "existing next-loop generation has a different intent"
            )
        if attempt_no in seen_attempt_numbers:
            raise NextLoopConflictError(
                "existing next-loop generation attempt sequence is invalid"
            )
        seen_attempt_numbers.add(attempt_no)
        highest_attempt_no = max(highest_attempt_no, attempt_no)

        if attempt.status == FAILED_STATUS or is_allowed_completed:
            continue
        if attempt.status in {"requested", "running"}:
            raise NextLoopConflictError(
                "next-loop generation is already in progress"
            )
        if attempt.status == COMPLETED_STATUS:
            raise NextLoopConflictError("next-loop output already exists")
        raise NextLoopConflictError(
            "existing next-loop generation status is invalid"
        )

    if seen_attempt_numbers != set(range(1, highest_attempt_no + 1)):
        raise NextLoopConflictError(
            "existing next-loop generation attempt sequence is invalid"
        )
    return highest_attempt_no + 1


def _no_op_response(run: PromotionRunRecord) -> NextLoopResponse:
    return NextLoopResponse(
        previous_promotion_run_id=run.promotion_run_id,
        next_promotion_run_id=None,
        promotion_id=run.promotion_id,
        loop_count=run.loop_count,
        segment_ids=list(run.segment_scope_json),
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


def _inspect_generation_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    generation_id: str,
    analysis_id: str,
    project_id: str,
    campaign_id: str,
    promotion_id: str,
    expected_segment_ids: Sequence[str],
    require_draft_per_segment: bool,
) -> GenerationCandidateSnapshot:
    if not candidates:
        raise NextLoopValidationError(
            "manual next-loop generation must create content candidates"
        )
    expected = set(expected_segment_ids)
    covered: set[str] = set()
    draft_segments: set[str] = set()
    selectable_segments: set[str] = set()
    content_ids: set[str] = set()
    content_option_keys: set[tuple[str, str]] = set()
    pending: list[tuple[str, str, str]] = []
    for candidate in candidates:
        candidate_generation_id = _candidate_text(candidate, "generation_id")
        candidate_analysis_id = _candidate_text(candidate, "analysis_id")
        candidate_project_id = _candidate_text(candidate, "project_id")
        candidate_campaign_id = _candidate_text(candidate, "campaign_id")
        candidate_promotion_id = _candidate_text(candidate, "promotion_id")
        segment_id = _candidate_text(candidate, "segment_id")
        content_id = _candidate_text(candidate, "content_id")
        content_option_id = _candidate_text(candidate, "content_option_id")
        status = _candidate_text(candidate, "status")
        if (
            candidate_generation_id != generation_id
            or candidate_analysis_id != analysis_id
            or candidate_project_id != project_id
            or candidate_campaign_id != campaign_id
            or candidate_promotion_id != promotion_id
            or segment_id not in expected
            or not content_id
            or not content_option_id
        ):
            raise NextLoopValidationError(
                "manual next-loop content candidates do not match the generation"
            )
        if status not in KNOWN_CANDIDATE_STATUSES:
            raise NextLoopValidationError(
                "manual next-loop content candidate status is invalid"
            )
        content_option_key = (segment_id, content_option_id)
        if content_id in content_ids or content_option_key in content_option_keys:
            raise NextLoopValidationError(
                "manual next-loop content candidates contain duplicate ids"
            )
        content_ids.add(content_id)
        content_option_keys.add(content_option_key)
        covered.add(segment_id)
        if status == "draft":
            draft_segments.add(segment_id)
            pending.append((segment_id, content_option_id, content_id))
        if status in SELECTABLE_CANDIDATE_STATUSES:
            selectable_segments.add(segment_id)
    if covered != expected:
        raise NextLoopValidationError(
            "manual next-loop content candidates must cover every failed segment"
        )
    if require_draft_per_segment and draft_segments != expected:
        raise NextLoopValidationError(
            "manual next-loop generation must create draft content for every segment"
        )
    return GenerationCandidateSnapshot(
        pending_content_ids=tuple(
            content_id for _segment, _option, content_id in sorted(pending)
        ),
        selectable_segment_ids=frozenset(selectable_segments),
        exhausted_segment_ids=frozenset(expected - selectable_segments),
        content_ids=frozenset(content_ids),
        content_option_keys=frozenset(content_option_keys),
    )


def _candidate_text(candidate: Mapping[str, Any], field_name: str) -> str:
    value = candidate.get(field_name)
    if value is None:
        return ""
    return str(value).strip()


def _context_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _context_id_set(value: Any) -> frozenset[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized) or len(set(normalized)) != len(
        normalized
    ):
        return None
    return frozenset(normalized)


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
        segment_ids=list(preparation.failed_segment_ids_json),
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
        error_type = (
            NextLoopGenerationFailedError
            if result.failure_persisted
            else NextLoopValidationError
        )
        raise error_type(
            "next-loop generation result must be completed before run creation"
        )
