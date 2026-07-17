from __future__ import annotations

import hashlib
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from app.generation.artifacts import (
    ArtifactIdentity,
    CreativeArtifactPublisher,
    RECOVERED_IMAGE_PROMPT_PREFIX,
    StaticCreativeArtifactPublisher,
    build_creative_metadata,
    creative_format_for_channel,
    failed_creative_metadata,
    image_prompt_sha256,
    merge_creative_metadata,
    pending_creative_metadata,
    safe_error_code,
)
from app.generation.brand_context import (
    BRAND_CONTEXT_PROMPT_VERSION,
    BrandContextProvider,
    BrandContextSnapshot,
    retrieval_snapshot_from_candidate_metadata,
    validate_brand_guardrails,
)
from app.generation.generator import (
    CONTENT_GENERATOR_VERSION,
    ContentGenerator,
    DeterministicContentGenerator,
    GeneratedContent,
)
from app.generation.image_tasks import ImageGenerationJob
from app.generation.repositories import (
    ContentCandidateRecord,
    GenerationRunRecord,
)
from app.generation.prompt_builder import (
    GenerationContext,
    GenerationContextBuilder,
    GenerationInputBuilder,
    GenerationPromptInput,
    GenerationStrategyPlan,
    GenerationStrategyPlanner,
    PROMPT_BUILDER_VERSION,
    PromptBuilder,
    PromotionPromptInput,
    TargetSegmentPromptInput,
)
from app.generation.report_builder import (
    GENERATION_REPORT_VERSION,
    GenerationReportBuilder,
)
from app.generation.schemas import (
    ContentCandidateResponse,
    ContentCandidateStatus,
    ContentChannel,
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
    ImageGenerationStatus,
)
from app.generation.submission import (
    GENERATION_REQUEST_SCHEMA_VERSION,
    INTERNAL_IDEMPOTENCY_KEY_PREFIX,
    build_generation_input_snapshot,
    generation_request_fingerprint,
)
from app.logging import log, log_context_scope, now_ms, duration_ms


MAX_CONTENT_IDENTIFIER_LENGTH = 100
CONTENT_SLUG_HASH_LENGTH = 16
MAX_DURABLE_CANDIDATE_WORKERS = 3


@dataclass(frozen=True, slots=True)
class _CandidateBuildTask:
    prompt_input: GenerationPromptInput
    generation_context: GenerationContext
    strategy_plan: GenerationStrategyPlan
    index: int


class GenerationRunWriter(Protocol):
    def create(self, record: GenerationRunRecord) -> dict[str, Any]:
        ...

    def list_ids_by_promotion(self, promotion_id: str) -> list[str]:
        ...


class ContentCandidateWriter(Protocol):
    def create(self, record: ContentCandidateRecord) -> dict[str, Any]:
        ...


class ImageGenerationScheduler(Protocol):
    def enqueue(self, job: ImageGenerationJob) -> None:
        ...


class GenerationInputReader(Protocol):
    def get_promotion_input(
        self,
        request: GenerationRequest,
    ) -> PromotionPromptInput | None:
        ...

    def list_target_segment_inputs(
        self,
        request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        ...

    def list_focus_target_segment_inputs(
        self,
        request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        ...


class GenerationRequestHandler(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        ...


class GenerationInputUnavailable(RuntimeError):
    """Raised when confirmed generation input rows are not ready yet."""


class BrandContextSnapshotReader(Protocol):
    def resolve_snapshot(self, *, project_id: str) -> BrandContextSnapshot | None:
        ...


DEMO_PROJECT_ID = "demo_project"
DEMO_DEFAULT_LANDING_URL = (
    "https://demo-shoppingmall.dev.loop-ad.org/hotel/jeju-ocean-breeze-006"
    "?destination=&checkIn=2026-08-01&checkOut=2026-08-03"
    "&adults=1&children=0&rooms=1&deal=summer"
)


@dataclass(frozen=True)
class NextLoopFocusGenerationRequest:
    project_id: str
    campaign_id: str
    promotion_id: str
    analysis_id: str
    focus_segment_ids: Sequence[str]
    loop_count: int
    source_promotion_run_id: str
    source_generation_id: str
    operator_instruction: str | None = None
    content_option_count: int = 1
    attempt_no: int | None = None
    candidate_status: ContentCandidateStatus = ContentCandidateStatus.APPROVED


@dataclass(frozen=True)
class NextLoopFocusGenerationResult:
    generation_id: str
    generated_segment_ids: list[str]
    status: GenerationStatus


@dataclass(frozen=True)
class DurableGenerationResult:
    generation_id: str
    content_candidates: tuple[ContentCandidateRecord, ...]
    output_json: dict[str, Any]
    generation_report_json: dict[str, Any]


CandidateCheckpoint = Callable[[ContentCandidateRecord], None]


class ArtifactFinalizationError(RuntimeError):
    """Raised after a failed artifact candidate has been checkpointed."""

    def __init__(self, candidate: ContentCandidateRecord) -> None:
        self.candidate = candidate
        super().__init__("creative artifact finalization failed")


class GenerationService:
    def __init__(
        self,
        *,
        generation_run_repository: GenerationRunWriter | None = None,
        content_candidate_repository: ContentCandidateWriter | None = None,
        generation_input_reader: GenerationInputReader | None = None,
        brand_context_snapshot_reader: BrandContextSnapshotReader | None = None,
        brand_context_provider: BrandContextProvider | None = None,
        generation_input_builder: GenerationInputBuilder | None = None,
        generation_context_builder: GenerationContextBuilder | None = None,
        generation_strategy_planner: GenerationStrategyPlanner | None = None,
        prompt_builder: PromptBuilder | None = None,
        content_generator: ContentGenerator | None = None,
        generation_model_version: str | None = None,
        artifact_publisher: CreativeArtifactPublisher | None = None,
        image_generation_scheduler: ImageGenerationScheduler | None = None,
        generation_report_builder: GenerationReportBuilder | None = None,
    ) -> None:
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository
        self._generation_input_reader = generation_input_reader
        self._brand_context_snapshot_reader = brand_context_snapshot_reader
        self._brand_context_provider = brand_context_provider
        self._generation_input_builder = (
            generation_input_builder or GenerationInputBuilder()
        )
        self._generation_context_builder = (
            generation_context_builder or GenerationContextBuilder()
        )
        self._generation_strategy_planner = (
            generation_strategy_planner or GenerationStrategyPlanner()
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._content_generator = content_generator or DeterministicContentGenerator()
        self._artifact_publisher = artifact_publisher or StaticCreativeArtifactPublisher()
        self._content_generator_version = _content_generator_version(
            self._content_generator
        )
        self._generation_model_version = (
            str(generation_model_version or "").strip()
            or self._content_generator_version
        )
        self._image_generation_scheduler = image_generation_scheduler
        self._generation_report_builder = (
            generation_report_builder or GenerationReportBuilder()
        )

    @log_context_scope
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": request.project_id,
                "campaignId": request.campaign_id,
                "promotionId": request.promotion_id,
                "analysisId": request.analysis_id,
            }
        )
        log.info("started", {"request": request})
        generation_id = self._next_generation_id(request.promotion_id)
        log.assign_context({"generationId": generation_id})
        prompt_inputs = self._build_prompt_inputs(request)
        log.info("generation_inputs_prepared", {"promptInputCount": len(prompt_inputs)})
        checkpointed_candidates: dict[str, ContentCandidateRecord] = {}
        try:
            content_candidates = self._build_content_candidate_records(
                request=request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
                checkpoint=lambda candidate: checkpointed_candidates.__setitem__(
                    candidate.content_id,
                    candidate,
                ),
            )
        except Exception as exc:
            log.warn("content_generation_failed", {"err": exc})
            generation_run = self._build_generation_run_record(
                request=request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
                content_candidates=[],
                status=GenerationStatus.FAILED,
                error_code=_safe_generation_error_code(exc),
                error_detail=_safe_generation_error_detail(exc),
            )
            self._save_generation_run(generation_run)
            self._save_content_candidates(tuple(checkpointed_candidates.values()))
            response = GenerationResponse(
                generation_id=generation_id,
                promotion_id=request.promotion_id,
                status=GenerationStatus.FAILED,
                content_candidates=[],
            )
            log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
            return response

        generation_run = self._build_generation_run_record(
            request=request,
            generation_id=generation_id,
            prompt_inputs=prompt_inputs,
            content_candidates=content_candidates,
            status=GenerationStatus.COMPLETED,
        )
        self._save_generation_run(generation_run)
        self._save_content_candidates(content_candidates)
        self._schedule_image_generation(content_candidates)

        response = GenerationResponse(
            generation_id=generation_id,
            promotion_id=request.promotion_id,
            status=GenerationStatus.COMPLETED,
            content_candidates=[
                ContentCandidateResponse.model_validate(candidate.to_public_values())
                for candidate in content_candidates
            ],
        )
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response

    def execute_durable(
        self,
        *,
        generation_id: str,
        prompt_inputs: Sequence[GenerationPromptInput],
        existing_candidates: Sequence[ContentCandidateRecord] = (),
        checkpoint: CandidateCheckpoint | None = None,
    ) -> DurableGenerationResult:
        """Run provider and artifact work without mutating durable job state.

        The job processor persists these deterministic candidates with a lease
        fence and is the only component allowed to move the parent run to a
        terminal state.
        """

        if not prompt_inputs:
            raise ValueError("generation input snapshot has no target segments")
        request = prompt_inputs[0].request
        if any(prompt_input.request != request for prompt_input in prompt_inputs):
            raise ValueError("generation input snapshot contains mixed requests")

        content_candidates = self._build_content_candidate_records(
            request=request,
            generation_id=generation_id,
            prompt_inputs=prompt_inputs,
            existing_candidates=existing_candidates,
            checkpoint=checkpoint,
            parallel_candidates=_supports_staged_generation(self._content_generator),
        )
        for candidate in content_candidates:
            _validate_durable_candidate_ready(candidate)

        generation_run = self._build_generation_run_record(
            request=request,
            generation_id=generation_id,
            prompt_inputs=prompt_inputs,
            content_candidates=content_candidates,
            status=GenerationStatus.COMPLETED,
        )
        return DurableGenerationResult(
            generation_id=generation_id,
            content_candidates=tuple(content_candidates),
            output_json=dict(generation_run.output_json or {}),
            generation_report_json=dict(generation_run.generation_report_json),
        )

    @log_context_scope
    def generate_focus(
        self,
        request: NextLoopFocusGenerationRequest,
    ) -> NextLoopFocusGenerationResult:
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": request.project_id,
                "campaignId": request.campaign_id,
                "promotionId": request.promotion_id,
                "analysisId": request.analysis_id,
                "promotionRunId": request.source_promotion_run_id,
                "generationId": request.source_generation_id,
            }
        )
        log.info("started", {"request": request})
        if request.content_option_count < 1:
            raise ValueError("content_option_count must be at least 1")
        if request.attempt_no is not None and request.attempt_no < 1:
            raise ValueError("attempt_no must be at least 1")
        generation_request = GenerationRequest(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            analysis_id=request.analysis_id,
            segment_ids=None,
            content_option_count=request.content_option_count,
            operator_instruction=request.operator_instruction,
        )
        generation_id = _next_loop_generation_id(
            promotion_id=request.promotion_id,
            loop_count=request.loop_count,
            attempt_no=request.attempt_no,
            source_promotion_run_id=request.source_promotion_run_id,
        )
        log.assign_context({"generationId": generation_id})
        prompt_inputs = self._build_focus_prompt_inputs(
            request=generation_request,
            focus_segment_ids=request.focus_segment_ids,
        )
        log.info("generation_inputs_prepared", {"promptInputCount": len(prompt_inputs)})
        checkpointed_candidates: dict[str, ContentCandidateRecord] = {}
        try:
            content_candidates = self._build_content_candidate_records(
                request=generation_request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
                candidate_status=request.candidate_status,
                checkpoint=lambda candidate: checkpointed_candidates.__setitem__(
                    candidate.content_id,
                    candidate,
                ),
            )
        except Exception as exc:
            log.warn("content_generation_failed", {"err": exc})
            generation_run = self._build_generation_run_record(
                request=generation_request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
                content_candidates=[],
                status=GenerationStatus.FAILED,
                error_code=_safe_generation_error_code(exc),
                error_detail=_safe_generation_error_detail(exc),
                source_context=_next_loop_source_context(request),
            )
            self._save_generation_run(generation_run)
            self._save_content_candidates(tuple(checkpointed_candidates.values()))
            response = NextLoopFocusGenerationResult(
                generation_id=generation_id,
                generated_segment_ids=[],
                status=GenerationStatus.FAILED,
            )
            log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
            return response

        generation_run = self._build_generation_run_record(
            request=generation_request,
            generation_id=generation_id,
            prompt_inputs=prompt_inputs,
            content_candidates=content_candidates,
            status=GenerationStatus.COMPLETED,
            source_context=_next_loop_source_context(request),
        )
        self._save_generation_run(generation_run)
        self._save_content_candidates(content_candidates)
        self._schedule_image_generation(content_candidates)

        response = NextLoopFocusGenerationResult(
            generation_id=generation_id,
            generated_segment_ids=[
                prompt_input.target_segment.segment_id for prompt_input in prompt_inputs
            ],
            status=GenerationStatus.COMPLETED,
        )
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response

    def _build_generation_run_record(
        self,
        *,
        request: GenerationRequest,
        generation_id: str,
        prompt_inputs: Sequence[GenerationPromptInput],
        content_candidates: Sequence[ContentCandidateRecord],
        status: GenerationStatus,
        error_code: str | None = None,
        error_detail: dict[str, Any] | None = None,
        source_context: dict[str, Any] | None = None,
    ) -> GenerationRunRecord:
        output_json = self._generation_report_builder.build_run_output(
            status=status,
            target_segment_count=len(prompt_inputs),
            content_candidate_metadata=[
                candidate.metadata_json for candidate in content_candidates
            ],
            error_code=error_code,
        )
        if any(prompt_input.brand_context is not None for prompt_input in prompt_inputs):
            output_json["retrieval_snapshot"] = (
                retrieval_snapshot_from_candidate_metadata(
                    [candidate.metadata_json for candidate in content_candidates]
                )
            )

        generation_report_json: dict[str, Any] = {
            "status": status.value,
            "schema_version": GENERATION_REQUEST_SCHEMA_VERSION,
            "content_candidate_count": len(content_candidates),
            "target_segment_count": len(prompt_inputs),
            "prompt_builder": PROMPT_BUILDER_VERSION,
            "content_generator": self._content_generator_version,
            "report_builder": GENERATION_REPORT_VERSION,
        }
        if error_code:
            generation_report_json["error_code"] = error_code
        if error_detail:
            generation_report_json["error_detail"] = error_detail

        if not prompt_inputs:
            raise ValueError("generation input snapshot requires prompt inputs")
        input_json = build_generation_input_snapshot(
            request=request,
            promotion=prompt_inputs[0].promotion,
            target_segments=[
                prompt_input.target_segment for prompt_input in prompt_inputs
            ],
            brand_context=prompt_inputs[0].brand_context,
            model_version=self._generation_model_version,
        )
        input_json.update(
            {
                "target_segment_ids": sorted(
                    prompt_input.target_segment.segment_id
                    for prompt_input in prompt_inputs
                ),
                "channel": prompt_inputs[0].promotion.channel.value,
            }
        )
        if source_context is not None:
            input_json["next_loop"] = source_context

        request_fingerprint = generation_request_fingerprint(input_json)
        idempotency_scope = "next-loop" if source_context is not None else "inline"
        terminal_at = datetime.now(timezone.utc)

        return GenerationRunRecord(
            generation_id=generation_id,
            analysis_id=request.analysis_id,
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            content_option_count=request.content_option_count,
            operator_instruction=request.operator_instruction,
            input_json=input_json,
            output_json=output_json,
            generation_report_json=generation_report_json,
            status=status.value,
            started_at=(
                terminal_at
                if status in (GenerationStatus.COMPLETED, GenerationStatus.FAILED)
                else None
            ),
            finished_at=(
                terminal_at
                if status in (GenerationStatus.COMPLETED, GenerationStatus.FAILED)
                else None
            ),
            idempotency_key=(
                f"{INTERNAL_IDEMPOTENCY_KEY_PREFIX}"
                f"{idempotency_scope}:{generation_id}"
            ),
            request_fingerprint=request_fingerprint,
        )

    def _build_prompt_inputs(
        self,
        request: GenerationRequest,
    ) -> list[GenerationPromptInput]:
        if self._generation_input_reader is not None:
            promotion = self._generation_input_reader.get_promotion_input(request)
            if promotion is None:
                log.warn("promotion_input_not_found", {"promotionId": request.promotion_id})
                raise GenerationInputUnavailable(
                    "promotion input was not found for generation"
                )

            target_segments = (
                self._generation_input_reader.list_target_segment_inputs(request)
            )
            if not target_segments:
                log.warn("target_segments_empty", {"analysisId": request.analysis_id})
                raise GenerationInputUnavailable(
                    "confirmed promotion_target_segments are required for generation"
                )
            _validate_requested_segment_ids(
                requested_segment_ids=request.segment_ids,
                target_segments=target_segments,
            )

            return self._build_generation_prompt_inputs(
                request=request,
                promotion=promotion,
                target_segments=target_segments,
            )

        return self._build_generation_prompt_inputs(
            request=request,
            promotion=_fixture_promotion_prompt_input(request),
            target_segments=[_fixture_target_segment_prompt_input(request)],
        )

    def _build_focus_prompt_inputs(
        self,
        *,
        request: GenerationRequest,
        focus_segment_ids: Sequence[str],
    ) -> list[GenerationPromptInput]:
        if self._generation_input_reader is None:
            log.warn("generation_input_reader_missing")
            raise ValueError("generation input reader is required for focus generation")

        promotion = self._generation_input_reader.get_promotion_input(request)
        if promotion is None:
            log.warn("promotion_input_not_found", {"promotionId": request.promotion_id})
            raise ValueError("promotion input was not found for focus generation")

        focus_segment_reader = getattr(
            self._generation_input_reader,
            "list_focus_target_segment_inputs",
            None,
        )
        if callable(focus_segment_reader):
            target_segments = focus_segment_reader(request)
        else:
            target_segments = self._generation_input_reader.list_target_segment_inputs(
                request
            )
        if not target_segments:
            log.warn("target_segments_empty", {"analysisId": request.analysis_id})
            raise ValueError(
                "promotion_target_segments are required for focus generation"
            )
        focus_ids = _focus_segment_ids(focus_segment_ids)
        target_segments_by_id = {
            target_segment.segment_id: target_segment
            for target_segment in target_segments
        }
        missing_segment_ids = [
            segment_id
            for segment_id in focus_ids
            if segment_id not in target_segments_by_id
        ]
        if missing_segment_ids:
            log.warn("focus_segments_invalid", {"missingSegmentIds": missing_segment_ids})
            raise ValueError("focus_segment_ids must match promotion_target_segments")

        return self._build_generation_prompt_inputs(
            request=request,
            promotion=promotion,
            target_segments=[
                target_segments_by_id[segment_id]
                for segment_id in focus_ids
            ],
        )

    def _build_generation_prompt_inputs(
        self,
        *,
        request: GenerationRequest,
        promotion: PromotionPromptInput,
        target_segments: Sequence[TargetSegmentPromptInput],
    ) -> list[GenerationPromptInput]:
        brand_context = self._resolve_brand_context_snapshot(request)
        if brand_context is None:
            return self._generation_input_builder.build(
                request=request,
                promotion=promotion,
                target_segments=target_segments,
            )
        return self._generation_input_builder.build(
            request=request,
            promotion=promotion,
            target_segments=target_segments,
            brand_context=brand_context,
        )

    def _resolve_brand_context_snapshot(
        self,
        request: GenerationRequest,
    ) -> BrandContextSnapshot | None:
        if self._brand_context_snapshot_reader is None:
            return None
        return self._brand_context_snapshot_reader.resolve_snapshot(
            project_id=request.project_id,
        )

    def _build_content_candidate_records(
        self,
        *,
        request: GenerationRequest,
        generation_id: str,
        prompt_inputs: Sequence[GenerationPromptInput],
        candidate_status: ContentCandidateStatus = ContentCandidateStatus.DRAFT,
        existing_candidates: Sequence[ContentCandidateRecord] = (),
        checkpoint: CandidateCheckpoint | None = None,
        parallel_candidates: bool = False,
    ) -> list[ContentCandidateRecord]:
        existing_by_id = {
            candidate.content_id: candidate for candidate in existing_candidates
        }
        tasks: list[_CandidateBuildTask] = []
        for raw_prompt_input in prompt_inputs:
            prompt_input = _prompt_input_with_resolved_landing_url(raw_prompt_input)
            generation_context = self._generation_context_builder.build(prompt_input)
            if prompt_input.brand_context is not None:
                if self._brand_context_provider is None:
                    raise ValueError(
                        "brand context provider is required for a snapshotted context"
                    )
                generation_context = replace(
                    generation_context,
                    brand_context=self._brand_context_provider.retrieve(
                        prompt_input,
                        generation_context,
                    ),
                )
            for index in range(1, request.content_option_count + 1):
                strategy_plan = self._generation_strategy_planner.build(
                    generation_context,
                    option_index=index,
                )
                tasks.append(
                    _CandidateBuildTask(
                        prompt_input=prompt_input,
                        generation_context=generation_context,
                        strategy_plan=strategy_plan,
                        index=index,
                    )
                )
        if not parallel_candidates or len(tasks) < 2:
            return [
                self._run_candidate_build_task(
                    task,
                    generation_id=generation_id,
                    status=candidate_status,
                    existing_by_id=existing_by_id,
                    checkpoint=checkpoint,
                )
                for task in tasks
            ]
        return self._build_content_candidates_in_parallel(
            tasks,
            generation_id=generation_id,
            status=candidate_status,
            existing_by_id=existing_by_id,
            checkpoint=checkpoint,
        )

    def _build_content_candidates_in_parallel(
        self,
        tasks: Sequence[_CandidateBuildTask],
        *,
        generation_id: str,
        status: ContentCandidateStatus,
        existing_by_id: Mapping[str, ContentCandidateRecord],
        checkpoint: CandidateCheckpoint | None,
    ) -> list[ContentCandidateRecord]:
        worker_count = min(len(tasks), MAX_DURABLE_CANDIDATE_WORKERS)
        log.info(
            "generation_candidates_parallel_started",
            {
                "generationId": generation_id,
                "candidateCount": len(tasks),
                "workerCount": worker_count,
            },
        )
        records_by_position: dict[int, ContentCandidateRecord] = {}
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="loop-ad-candidate",
        ) as executor:
            positions_by_future: dict[Future[ContentCandidateRecord], int] = {
                executor.submit(
                    self._run_candidate_build_task,
                    task,
                    generation_id=generation_id,
                    status=status,
                    existing_by_id=existing_by_id,
                    checkpoint=checkpoint,
                ): position
                for position, task in enumerate(tasks)
            }
            try:
                for future in as_completed(positions_by_future):
                    records_by_position[positions_by_future[future]] = future.result()
            except Exception:
                for future in positions_by_future:
                    future.cancel()
                raise
        records = [records_by_position[position] for position in range(len(tasks))]
        log.info(
            "generation_candidates_parallel_completed",
            {
                "generationId": generation_id,
                "candidateCount": len(records),
                "workerCount": worker_count,
            },
        )
        return records

    def _run_candidate_build_task(
        self,
        task: _CandidateBuildTask,
        *,
        generation_id: str,
        status: ContentCandidateStatus,
        existing_by_id: Mapping[str, ContentCandidateRecord],
        checkpoint: CandidateCheckpoint | None,
    ) -> ContentCandidateRecord:
        return self._build_content_candidate_record(
            generation_id=generation_id,
            prompt_input=task.prompt_input,
            generation_context=task.generation_context,
            strategy_plan=task.strategy_plan,
            index=task.index,
            status=status,
            existing_by_id=existing_by_id,
            checkpoint=checkpoint,
        )

    def _build_content_candidate_record(
        self,
        *,
        generation_id: str,
        prompt_input: GenerationPromptInput,
        generation_context: GenerationContext,
        strategy_plan: GenerationStrategyPlan,
        index: int,
        status: ContentCandidateStatus,
        existing_by_id: dict[str, ContentCandidateRecord],
        checkpoint: CandidateCheckpoint | None,
    ) -> ContentCandidateRecord:
        channel = prompt_input.promotion.channel
        channel_slug = _channel_slug(channel)
        segment_slug = _segment_slug(prompt_input.target_segment)
        generation_slug = _general_generation_attempt_slug(
            generation_id=generation_id,
            promotion_id=prompt_input.request.promotion_id,
        )
        content_slug = (
            f"{segment_slug}_{generation_slug}" if generation_slug else segment_slug
        )
        content_slug = _bounded_content_slug(
            channel_slug=channel_slug,
            content_slug=content_slug,
            index=index,
        )
        segment_id = prompt_input.target_segment.segment_id
        content_id = f"content_{channel_slug}_{content_slug}_{index:03d}"
        content_option_id = f"{channel_slug}_{content_slug}_option_{index:03d}"
        identity = ArtifactIdentity(
            project_id=prompt_input.request.project_id,
            promotion_id=prompt_input.request.promotion_id,
            generation_id=generation_id,
            content_id=content_id,
        )
        existing_candidate = existing_by_id.get(content_id)
        if existing_candidate is not None:
            _validate_existing_candidate_identity(
                existing_candidate,
                identity=identity,
                campaign_id=prompt_input.request.campaign_id,
                analysis_id=prompt_input.request.analysis_id,
                segment_id=segment_id,
                content_option_id=content_option_id,
                channel=channel,
            )
            if _candidate_is_ready(existing_candidate):
                return existing_candidate
            if _candidate_can_resume_image(existing_candidate):
                generated_content = _generated_content_from_candidate(
                    existing_candidate
                )
                generated_content = _ensure_staged_image(
                    content_generator=self._content_generator,
                    channel=channel,
                    generated_content=generated_content,
                    identity=identity,
                )
                image_candidate = _candidate_with_generated_image(
                    existing_candidate,
                    generated_content,
                )
                if checkpoint is not None:
                    checkpoint(image_candidate)
                return self._finalize_content_candidate(
                    candidate=image_candidate,
                    identity=identity,
                    checkpoint=checkpoint,
                )
            if _candidate_can_resume_artifact(existing_candidate):
                return self._finalize_content_candidate(
                    candidate=existing_candidate,
                    identity=identity,
                    checkpoint=checkpoint,
                )

        prompt_result = self._prompt_builder.build(
            prompt_input,
            generation_context=generation_context,
            strategy_plan=strategy_plan,
        )
        staged_generation = _supports_staged_generation(self._content_generator)
        if staged_generation:
            generate_source = getattr(self._content_generator, "generate_source")
            generated_content = generate_source(
                prompt_input=prompt_input,
                prompt_result=prompt_result,
                option_index=index,
                artifact_identity=identity,
            )
        else:
            generated_content = self._content_generator.generate(
                prompt_input=prompt_input,
                prompt_result=prompt_result,
                option_index=index,
                artifact_identity=identity,
            )
        content_values = generated_content.to_record_values(channel)
        validate_brand_guardrails(
            generation_context.brand_context,
            content_values=content_values,
        )
        candidate_report = self._generation_report_builder.build_candidate_report(
            prompt_input=prompt_input,
            prompt_result=prompt_result,
            content_id=content_id,
            content_option_id=content_option_id,
            content_generator_version=self._content_generator_version,
            content_values=content_values,
            status=status.value,
        )
        image_generation_status = (
            ImageGenerationStatus.NOT_REQUIRED.value
            if channel == ContentChannel.SMS
            else (
                ImageGenerationStatus.PENDING.value
                if staged_generation
                else (
                    ImageGenerationStatus.COMPLETED.value
                    if content_values["image_url"]
                    else ImageGenerationStatus.PENDING.value
                )
            )
        )
        creative_metadata = pending_creative_metadata(
            channel=channel,
            content_values=_content_values_with_generated_renderer(
                content_values,
                generated_content,
            ),
        )
        creative_metadata["model"] = {
            "provider": _content_generator_provider(self._content_generator),
            "model_version": self._generation_model_version,
            "prompt_version": BRAND_CONTEXT_PROMPT_VERSION,
        }
        if generation_context.brand_context is not None:
            creative_metadata["lineage"] = generation_context.brand_context.lineage(
                provider_request_id=_provider_request_id(
                    generation_id=generation_id,
                    content_id=content_id,
                    generation_prompt=prompt_result.generation_prompt,
                )
            )
        image_metadata = _canonical_image_metadata(generated_content)
        if image_metadata is not None:
            creative_metadata["image"] = image_metadata
        metadata_json = merge_creative_metadata(
            candidate_report.metadata_json,
            creative_metadata,
        )
        if generated_content.image_artifact is not None:
            metadata_json["image"] = generated_content.image_artifact.to_metadata()
        elif generated_content.image_url:
            metadata_json["image"] = {"public_url": generated_content.image_url}

        pending_candidate = ContentCandidateRecord(
            content_id=content_id,
            content_option_id=content_option_id,
            project_id=prompt_input.request.project_id,
            campaign_id=prompt_input.request.campaign_id,
            promotion_id=prompt_input.request.promotion_id,
            analysis_id=prompt_input.request.analysis_id,
            generation_id=generation_id,
            segment_id=segment_id,
            channel=channel,
            subject=content_values["subject"],
            preheader=content_values["preheader"],
            title=content_values["title"],
            body=content_values["body"],
            cta=content_values["cta"],
            message=content_values["message"],
            image_prompt=content_values["image_prompt"],
            image_url=content_values["image_url"],
            landing_url=content_values["landing_url"],
            generation_prompt=prompt_result.generation_prompt,
            reason_summary=candidate_report.reason_summary,
            data_evidence_json=candidate_report.data_evidence_json,
            message_strategy=candidate_report.message_strategy,
            metadata_json=metadata_json,
            status=status.value,
            creative_format=creative_format_for_channel(channel).value,
            image_generation_status=image_generation_status,
            artifact_status=(
                "not_required" if channel == ContentChannel.SMS else "pending"
            ),
        )
        if checkpoint is not None:
            checkpoint(pending_candidate)
        if staged_generation and channel != ContentChannel.SMS:
            generated_content = _ensure_staged_image(
                content_generator=self._content_generator,
                channel=channel,
                generated_content=generated_content,
                identity=identity,
            )
            pending_candidate = _candidate_with_generated_image(
                pending_candidate,
                generated_content,
            )
            if checkpoint is not None:
                checkpoint(pending_candidate)
        return self._finalize_content_candidate(
            candidate=pending_candidate,
            identity=identity,
            checkpoint=checkpoint,
        )

    def _finalize_content_candidate(
        self,
        *,
        candidate: ContentCandidateRecord,
        identity: ArtifactIdentity,
        checkpoint: CandidateCheckpoint | None,
    ) -> ContentCandidateRecord:
        content_values = _content_values_with_candidate_renderer(
            candidate,
            candidate.to_record_values(),
        )
        try:
            creative_metadata = build_creative_metadata(
                channel=candidate.channel,
                identity=identity,
                content_values=content_values,
                artifact_publisher=self._artifact_publisher,
            )
            creative_metadata = _creative_patch_with_candidate_image(
                candidate,
                creative_metadata,
            )
        except Exception as exc:
            error_code = safe_error_code(exc)
            failed_metadata = failed_creative_metadata(
                channel=candidate.channel,
                content_values=content_values,
                error_code=error_code,
            )
            failed_metadata = _creative_patch_with_candidate_image(
                candidate,
                failed_metadata,
            )
            failed_candidate = replace(
                candidate,
                metadata_json=merge_creative_metadata(
                    candidate.metadata_json,
                    failed_metadata,
                ),
                artifact_status="failed",
                artifact_storage_key=None,
                artifact_public_url=None,
                artifact_sha256=None,
                artifact_content_type=None,
                artifact_error_code=error_code,
                artifact_published_at=None,
            )
            if checkpoint is not None:
                checkpoint(failed_candidate)
            raise ArtifactFinalizationError(failed_candidate) from exc

        artifact = creative_metadata["artifact"]
        artifact_status = str(artifact["artifact_status"])
        artifact_published_at = (
            candidate.artifact_published_at or datetime.now(timezone.utc)
            if artifact_status == "published"
            else None
        )
        if artifact_published_at is not None:
            creative_metadata = {
                **creative_metadata,
                "artifact": {
                    **artifact,
                    "published_at": artifact_published_at.isoformat(),
                },
            }
            artifact = creative_metadata["artifact"]
        finalized_candidate = replace(
            candidate,
            metadata_json=merge_creative_metadata(
                candidate.metadata_json,
                creative_metadata,
            ),
            artifact_status=artifact_status,
            artifact_storage_key=_optional_artifact_text(artifact, "storage_key"),
            artifact_public_url=_optional_artifact_text(artifact, "public_url"),
            artifact_sha256=_optional_artifact_text(artifact, "sha256"),
            artifact_content_type=_optional_artifact_text(artifact, "content_type"),
            artifact_error_code=_optional_artifact_text(artifact, "error_code"),
            artifact_published_at=artifact_published_at,
        )
        if checkpoint is not None:
            checkpoint(finalized_candidate)
        return finalized_candidate

    def _save_generation_run(self, generation_run: GenerationRunRecord) -> None:
        if self._generation_run_repository is None:
            return
        self._generation_run_repository.create(generation_run)
        log.info("generation_run_created", {"generationRun": generation_run})

    def _save_content_candidates(
        self,
        content_candidates: Sequence[ContentCandidateRecord],
    ) -> None:
        if self._content_candidate_repository is None:
            return
        for content_candidate in content_candidates:
            self._content_candidate_repository.create(content_candidate)
        log.info("content_candidates_created", {"contentCandidateCount": len(content_candidates)})

    def _schedule_image_generation(
        self,
        content_candidates: Sequence[ContentCandidateRecord],
    ) -> None:
        if (
            self._content_candidate_repository is None
            or self._image_generation_scheduler is None
        ):
            return

        for content_candidate in content_candidates:
            if (
                content_candidate.channel == ContentChannel.ONSITE_BANNER
                and content_candidate.image_prompt
                and not content_candidate.image_url
            ):
                self._image_generation_scheduler.enqueue(
                    ImageGenerationJob(
                        identity=ArtifactIdentity(
                            project_id=content_candidate.project_id,
                            promotion_id=content_candidate.promotion_id,
                            generation_id=content_candidate.generation_id,
                            content_id=content_candidate.content_id,
                        ),
                        image_prompt=content_candidate.image_prompt,
                    )
                )
                log.info("image_generation_queued", {"contentId": content_candidate.content_id})

    def _next_generation_id(self, promotion_id: str) -> str:
        base_generation_id = _generation_id_from_promotion(promotion_id)
        if self._generation_run_repository is None:
            return base_generation_id

        existing_generation_ids = set(
            self._generation_run_repository.list_ids_by_promotion(promotion_id)
        )
        if base_generation_id not in existing_generation_ids:
            return base_generation_id

        run_number = 2
        while True:
            candidate_generation_id = _generation_id_from_promotion(
                promotion_id,
                generation_run_number=run_number,
            )
            if candidate_generation_id not in existing_generation_ids:
                return candidate_generation_id
            run_number += 1


def _generation_id_from_promotion(
    promotion_id: str,
    *,
    loop_count: int | None = None,
    attempt_no: int | None = None,
    generation_run_number: int | None = None,
) -> str:
    promotion_slug = promotion_id.removeprefix("promo_")
    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", promotion_slug).strip("_")
    if loop_count is not None:
        generation_id = f"generation_{safe_slug or 'content'}_loop_{loop_count}"
        if attempt_no is not None:
            return f"{generation_id}_attempt_{attempt_no}"
        return generation_id
    generation_id = f"generation_{safe_slug or 'content'}"
    if generation_run_number is not None and generation_run_number > 1:
        return f"{generation_id}_run_{generation_run_number}"
    return generation_id


def _next_loop_generation_id(
    *,
    promotion_id: str,
    loop_count: int,
    source_promotion_run_id: str,
    attempt_no: int | None = None,
) -> str:
    promotion_slug = promotion_id.removeprefix("promo_")
    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", promotion_slug).strip("_")
    lineage_digest = hashlib.sha256(
        source_promotion_run_id.encode("utf-8")
    ).hexdigest()[:12]
    suffix = f"_loop_{loop_count}_{lineage_digest}"
    if attempt_no is not None:
        suffix = f"{suffix}_attempt_{attempt_no}"
    prefix = "generation"
    max_slug_length = MAX_CONTENT_IDENTIFIER_LENGTH - len(prefix) - len(suffix) - 1
    safe_slug = safe_slug[:max_slug_length].rstrip("_") or "content"
    return f"{prefix}_{safe_slug}{suffix}"


def _general_generation_attempt_slug(
    *,
    generation_id: str,
    promotion_id: str,
) -> str | None:
    base_generation_id = _generation_id_from_promotion(promotion_id)
    if generation_id == base_generation_id:
        return None
    prefix = f"{base_generation_id}_"
    if generation_id.startswith(prefix):
        attempt_slug = generation_id.removeprefix(prefix)
        if re.fullmatch(
            r"(?:run_[2-9][0-9]*|loop_[2-9][0-9]*(?:(?:_[0-9a-f]{12})?(?:_attempt_[1-9][0-9]*)?)?)",
            attempt_slug,
        ):
            return attempt_slug

    # Durable submission IDs end in the request digest. Their promotion slug is
    # bounded independently, so a long promotion_id may no longer match the
    # unbounded legacy prefix above. Preserve the digest in candidate IDs to
    # keep content_id globally unique across idempotent generation requests.
    durable_digest = re.search(r"_([0-9a-f]{16})$", generation_id)
    if durable_digest is not None:
        return durable_digest.group(1)
    return None


def _validate_requested_segment_ids(
    *,
    requested_segment_ids: Sequence[str] | None,
    target_segments: Sequence[TargetSegmentPromptInput],
) -> None:
    if requested_segment_ids is None:
        return

    requested_ids = set(requested_segment_ids)
    actual_ids = {target_segment.segment_id for target_segment in target_segments}
    if actual_ids != requested_ids:
        raise GenerationInputUnavailable(
            "segment_ids must match approved promotion_target_segments"
        )


def _focus_segment_ids(values: Sequence[str]) -> list[str]:
    cleaned = [str(value).strip() for value in values]
    if not cleaned:
        raise ValueError("focus_segment_ids must contain at least one segment")
    if any(not value for value in cleaned):
        raise ValueError("focus_segment_ids must not contain empty values")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("focus_segment_ids must not contain duplicates")
    return cleaned


def _next_loop_source_context(
    request: NextLoopFocusGenerationRequest,
) -> dict[str, Any]:
    return {
        "loop_count": request.loop_count,
        "source_promotion_run_id": request.source_promotion_run_id,
        "source_generation_id": request.source_generation_id,
        "focus_segment_ids": list(request.focus_segment_ids),
        "content_option_count": request.content_option_count,
        "attempt_no": request.attempt_no,
        "candidate_status": request.candidate_status.value,
    }


def _fixture_promotion_prompt_input(
    request: GenerationRequest,
) -> PromotionPromptInput:
    return PromotionPromptInput(
        project_id=request.project_id,
        campaign_id=request.campaign_id,
        promotion_id=request.promotion_id,
        channel=ContentChannel.ONSITE_BANNER,
        goal_metric="booking_conversion_rate",
        goal_target_value="0.030000",
        goal_basis="all_segments",
        message_brief="Drive hotel booking conversion for summer stays.",
        landing_url="https://demo-stay.example.com/summer",
        offer_type="hotel_deal",
        landing_type="hotel_detail_page",
    )


def _fixture_target_segment_prompt_input(
    request: GenerationRequest,
) -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id=request.analysis_id,
        promotion_id=request.promotion_id,
        segment_id="seg_repeat_hotel_no_booking",
        segment_name="Repeat hotel viewers without booking",
        content_slug="repeat_hotel",
        content_brief_json={
            "message_direction": (
                "Emphasize refundable rooms, same-weekend availability, "
                "and a clear hotel deals CTA."
            ),
            "keywords": [
                "refundable rooms",
                "same-weekend availability",
                "hotel deals",
            ],
        },
        segment_vector_id="segvec_repeat_hotel_v1",
        estimated_size=1342,
        priority="high",
        natural_language_query="repeat hotel viewers who did not book",
        generated_sql=None,
        sample_ratio="0.018000",
        source="system_default",
        query_preview_id=None,
    )


def _channel_slug(channel: ContentChannel) -> str:
    if channel == ContentChannel.ONSITE_BANNER:
        return "banner"
    return channel.value


def _segment_slug(target_segment: TargetSegmentPromptInput) -> str:
    if target_segment.content_slug:
        return target_segment.content_slug
    segment_id = target_segment.segment_id.removeprefix("seg_")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", segment_id).strip("_") or "segment"


def _bounded_content_slug(
    *,
    channel_slug: str,
    content_slug: str,
    index: int,
) -> str:
    content_id_overhead = len(f"content_{channel_slug}__{index:03d}")
    content_option_id_overhead = len(f"{channel_slug}__option_{index:03d}")
    max_slug_length = MAX_CONTENT_IDENTIFIER_LENGTH - max(
        content_id_overhead,
        content_option_id_overhead,
    )
    if len(content_slug) <= max_slug_length:
        return content_slug

    digest = hashlib.sha256(content_slug.encode("utf-8")).hexdigest()[
        :CONTENT_SLUG_HASH_LENGTH
    ]
    prefix_length = max_slug_length - CONTENT_SLUG_HASH_LENGTH - 1
    return f"{content_slug[:prefix_length]}_{digest}"


def _safe_generation_error_code(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return "content_generation_validation_failed"
    return "content_generation_failed"


def _safe_generation_error_detail(exc: Exception) -> dict[str, Any] | None:
    if not isinstance(exc, ValueError):
        return None

    message = str(exc)
    missing_fields_match = re.fullmatch(
        r"(?P<channel>[a-z_]+) generated content is missing required fields: "
        r"(?P<fields>[a-z_, ]+)",
        message,
    )
    if missing_fields_match:
        return {
            "reason": "missing_required_fields",
            "channel": missing_fields_match.group("channel"),
            "missing_fields": [
                field.strip()
                for field in missing_fields_match.group("fields").split(",")
                if field.strip()
            ],
        }

    if "landing_url" in message:
        return {"reason": "missing_landing_url"}

    return {"reason": "validation_failed"}


def _prompt_input_with_resolved_landing_url(
    prompt_input: GenerationPromptInput,
) -> GenerationPromptInput:
    landing_url = prompt_input.promotion.landing_url
    if landing_url:
        return prompt_input
    if prompt_input.promotion.project_id == DEMO_PROJECT_ID:
        return replace(
            prompt_input,
            promotion=replace(
                prompt_input.promotion,
                landing_url=DEMO_DEFAULT_LANDING_URL,
            ),
        )
    raise ValueError("promotion.landing_url is required to generate content")


def _content_generator_version(content_generator: ContentGenerator) -> str:
    version = str(getattr(content_generator, "version", CONTENT_GENERATOR_VERSION))
    return version.strip() or CONTENT_GENERATOR_VERSION


def _content_generator_provider(content_generator: ContentGenerator) -> str:
    if "external" in type(content_generator).__name__.lower():
        return "openai"
    return "deterministic"


def _provider_request_id(
    *,
    generation_id: str,
    content_id: str,
    generation_prompt: str,
) -> str:
    digest = hashlib.sha256(
        f"{generation_id}\x1f{content_id}\x1f{generation_prompt}".encode("utf-8")
    ).hexdigest()[:24]
    return f"loopad:{generation_id}:{digest}"


def _canonical_image_metadata(
    generated_content: GeneratedContent,
) -> dict[str, Any] | None:
    if not generated_content.image_prompt and not generated_content.image_url:
        return None
    metadata: dict[str, Any] = {
        **_canonical_image_prompt_metadata(generated_content.image_prompt),
        "public_url": generated_content.image_url,
    }
    stored = generated_content.image_artifact
    if stored is not None:
        metadata.update(
            {
                "storage_key": stored.storage_key,
                "public_url": stored.public_url,
                "sha256": stored.sha256,
                "byte_size": stored.bytes,
                "content_type": stored.content_type,
            }
        )
    return {key: value for key, value in metadata.items() if value is not None}


def _supports_staged_generation(content_generator: ContentGenerator) -> bool:
    return callable(getattr(content_generator, "generate_source", None)) and callable(
        getattr(content_generator, "ensure_image", None)
    )


def _ensure_staged_image(
    *,
    content_generator: ContentGenerator,
    channel: ContentChannel,
    generated_content: GeneratedContent,
    identity: ArtifactIdentity,
) -> GeneratedContent:
    ensure_image = getattr(content_generator, "ensure_image", None)
    if not callable(ensure_image):
        raise RuntimeError(
            "pending image generation cannot resume with this content generator"
        )
    return ensure_image(
        channel=channel,
        content=generated_content,
        artifact_identity=identity,
    )


def _generated_content_from_candidate(
    candidate: ContentCandidateRecord,
) -> GeneratedContent:
    creative = candidate.metadata_json.get("creative")
    renderer = creative.get("renderer") if isinstance(creative, Mapping) else None
    renderer_version = (
        str(renderer.get("version"))
        if isinstance(renderer, Mapping) and renderer.get("version")
        else None
    )
    template_version = (
        str(renderer.get("template_version"))
        if isinstance(renderer, Mapping) and renderer.get("template_version")
        else None
    )
    content = GeneratedContent(
        subject=candidate.subject,
        preheader=candidate.preheader,
        title=candidate.title,
        body=candidate.body,
        cta=candidate.cta,
        message=candidate.message,
        image_prompt=candidate.image_prompt,
        image_url=candidate.image_url,
        landing_url=candidate.landing_url,
        artifact_renderer_version=renderer_version,
        artifact_template_version=template_version,
    )
    content.to_record_values(candidate.channel)
    return content


def _candidate_with_generated_image(
    candidate: ContentCandidateRecord,
    generated_content: GeneratedContent,
) -> ContentCandidateRecord:
    generated_content.to_record_values(candidate.channel)
    image_metadata = _canonical_image_metadata(generated_content)
    metadata_json = dict(candidate.metadata_json)
    if image_metadata is not None:
        metadata_json = merge_creative_metadata(
            metadata_json,
            {"image": image_metadata},
        )
    if generated_content.image_artifact is not None:
        metadata_json["image"] = generated_content.image_artifact.to_metadata()
    elif generated_content.image_url:
        metadata_json["image"] = {"public_url": generated_content.image_url}
    metadata_json["image_prompt"] = generated_content.image_prompt
    metadata_json["image_url"] = generated_content.image_url
    return replace(
        candidate,
        image_prompt=generated_content.image_prompt,
        image_url=generated_content.image_url,
        metadata_json=metadata_json,
        image_generation_status=(
            ImageGenerationStatus.COMPLETED.value
            if generated_content.image_url
            else ImageGenerationStatus.PENDING.value
        ),
    )


def _content_values_with_generated_renderer(
    content_values: Mapping[str, str | None],
    generated_content: GeneratedContent,
) -> dict[str, str | None]:
    values = dict(content_values)
    if generated_content.artifact_renderer_version:
        values["renderer_version"] = generated_content.artifact_renderer_version
    if generated_content.artifact_template_version:
        values["template_version"] = generated_content.artifact_template_version
    return values


def _content_values_with_candidate_renderer(
    candidate: ContentCandidateRecord,
    content_values: Mapping[str, str | None],
) -> dict[str, str | None]:
    values = dict(content_values)
    creative = candidate.metadata_json.get("creative")
    renderer = creative.get("renderer") if isinstance(creative, Mapping) else None
    if isinstance(renderer, Mapping):
        renderer_version = renderer.get("version")
        template_version = renderer.get("template_version")
        if renderer_version:
            values["renderer_version"] = str(renderer_version)
        if template_version:
            values["template_version"] = str(template_version)
    return values


def _creative_patch_with_candidate_image(
    candidate: ContentCandidateRecord,
    creative_patch: Mapping[str, Any],
) -> dict[str, Any]:
    if candidate.channel == ContentChannel.SMS:
        return dict(creative_patch)
    existing_creative = candidate.metadata_json.get("creative")
    existing_image = (
        existing_creative.get("image")
        if isinstance(existing_creative, Mapping)
        else None
    )
    legacy_image = candidate.metadata_json.get("image")
    canonical: dict[str, Any] = {
        **_canonical_image_prompt_metadata(candidate.image_prompt),
        "public_url": candidate.image_url,
    }
    if isinstance(legacy_image, Mapping):
        canonical.update(
            {
                "storage_key": legacy_image.get("storage_key"),
                "public_url": legacy_image.get("public_url")
                or candidate.image_url,
                "sha256": legacy_image.get("sha256"),
                "byte_size": legacy_image.get("byte_size")
                if legacy_image.get("byte_size") is not None
                else legacy_image.get("bytes"),
                "content_type": legacy_image.get("content_type"),
            }
        )
    canonical = {key: value for key, value in canonical.items() if value is not None}
    if isinstance(existing_image, Mapping):
        canonical = {**dict(existing_image), **canonical}
    if not canonical:
        return dict(creative_patch)
    return {**dict(creative_patch), "image": canonical}


def _canonical_image_prompt_metadata(image_prompt: str | None) -> dict[str, Any]:
    if not image_prompt:
        return {}
    if image_prompt.startswith(RECOVERED_IMAGE_PROMPT_PREFIX):
        return {
            "prompt_sha256": image_prompt_sha256(image_prompt),
            "prompt_recovered": False,
        }
    return {"prompt": image_prompt}


def _optional_artifact_text(value: dict[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    text = str(item).strip()
    return text or None


def _validate_existing_candidate_identity(
    candidate: ContentCandidateRecord,
    *,
    identity: ArtifactIdentity,
    campaign_id: str,
    analysis_id: str,
    segment_id: str,
    content_option_id: str,
    channel: ContentChannel,
) -> None:
    expected = {
        "content_id": identity.content_id,
        "project_id": identity.project_id,
        "promotion_id": identity.promotion_id,
        "generation_id": identity.generation_id,
        "campaign_id": campaign_id,
        "analysis_id": analysis_id,
        "segment_id": segment_id,
        "content_option_id": content_option_id,
        "channel": channel,
    }
    mismatched = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(candidate, field_name) != expected_value
    ]
    if mismatched:
        raise ValueError(
            "existing generation candidate identity does not match the input snapshot"
        )


def _candidate_is_ready(candidate: ContentCandidateRecord) -> bool:
    try:
        _validate_durable_candidate_ready(candidate)
    except ValueError:
        return False
    return True


def _candidate_can_resume_image(candidate: ContentCandidateRecord) -> bool:
    return bool(
        candidate.channel != ContentChannel.SMS
        and candidate.image_prompt
        and candidate.image_generation_status in {"pending", "failed"}
        and candidate.artifact_status in {"pending", "failed"}
    )


def _candidate_can_resume_artifact(candidate: ContentCandidateRecord) -> bool:
    if candidate.channel == ContentChannel.SMS:
        return False
    return bool(
        candidate.image_generation_status == "completed"
        and candidate.image_url
        and candidate.artifact_status in {"pending", "failed", "published"}
        and candidate.artifact_error_code != "artifact_hash_conflict"
    )


def _validate_durable_candidate_ready(candidate: ContentCandidateRecord) -> None:
    if candidate.channel == ContentChannel.SMS:
        if (
            not candidate.message
            or candidate.creative_format != "sms_text"
            or candidate.image_generation_status != "not_required"
            or candidate.artifact_status != "not_required"
        ):
            raise ValueError("SMS candidate is not ready for completion")
        return

    if (
        candidate.image_generation_status != "completed"
        or not candidate.image_url
        or candidate.artifact_status != "published"
        or not candidate.artifact_storage_key
        or not candidate.artifact_public_url
        or not candidate.artifact_sha256
        or not candidate.artifact_content_type
        or candidate.artifact_published_at is None
    ):
        raise ValueError("HTML candidate artifact is not ready for completion")
