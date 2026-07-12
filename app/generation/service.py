from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence

from app.generation.artifacts import (
    CreativeArtifactPublisher,
    StaticCreativeArtifactPublisher,
    build_creative_metadata,
)
from app.generation.generator import (
    CONTENT_GENERATOR_VERSION,
    ContentGenerator,
    DeterministicContentGenerator,
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
)
from app.logging import log, log_context_scope, now_ms, duration_ms


MAX_CONTENT_IDENTIFIER_LENGTH = 100
CONTENT_SLUG_HASH_LENGTH = 16


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


@dataclass(frozen=True)
class NextLoopFocusGenerationResult:
    generation_id: str
    generated_segment_ids: list[str]
    status: GenerationStatus


class GenerationService:
    def __init__(
        self,
        *,
        generation_run_repository: GenerationRunWriter | None = None,
        content_candidate_repository: ContentCandidateWriter | None = None,
        generation_input_reader: GenerationInputReader | None = None,
        generation_input_builder: GenerationInputBuilder | None = None,
        generation_context_builder: GenerationContextBuilder | None = None,
        generation_strategy_planner: GenerationStrategyPlanner | None = None,
        prompt_builder: PromptBuilder | None = None,
        content_generator: ContentGenerator | None = None,
        artifact_publisher: CreativeArtifactPublisher | None = None,
        image_generation_scheduler: ImageGenerationScheduler | None = None,
        generation_report_builder: GenerationReportBuilder | None = None,
    ) -> None:
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository
        self._generation_input_reader = generation_input_reader
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
        try:
            content_candidates = self._build_content_candidate_records(
                request=request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
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
        generation_request = GenerationRequest(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            analysis_id=request.analysis_id,
            segment_ids=None,
            content_option_count=1,
            operator_instruction=request.operator_instruction,
        )
        generation_id = _generation_id_from_promotion(
            request.promotion_id,
            loop_count=request.loop_count,
        )
        log.assign_context({"generationId": generation_id})
        prompt_inputs = self._build_focus_prompt_inputs(
            request=generation_request,
            focus_segment_ids=request.focus_segment_ids,
        )
        log.info("generation_inputs_prepared", {"promptInputCount": len(prompt_inputs)})
        try:
            content_candidates = self._build_content_candidate_records(
                request=generation_request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
                # v1.7 section 6.8 does not specify a separate approval gate for
                # next-loop regenerated content. Temporarily approve the
                # single internal focus candidate so /next-loop can create
                # the next run; this may change if the team chooses draft +
                # explicit approval for next-loop content.
                candidate_status=ContentCandidateStatus.APPROVED,
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

        generation_report_json: dict[str, Any] = {
            "status": status.value,
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

        input_json: dict[str, Any] = {
            "project_id": request.project_id,
            "campaign_id": request.campaign_id,
            "promotion_id": request.promotion_id,
            "analysis_id": request.analysis_id,
            "content_option_count": request.content_option_count,
            "operator_instruction": request.operator_instruction,
            "target_segment_ids": [
                prompt_input.target_segment.segment_id
                for prompt_input in prompt_inputs
            ],
            "channel": prompt_inputs[0].promotion.channel.value
            if prompt_inputs
            else None,
        }
        if source_context is not None:
            input_json["next_loop"] = source_context

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

            return self._generation_input_builder.build(
                request=request,
                promotion=promotion,
                target_segments=target_segments,
            )

        return self._generation_input_builder.build(
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

        return self._generation_input_builder.build(
            request=request,
            promotion=promotion,
            target_segments=[
                target_segments_by_id[segment_id]
                for segment_id in focus_ids
            ],
        )

    def _build_content_candidate_records(
        self,
        *,
        request: GenerationRequest,
        generation_id: str,
        prompt_inputs: Sequence[GenerationPromptInput],
        candidate_status: ContentCandidateStatus = ContentCandidateStatus.DRAFT,
    ) -> list[ContentCandidateRecord]:
        records: list[ContentCandidateRecord] = []
        for raw_prompt_input in prompt_inputs:
            prompt_input = _prompt_input_with_resolved_landing_url(raw_prompt_input)
            generation_context = self._generation_context_builder.build(prompt_input)
            for index in range(1, request.content_option_count + 1):
                strategy_plan = self._generation_strategy_planner.build(
                    generation_context,
                    option_index=index,
                )
                records.append(
                    self._build_content_candidate_record(
                        generation_id=generation_id,
                        prompt_input=prompt_input,
                        generation_context=generation_context,
                        strategy_plan=strategy_plan,
                        index=index,
                        status=candidate_status,
                    )
                )
        return records

    def _build_content_candidate_record(
        self,
        *,
        generation_id: str,
        prompt_input: GenerationPromptInput,
        generation_context: GenerationContext,
        strategy_plan: GenerationStrategyPlan,
        index: int,
        status: ContentCandidateStatus,
    ) -> ContentCandidateRecord:
        prompt_result = self._prompt_builder.build(
            prompt_input,
            generation_context=generation_context,
            strategy_plan=strategy_plan,
        )
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
        generated_content = self._content_generator.generate(
            prompt_input=prompt_input,
            prompt_result=prompt_result,
            option_index=index,
        )
        content_values = generated_content.to_record_values(channel)
        candidate_report = self._generation_report_builder.build_candidate_report(
            prompt_input=prompt_input,
            prompt_result=prompt_result,
            content_id=content_id,
            content_option_id=content_option_id,
            content_generator_version=self._content_generator_version,
            content_values=content_values,
            status=status.value,
        )
        creative_metadata = build_creative_metadata(
            channel=channel,
            content_id=content_id,
            content_values=content_values,
            artifact_publisher=self._artifact_publisher,
        )

        return ContentCandidateRecord(
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
            metadata_json={
                **candidate_report.metadata_json,
                "creative": creative_metadata,
            },
            status=status.value,
        )

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
                        content_id=content_candidate.content_id,
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
    generation_run_number: int | None = None,
) -> str:
    promotion_slug = promotion_id.removeprefix("promo_")
    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", promotion_slug).strip("_")
    if loop_count is not None:
        return f"generation_{safe_slug or 'content'}_loop_{loop_count}"
    generation_id = f"generation_{safe_slug or 'content'}"
    if generation_run_number is not None and generation_run_number > 1:
        return f"{generation_id}_run_{generation_run_number}"
    return generation_id


def _general_generation_attempt_slug(
    *,
    generation_id: str,
    promotion_id: str,
) -> str | None:
    base_generation_id = _generation_id_from_promotion(promotion_id)
    prefix = f"{base_generation_id}_"
    if not generation_id.startswith(prefix):
        return None
    attempt_slug = generation_id.removeprefix(prefix)
    if re.fullmatch(r"(?:run|loop)_[2-9][0-9]*", attempt_slug):
        return attempt_slug
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
