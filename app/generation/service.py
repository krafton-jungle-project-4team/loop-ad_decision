from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

from app.generation.generator import (
    CONTENT_GENERATOR_VERSION,
    ContentGenerator,
    DeterministicContentGenerator,
)
from app.generation.repositories import (
    ContentCandidateRecord,
    GenerationRunRecord,
)
from app.generation.prompt_builder import (
    GenerationInputBuilder,
    GenerationPromptInput,
    PROMPT_BUILDER_VERSION,
    PromptBuilder,
    PromotionPromptInput,
    TargetSegmentPromptInput,
)
from app.generation.schemas import (
    ContentCandidateResponse,
    ContentCandidateStatus,
    ContentChannel,
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
)


class GenerationRunWriter(Protocol):
    def create(self, record: GenerationRunRecord) -> dict[str, Any]:
        ...


class ContentCandidateWriter(Protocol):
    def create(self, record: ContentCandidateRecord) -> dict[str, Any]:
        ...


class GenerationRequestHandler(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        ...


class GenerationService:
    def __init__(
        self,
        *,
        generation_run_repository: GenerationRunWriter | None = None,
        content_candidate_repository: ContentCandidateWriter | None = None,
        generation_input_builder: GenerationInputBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        content_generator: ContentGenerator | None = None,
    ) -> None:
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository
        self._generation_input_builder = (
            generation_input_builder or GenerationInputBuilder()
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._content_generator = content_generator or DeterministicContentGenerator()
        self._content_generator_version = _content_generator_version(
            self._content_generator
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        generation_id = _generation_id_from_promotion(request.promotion_id)
        prompt_inputs = self._build_prompt_inputs(request)
        try:
            content_candidates = self._build_content_candidate_records(
                request=request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
            )
        except Exception as exc:
            generation_run = self._build_generation_run_record(
                request=request,
                generation_id=generation_id,
                prompt_inputs=prompt_inputs,
                content_candidates=[],
                status=GenerationStatus.FAILED,
                error_code=_safe_generation_error_code(exc),
            )
            self._save_generation_run(generation_run)
            return GenerationResponse(
                generation_id=generation_id,
                promotion_id=request.promotion_id,
                status=GenerationStatus.FAILED,
                content_candidates=[],
            )

        generation_run = self._build_generation_run_record(
            request=request,
            generation_id=generation_id,
            prompt_inputs=prompt_inputs,
            content_candidates=content_candidates,
            status=GenerationStatus.COMPLETED,
        )
        self._save_generation_run(generation_run)
        self._save_content_candidates(content_candidates)

        return GenerationResponse(
            generation_id=generation_id,
            promotion_id=request.promotion_id,
            status=GenerationStatus.COMPLETED,
            content_candidates=[
                ContentCandidateResponse.model_validate(candidate.to_public_values())
                for candidate in content_candidates
            ],
        )

    def _build_generation_run_record(
        self,
        *,
        request: GenerationRequest,
        generation_id: str,
        prompt_inputs: Sequence[GenerationPromptInput],
        content_candidates: Sequence[ContentCandidateRecord],
        status: GenerationStatus,
        error_code: str | None = None,
    ) -> GenerationRunRecord:
        output_json: dict[str, Any] = {
            "content_candidate_ids": [
                candidate.content_id for candidate in content_candidates
            ],
        }
        if error_code:
            output_json["error_code"] = error_code

        generation_report_json: dict[str, Any] = {
            "status": status.value,
            "content_candidate_count": len(content_candidates),
            "target_segment_count": len(prompt_inputs),
            "prompt_builder": PROMPT_BUILDER_VERSION,
            "content_generator": self._content_generator_version,
        }
        if error_code:
            generation_report_json["error_code"] = error_code

        return GenerationRunRecord(
            generation_id=generation_id,
            analysis_id=request.analysis_id,
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            content_option_count=request.content_option_count,
            operator_instruction=request.operator_instruction,
            input_json={
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
            },
            output_json=output_json,
            generation_report_json=generation_report_json,
            status=status.value,
        )

    def _build_prompt_inputs(
        self,
        request: GenerationRequest,
    ) -> list[GenerationPromptInput]:
        return self._generation_input_builder.build(
            request=request,
            promotion=_fixture_promotion_prompt_input(request),
            target_segments=[_fixture_target_segment_prompt_input(request)],
        )

    def _build_content_candidate_records(
        self,
        *,
        request: GenerationRequest,
        generation_id: str,
        prompt_inputs: Sequence[GenerationPromptInput],
    ) -> list[ContentCandidateRecord]:
        return [
            self._build_content_candidate_record(
                generation_id=generation_id,
                prompt_input=prompt_input,
                index=index,
            )
            for prompt_input in prompt_inputs
            for index in range(1, request.content_option_count + 1)
        ]

    def _build_content_candidate_record(
        self,
        *,
        generation_id: str,
        prompt_input: GenerationPromptInput,
        index: int,
    ) -> ContentCandidateRecord:
        prompt_result = self._prompt_builder.build(prompt_input)
        channel = prompt_input.promotion.channel
        channel_slug = _channel_slug(channel)
        segment_slug = _segment_slug(prompt_input.target_segment)
        segment_id = prompt_input.target_segment.segment_id
        segment_name = prompt_input.target_segment.segment_name
        content_id = f"content_{channel_slug}_{segment_slug}_{index:03d}"
        content_option_id = f"{channel_slug}_{segment_slug}_option_{index:03d}"
        generated_content = self._content_generator.generate(
            prompt_input=prompt_input,
            prompt_result=prompt_result,
            option_index=index,
        )
        content_values = generated_content.to_record_values(channel)

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
            landing_url=content_values["landing_url"],
            generation_prompt=prompt_result.generation_prompt,
            reason_summary=prompt_result.reason_summary,
            data_evidence_json=prompt_result.data_evidence_json,
            message_strategy=prompt_result.message_strategy,
            metadata_json={
                **prompt_result.metadata_json,
                "content_id": content_id,
                "content_option_id": content_option_id,
                "segment_id": segment_id,
                "segment_name": segment_name,
                "channel": channel.value,
                "content_generator_version": self._content_generator_version,
                **content_values,
                "status": ContentCandidateStatus.DRAFT.value,
            },
            status=ContentCandidateStatus.DRAFT.value,
        )

    def _save_generation_run(self, generation_run: GenerationRunRecord) -> None:
        if self._generation_run_repository is None:
            return
        self._generation_run_repository.create(generation_run)

    def _save_content_candidates(
        self,
        content_candidates: Sequence[ContentCandidateRecord],
    ) -> None:
        if self._content_candidate_repository is None:
            return
        for content_candidate in content_candidates:
            self._content_candidate_repository.create(content_candidate)


def _generation_id_from_promotion(promotion_id: str) -> str:
    promotion_slug = promotion_id.removeprefix("promo_")
    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", promotion_slug).strip("_")
    return f"generation_{safe_slug or 'content'}"


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


def _safe_generation_error_code(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return "content_generation_validation_failed"
    return "content_generation_failed"


def _content_generator_version(content_generator: ContentGenerator) -> str:
    version = str(getattr(content_generator, "version", CONTENT_GENERATOR_VERSION))
    return version.strip() or CONTENT_GENERATOR_VERSION
