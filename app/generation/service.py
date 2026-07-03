from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

from app.generation.repositories import (
    ContentCandidateRecord,
    GenerationRunRecord,
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
    ) -> None:
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        generation_id = _generation_id_from_promotion(request.promotion_id)
        content_candidates = self._build_content_candidate_records(
            request=request,
            generation_id=generation_id,
        )
        generation_run = self._build_generation_run_record(
            request=request,
            generation_id=generation_id,
            content_candidates=content_candidates,
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
        content_candidates: Sequence[ContentCandidateRecord],
    ) -> GenerationRunRecord:
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
            },
            output_json={
                "content_candidate_ids": [
                    candidate.content_id for candidate in content_candidates
                ],
            },
            generation_report_json={
                "status": GenerationStatus.COMPLETED.value,
                "content_candidate_count": len(content_candidates),
            },
            status=GenerationStatus.COMPLETED.value,
        )

    def _build_content_candidate_records(
        self,
        *,
        request: GenerationRequest,
        generation_id: str,
    ) -> list[ContentCandidateRecord]:
        channel = ContentChannel.ONSITE_BANNER
        channel_slug = "banner"
        segment_slug = "repeat_hotel"
        segment_id = "seg_repeat_hotel_no_booking"
        segment_name = "Repeat hotel viewers without booking"

        return [
            self._build_content_candidate_record(
                request=request,
                generation_id=generation_id,
                channel=channel,
                channel_slug=channel_slug,
                segment_slug=segment_slug,
                segment_id=segment_id,
                segment_name=segment_name,
                index=index,
            )
            for index in range(1, request.content_option_count + 1)
        ]

    def _build_content_candidate_record(
        self,
        *,
        request: GenerationRequest,
        generation_id: str,
        channel: ContentChannel,
        channel_slug: str,
        segment_slug: str,
        segment_id: str,
        segment_name: str,
        index: int,
    ) -> ContentCandidateRecord:
        content_id = f"content_{channel_slug}_{segment_slug}_{index:03d}"
        content_option_id = f"{channel_slug}_{segment_slug}_option_{index:03d}"
        title = "Book this weekend's rooms before they are gone"
        body = (
            "Show repeat hotel viewers a refundable summer offer while "
            "rooms are still available."
        )
        cta = "View hotel deals"
        image_prompt = "modern hotel room summer promotion banner, clean, bright, travel"
        landing_url = "https://demo-stay.example.com/summer"

        return ContentCandidateRecord(
            content_id=content_id,
            content_option_id=content_option_id,
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            analysis_id=request.analysis_id,
            generation_id=generation_id,
            segment_id=segment_id,
            channel=channel,
            title=title,
            body=body,
            cta=cta,
            image_prompt=image_prompt,
            landing_url=landing_url,
            generation_prompt=(
                "Create an onsite banner for repeat hotel viewers without booking. "
                "Highlight refundable rooms, urgency, and a clear hotel deals CTA."
            ),
            reason_summary=(
                "Uses the analysis target segment and hotel availability message "
                "to produce an onsite banner draft."
            ),
            data_evidence_json={
                "analysis_id": request.analysis_id,
                "segment_id": segment_id,
                "segment_name": segment_name,
            },
            message_strategy=(
                "Emphasize refundable booking and same-weekend hotel availability."
            ),
            metadata_json={
                "content_id": content_id,
                "content_option_id": content_option_id,
                "segment_id": segment_id,
                "segment_name": segment_name,
                "channel": channel.value,
                "title": title,
                "body": body,
                "cta": cta,
                "image_prompt": image_prompt,
                "landing_url": landing_url,
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
