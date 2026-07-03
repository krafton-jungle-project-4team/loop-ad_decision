from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence

from app.analysis.repositories import (
    HotelMarketingProfileRecord,
    PromotionAnalysisWrite,
    PromotionRecord,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
    SegmentVectorRecord,
)
from app.analysis.schemas import AnalysisRequest, AnalysisStatus, Channel, GoalMetric


MAX_DEFAULT_TARGET_SEGMENTS = 4

CUSTOM_SEGMENT_SOURCES = {"custom_chatkit", "manual_rule"}

DEFAULT_SEGMENT_IDS_BY_CHANNEL = {
    Channel.EMAIL.value: (
        "seg_mobile_user",
        "seg_family_trip",
        "seg_near_checkin",
        "seg_existing_all",
    ),
    Channel.ONSITE_BANNER.value: (
        "seg_family_trip",
        "seg_mobile_user",
        "seg_repeat_hotel_no_booking",
        "seg_near_checkin",
    ),
    Channel.SMS.value: (
        "seg_near_checkin",
        "seg_mobile_user",
        "seg_family_trip",
        "seg_existing_all",
    ),
}

SEGMENT_CONTENT_HINTS = {
    "seg_mobile_user": (
        "Reduce steps and emphasize mobile-friendly booking.",
        ("mobile booking", "quick checkout", "easy reservation"),
    ),
    "seg_family_trip": (
        "Highlight family rooms, breakfast, and flexible cancellation.",
        ("family room", "breakfast included", "flexible cancellation"),
    ),
    "seg_couple_trip": (
        "Highlight two-person stays and convenient hotel conditions.",
        ("couple stay", "hotel deal", "late checkout"),
    ),
    "seg_package_trip": (
        "Emphasize bundled stay benefits and clear package value.",
        ("package deal", "bundled stay", "travel value"),
    ),
    "seg_long_stay": (
        "Highlight benefits for longer stays and stable availability.",
        ("long stay", "weekly rate", "room availability"),
    ),
    "seg_near_checkin": (
        "Emphasize near check-in availability and low-friction booking.",
        ("near check-in", "same-day availability", "free cancellation"),
    ),
    "seg_repeat_hotel_no_booking": (
        "Emphasize free cancellation, same-day availability, and breakfast benefits.",
        ("free cancellation", "same-day availability", "breakfast included"),
    ),
    "seg_existing_all": (
        "Use a broad hotel booking message for existing users.",
        ("hotel deal", "seasonal stay", "booking benefit"),
    ),
}

RELATED_TERMS_BY_GOAL = {
    GoalMetric.INFLOW_RATE.value: (
        "inflow",
        "visit",
        "traffic",
        "click",
        "mobile",
        "banner",
        "email",
        "sms",
    ),
    GoalMetric.BOOKING_CONVERSION_RATE.value: (
        "booking",
        "book",
        "hotel",
        "stay",
        "checkin",
        "reservation",
    ),
    GoalMetric.FUNNEL_STEP_RATE.value: (
        "funnel",
        "step",
        "search",
        "detail",
        "booking",
        "checkin",
    ),
}


class PromotionReader(Protocol):
    def get_for_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionRecord | None:
        ...


class SegmentDefinitionReader(Protocol):
    def list_active(
        self,
        *,
        project_id: str,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        ...


class HotelProfileReader(Protocol):
    def list_marketing_profiles(
        self,
        *,
        project_id: str,
    ) -> list[HotelMarketingProfileRecord]:
        ...


class PromotionAnalysisWriter(Protocol):
    def save_analysis(self, analysis: PromotionAnalysisWrite) -> None:
        ...

    def save_target_segments(
        self,
        target_segments: Sequence[PromotionTargetSegmentWrite],
    ) -> None:
        ...


class SegmentVectorWriter(Protocol):
    def save(self, vector: SegmentVectorRecord) -> None:
        ...


class AnalysisRequestHandler(Protocol):
    def analyze(self, request: AnalysisRequest) -> "PromotionAnalysisResult":
        ...


@dataclass(frozen=True)
class PromotionAnalysisResult:
    analysis: PromotionAnalysisWrite
    target_segments: list[PromotionTargetSegmentWrite]


@dataclass(frozen=True)
class SegmentCandidate:
    definition: SegmentDefinitionRecord
    profile: HotelMarketingProfileRecord | None

    @property
    def segment_id(self) -> str:
        return self.definition.segment_id

    @property
    def estimated_size(self) -> int:
        return self.definition.sample_size


class PromotionNotFoundError(Exception):
    pass


class SegmentSelectionError(Exception):
    pass


class PromotionAnalysisService:
    def __init__(
        self,
        *,
        promotion_repository: PromotionReader,
        segment_definition_repository: SegmentDefinitionReader,
        hotel_profile_repository: HotelProfileReader,
        promotion_analysis_repository: PromotionAnalysisWriter,
        segment_vector_repository: SegmentVectorWriter | None = None,
        max_default_target_segments: int = MAX_DEFAULT_TARGET_SEGMENTS,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._segment_definition_repository = segment_definition_repository
        self._hotel_profile_repository = hotel_profile_repository
        self._promotion_analysis_repository = promotion_analysis_repository
        self._segment_vector_repository = segment_vector_repository
        self._max_default_target_segments = max_default_target_segments

    def analyze(self, request: AnalysisRequest) -> PromotionAnalysisResult:
        promotion = self._get_promotion(request)
        segment_definitions = self._segment_definition_repository.list_active(
            project_id=request.project_id,
        )
        hotel_profiles = self._hotel_profile_repository.list_marketing_profiles(
            project_id=request.project_id,
        )
        candidates = self._build_candidates(segment_definitions, hotel_profiles)
        selected_candidates = self._select_candidates(
            promotion=promotion,
            request=request,
            candidates=candidates,
        )
        if not selected_candidates:
            raise SegmentSelectionError(
                "no active segment candidates matched analysis request"
            )

        analysis_id = f"analysis_{promotion.promotion_id}"
        target_segments = [
            self._build_target_segment(
                analysis_id=analysis_id,
                promotion=promotion,
                candidate=candidate,
                rank=rank,
                operator_instruction=request.operator_instruction,
                segment_vector_id=_segment_vector_id(candidate.segment_id)
                if self._segment_vector_repository is not None
                else None,
            )
            for rank, candidate in enumerate(selected_candidates)
        ]
        analysis = self._build_analysis(
            analysis_id=analysis_id,
            promotion=promotion,
            request=request,
            segment_definitions=segment_definitions,
            candidates=candidates,
            target_segments=target_segments,
        )

        self._promotion_analysis_repository.save_analysis(analysis)
        self._save_segment_vectors(
            analysis_id=analysis_id,
            promotion=promotion,
            target_segments=target_segments,
        )
        self._promotion_analysis_repository.save_target_segments(target_segments)
        return PromotionAnalysisResult(
            analysis=analysis,
            target_segments=target_segments,
        )

    def _get_promotion(self, request: AnalysisRequest) -> PromotionRecord:
        promotion = self._promotion_repository.get_for_analysis(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
        )
        if promotion is None:
            raise PromotionNotFoundError(
                f"promotion not found for analysis: {request.promotion_id}"
            )
        return promotion

    def _build_candidates(
        self,
        segment_definitions: Sequence[SegmentDefinitionRecord],
        hotel_profiles: Sequence[HotelMarketingProfileRecord],
    ) -> dict[str, SegmentCandidate]:
        profiles_by_segment = {
            profile.profile_name: profile for profile in hotel_profiles
        }
        return {
            segment.segment_id: SegmentCandidate(
                definition=segment,
                profile=profiles_by_segment.get(segment.segment_id),
            )
            for segment in segment_definitions
        }

    def _select_candidates(
        self,
        *,
        promotion: PromotionRecord,
        request: AnalysisRequest,
        candidates: Mapping[str, SegmentCandidate],
    ) -> list[SegmentCandidate]:
        if request.focus_segment_ids:
            return [
                candidates[segment_id]
                for segment_id in request.focus_segment_ids
                if segment_id in candidates
            ]

        ordered_ids: list[str] = []
        for candidate in sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.estimated_size, candidate.segment_id),
        ):
            if self._is_related_custom_segment(candidate.definition, promotion):
                ordered_ids.append(candidate.segment_id)

        ordered_ids.extend(DEFAULT_SEGMENT_IDS_BY_CHANNEL.get(promotion.channel, ()))
        ordered_ids.extend(
            profile.profile_name
            for profile in sorted(
                (
                    candidate.profile
                    for candidate in candidates.values()
                    if candidate.profile
                ),
                key=lambda profile: (
                    -int(profile.profile_json.get("event_count", 0)),
                    profile.profile_name,
                ),
            )
        )
        ordered_ids.extend(
            candidate.segment_id
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: (-candidate.estimated_size, candidate.segment_id),
            )
            if candidate.definition.source in {"system_default", "ai_suggested"}
        )

        selected: list[SegmentCandidate] = []
        seen: set[str] = set()
        for segment_id in ordered_ids:
            if segment_id in seen or segment_id not in candidates:
                continue
            selected.append(candidates[segment_id])
            seen.add(segment_id)
            if len(selected) == self._max_default_target_segments:
                break
        return selected

    def _is_related_custom_segment(
        self,
        segment: SegmentDefinitionRecord,
        promotion: PromotionRecord,
    ) -> bool:
        if segment.source not in CUSTOM_SEGMENT_SOURCES:
            return False

        searchable = " ".join(
            [
                segment.segment_id,
                segment.segment_name,
                segment.natural_language_query or "",
                segment.generated_sql or "",
                str(segment.rule_json),
                str(segment.profile_json),
                promotion.message_brief or "",
                promotion.landing_url or "",
            ]
        ).lower()
        goal_terms = RELATED_TERMS_BY_GOAL.get(promotion.goal_metric, ())
        channel_terms = (promotion.channel.replace("_", " "), promotion.channel)
        return any(term in searchable for term in (*goal_terms, *channel_terms))

    def _build_target_segment(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        candidate: SegmentCandidate,
        rank: int,
        operator_instruction: str | None,
        segment_vector_id: str | None,
    ) -> PromotionTargetSegmentWrite:
        segment = candidate.definition
        content_brief_json = self._build_content_brief_json(
            segment=segment,
            operator_instruction=operator_instruction,
        )
        profile_json = dict(segment.profile_json)
        if candidate.profile is not None:
            profile_json["hotel_profile"] = dict(candidate.profile.profile_json)

        return PromotionTargetSegmentWrite(
            analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_id=segment.segment_id,
            segment_name=segment.segment_name,
            rule_json=segment.rule_json,
            profile_json=profile_json,
            content_brief_json=content_brief_json,
            data_evidence_json=self._build_data_evidence_json(candidate),
            segment_vector_id=segment_vector_id,
            estimated_size=max(segment.sample_size, 0),
            priority=self._priority_for_segment(
                estimated_size=segment.sample_size,
                min_sample_size=promotion.min_sample_size,
                rank=rank,
            ),
            status="planned",
        )

    def _build_content_brief_json(
        self,
        *,
        segment: SegmentDefinitionRecord,
        operator_instruction: str | None,
    ) -> dict[str, Any]:
        message_direction, keywords = SEGMENT_CONTENT_HINTS.get(
            segment.segment_id,
            (
                "Use a hotel booking message tailored to this segment.",
                ("hotel booking", "seasonal stay", "booking benefit"),
            ),
        )
        brief: dict[str, Any] = {
            "message_direction": message_direction,
            "keywords": list(keywords),
        }
        if operator_instruction:
            brief["operator_instruction"] = operator_instruction
        return brief

    def _build_data_evidence_json(
        self,
        candidate: SegmentCandidate,
    ) -> dict[str, Any]:
        segment = candidate.definition
        evidence: dict[str, Any] = {
            "source": segment.source,
            "sample_size": segment.sample_size,
            "sample_ratio": _json_decimal(segment.sample_ratio),
            "total_eligible_user_count": segment.total_eligible_user_count,
        }
        if candidate.profile is not None:
            evidence["hotel_profile"] = dict(candidate.profile.profile_json)
        return evidence

    def _priority_for_segment(
        self,
        *,
        estimated_size: int,
        min_sample_size: int,
        rank: int,
    ) -> str:
        if estimated_size < min_sample_size:
            return "low"
        if rank < 2:
            return "high"
        return "medium"

    def _build_analysis(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        request: AnalysisRequest,
        segment_definitions: Sequence[SegmentDefinitionRecord],
        candidates: Mapping[str, SegmentCandidate],
        target_segments: Sequence[PromotionTargetSegmentWrite],
    ) -> PromotionAnalysisWrite:
        return PromotionAnalysisWrite(
            analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            status=AnalysisStatus.COMPLETED.value,
            focus_segment_ids_json=request.focus_segment_ids,
            operator_instruction=request.operator_instruction,
            input_snapshot_json={
                "promotion": _promotion_snapshot(promotion),
                "available_segment_definitions": [
                    _segment_definition_snapshot(segment)
                    for segment in segment_definitions
                ],
                "focus_segment_ids": request.focus_segment_ids,
                "operator_instruction": request.operator_instruction,
            },
            profile_summary_json={
                "total_eligible_users": _total_eligible_users(segment_definitions),
                "candidate_segment_count": len(candidates),
                "selected_segment_count": len(target_segments),
                "reason": (
                    "Selected hotel audience segments by channel, goal metric, "
                    "and active segment definitions."
                ),
            },
            output_json={
                "selected_segment_ids": [
                    segment.segment_id for segment in target_segments
                ],
                "target_segment_count": len(target_segments),
            },
        )

    def _save_segment_vectors(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        target_segments: Sequence[PromotionTargetSegmentWrite],
    ) -> None:
        if self._segment_vector_repository is None:
            return
        for target_segment in target_segments:
            if target_segment.segment_vector_id is None:
                continue
            self._segment_vector_repository.save(
                SegmentVectorRecord(
                    segment_vector_id=target_segment.segment_vector_id,
                    project_id=promotion.project_id,
                    promotion_id=promotion.promotion_id,
                    promotion_run_id=None,
                    analysis_id=analysis_id,
                    segment_id=target_segment.segment_id,
                    vector_dim=64,
                    vector_values=_fixture_vector_values(target_segment.segment_id),
                    vector_version="v1",
                    source="decision_analysis",
                )
            )


def _promotion_snapshot(promotion: PromotionRecord) -> dict[str, Any]:
    return {
        "project_id": promotion.project_id,
        "campaign_id": promotion.campaign_id,
        "promotion_id": promotion.promotion_id,
        "channel": promotion.channel,
        "goal_metric": promotion.goal_metric,
        "goal_target_value": _json_decimal(promotion.goal_target_value),
        "goal_basis": promotion.goal_basis,
        "min_sample_size": promotion.min_sample_size,
        "landing_url": promotion.landing_url,
        "message_brief": promotion.message_brief,
    }


def _segment_definition_snapshot(segment: SegmentDefinitionRecord) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "segment_name": segment.segment_name,
        "source": segment.source,
        "sample_size": segment.sample_size,
        "total_eligible_user_count": segment.total_eligible_user_count,
        "sample_ratio": _json_decimal(segment.sample_ratio),
        "status": segment.status,
    }


def _total_eligible_users(
    segment_definitions: Sequence[SegmentDefinitionRecord],
) -> int:
    if not segment_definitions:
        return 0
    return max(segment.total_eligible_user_count for segment in segment_definitions)


def _json_decimal(value: Decimal) -> str:
    return format(value, "f")


def _segment_vector_id(segment_id: str) -> str:
    segment_slug = segment_id.removeprefix("seg_")
    return f"segvec_{segment_slug}_v1"


def _fixture_vector_values(segment_id: str) -> list[float]:
    digest = hashlib.sha256(segment_id.encode("utf-8")).digest()
    raw_values = [(digest[index % len(digest)] / 255.0) - 0.5 for index in range(64)]
    norm = math.sqrt(sum(value * value for value in raw_values)) or 1.0
    return [round(value / norm, 8) for value in raw_values]
