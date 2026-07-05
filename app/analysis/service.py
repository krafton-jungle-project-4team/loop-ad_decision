from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence

from app.analysis.repositories import (
    HotelMarketingProfileRecord,
    PromotionAnalysisWrite,
    PromotionRecord,
    PromotionSegmentSuggestionWrite,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
)
from app.analysis.schemas import AnalysisRequest, AnalysisStatus, Channel, GoalMetric
from app.analysis.vector_service import SegmentVectorBuildRequest, SegmentVectorBuildResult


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
        campaign_id: str | None = None,
        promotion_id: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        ...

    def save_ai_suggested(
        self,
        segments: Sequence[SegmentDefinitionRecord],
    ) -> None:
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

    def save_segment_suggestions(
        self,
        suggestions: Sequence[PromotionSegmentSuggestionWrite],
    ) -> None:
        ...


class SegmentVectorPreparer(Protocol):
    def prepare_segment_vector(
        self,
        request: SegmentVectorBuildRequest,
    ) -> SegmentVectorBuildResult:
        ...


class SegmentDefinitionSuggester(Protocol):
    def suggest_segments(
        self,
        *,
        promotion: PromotionRecord,
    ) -> list[SegmentDefinitionRecord]:
        ...


@dataclass(frozen=True)
class PromotionAnalysisResult:
    analysis: PromotionAnalysisWrite
    target_segments: list[PromotionTargetSegmentWrite]
    segment_suggestions: list[PromotionSegmentSuggestionWrite] = field(default_factory=list)


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
        segment_vector_service: SegmentVectorPreparer | None = None,
        segment_suggester: SegmentDefinitionSuggester | None = None,
        max_default_target_segments: int = MAX_DEFAULT_TARGET_SEGMENTS,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._segment_definition_repository = segment_definition_repository
        self._hotel_profile_repository = hotel_profile_repository
        self._promotion_analysis_repository = promotion_analysis_repository
        self._segment_vector_service = segment_vector_service
        self._segment_suggester = segment_suggester
        self._max_default_target_segments = max_default_target_segments

    def analyze(self, request: AnalysisRequest) -> PromotionAnalysisResult:
        promotion = self._get_promotion(request)
        segment_definitions = self._segment_definition_repository.list_active(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
        )
        suggested_segment_definitions = self._suggest_segment_definitions(promotion)
        if suggested_segment_definitions:
            self._segment_definition_repository.save_ai_suggested(
                suggested_segment_definitions
            )
            segment_definitions = _merge_segment_definitions(
                segment_definitions,
                suggested_segment_definitions,
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
            raise SegmentSelectionError("no active segment candidates matched analysis request")

        analysis_id = f"analysis_{promotion.promotion_id}"
        analysis = self._build_analysis(
            analysis_id=analysis_id,
            promotion=promotion,
            request=request,
            segment_definitions=segment_definitions,
            candidates=candidates,
            selected_segment_ids=[
                candidate.segment_id for candidate in selected_candidates
            ],
        )

        self._promotion_analysis_repository.save_analysis(analysis)
        target_segments = [
            self._build_target_segment(
                analysis_id=analysis_id,
                promotion=promotion,
                candidate=candidate,
                rank=rank,
                operator_instruction=request.operator_instruction,
                segment_vector_id=self._prepare_segment_vector_id(
                    analysis_id=analysis_id,
                    promotion=promotion,
                    candidate=candidate,
                ),
            )
            for rank, candidate in enumerate(selected_candidates)
        ]
        segment_suggestions = [
            self._build_segment_suggestion(
                analysis_id=analysis_id,
                promotion=promotion,
                target_segment=target_segment,
                candidate=selected_candidates[rank],
                rank=rank,
            )
            for rank, target_segment in enumerate(target_segments)
        ]
        self._promotion_analysis_repository.save_segment_suggestions(segment_suggestions)
        return PromotionAnalysisResult(
            analysis=analysis,
            target_segments=target_segments,
            segment_suggestions=segment_suggestions,
        )

    def _suggest_segment_definitions(
        self,
        promotion: PromotionRecord,
    ) -> list[SegmentDefinitionRecord]:
        if self._segment_suggester is None:
            return []
        return self._segment_suggester.suggest_segments(promotion=promotion)

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
        profiles_by_segment = {profile.profile_name: profile for profile in hotel_profiles}
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
        ordered_ids: list[str] = []
        for candidate in sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.estimated_size, candidate.segment_id),
        ):
            if self._is_related_custom_segment(candidate.definition, promotion):
                ordered_ids.append(candidate.segment_id)

        ordered_ids.extend(
            candidate.segment_id
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: (
                    -_ai_segment_score(candidate.definition),
                    -candidate.estimated_size,
                    candidate.segment_id,
                ),
            )
            if candidate.definition.source == "ai_suggested"
        )
        ordered_ids.extend(DEFAULT_SEGMENT_IDS_BY_CHANNEL.get(promotion.channel, ()))
        ordered_ids.extend(
            profile.profile_name
            for profile in sorted(
                (candidate.profile for candidate in candidates.values() if candidate.profile),
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

    def _build_segment_suggestion(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        target_segment: PromotionTargetSegmentWrite,
        candidate: SegmentCandidate,
        rank: int,
    ) -> PromotionSegmentSuggestionWrite:
        segment = candidate.definition
        return PromotionSegmentSuggestionWrite(
            suggestion_id=_suggestion_id(
                analysis_id=analysis_id,
                segment_id=segment.segment_id,
            ),
            analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_id=segment.segment_id,
            suggested_rank=rank + 1,
            suggestion_source=_suggestion_source(segment),
            status="suggested",
            score_json={
                "rank": rank + 1,
                "estimated_size": target_segment.estimated_size,
                "priority": target_segment.priority,
                "cluster_score": _ai_segment_score(segment),
            },
            reason_json={
                "channel": promotion.channel,
                "goal_metric": promotion.goal_metric,
                "segment_source": segment.source,
                "has_hotel_profile": candidate.profile is not None,
            },
            metadata_json={
                "segment_name": target_segment.segment_name,
                "segment_vector_id": target_segment.segment_vector_id,
                "content_brief": target_segment.content_brief_json,
                "data_evidence": target_segment.data_evidence_json,
            },
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
        selected_segment_ids: Sequence[str],
    ) -> PromotionAnalysisWrite:
        return PromotionAnalysisWrite(
            analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            status=AnalysisStatus.COMPLETED.value,
            focus_segment_ids_json=None,
            operator_instruction=request.operator_instruction,
            input_snapshot_json={
                "promotion": _promotion_snapshot(promotion),
                "available_segment_definitions": [
                    _segment_definition_snapshot(segment)
                    for segment in segment_definitions
                ],
                "operator_instruction": request.operator_instruction,
            },
            profile_summary_json={
                "total_eligible_users": _total_eligible_users(segment_definitions),
                "candidate_segment_count": len(candidates),
                "selected_segment_count": len(selected_segment_ids),
                "reason": (
                    "Selected hotel audience segments by channel, goal metric, "
                    "and active segment definitions."
                ),
            },
            output_json={
                "selected_segment_ids": list(selected_segment_ids),
                "target_segment_count": len(selected_segment_ids),
            },
        )

    def _prepare_segment_vector_id(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        candidate: SegmentCandidate,
    ) -> str | None:
        if self._segment_vector_service is None:
            return None

        result = self._segment_vector_service.prepare_segment_vector(
            SegmentVectorBuildRequest(
                project_id=promotion.project_id,
                promotion_id=promotion.promotion_id,
                analysis_id=analysis_id,
                segment_id=candidate.segment_id,
                candidate_user_ids=_candidate_user_ids(candidate.definition.rule_json),
            )
        )
        return result.segment_vector_id


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
        "campaign_id": segment.campaign_id,
        "promotion_id": segment.promotion_id,
        "segment_name": segment.segment_name,
        "source": segment.source,
        "sample_size": segment.sample_size,
        "total_eligible_user_count": segment.total_eligible_user_count,
        "sample_ratio": _json_decimal(segment.sample_ratio),
        "status": segment.status,
    }


def _merge_segment_definitions(
    stored_segments: Sequence[SegmentDefinitionRecord],
    suggested_segments: Sequence[SegmentDefinitionRecord],
) -> list[SegmentDefinitionRecord]:
    merged = {segment.segment_id: segment for segment in stored_segments}
    for segment in suggested_segments:
        existing = merged.get(segment.segment_id)
        if existing is None or existing.source == "ai_suggested":
            merged[segment.segment_id] = segment
    return list(merged.values())


def _ai_segment_score(segment: SegmentDefinitionRecord) -> float:
    raw_score = segment.profile_json.get("cluster_score", 0.0)
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def _suggestion_source(segment: SegmentDefinitionRecord) -> str:
    if segment.source == "ai_suggested":
        return "ai_generated"
    return "ai_ranked_existing"


def _suggestion_id(*, analysis_id: str, segment_id: str) -> str:
    digest = hashlib.sha1(  # noqa: S324 - stable non-security identifier.
        f"{analysis_id}:{segment_id}".encode("utf-8"),
    ).hexdigest()[:24]
    return f"sugg_{digest}"


def _total_eligible_users(
    segment_definitions: Sequence[SegmentDefinitionRecord],
) -> int:
    if not segment_definitions:
        return 0
    return max(segment.total_eligible_user_count for segment in segment_definitions)


def _candidate_user_ids(rule_json: Mapping[str, Any]) -> list[str]:
    raw_user_ids = rule_json.get("candidate_user_ids") or rule_json.get("user_ids")
    if isinstance(raw_user_ids, str) or not isinstance(raw_user_ids, Sequence):
        return []
    return [str(user_id) for user_id in raw_user_ids]


def _json_decimal(value: Decimal) -> str:
    return format(value, "f")
