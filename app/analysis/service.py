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

CHANNEL_AFFINITY_BY_SEGMENT = {
    Channel.EMAIL.value: {
        "seg_mobile_user": 1.0,
        "seg_family_trip": 0.9,
        "seg_near_checkin": 0.65,
        "seg_existing_all": 0.6,
        "seg_package_trip": 0.82,
        "seg_long_stay": 0.78,
        "seg_repeat_hotel_no_booking": 0.68,
    },
    Channel.ONSITE_BANNER.value: {
        "seg_family_trip": 1.0,
        "seg_mobile_user": 0.9,
        "seg_repeat_hotel_no_booking": 0.72,
        "seg_near_checkin": 0.65,
        "seg_package_trip": 0.7,
        "seg_long_stay": 0.62,
        "seg_existing_all": 0.55,
    },
    Channel.SMS.value: {
        "seg_near_checkin": 1.0,
        "seg_mobile_user": 0.9,
        "seg_family_trip": 0.8,
        "seg_existing_all": 0.58,
        "seg_repeat_hotel_no_booking": 0.76,
        "seg_package_trip": 0.58,
        "seg_long_stay": 0.45,
    },
}

GOAL_AFFINITY_BY_SEGMENT = {
    GoalMetric.INFLOW_RATE.value: {
        "seg_mobile_user": 0.86,
        "seg_existing_all": 0.78,
        "seg_family_trip": 0.68,
        "seg_near_checkin": 0.64,
        "seg_repeat_hotel_no_booking": 0.6,
    },
    GoalMetric.BOOKING_CONVERSION_RATE.value: {
        "seg_repeat_hotel_no_booking": 0.95,
        "seg_near_checkin": 0.9,
        "seg_package_trip": 0.82,
        "seg_family_trip": 0.75,
        "seg_mobile_user": 0.65,
        "seg_existing_all": 0.48,
    },
    GoalMetric.FUNNEL_STEP_RATE.value: {
        "seg_near_checkin": 0.88,
        "seg_repeat_hotel_no_booking": 0.82,
        "seg_mobile_user": 0.76,
        "seg_family_trip": 0.72,
        "seg_existing_all": 0.55,
    },
}

FIT_SCORE_WEIGHTS = {
    "cluster_quality": 0.10,
    "sample_reliability": 0.20,
    "goal_alignment": 0.25,
    "channel_affinity": 0.35,
    "hotel_profile": 0.10,
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

    def summarize_user_ids(
        self,
        *,
        project_id: str,
        profile_name: str,
        user_ids: Sequence[str],
    ) -> HotelMarketingProfileRecord | None:
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


@dataclass(frozen=True)
class SegmentFitScore:
    promotion_fit_score: float
    cluster_quality_score: float
    sample_reliability_score: float
    goal_alignment_score: float
    channel_affinity_score: float
    hotel_profile_score: float
    rationale: tuple[str, ...]


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
        candidates = self._build_candidates(
            project_id=request.project_id,
            segment_definitions=segment_definitions,
            hotel_profiles=hotel_profiles,
        )
        fit_scores = _score_candidates(
            promotion=promotion,
            candidates=candidates,
        )
        selected_candidates = self._select_candidates(
            promotion=promotion,
            request=request,
            candidates=candidates,
            fit_scores=fit_scores,
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
                fit_score=fit_scores[selected_candidates[rank].segment_id],
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
        *,
        project_id: str,
        segment_definitions: Sequence[SegmentDefinitionRecord],
        hotel_profiles: Sequence[HotelMarketingProfileRecord],
    ) -> dict[str, SegmentCandidate]:
        profiles_by_segment = {profile.profile_name: profile for profile in hotel_profiles}
        candidates: dict[str, SegmentCandidate] = {}
        for segment in segment_definitions:
            profile = profiles_by_segment.get(segment.segment_id)
            if profile is None and segment.source == "ai_suggested":
                profile = self._summarize_ai_segment_profile(
                    project_id=project_id,
                    segment=segment,
                )
            candidates[segment.segment_id] = SegmentCandidate(
                definition=segment,
                profile=profile,
            )
        return candidates

    def _summarize_ai_segment_profile(
        self,
        *,
        project_id: str,
        segment: SegmentDefinitionRecord,
    ) -> HotelMarketingProfileRecord | None:
        candidate_user_ids = _candidate_user_ids(segment.rule_json)
        if not candidate_user_ids:
            return None
        return self._hotel_profile_repository.summarize_user_ids(
            project_id=project_id,
            profile_name=segment.segment_id,
            user_ids=candidate_user_ids,
        )

    def _select_candidates(
        self,
        *,
        promotion: PromotionRecord,
        request: AnalysisRequest,
        candidates: Mapping[str, SegmentCandidate],
        fit_scores: Mapping[str, SegmentFitScore],
    ) -> list[SegmentCandidate]:
        ordered_ids: list[str] = []
        for candidate in sorted(
            candidates.values(),
            key=lambda candidate: _fit_sort_key(candidate, fit_scores),
        ):
            if self._is_related_custom_segment(candidate.definition, promotion):
                ordered_ids.append(candidate.segment_id)

        ordered_ids.extend(
            candidate.segment_id
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: _fit_sort_key(candidate, fit_scores),
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
                key=lambda candidate: _fit_sort_key(candidate, fit_scores),
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
        fit_score: SegmentFitScore,
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
                "promotion_fit_score": fit_score.promotion_fit_score,
                "cluster_score": _ai_segment_score(segment),
                "cluster_quality_score": fit_score.cluster_quality_score,
                "sample_reliability_score": fit_score.sample_reliability_score,
                "goal_alignment_score": fit_score.goal_alignment_score,
                "channel_affinity_score": fit_score.channel_affinity_score,
                "hotel_profile_score": fit_score.hotel_profile_score,
            },
            reason_json={
                "channel": promotion.channel,
                "goal_metric": promotion.goal_metric,
                "segment_source": segment.source,
                "has_hotel_profile": candidate.profile is not None,
                "fit_model": "rule_based_v1",
                "fit_rationale": list(fit_score.rationale),
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


def _score_candidates(
    *,
    promotion: PromotionRecord,
    candidates: Mapping[str, SegmentCandidate],
) -> dict[str, SegmentFitScore]:
    max_profile_event_count = max(
        (
            int(candidate.profile.profile_json.get("event_count", 0))
            for candidate in candidates.values()
            if candidate.profile is not None
        ),
        default=0,
    )
    return {
        candidate.segment_id: _score_candidate(
            promotion=promotion,
            candidate=candidate,
            max_profile_event_count=max_profile_event_count,
        )
        for candidate in candidates.values()
    }


def _score_candidate(
    *,
    promotion: PromotionRecord,
    candidate: SegmentCandidate,
    max_profile_event_count: int,
) -> SegmentFitScore:
    cluster_quality_score = _clamp01(_ai_segment_score(candidate.definition))
    sample_reliability_score = _sample_reliability_score(
        sample_size=candidate.estimated_size,
        min_sample_size=promotion.min_sample_size,
    )
    goal_alignment_score = _goal_alignment_score(
        promotion=promotion,
        candidate=candidate,
    )
    channel_affinity_score = _channel_affinity_score(
        promotion=promotion,
        candidate=candidate,
    )
    hotel_profile_score = _hotel_profile_score(
        profile=candidate.profile,
        max_profile_event_count=max_profile_event_count,
    )
    promotion_fit_score = round(
        cluster_quality_score * FIT_SCORE_WEIGHTS["cluster_quality"]
        + sample_reliability_score * FIT_SCORE_WEIGHTS["sample_reliability"]
        + goal_alignment_score * FIT_SCORE_WEIGHTS["goal_alignment"]
        + channel_affinity_score * FIT_SCORE_WEIGHTS["channel_affinity"]
        + hotel_profile_score * FIT_SCORE_WEIGHTS["hotel_profile"],
        6,
    )
    return SegmentFitScore(
        promotion_fit_score=promotion_fit_score,
        cluster_quality_score=round(cluster_quality_score, 6),
        sample_reliability_score=round(sample_reliability_score, 6),
        goal_alignment_score=round(goal_alignment_score, 6),
        channel_affinity_score=round(channel_affinity_score, 6),
        hotel_profile_score=round(hotel_profile_score, 6),
        rationale=_fit_rationale(
            promotion=promotion,
            has_profile=candidate.profile is not None,
            cluster_quality_score=cluster_quality_score,
            sample_reliability_score=sample_reliability_score,
            goal_alignment_score=goal_alignment_score,
            channel_affinity_score=channel_affinity_score,
            hotel_profile_score=hotel_profile_score,
        ),
    )


def _fit_sort_key(
    candidate: SegmentCandidate,
    fit_scores: Mapping[str, SegmentFitScore],
) -> tuple[float, int, str]:
    fit_score = fit_scores[candidate.segment_id]
    return (
        -fit_score.promotion_fit_score,
        -candidate.estimated_size,
        candidate.segment_id,
    )


def _sample_reliability_score(
    *,
    sample_size: int,
    min_sample_size: int,
) -> float:
    if min_sample_size <= 0:
        return 1.0
    return _clamp01(sample_size / min_sample_size)


def _goal_alignment_score(
    *,
    promotion: PromotionRecord,
    candidate: SegmentCandidate,
) -> float:
    segment = candidate.definition
    segment_score = GOAL_AFFINITY_BY_SEGMENT.get(promotion.goal_metric, {}).get(
        segment.segment_id,
        0.0,
    )
    profile_score = _goal_profile_score(
        goal_metric=promotion.goal_metric,
        profile=candidate.profile,
    )
    text_score = _term_match_score(
        _candidate_searchable_text(segment),
        RELATED_TERMS_BY_GOAL.get(promotion.goal_metric, ()),
    )
    return _clamp01(max(segment_score, profile_score, text_score))


def _goal_profile_score(
    *,
    goal_metric: str,
    profile: HotelMarketingProfileRecord | None,
) -> float:
    if profile is None:
        return 0.0
    profile_json = profile.profile_json
    event_count = _float_value(profile_json.get("event_count"))
    booking_count = _float_value(profile_json.get("booking_count"))
    booking_rate = booking_count / event_count if event_count > 0 else 0.0
    mobile_ratio = _float_value(profile_json.get("mobile_ratio"))
    package_ratio = _float_value(profile_json.get("package_ratio"))
    avg_days_until_checkin = _float_value(
        profile_json.get("avg_days_until_checkin"),
        default=999.0,
    )

    if goal_metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
        return _clamp01(booking_rate / 0.08)
    if goal_metric == GoalMetric.INFLOW_RATE.value:
        return _clamp01((mobile_ratio * 0.7) + (package_ratio * 0.3))
    if goal_metric == GoalMetric.FUNNEL_STEP_RATE.value:
        near_checkin_score = 1.0 if 0 <= avg_days_until_checkin <= 7 else 0.4
        return _clamp01((mobile_ratio * 0.4) + (near_checkin_score * 0.6))
    return 0.0


def _channel_affinity_score(
    *,
    promotion: PromotionRecord,
    candidate: SegmentCandidate,
) -> float:
    segment = candidate.definition
    segment_score = CHANNEL_AFFINITY_BY_SEGMENT.get(promotion.channel, {}).get(
        segment.segment_id,
        0.0,
    )
    profile_score = _channel_profile_score(
        channel=promotion.channel,
        profile=candidate.profile,
    )
    channel_terms = (promotion.channel.replace("_", " "), promotion.channel)
    text_score = _term_match_score(_candidate_searchable_text(segment), channel_terms)
    return _clamp01(max(segment_score, profile_score, text_score))


def _channel_profile_score(
    *,
    channel: str,
    profile: HotelMarketingProfileRecord | None,
) -> float:
    if profile is None:
        return 0.0
    profile_json = profile.profile_json
    mobile_ratio = _float_value(profile_json.get("mobile_ratio"))
    package_ratio = _float_value(profile_json.get("package_ratio"))
    avg_stay_nights = _float_value(profile_json.get("avg_stay_nights"))
    avg_days_until_checkin = _float_value(
        profile_json.get("avg_days_until_checkin"),
        default=999.0,
    )

    if channel == Channel.SMS.value:
        near_checkin_score = 1.0 if 0 <= avg_days_until_checkin <= 7 else 0.3
        return _clamp01((mobile_ratio * 0.7) + (near_checkin_score * 0.3))
    if channel == Channel.EMAIL.value:
        long_stay_score = 1.0 if avg_stay_nights >= 3 else 0.5
        return _clamp01((package_ratio * 0.5) + (long_stay_score * 0.5))
    if channel == Channel.ONSITE_BANNER.value:
        return _clamp01((mobile_ratio * 0.5) + (package_ratio * 0.2) + 0.3)
    return 0.0


def _hotel_profile_score(
    *,
    profile: HotelMarketingProfileRecord | None,
    max_profile_event_count: int,
) -> float:
    if profile is None:
        return 0.0
    profile_json = profile.profile_json
    event_count = _float_value(profile_json.get("event_count"))
    event_score = (
        _clamp01(event_count / max_profile_event_count)
        if max_profile_event_count > 0
        else 0.0
    )
    metric_keys = (
        "booking_count",
        "mobile_ratio",
        "package_ratio",
        "avg_stay_nights",
        "avg_days_until_checkin",
    )
    completeness_score = sum(
        1 for key in metric_keys if profile_json.get(key) is not None
    ) / len(metric_keys)
    return _clamp01((event_score * 0.7) + (completeness_score * 0.3))


def _fit_rationale(
    *,
    promotion: PromotionRecord,
    has_profile: bool,
    cluster_quality_score: float,
    sample_reliability_score: float,
    goal_alignment_score: float,
    channel_affinity_score: float,
    hotel_profile_score: float,
) -> tuple[str, ...]:
    rationale: list[str] = []
    if cluster_quality_score >= 0.7:
        rationale.append("cluster_quality_high")
    if sample_reliability_score >= 1.0:
        rationale.append("sample_size_meets_minimum")
    elif sample_reliability_score < 0.5:
        rationale.append("sample_size_below_minimum")
    if goal_alignment_score >= 0.65:
        rationale.append(f"goal_{promotion.goal_metric}_aligned")
    if channel_affinity_score >= 0.65:
        rationale.append(f"channel_{promotion.channel}_aligned")
    if has_profile and hotel_profile_score > 0:
        rationale.append("hotel_profile_used")
    return tuple(rationale)


def _candidate_searchable_text(segment: SegmentDefinitionRecord) -> str:
    return " ".join(
        [
            segment.segment_id,
            segment.segment_name,
            segment.natural_language_query or "",
            segment.generated_sql or "",
            str(segment.rule_json),
            str(segment.profile_json),
        ]
    ).lower()


def _term_match_score(text: str, terms: Sequence[str]) -> float:
    if not terms:
        return 0.0
    matched_count = sum(1 for term in terms if term and term.lower() in text)
    return _clamp01(matched_count / min(len(terms), 3))


def _float_value(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
