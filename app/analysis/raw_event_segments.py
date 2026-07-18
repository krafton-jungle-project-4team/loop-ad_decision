from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

from app.audience_contract import (
    SEGMENT_AUDIENCE_CONTRACT,
    SegmentAudienceContractError,
)
from app.analysis.audience_selection import (
    AudienceSelectionDecision,
    AudienceSelectionPolicyProtocol,
    all_matching_audience_selection_policy,
)
from app.analysis.repositories import (
    PromotionRecord,
    RawEventUserSignalRecord,
    SegmentDefinitionRecord,
)
from app.analysis.segment_audience_templates import (
    RegisteredSegmentAudienceBinder,
)
from app.analysis.segment_performance import (
    ContextualBookingHeuristicPredictor,
    SegmentPerformanceFeatures,
    SegmentPerformancePredictor,
    candidate_type_prediction_support,
    predict_segment_performance,
)
from app.config import Settings
from app.generation.adapters import (
    OPENAI_RESPONSES_URL,
    JsonTransport,
    _parse_output_json,
    _post_json,
)
from app.logging import duration_ms, log, log_context_scope


RAW_EVENT_SEGMENT_VERSION = "raw-event-segment.v7"
RAW_EVENT_INTENT_COMPILER_VERSION = "raw-event-intent.v2"
INTENT_EXTRACTOR_VERSION = "dec.segment-intent.v2"
EXPECTED_RATE_PRIOR_USER_COUNT = 30.0
PRIMARY_RECOMMENDATION_MIN_RELIABILITY = 0.75
MAX_RANK_USER_OVERLAP = 0.70

CANDIDATE_TYPE_ORDER = (
    "intent_matched",
    "target_destination_affinity",
    "funnel_recovery",
    "benefit_value_seeker",
    "promotion_responsive",
    "general_destination_explorer",
)
EXCLUDED_BEHAVIOR_VALUES = (
    "booking_complete",
    "booking_cancel",
    "booking_start",
    "promotion_response",
    "hotel_search",
    "hotel_detail_view",
)
AUDIENCE_HINT_VALUES = (
    "20s_30s",
    "male",
    "female",
    "travel_ready",
)

DEFAULT_SCORE_WEIGHTS: Mapping[str, float] = {
    "promotion_condition_match": 0.15,
    "expected_goal_performance": 0.70,
    "behavior_lift_vs_baseline": 0.06,
    "sample_reliability": 0.06,
    "rank_distinctiveness": 0.03,
}

# Destination-specific candidates already require a matching destination event.
# Keep condition fit as a tie-breaker instead of counting the same signal twice.
DESTINATION_SCORE_WEIGHTS: Mapping[str, float] = {
    "promotion_condition_match": 0.10,
    "expected_goal_performance": 0.85,
    "behavior_lift_vs_baseline": 0.02,
    "sample_reliability": 0.02,
    "rank_distinctiveness": 0.01,
}

CANDIDATE_TYPE_LABELS: Mapping[str, Mapping[str, str]] = {
    "intent_matched": {
        "strategy_role": "프로모션 조건 정합형",
        "fallback_title": "프로모션 조건과 맞는 숙소 관심 고객",
    },
    "funnel_recovery": {
        "strategy_role": "예약 이탈 회수형",
        "fallback_title": "예약 직전 이탈 고객",
    },
    "promotion_responsive": {
        "strategy_role": "프로모션 반응 확장형",
        "fallback_title": "프로모션 반응이 확인된 고객",
    },
    "target_destination_affinity": {
        "strategy_role": "이번 목적지 반복 관심형",
        "fallback_title": "이번 여행지를 반복 탐색한 고객",
    },
    "general_destination_explorer": {
        "strategy_role": "다목적지 탐색 확장형",
        "fallback_title": "여러 여행지를 비교 탐색한 고객",
    },
    "benefit_value_seeker": {
        "strategy_role": "혜택 민감형",
        "fallback_title": "할인과 혜택에 반응할 고객",
    },
}

# Stable ontology labels used for deterministic fallbacks and UI chips.
# These labels do not select users or decide candidate selection.
CONDITION_LABELS: Mapping[str, tuple[str, str]] = {
    "hotel_product_interest": ("숙소 관심 행동", "숙소 관심"),
    "recent_destination_search": ("목적지 숙소 검색", "목적지 검색"),
    "summer_checkin_search": ("여름 체크인 숙소 검색", "여름 체크인"),
    "winter_checkin_search": ("겨울 체크인 숙소 검색", "겨울 체크인"),
    "hotel_detail_view": ("호텔 상세 조회", "호텔 상세 조회"),
    "promotion_response": ("프로모션 반응", "프로모션 반응"),
    "campaign_landing": ("캠페인 랜딩", "캠페인 랜딩"),
    "booking_start_without_complete": ("예약 시작 후 미완료", "예약 미완료"),
    "target_destination_affinity": ("이번 목적지 반복 관심", "이번 목적지 반복"),
    "general_destination_exploration": ("여러 목적지 비교 탐색", "다목적지 탐색"),
    "benefit_interest": ("혜택 조건 관심", "혜택 관심"),
    "price_sensitive": ("가격 비교 행동", "가격 비교"),
    "free_cancellation_interest": ("무료 취소 관심", "무료 취소"),
    "breakfast_interest": ("조식 포함 관심", "조식 포함"),
    "profile_hint": ("고객 프로필 조건", "프로필 조건"),
}

SEASON_MONTHS: Mapping[str, tuple[int, ...]] = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "fall": (9, 10, 11),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}

DESTINATION_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "jeju": ("jeju", "제주"),
    "okinawa": ("okinawa", "오키나와"),
    "japan": ("japan", "일본", "도쿄", "오사카", "삿포로"),
    "busan": ("busan", "부산"),
    "seoul": ("seoul", "서울"),
    "gangneung": ("gangneung", "강릉"),
    "yeosu": ("yeosu", "여수"),
}


class PromotionIntentExtractor(Protocol):
    def extract(
        self,
        promotion: PromotionRecord,
        *,
        segment_instruction: str | None = None,
    ) -> "PromotionIntent":
        ...


@dataclass(frozen=True)
class PromotionIntent:
    summary: str
    product: str
    season: tuple[str, ...]
    destinations: tuple[str, ...]
    benefits: tuple[str, ...]
    audience_hints: tuple[str, ...]
    channel: str
    goal_metric: str
    funnel_goal: str
    desired_behaviors: tuple[str, ...]
    explicit_conditions: tuple[str, ...]
    excluded_behaviors: tuple[str, ...] = ()
    requested_candidate_types: tuple[str, ...] = ()
    source: str = "deterministic"

    def to_json(self) -> dict[str, Any]:
        return {
            "version": INTENT_EXTRACTOR_VERSION,
            "source": self.source,
            "summary": self.summary,
            "product": self.product,
            "season": list(self.season),
            "destinations": list(self.destinations),
            "benefits": list(self.benefits),
            "audience_hints": list(self.audience_hints),
            "channel": self.channel,
            "goal_metric": self.goal_metric,
            "funnel_goal": self.funnel_goal,
            "desired_behaviors": list(self.desired_behaviors),
            "excluded_behaviors": list(self.excluded_behaviors),
            "explicit_conditions": list(self.explicit_conditions),
            "requested_candidate_types": list(self.requested_candidate_types),
        }


@dataclass(frozen=True)
class CompiledRawEventCondition:
    key: str
    label: str
    support: str
    event_names: tuple[str, ...]
    property_keys: tuple[str, ...]
    weight: float

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "support": self.support,
            "event_names": list(self.event_names),
            "property_keys": list(self.property_keys),
            "weight": round(self.weight, 6),
        }


@dataclass(frozen=True)
class RawEventIntentCompilation:
    compiled_conditions: tuple[CompiledRawEventCondition, ...]
    unsupported_conditions: tuple[str, ...]
    compiler_version: str = RAW_EVENT_INTENT_COMPILER_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "compiled_conditions": [
                condition.to_json() for condition in self.compiled_conditions
            ],
            "unsupported_conditions": list(self.unsupported_conditions),
            "compiler_version": self.compiler_version,
        }


@dataclass(frozen=True)
class RawEventAudienceParameters:
    destination_ids: tuple[str, ...] = ()
    season_months: tuple[int, ...] = ()
    benefit_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RawEventCandidate:
    candidate_type: str
    strategy_role: str
    title: str
    reason: str
    action_hint: str
    candidate_user_ids: tuple[str, ...]
    matched_condition_keys: tuple[str, ...]
    missing_condition_keys: tuple[str, ...]
    signal_chips: tuple[str, ...]
    signal_metrics: Mapping[str, Any]
    promotion_condition_match: float
    predicted_goal_rate: float
    expected_goal_performance: float
    behavior_lift_vs_baseline: float
    sample_reliability: float
    destination_context_required: bool
    performance_features: SegmentPerformanceFeatures
    performance_model_metadata: Mapping[str, Any]
    audience_selection: AudienceSelectionDecision
    audience_parameters: RawEventAudienceParameters
    rank_distinctiveness: float = 1.0

    @property
    def sample_size(self) -> int:
        return len(self.candidate_user_ids)


class DeterministicPromotionIntentExtractor:
    def extract(
        self,
        promotion: PromotionRecord,
        *,
        segment_instruction: str | None = None,
    ) -> PromotionIntent:
        return _fallback_intent(
            promotion=promotion,
            segment_instruction=segment_instruction,
            source="deterministic",
        )


class OpenAIPromotionIntentExtractor:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = OPENAI_RESPONSES_URL,
        timeout_seconds: float = 15.0,
        fallback_extractor: PromotionIntentExtractor | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._fallback_extractor = fallback_extractor or DeterministicPromotionIntentExtractor()
        self._transport = transport or _post_json

    @log_context_scope
    def extract(
        self,
        promotion: PromotionRecord,
        *,
        segment_instruction: str | None = None,
    ) -> PromotionIntent:
        payload = {
            "model": self._model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _intent_system_instruction(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _intent_user_instruction(
                                promotion,
                                segment_instruction=segment_instruction,
                            ),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "promotion_segment_intent",
                    "strict": True,
                    "schema": _intent_schema(),
                }
            },
            "temperature": 0.1,
            "max_output_tokens": 900,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started_at = perf_counter()
        log.assign_context(
            {
                "promotionId": promotion.promotion_id,
                "provider": "openai",
                "model": self._model,
            }
        )
        log.info(
            "provider_request_prepared",
            {
                "providerOperation": "promotion_intent_extraction",
                "endpoint": self._endpoint,
            },
        )
        try:
            response_payload = self._transport(
                self._endpoint,
                headers,
                payload,
                self._timeout_seconds,
            )
            intent = _intent_from_payload(
                _parse_output_json(response_payload),
                promotion=promotion,
                segment_instruction=segment_instruction,
                source="openai",
            )
        except Exception as exc:
            log.warn(
                "provider_request_failed",
                {
                    "providerOperation": "promotion_intent_extraction",
                    "endpoint": self._endpoint,
                    "err": exc,
                    "durationMs": duration_ms(started_at),
                    "fallback": "deterministic",
                },
            )
            return self._fallback_extractor.extract(
                promotion,
                segment_instruction=segment_instruction,
            )
        log.info(
            "provider_request_completed",
            {
                "providerOperation": "promotion_intent_extraction",
                "endpoint": self._endpoint,
                "durationMs": duration_ms(started_at),
            },
        )
        return intent


def build_promotion_intent_extractor(settings: Settings) -> PromotionIntentExtractor:
    if settings.env == "test" or _is_placeholder_api_key(settings.openai_api_key):
        return DeterministicPromotionIntentExtractor()
    return OpenAIPromotionIntentExtractor(
        api_key=settings.openai_api_key,
        model=settings.openai_content_model,
    )


def compile_raw_event_intent(intent: PromotionIntent) -> RawEventIntentCompilation:
    conditions: list[CompiledRawEventCondition] = [
        CompiledRawEventCondition(
            key="hotel_product_interest",
            label=CONDITION_LABELS["hotel_product_interest"][0],
            support="direct",
            event_names=("hotel_search", "hotel_click", "hotel_detail_view"),
            property_keys=("destination_id", "destination_name", "hotel_city"),
            weight=0.18,
        )
    ]
    unsupported: list[str] = []

    if intent.destinations:
        conditions.append(
            CompiledRawEventCondition(
                key="recent_destination_search",
                label=CONDITION_LABELS["recent_destination_search"][0],
                support="direct",
                event_names=("hotel_search", "hotel_detail_view"),
                property_keys=("destination_id", "destination_name", "hotel_city", "hotel_country"),
                weight=0.32,
            )
        )

    if "summer" in intent.season:
        conditions.append(
            CompiledRawEventCondition(
                key="summer_checkin_search",
                label=CONDITION_LABELS["summer_checkin_search"][0],
                support="direct",
                event_names=("hotel_search",),
                property_keys=("checkin_date", "checkout_date"),
                weight=0.15,
            )
        )
    if "winter" in intent.season:
        conditions.append(
            CompiledRawEventCondition(
                key="winter_checkin_search",
                label=CONDITION_LABELS["winter_checkin_search"][0],
                support="direct",
                event_names=("hotel_search",),
                property_keys=("checkin_date", "checkout_date"),
                weight=0.15,
            )
        )

    desired_conditions = {
        "hotel_detail_view": ("hotel_detail_view", ("hotel_detail_view",), ()),
        "promotion_response": (
            "promotion_response",
            ("promotion_click", "campaign_landing"),
            (),
        ),
        "booking_start_without_complete": (
            "booking_start_without_complete",
            ("booking_start", "booking_complete"),
            (),
        ),
        "recent_destination_search": (
            "recent_destination_search",
            ("hotel_search",),
            ("destination_id", "destination_name"),
        ),
        "price_sensitive": (
            "price_sensitive",
            ("hotel_search", "hotel_click"),
            ("deal", "price"),
        ),
    }
    for desired_behavior in intent.desired_behaviors:
        condition = desired_conditions.get(desired_behavior)
        if condition is None:
            unsupported.append(desired_behavior)
            continue
        key, event_names, property_keys = condition
        if any(existing.key == key for existing in conditions):
            continue
        conditions.append(
            CompiledRawEventCondition(
                key=key,
                label=CONDITION_LABELS.get(key, (key, key))[0],
                support="direct",
                event_names=tuple(event_names),
                property_keys=tuple(property_keys),
                weight=0.16,
            )
        )

    if any(benefit in intent.benefits for benefit in ("discount", "early_booking")):
        conditions.append(
            CompiledRawEventCondition(
                key="benefit_interest",
                label=CONDITION_LABELS["benefit_interest"][0],
                support="direct",
                event_names=("hotel_search", "promotion_click"),
                property_keys=("deal", "price", "free_cancellation", "breakfast_included"),
                weight=0.14,
            )
        )

    if any(hint in intent.audience_hints for hint in ("20s_30s", "male", "female")):
        conditions.append(
            CompiledRawEventCondition(
                key="profile_hint",
                label=CONDITION_LABELS["profile_hint"][0],
                support="direct",
                event_names=(),
                property_keys=("age_group", "gender", "user_segment", "preferred_category"),
                weight=0.08,
            )
        )

    return RawEventIntentCompilation(
        compiled_conditions=tuple(_dedupe_conditions(conditions)),
        unsupported_conditions=tuple(dict.fromkeys(unsupported)),
    )


def generate_raw_event_segment_definitions(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    max_suggested_segments: int,
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor | None = None,
    audience_selection_policy: AudienceSelectionPolicyProtocol | None = None,
) -> list[SegmentDefinitionRecord]:
    candidates = _generate_raw_event_candidates(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        min_sample_size=min_sample_size,
        performance_predictor=performance_predictor,
        audience_selection_policy=audience_selection_policy,
        enforce_prediction_support=True,
    )
    if not candidates:
        return []
    total_eligible_user_count = len(profiles)
    selected_candidates = _select_candidate_portfolio(
        candidates,
        max_suggested_segments=max_suggested_segments,
    )
    return [
        _segment_definition_from_candidate(
            promotion=promotion,
            intent=intent,
            compilation=compilation,
            candidate=candidate,
            position=position,
            total_eligible_user_count=total_eligible_user_count,
            selected_candidates=selected_candidates,
        )
        for position, candidate in enumerate(selected_candidates)
    ]


def generate_raw_event_segment_candidate_pool(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor | None = None,
    enforce_prediction_support: bool = True,
) -> list[SegmentDefinitionRecord]:
    """Return every eligible candidate before overlap and rank pruning."""
    candidates = _generate_raw_event_candidates(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        min_sample_size=min_sample_size,
        performance_predictor=performance_predictor,
        audience_selection_policy=None,
        enforce_prediction_support=enforce_prediction_support,
    )
    total_eligible_user_count = len(profiles)
    return [
        _segment_definition_from_candidate(
            promotion=promotion,
            intent=intent,
            compilation=compilation,
            candidate=candidate,
            position=position,
            total_eligible_user_count=total_eligible_user_count,
            selected_candidates=None,
        )
        for position, candidate in enumerate(candidates)
    ]


def _generate_raw_event_candidates(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor | None,
    audience_selection_policy: AudienceSelectionPolicyProtocol | None,
    enforce_prediction_support: bool,
) -> list[_RawEventCandidate]:
    if len(profiles) < min_sample_size:
        return []
    baseline = _baseline_metrics(profiles)
    eligible_profiles = _exclude_profiles(profiles, intent.excluded_behaviors)
    if len(eligible_profiles) < min_sample_size:
        return []
    predictor = performance_predictor or ContextualBookingHeuristicPredictor()
    candidate_factories: tuple[
        tuple[str, Callable[..., _RawEventCandidate | None]],
        ...,
    ] = (
        ("intent_matched", _intent_matched_candidate),
        (
            "target_destination_affinity",
            _target_destination_affinity_candidate,
        ),
        ("funnel_recovery", _funnel_recovery_candidate),
        ("benefit_value_seeker", _benefit_value_seeker_candidate),
        ("promotion_responsive", _promotion_responsive_candidate),
        (
            "general_destination_explorer",
            _general_destination_explorer_candidate,
        ),
    )
    raw_candidates: list[_RawEventCandidate | None] = []
    for candidate_type, candidate_factory in candidate_factories:
        if (
            intent.requested_candidate_types
            and candidate_type not in intent.requested_candidate_types
        ):
            continue
        if enforce_prediction_support:
            support = candidate_type_prediction_support(
                predictor,
                goal_metric=promotion.goal_metric,
                candidate_type=candidate_type,
            )
            if not support.supported:
                log.info(
                    "candidate_type_prediction_unsupported",
                    {
                        "candidateType": candidate_type,
                        "goalMetric": promotion.goal_metric,
                        "modelVersion": predictor.version,
                        "trainingExampleCount": support.training_example_count,
                        "reason": support.reason,
                    },
                )
                continue
        raw_candidates.append(
            candidate_factory(
                promotion=promotion,
                intent=intent,
                compilation=compilation,
                profiles=eligible_profiles,
                baseline=baseline,
                min_sample_size=min_sample_size,
                performance_predictor=predictor,
            )
        )
    candidates = [candidate for candidate in raw_candidates if candidate is not None]
    if audience_selection_policy is not None:
        candidates = [
            _apply_audience_selection(
                candidate=candidate,
                promotion=promotion,
                all_profiles=profiles,
                baseline=baseline,
                min_sample_size=min_sample_size,
                performance_predictor=predictor,
                audience_selection_policy=audience_selection_policy,
            )
            for candidate in candidates
        ]
    return _normalize_expected_performance(
        candidates
    )


def destination_terms_from_intent(intent: PromotionIntent) -> tuple[str, ...]:
    terms: list[str] = []
    for destination in intent.destinations:
        normalized = destination.strip().lower()
        terms.extend(DESTINATION_KEYWORDS.get(normalized, (normalized,)))
    return tuple(dict.fromkeys(term for term in terms if term))


def season_months_from_intent(intent: PromotionIntent) -> tuple[int, ...]:
    months: list[int] = []
    for season in intent.season:
        months.extend(SEASON_MONTHS.get(season.strip().lower(), ()))
    return tuple(dict.fromkeys(months))


def _audience_parameters_from_intent(
    *,
    candidate_type: str,
    intent: PromotionIntent | None,
) -> RawEventAudienceParameters:
    """Freeze structured candidate parameters before SegmentDefinition storage.

    The runtime binder receives only this value. It never reads the promotion or
    re-extracts meaning from text.
    """

    if intent is None:
        return RawEventAudienceParameters()
    destination_ids: Sequence[str] = ()
    season_months: Sequence[int] = ()
    benefit_keys: Sequence[str] = ()
    if candidate_type in {
        "intent_matched",
        "target_destination_affinity",
        "funnel_recovery",
        "benefit_value_seeker",
    }:
        destination_ids = intent.destinations
    if candidate_type == "intent_matched":
        season_months = season_months_from_intent(intent)
    if candidate_type == "benefit_value_seeker":
        benefit_keys = intent.benefits
    return RawEventAudienceParameters(
        destination_ids=tuple(destination_ids),
        season_months=tuple(season_months),
        benefit_keys=tuple(benefit_keys),
    )


def _intent_matched_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor,
) -> _RawEventCandidate | None:
    requires_destination = bool(intent.destinations)
    requires_season = bool(intent.season)
    requires_profile = _has_profile_constraint(intent.audience_hints)
    matched_profiles = [
        profile
        for profile in profiles
        if (profile.hotel_search_count + profile.hotel_detail_view_count) > 0
        and (not requires_destination or profile.destination_match_count > 0)
        and (not requires_season or profile.season_match_count > 0)
        and _matches_profile_constraints(profile, intent.audience_hints)
    ]
    if (
        len(matched_profiles) < min_sample_size
        and not requires_destination
        and not requires_season
        and not requires_profile
    ):
        matched_profiles = [
            profile
            for profile in profiles
            if (profile.destination_match_count + profile.season_match_count) > 0
            or (profile.hotel_search_count + profile.hotel_detail_view_count) >= 2
        ]
    matched_condition_keys = ["hotel_product_interest"]
    if any(profile.destination_match_count > 0 for profile in matched_profiles):
        matched_condition_keys.append("recent_destination_search")
    if any(profile.season_match_count > 0 for profile in matched_profiles):
        matched_condition_keys.append("summer_checkin_search" if "summer" in intent.season else "winter_checkin_search")
    if any(profile.hotel_detail_view_count > 0 for profile in matched_profiles):
        matched_condition_keys.append("hotel_detail_view")
    if requires_profile:
        matched_condition_keys.append("profile_hint")
    return _candidate_from_profiles(
        candidate_type="intent_matched",
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=tuple(dict.fromkeys(matched_condition_keys)),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            matched_condition_keys,
        ),
        performance_predictor=performance_predictor,
    )


def _target_destination_affinity_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor,
) -> _RawEventCandidate | None:
    if not intent.destinations:
        return None
    matched_profiles = [
        profile
        for profile in profiles
        if profile.destination_match_count >= 2
        and (profile.hotel_search_count + profile.hotel_detail_view_count) > 0
    ]
    matched_condition_keys = [
        "target_destination_affinity",
        "hotel_product_interest",
        "recent_destination_search",
    ]
    if any(profile.hotel_detail_view_count > 0 for profile in matched_profiles):
        matched_condition_keys.append("hotel_detail_view")
    return _candidate_from_profiles(
        candidate_type="target_destination_affinity",
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=tuple(matched_condition_keys),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            matched_condition_keys,
        ),
        performance_predictor=performance_predictor,
    )


def _funnel_recovery_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor,
) -> _RawEventCandidate | None:
    matched_profiles = [
        profile
        for profile in profiles
        if profile.booking_start_count > profile.booking_complete_count
        and (not intent.destinations or profile.destination_match_count > 0)
    ]
    matched_condition_keys = ["booking_start_without_complete"]
    if any(
        profile.hotel_search_count + profile.hotel_detail_view_count > 0
        for profile in matched_profiles
    ):
        matched_condition_keys.append("hotel_product_interest")
    if any(profile.hotel_detail_view_count > 0 for profile in matched_profiles):
        matched_condition_keys.append("hotel_detail_view")
    if intent.destinations:
        matched_condition_keys.append("recent_destination_search")
    return _candidate_from_profiles(
        candidate_type="funnel_recovery",
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=tuple(matched_condition_keys),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            matched_condition_keys,
        ),
        performance_predictor=performance_predictor,
    )


def _promotion_responsive_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor,
) -> _RawEventCandidate | None:
    matched_profiles = [
        profile
        for profile in profiles
        if profile.promotion_click_count > 0
        or profile.campaign_landing_count > 0
        or (
            profile.promotion_impression_count > 0
            and profile.promotion_click_count / profile.promotion_impression_count >= 0.15
        )
    ]
    return _candidate_from_profiles(
        candidate_type="promotion_responsive",
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=("promotion_response", "campaign_landing"),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            ("promotion_response", "campaign_landing"),
        ),
        performance_predictor=performance_predictor,
    )


def _general_destination_explorer_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor,
) -> _RawEventCandidate | None:
    if intent.destinations:
        return None
    matched_profiles = [
        profile
        for profile in profiles
        if (
            len(profile.destination_values) >= 2
            or len(profile.hotel_market_values) >= 2
            or len(profile.hotel_cluster_values) >= 2
        )
        and (profile.hotel_search_count + profile.hotel_detail_view_count) > 0
    ]
    matched_condition_keys = (
        "general_destination_exploration",
        "hotel_product_interest",
    )
    return _candidate_from_profiles(
        candidate_type="general_destination_explorer",
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=matched_condition_keys,
        missing_condition_keys=_missing_condition_keys(
            compilation,
            matched_condition_keys,
        ),
        performance_predictor=performance_predictor,
    )


def _benefit_value_seeker_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor,
) -> _RawEventCandidate | None:
    matched_profiles = [
        profile
        for profile in profiles
        if (
            profile.deal_event_count
            + profile.free_cancellation_count
            + profile.breakfast_included_count
            + profile.price_event_count
        )
        > 0
        and (not intent.destinations or profile.destination_match_count > 0)
    ]
    matched_condition_keys = [
        "benefit_interest",
        "price_sensitive",
        "free_cancellation_interest",
        "breakfast_interest",
    ]
    if intent.destinations:
        matched_condition_keys.append("recent_destination_search")
    if any(
        profile.hotel_search_count + profile.hotel_detail_view_count > 0
        for profile in matched_profiles
    ):
        matched_condition_keys.append("hotel_product_interest")
    return _candidate_from_profiles(
        candidate_type="benefit_value_seeker",
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=tuple(matched_condition_keys),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            matched_condition_keys,
        ),
        performance_predictor=performance_predictor,
    )


def _candidate_from_profiles(
    *,
    candidate_type: str,
    promotion: PromotionRecord,
    intent: PromotionIntent | None,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    matched_condition_keys: Sequence[str],
    missing_condition_keys: Sequence[str],
    performance_predictor: SegmentPerformancePredictor,
) -> _RawEventCandidate | None:
    if len(profiles) < min_sample_size:
        return None
    matching_profile_count = len(profiles)
    ordered_profiles = sorted(
        profiles,
        key=lambda profile: (
            -_profile_strength(profile, candidate_type=candidate_type),
            profile.user_id,
        ),
    )
    # Keep every matching user until a selection ratio is calibrated by backtest.
    candidate_profiles = ordered_profiles
    candidate_user_ids = tuple(profile.user_id for profile in candidate_profiles)
    audience_selection = all_matching_audience_selection_policy().decide(
        goal_metric=promotion.goal_metric,
        candidate_type=candidate_type,
        matching_user_count=matching_profile_count,
    )
    signal_metrics = _signal_metrics(
        candidate_profiles,
        matching_profile_count=matching_profile_count,
    )
    type_labels = CANDIDATE_TYPE_LABELS[candidate_type]
    matched_condition_labels = [
        CONDITION_LABELS.get(condition_key, (condition_key, condition_key))[0]
        for condition_key in matched_condition_keys
    ]
    promotion_condition_match = _condition_match_score(
        compilation=compilation,
        matched_condition_keys=matched_condition_keys,
    )
    sample_reliability = _sample_reliability(
        sample_size=len(candidate_profiles),
        min_sample_size=min_sample_size,
    )
    performance_features = _performance_features(
        candidate_type=candidate_type,
        profiles=candidate_profiles,
        signal_metrics=signal_metrics,
        baseline=baseline,
        promotion_condition_match=promotion_condition_match,
        destination_context_required=bool(intent and intent.destinations),
        sample_reliability=sample_reliability,
    )
    predicted_goal_rate, prediction_metadata = _expected_goal_performance(
        promotion=promotion,
        profiles=candidate_profiles,
        baseline=baseline,
        performance_features=performance_features,
        performance_predictor=performance_predictor,
    )
    return _RawEventCandidate(
        candidate_type=candidate_type,
        strategy_role=type_labels["strategy_role"],
        title=type_labels["fallback_title"],
        reason=_fallback_candidate_reason(
            candidate_type=candidate_type,
            matched_condition_labels=matched_condition_labels,
        ),
        action_hint=_fallback_candidate_action_hint(
            candidate_type=candidate_type,
            promotion=promotion,
        ),
        candidate_user_ids=candidate_user_ids,
        matched_condition_keys=tuple(dict.fromkeys(matched_condition_keys)),
        missing_condition_keys=tuple(dict.fromkeys(missing_condition_keys)),
        signal_chips=_signal_chips(matched_condition_keys, candidate_type=candidate_type),
        signal_metrics=signal_metrics,
        promotion_condition_match=promotion_condition_match,
        predicted_goal_rate=predicted_goal_rate,
        expected_goal_performance=predicted_goal_rate,
        behavior_lift_vs_baseline=_behavior_lift(
            profiles=candidate_profiles,
            baseline=baseline,
            candidate_type=candidate_type,
        ),
        sample_reliability=sample_reliability,
        destination_context_required=bool(intent and intent.destinations),
        performance_features=performance_features,
        performance_model_metadata={
            **dict(performance_predictor.metadata()),
            **dict(prediction_metadata),
        },
        audience_selection=audience_selection,
        audience_parameters=_audience_parameters_from_intent(
            candidate_type=candidate_type,
            intent=intent,
        ),
    )


def _apply_audience_selection(
    *,
    candidate: _RawEventCandidate,
    promotion: PromotionRecord,
    all_profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    performance_predictor: SegmentPerformancePredictor,
    audience_selection_policy: AudienceSelectionPolicyProtocol,
) -> _RawEventCandidate:
    decision = audience_selection_policy.decide(
        goal_metric=promotion.goal_metric,
        candidate_type=candidate.candidate_type,
        matching_user_count=candidate.sample_size,
    )
    if not decision.selection_limited:
        return replace(candidate, audience_selection=decision)

    profiles_by_user_id = {profile.user_id: profile for profile in all_profiles}
    selected_profiles = [
        profile
        for user_id in candidate.candidate_user_ids[: decision.selected_user_count]
        if (profile := profiles_by_user_id.get(user_id)) is not None
    ]
    if len(selected_profiles) != decision.selected_user_count:
        return replace(
            candidate,
            audience_selection=all_matching_audience_selection_policy(
                calibration_status="runtime_fallback",
                fallback_reason="selected_profiles_unavailable",
            ).decide(
                goal_metric=promotion.goal_metric,
                candidate_type=candidate.candidate_type,
                matching_user_count=candidate.sample_size,
            ),
        )

    matching_profile_count = candidate.sample_size
    signal_metrics = _signal_metrics(
        selected_profiles,
        matching_profile_count=matching_profile_count,
    )
    sample_reliability = _sample_reliability(
        sample_size=len(selected_profiles),
        min_sample_size=min_sample_size,
    )
    performance_features = _performance_features(
        candidate_type=candidate.candidate_type,
        profiles=selected_profiles,
        signal_metrics=signal_metrics,
        baseline=baseline,
        promotion_condition_match=candidate.promotion_condition_match,
        destination_context_required=candidate.destination_context_required,
        sample_reliability=sample_reliability,
    )
    predicted_goal_rate, prediction_metadata = _expected_goal_performance(
        promotion=promotion,
        profiles=selected_profiles,
        baseline=baseline,
        performance_features=performance_features,
        performance_predictor=performance_predictor,
    )
    return replace(
        candidate,
        candidate_user_ids=tuple(profile.user_id for profile in selected_profiles),
        signal_metrics=signal_metrics,
        predicted_goal_rate=predicted_goal_rate,
        expected_goal_performance=predicted_goal_rate,
        behavior_lift_vs_baseline=_behavior_lift(
            profiles=selected_profiles,
            baseline=baseline,
            candidate_type=candidate.candidate_type,
        ),
        sample_reliability=sample_reliability,
        performance_features=performance_features,
        performance_model_metadata={
            **dict(performance_predictor.metadata()),
            **dict(prediction_metadata),
        },
        audience_selection=decision,
    )


def _fallback_candidate_reason(
    *,
    candidate_type: str,
    matched_condition_labels: Sequence[str],
) -> str:
    if matched_condition_labels:
        return (
            "실제 SDK 이벤트에서 "
            + ", ".join(matched_condition_labels[:3])
            + " 조건이 확인된 고객군입니다."
        )
    return f"{candidate_type} 조건에 해당하는 행동 신호가 확인된 고객군입니다."


def _fallback_candidate_action_hint(
    *,
    candidate_type: str,
    promotion: PromotionRecord,
) -> str:
    if promotion.goal_metric == "booking_conversion_rate":
        return "예약 시작과 예약 완료 지표를 우선 확인하며 발송 결과를 비교하세요."
    if promotion.goal_metric == "inflow_rate":
        return "클릭과 랜딩 유입 지표를 우선 확인하며 발송 결과를 비교하세요."
    return f"{candidate_type} 후보의 다음 퍼널 이동 지표를 우선 확인하세요."


def _normalize_expected_performance(
    candidates: Sequence[_RawEventCandidate],
) -> list[_RawEventCandidate]:
    highest_predicted_rate = max(
        (candidate.predicted_goal_rate for candidate in candidates),
        default=0.0,
    )
    if highest_predicted_rate <= 0:
        return [
            replace(candidate, expected_goal_performance=0.0)
            for candidate in candidates
        ]
    return [
        replace(
            candidate,
            expected_goal_performance=_clamp01(
                candidate.predicted_goal_rate / highest_predicted_rate
            ),
        )
        for candidate in candidates
    ]


def _select_candidate_portfolio(
    candidates: Sequence[_RawEventCandidate],
    *,
    max_suggested_segments: int,
) -> list[_RawEventCandidate]:
    remaining = list(candidates)
    selected: list[_RawEventCandidate] = []
    while remaining and len(selected) < max_suggested_segments:
        scored = [
            _with_distinctiveness(
                candidate,
                selected_candidates=selected,
            )
            for candidate in remaining
        ]
        distinct_candidates = [
            candidate
            for candidate in scored
            if not selected
            or _maximum_user_overlap(candidate, selected_candidates=selected)
            < MAX_RANK_USER_OVERLAP
        ]
        if not distinct_candidates:
            break
        next_candidate = max(
            distinct_candidates,
            key=lambda candidate: (
                _recommendation_tier_priority(candidate),
                _final_score(candidate),
                _expected_goal_achievement_count(candidate),
                -CANDIDATE_TYPE_ORDER.index(candidate.candidate_type),
                candidate.sample_size,
            ),
        )
        selected.append(next_candidate)
        remaining = [
            candidate
            for candidate in remaining
            if candidate.candidate_type != next_candidate.candidate_type
        ]
    return selected


def _with_distinctiveness(
    candidate: _RawEventCandidate,
    *,
    selected_candidates: Sequence[_RawEventCandidate],
) -> _RawEventCandidate:
    if not selected_candidates:
        return candidate
    user_distinctiveness = 1.0 - _maximum_user_overlap(
        candidate,
        selected_candidates=selected_candidates,
    )
    chip_distinctiveness = 1.0 - max(
        (
            _set_jaccard_similarity(candidate.signal_chips, selected.signal_chips)
            for selected in selected_candidates
        ),
        default=0.0,
    )
    distinctiveness = 0.8 * user_distinctiveness + 0.2 * chip_distinctiveness
    return _RawEventCandidate(
        candidate_type=candidate.candidate_type,
        strategy_role=candidate.strategy_role,
        title=candidate.title,
        reason=candidate.reason,
        action_hint=candidate.action_hint,
        candidate_user_ids=candidate.candidate_user_ids,
        matched_condition_keys=candidate.matched_condition_keys,
        missing_condition_keys=candidate.missing_condition_keys,
        signal_chips=candidate.signal_chips,
        signal_metrics=candidate.signal_metrics,
        promotion_condition_match=candidate.promotion_condition_match,
        predicted_goal_rate=candidate.predicted_goal_rate,
        expected_goal_performance=candidate.expected_goal_performance,
        behavior_lift_vs_baseline=candidate.behavior_lift_vs_baseline,
        sample_reliability=candidate.sample_reliability,
        destination_context_required=candidate.destination_context_required,
        performance_features=candidate.performance_features,
        performance_model_metadata=candidate.performance_model_metadata,
        audience_selection=candidate.audience_selection,
        audience_parameters=candidate.audience_parameters,
        rank_distinctiveness=max(0.0, min(1.0, distinctiveness)),
    )


def _maximum_user_overlap(
    candidate: _RawEventCandidate,
    *,
    selected_candidates: Sequence[_RawEventCandidate],
) -> float:
    return max(
        (
            _set_jaccard_similarity(
                candidate.candidate_user_ids,
                selected.candidate_user_ids,
            )
            for selected in selected_candidates
        ),
        default=0.0,
    )

def _set_jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_values = set(left)
    right_values = set(right)
    union = left_values | right_values
    if not union:
        return 0.0
    return len(left_values & right_values) / len(union)


def _segment_definition_from_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    candidate: _RawEventCandidate,
    position: int,
    total_eligible_user_count: int,
    selected_candidates: Sequence[_RawEventCandidate] | None,
) -> SegmentDefinitionRecord:
    sample_ratio = _sample_ratio(
        sample_size=candidate.sample_size,
        total_eligible_user_count=total_eligible_user_count,
    )
    score_components = _score_components(candidate)
    matched_conditions = [
        CONDITION_LABELS.get(key, (key, key))[0] for key in candidate.matched_condition_keys
    ]
    missing_conditions = [
        CONDITION_LABELS.get(key, (key, key))[0] for key in candidate.missing_condition_keys
    ]
    audience_summary = _candidate_audience_summary(
        candidate=candidate,
        total_eligible_user_count=total_eligible_user_count,
        sample_ratio=sample_ratio,
    )
    matching_user_count = int(
        candidate.signal_metrics.get("matching_profile_count", candidate.sample_size)
        or candidate.sample_size
    )
    matching_user_ratio = _safe_rate(
        matching_user_count,
        total_eligible_user_count,
    )
    selection_ratio_within_matching = _safe_rate(
        candidate.sample_size,
        matching_user_count,
    )
    selection_decision = candidate.audience_selection
    recommendation_tier = _recommendation_tier(candidate)
    audience = {
        "total_eligible_user_count": total_eligible_user_count,
        "matching_user_count": matching_user_count,
        "selected_user_count": candidate.sample_size,
        "selected_user_ratio": round(float(sample_ratio), 6),
        "matching_user_ratio": round(matching_user_ratio, 6),
        "selection_ratio_within_matching": round(
            selection_ratio_within_matching,
            6,
        ),
        "selection_limited": selection_decision.selection_limited,
        "selection_basis": (
            "behavior_strength_within_candidate"
            if selection_decision.selection_limited
            else "candidate_condition_match"
        ),
        "selection_limit": (
            candidate.sample_size if selection_decision.selection_limited else None
        ),
        "selected_user_role": "recommended_audience",
        "selection_policy": selection_decision.to_metadata(),
    }
    performance_estimate = _performance_estimate(
        promotion=promotion,
        candidate=candidate,
    )
    strategy_summary = _strategy_difference_summary(candidate)
    selection_consideration = _selection_consideration_summary(candidate)
    display_copy = {
        "title": candidate.title,
        "strategy_role": candidate.strategy_role,
        **recommendation_tier,
        "audience_summary": audience_summary,
        "audience": audience,
        "performance_estimate": performance_estimate,
        "signal_chips": list(candidate.signal_chips),
        "reason": candidate.reason,
        "strength_summary": strategy_summary,
        "tradeoff_summary": selection_consideration,
        "action_hint": candidate.action_hint,
    }
    segment_id = _raw_event_segment_id(
        promotion_id=promotion.promotion_id,
        candidate_type=candidate.candidate_type,
        candidate_user_ids=candidate.candidate_user_ids,
    )
    try:
        audience_spec = RegisteredSegmentAudienceBinder().bind(
            candidate_type=candidate.candidate_type,
            destination_ids=candidate.audience_parameters.destination_ids,
            season_months=candidate.audience_parameters.season_months,
            benefit_keys=candidate.audience_parameters.benefit_keys,
        )
    except ValueError as exc:
        raise SegmentAudienceContractError(
            code="segment_audience_template_binding_invalid",
            segment_id=segment_id,
            reason=str(exc),
        ) from exc
    profile_json: dict[str, Any] = {
        "primary_segment": segment_id,
        "source": "raw_event_intent",
        "strategy_role": candidate.strategy_role,
        "candidate_type": candidate.candidate_type,
        **recommendation_tier,
        "portfolio_position": position + 1,
        "score_components": score_components,
        "matched_conditions": matched_conditions,
        "missing_conditions": missing_conditions,
        "signal_chips": list(candidate.signal_chips),
        "audience": audience,
        "performance_estimate": performance_estimate,
        "performance_features": candidate.performance_features.to_json(),
        "signal_metrics": {
            **dict(candidate.signal_metrics),
            "sample_size": candidate.sample_size,
            "total_eligible_user_count": total_eligible_user_count,
        },
        "promotion_intent": intent.to_json(),
        "compiled_intent": compilation.to_json(),
        "display_copy": display_copy,
        "recommendation_score": score_components["final_score"],
        "selection_basis": {
            "primary_component": "recommendation_tier",
            "metric": promotion.goal_metric,
            "metric_label": performance_estimate["label"],
            "method": "diversified_candidate_portfolio",
            "expected_goal_achievement_count": performance_estimate[
                "expected_count"
            ],
            "internal_position": position + 1,
            "portfolio_size": (
                len(selected_candidates) if selected_candidates is not None else None
            ),
        },
    }
    primary_signals = [
        key for key in candidate.matched_condition_keys if key.strip()
    ][:3]
    if primary_signals:
        profile_json["primary_signals"] = primary_signals

    return SegmentDefinitionRecord(
        segment_id=segment_id,
        project_id=promotion.project_id,
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        segment_name=candidate.title,
        source="ai_suggested",
        query_preview_id=None,
        natural_language_query=(
            f"{candidate.strategy_role}: {', '.join(matched_conditions[:3])} 조건을 "
            "실제 SDK 행동 이벤트에서 만족한 고객군입니다."
        ),
        generated_sql=None,
        rule_json={
            "source": "raw_event_intent",
            "candidate_type": candidate.candidate_type,
            "compiled_conditions": list(candidate.matched_condition_keys),
            "candidate_user_ids": list(candidate.candidate_user_ids),
            "fallback_used": False,
            "version": RAW_EVENT_SEGMENT_VERSION,
            "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
            "segment_audience_spec": dict(audience_spec),
        },
        profile_json=profile_json,
        sample_size=candidate.sample_size,
        total_eligible_user_count=total_eligible_user_count,
        sample_ratio=sample_ratio,
        status="active",
    )


def _candidate_audience_summary(
    *,
    candidate: _RawEventCandidate,
    total_eligible_user_count: int,
    sample_ratio: Decimal,
) -> str:
    matching_user_count = int(
        candidate.signal_metrics.get("matching_profile_count", candidate.sample_size)
        or candidate.sample_size
    )
    summary = (
        f"분석 대상 {total_eligible_user_count}명 중 "
        f"조건 일치 {matching_user_count}명"
    )
    selection = candidate.audience_selection
    if selection.selection_limited:
        return (
            f"{summary} 중 행동 신호 상위 {candidate.sample_size}명 추천 "
            f"({selection.applied_ratio * 100:g}%)"
        )
    return f"{summary} · 조건 일치자 전체를 추천 대상으로 사용"


def _intent_system_instruction() -> str:
    return (
        "당신은 숙박/여행 프로모션을 세그먼트 추천 조건으로 구조화하는 분석기입니다. "
        "반드시 입력에 포함된 정보만 사용하고, 추정이 필요한 경우 넓은 의도 표현으로 남기세요. "
        "segment_instruction은 운영자가 명시한 고객군 제약입니다. 목적지, 행동, 제외 조건을 "
        "생략하지 말고 프로모션 기본 설명과 충돌하면 segment_instruction을 우선하세요. "
        "segment_instruction에 '후속 요청:'이 여러 번 나오면 뒤에 나온 요청이 앞선 요청을 "
        "구체화하거나 변경한 것으로 해석하고, 서로 충돌하는 조건은 가장 마지막 요청을 우선하세요. "
        "requested_candidate_types에는 운영자가 특정한 전략만 넣고, 특정 전략을 요구하지 않았다면 "
        "빈 배열을 반환하세요. "
        "excluded_behaviors에는 사용자가 명시적으로 제외해 달라고 한 행동만 넣으세요. "
        "최종 고객 선정이나 순위 결정은 하지 말고 조건 추출만 수행하세요."
    )


def _intent_user_instruction(
    promotion: PromotionRecord,
    *,
    segment_instruction: str | None = None,
) -> str:
    cleaned_instruction = _clean_segment_instruction(segment_instruction)
    return "\n".join(
        [
            "프로모션 입력을 SDK raw_events 기반 세그먼트 추천 의도로 구조화하세요.",
            "운영자 요청이 있으면 하드 제약으로 반영하되 입력에 없는 사실을 만들지 마세요.",
            (
                "requested_candidate_types 허용값: intent_matched, "
                "target_destination_affinity, funnel_recovery, benefit_value_seeker, "
                "promotion_responsive, general_destination_explorer"
            ),
            (
                "excluded_behaviors 허용값: booking_complete, booking_cancel, "
                "booking_start, promotion_response, hotel_search, hotel_detail_view"
            ),
            f"- channel: {promotion.channel}",
            f"- goal_metric: {promotion.goal_metric}",
            f"- goal_basis: {promotion.goal_basis}",
            f"- goal_target_value: {promotion.goal_target_value}",
            f"- landing_url: {promotion.landing_url or '-'}",
            f"- message_brief: {promotion.message_brief or '-'}",
            f"- segment_instruction: {cleaned_instruction or '-'}",
        ]
    )


def _intent_schema() -> dict[str, Any]:
    array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 8,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "product",
            "season",
            "destinations",
            "benefits",
            "audience_hints",
            "channel",
            "goal_metric",
            "funnel_goal",
            "desired_behaviors",
            "excluded_behaviors",
            "explicit_conditions",
            "requested_candidate_types",
        ],
        "properties": {
            "summary": {"type": "string"},
            "product": {"type": "string"},
            "season": array_schema,
            "destinations": array_schema,
            "benefits": array_schema,
            "audience_hints": {
                "type": "array",
                "items": {"type": "string", "enum": list(AUDIENCE_HINT_VALUES)},
                "maxItems": len(AUDIENCE_HINT_VALUES),
            },
            "channel": {"type": "string"},
            "goal_metric": {"type": "string"},
            "funnel_goal": {"type": "string"},
            "desired_behaviors": array_schema,
            "excluded_behaviors": {
                "type": "array",
                "items": {"type": "string", "enum": list(EXCLUDED_BEHAVIOR_VALUES)},
                "maxItems": len(EXCLUDED_BEHAVIOR_VALUES),
            },
            "explicit_conditions": array_schema,
            "requested_candidate_types": {
                "type": "array",
                "items": {"type": "string", "enum": list(CANDIDATE_TYPE_ORDER)},
                "maxItems": len(CANDIDATE_TYPE_ORDER),
            },
        },
    }


def _intent_from_payload(
    payload: Mapping[str, Any],
    *,
    promotion: PromotionRecord,
    segment_instruction: str | None = None,
    source: str,
) -> PromotionIntent:
    fallback = _fallback_intent(
        promotion=promotion,
        segment_instruction=segment_instruction,
        source=source,
    )
    return PromotionIntent(
        summary=_safe_text(payload.get("summary")) or fallback.summary,
        product=_safe_text(payload.get("product")) or fallback.product,
        season=tuple(_safe_text_list(payload.get("season"))) or fallback.season,
        destinations=tuple(_safe_text_list(payload.get("destinations")))
        or fallback.destinations,
        benefits=tuple(_safe_text_list(payload.get("benefits"))) or fallback.benefits,
        audience_hints=tuple(
            hint
            for hint in _safe_text_list(payload.get("audience_hints"))
            if hint in AUDIENCE_HINT_VALUES
        )
        or fallback.audience_hints,
        channel=_safe_text(payload.get("channel")) or promotion.channel,
        goal_metric=_safe_text(payload.get("goal_metric")) or promotion.goal_metric,
        funnel_goal=_safe_text(payload.get("funnel_goal")) or fallback.funnel_goal,
        desired_behaviors=tuple(_safe_text_list(payload.get("desired_behaviors")))
        or fallback.desired_behaviors,
        excluded_behaviors=tuple(
            behavior
            for behavior in _safe_text_list(payload.get("excluded_behaviors"))
            if behavior in EXCLUDED_BEHAVIOR_VALUES
        )
        or fallback.excluded_behaviors,
        explicit_conditions=tuple(_safe_text_list(payload.get("explicit_conditions")))
        or fallback.explicit_conditions,
        requested_candidate_types=tuple(
            candidate_type
            for candidate_type in _safe_text_list(
                payload.get("requested_candidate_types")
            )
            if candidate_type in CANDIDATE_TYPE_ORDER
        )
        or fallback.requested_candidate_types,
        source=source,
    )


def _fallback_intent(
    *,
    promotion: PromotionRecord,
    segment_instruction: str | None = None,
    source: str,
) -> PromotionIntent:
    searchable = _promotion_searchable_text(
        promotion,
        segment_instruction=segment_instruction,
    )
    season = _extract_seasons(searchable)
    destinations = _extract_destinations(searchable)
    benefits = _extract_benefits(searchable)
    desired_behaviors = ["hotel_detail_view"]
    if promotion.goal_metric == "booking_conversion_rate":
        desired_behaviors.append("booking_start_without_complete")
    if promotion.goal_metric == "inflow_rate":
        desired_behaviors.extend(["promotion_response", "recent_destination_search"])
    if "discount" in benefits or "early_booking" in benefits:
        desired_behaviors.append("price_sensitive")
    explicit_conditions = [
        *season,
        *destinations,
        *benefits,
        "hotel",
        promotion.channel,
    ]
    requested_candidate_types = _extract_requested_candidate_types(
        segment_instruction
    )
    excluded_behaviors = _extract_excluded_behaviors(segment_instruction)
    return PromotionIntent(
        summary=_fallback_summary(season=season, destinations=destinations, benefits=benefits),
        product="hotel",
        season=tuple(season),
        destinations=tuple(destinations),
        benefits=tuple(benefits),
        audience_hints=tuple(_extract_audience_hints(searchable)),
        channel=promotion.channel,
        goal_metric=promotion.goal_metric,
        funnel_goal=_funnel_goal(promotion.goal_metric),
        desired_behaviors=tuple(dict.fromkeys(desired_behaviors)),
        excluded_behaviors=tuple(excluded_behaviors),
        explicit_conditions=tuple(dict.fromkeys(explicit_conditions)),
        requested_candidate_types=tuple(requested_candidate_types),
        source=source,
    )


def _promotion_searchable_text(
    promotion: PromotionRecord,
    *,
    segment_instruction: str | None = None,
) -> str:
    parsed_url = urlparse(promotion.landing_url or "")
    query_values = " ".join(
        value
        for values in parse_qs(parsed_url.query).values()
        for value in values
    )
    return " ".join(
        [
            promotion.channel,
            promotion.goal_metric,
            promotion.message_brief or "",
            promotion.landing_url or "",
            parsed_url.path,
            query_values,
            _clean_segment_instruction(segment_instruction) or "",
        ]
    ).lower()


def _clean_segment_instruction(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def _extract_requested_candidate_types(value: str | None) -> list[str]:
    searchable = (_clean_segment_instruction(value) or "").lower()
    if not searchable:
        return []

    requested: list[str] = []
    destinations = _extract_destinations(searchable)
    if (
        "예약" in searchable
        and any(
            term in searchable
            for term in ("이탈", "미완료", "중단", "완료하지", "결제 전")
        )
    ):
        requested.append("funnel_recovery")
    if any(
        term in searchable
        for term in (
            "할인",
            "혜택",
            "가격 비교",
            "무료 취소",
            "조식 포함",
            "블랙프라이데이",
            "black friday",
        )
    ):
        requested.append("benefit_value_seeker")
    if any(term in searchable for term in ("프로모션", "캠페인", "광고")) and any(
        term in searchable for term in ("반응", "클릭", "랜딩", "유입")
    ):
        requested.append("promotion_responsive")
    if destinations and any(
        term in searchable for term in ("반복", "여러 번", "재검색", "다시 검색")
    ):
        requested.append("target_destination_affinity")
    if destinations and any(
        term in searchable for term in ("검색", "조회", "관심", "탐색", "숙소", "호텔")
    ):
        requested.append("intent_matched")
    if any(
        term in searchable
        for term in ("여러 목적지", "다목적지", "여행지를 비교", "목적지 비교")
    ):
        requested.append("general_destination_explorer")
    if any(term in searchable for term in ("연령", "20대", "30대", "남성", "여성")):
        requested.append("intent_matched")
    return list(dict.fromkeys(requested))


def _extract_excluded_behaviors(value: str | None) -> list[str]:
    searchable = (_clean_segment_instruction(value) or "").lower()
    if not searchable:
        return []
    if not any(
        term in searchable
        for term in ("제외", "빼", "않은", "없는", "미완료", "exclude", "without")
    ):
        return []

    excluded: list[str] = []
    if any(
        term in searchable
        for term in ("예약 완료", "예약한 고객", "예약한 사람", "booking complete")
    ):
        excluded.append("booking_complete")
    if any(term in searchable for term in ("예약 취소", "booking cancel")):
        excluded.append("booking_cancel")
    if any(term in searchable for term in ("예약 시작", "booking start")):
        excluded.append("booking_start")
    if any(term in searchable for term in ("프로모션 반응", "광고 반응", "promotion response")):
        excluded.append("promotion_response")
    if any(term in searchable for term in ("숙소 검색", "호텔 검색", "hotel search")):
        excluded.append("hotel_search")
    if any(term in searchable for term in ("상세 조회", "hotel detail")):
        excluded.append("hotel_detail_view")
    return list(dict.fromkeys(excluded))


def _extract_seasons(searchable: str) -> list[str]:
    seasons: list[str] = []
    if any(term in searchable for term in ("summer", "여름", "휴가")):
        seasons.append("summer")
    if any(term in searchable for term in ("winter", "겨울", "스키", "삿포로")):
        seasons.append("winter")
    if any(term in searchable for term in ("spring", "봄")):
        seasons.append("spring")
    if any(term in searchable for term in ("fall", "autumn", "가을")):
        seasons.append("fall")
    return seasons


def _extract_destinations(searchable: str) -> list[str]:
    destinations: list[str] = []
    for canonical, aliases in DESTINATION_KEYWORDS.items():
        if any(alias in searchable for alias in aliases):
            destinations.append(canonical)
    return destinations


def _extract_benefits(searchable: str) -> list[str]:
    benefits: list[str] = []
    if any(
        term in searchable
        for term in (
            "discount",
            "deal",
            "sale",
            "할인",
            "특가",
            "혜택",
            "블랙프라이데이",
            "black friday",
        )
    ):
        benefits.append("discount")
    if any(term in searchable for term in ("early", "조기", "얼리")):
        benefits.append("early_booking")
    if any(term in searchable for term in ("review", "후기", "추천")):
        benefits.append("review_based_recommendation")
    if any(term in searchable for term in ("free cancellation", "무료 취소")):
        benefits.append("free_cancellation")
    if any(term in searchable for term in ("breakfast", "조식")):
        benefits.append("breakfast_included")
    return benefits


def _extract_audience_hints(searchable: str) -> list[str]:
    hints: list[str] = []
    if any(term in searchable for term in ("20-30", "20~30", "20대", "30대")):
        hints.append("20s_30s")
    if any(term in searchable for term in ("male", "남성")):
        hints.append("male")
    if any(term in searchable for term in ("female", "여성")):
        hints.append("female")
    if any(term in searchable for term in ("travel", "여행", "휴가")):
        hints.append("travel_ready")
    return hints


def _fallback_summary(
    *,
    season: Sequence[str],
    destinations: Sequence[str],
    benefits: Sequence[str],
) -> str:
    parts = []
    if season:
        parts.append("/".join(season))
    if destinations:
        parts.append("/".join(destinations))
    parts.append("숙소 예약")
    if benefits:
        parts.append("/".join(benefits))
    return " ".join(parts) + " 프로모션"


def _funnel_goal(goal_metric: str) -> str:
    if goal_metric == "booking_conversion_rate":
        return "booking_start_or_complete"
    if goal_metric == "inflow_rate":
        return "landing_or_search"
    return "next_funnel_step"


def _dedupe_conditions(
    conditions: Iterable[CompiledRawEventCondition],
) -> list[CompiledRawEventCondition]:
    deduped: dict[str, CompiledRawEventCondition] = {}
    for condition in conditions:
        deduped.setdefault(condition.key, condition)
    return list(deduped.values())


def _exclude_profiles(
    profiles: Sequence[RawEventUserSignalRecord],
    excluded_behaviors: Sequence[str],
) -> list[RawEventUserSignalRecord]:
    excluded = set(excluded_behaviors)
    if not excluded:
        return list(profiles)

    def is_excluded(profile: RawEventUserSignalRecord) -> bool:
        return any(
            (
                behavior == "booking_complete"
                and profile.booking_complete_count > 0
            )
            or (behavior == "booking_cancel" and profile.booking_cancel_count > 0)
            or (behavior == "booking_start" and profile.booking_start_count > 0)
            or (
                behavior == "promotion_response"
                and (
                    profile.promotion_click_count > 0
                    or profile.campaign_redirect_click_count > 0
                    or profile.campaign_landing_count > 0
                )
            )
            or (behavior == "hotel_search" and profile.hotel_search_count > 0)
            or (
                behavior == "hotel_detail_view"
                and profile.hotel_detail_view_count > 0
            )
            for behavior in excluded
        )

    return [profile for profile in profiles if not is_excluded(profile)]


def _has_profile_constraint(audience_hints: Sequence[str]) -> bool:
    return any(hint in {"20s_30s", "male", "female"} for hint in audience_hints)


def _matches_profile_constraints(
    profile: RawEventUserSignalRecord,
    audience_hints: Sequence[str],
) -> bool:
    hints = set(audience_hints)
    if "20s_30s" in hints and not any(
        _is_twenty_or_thirty_age_group(value)
        for value in profile.age_group_values
    ):
        return False

    requested_genders = hints & {"male", "female"}
    if len(requested_genders) == 1:
        requested_gender = next(iter(requested_genders))
        if not any(
            _normalized_gender(value) == requested_gender
            for value in profile.gender_values
        ):
            return False
    return True


def _is_twenty_or_thirty_age_group(value: str) -> bool:
    normalized = (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "-")
        .replace("~", "-")
    )
    return normalized in {
        "20",
        "20s",
        "20대",
        "20-29",
        "30",
        "30s",
        "30대",
        "30-39",
    }


def _normalized_gender(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"male", "m", "남", "남성"}:
        return "male"
    if normalized in {"female", "f", "여", "여성"}:
        return "female"
    return None


def _baseline_metrics(
    profiles: Sequence[RawEventUserSignalRecord],
) -> dict[str, float]:
    total = max(len(profiles), 1)
    return {
        "hotel_search": sum(profile.hotel_search_count for profile in profiles) / total,
        "hotel_detail_view": sum(profile.hotel_detail_view_count for profile in profiles) / total,
        "promotion_click": sum(profile.promotion_click_count for profile in profiles) / total,
        "campaign_landing": sum(profile.campaign_landing_count for profile in profiles) / total,
        "booking_start": sum(profile.booking_start_count for profile in profiles) / total,
        "booking_complete": sum(profile.booking_complete_count for profile in profiles) / total,
        "destination_match": sum(
            profile.destination_match_count for profile in profiles
        )
        / total,
        "hotel_search_user_rate": _user_rate(
            profiles,
            lambda profile: profile.hotel_search_count > 0,
        ),
        "hotel_detail_view_user_rate": _user_rate(
            profiles,
            lambda profile: profile.hotel_detail_view_count > 0,
        ),
        "promotion_click_user_rate": _user_rate(
            profiles,
            lambda profile: profile.promotion_click_count > 0,
        ),
        "campaign_landing_user_rate": _user_rate(
            profiles,
            lambda profile: profile.campaign_landing_count > 0,
        ),
        "booking_start_user_rate": _user_rate(
            profiles,
            lambda profile: profile.booking_start_count > 0,
        ),
        "booking_complete_user_rate": _user_rate(
            profiles,
            lambda profile: profile.booking_complete_count > 0,
        ),
        "destination_match_user_rate": _user_rate(
            profiles,
            lambda profile: profile.destination_match_count > 0,
        ),
        "benefit": sum(
            profile.deal_event_count
            + profile.free_cancellation_count
            + profile.breakfast_included_count
            + profile.price_event_count
            for profile in profiles
        )
        / total,
    }


def _profile_strength(profile: RawEventUserSignalRecord, *, candidate_type: str) -> float:
    if candidate_type == "intent_matched":
        return (
            2.0 * profile.destination_match_count
            + 1.5 * profile.season_match_count
            + profile.hotel_detail_view_count
            + 0.5 * profile.hotel_search_count
        )
    if candidate_type == "funnel_recovery":
        return (
            2.0 * profile.destination_match_count
            + 2.0 * max(profile.booking_start_count - profile.booking_complete_count, 0)
            + profile.hotel_detail_view_count
        )
    if candidate_type == "promotion_responsive":
        return (
            2.0 * profile.promotion_click_count
            + profile.campaign_landing_count
            + 0.25 * profile.promotion_impression_count
        )
    if candidate_type == "target_destination_affinity":
        return (
            2.0 * profile.destination_match_count
            + profile.hotel_detail_view_count
            + 0.5 * profile.hotel_search_count
        )
    if candidate_type == "general_destination_explorer":
        return (
            len(profile.destination_values)
            + len(profile.hotel_market_values)
            + profile.hotel_search_count
        )
    if candidate_type == "benefit_value_seeker":
        return (
            2.0 * profile.destination_match_count
            + profile.deal_event_count
            + profile.free_cancellation_count
            + profile.breakfast_included_count
            + 0.5 * profile.price_event_count
        )
    return profile.event_count


def _signal_metrics(
    profiles: Sequence[RawEventUserSignalRecord],
    *,
    matching_profile_count: int,
) -> dict[str, Any]:
    sample_size = len(profiles)
    hotel_search_user_count = _user_count(
        profiles,
        lambda profile: profile.hotel_search_count > 0,
    )
    hotel_detail_view_user_count = _user_count(
        profiles,
        lambda profile: profile.hotel_detail_view_count > 0,
    )
    campaign_landing_user_count = _user_count(
        profiles,
        lambda profile: profile.campaign_landing_count > 0,
    )
    booking_start_user_count = _user_count(
        profiles,
        lambda profile: profile.booking_start_count > 0,
    )
    booking_complete_user_count = _user_count(
        profiles,
        lambda profile: profile.booking_complete_count > 0,
    )
    destination_match_user_count = _user_count(
        profiles,
        lambda profile: profile.destination_match_count > 0,
    )
    return {
        "profile_count": sample_size,
        "matching_profile_count": matching_profile_count,
        "hotel_search_count": sum(profile.hotel_search_count for profile in profiles),
        "hotel_search_user_count": hotel_search_user_count,
        "hotel_detail_view_count": sum(
            profile.hotel_detail_view_count for profile in profiles
        ),
        "hotel_detail_view_user_count": hotel_detail_view_user_count,
        "promotion_impression_count": sum(
            profile.promotion_impression_count for profile in profiles
        ),
        "promotion_click_count": sum(
            profile.promotion_click_count for profile in profiles
        ),
        "campaign_landing_count": sum(
            profile.campaign_landing_count for profile in profiles
        ),
        "campaign_landing_user_count": campaign_landing_user_count,
        "campaign_landing_user_rate": _safe_rate(
            campaign_landing_user_count,
            sample_size,
        ),
        "booking_start_count": sum(profile.booking_start_count for profile in profiles),
        "booking_start_user_count": booking_start_user_count,
        "booking_start_user_rate": _safe_rate(
            booking_start_user_count,
            sample_size,
        ),
        "booking_complete_count": sum(
            profile.booking_complete_count for profile in profiles
        ),
        "booking_complete_user_count": booking_complete_user_count,
        "booking_complete_user_rate": _safe_rate(
            booking_complete_user_count,
            sample_size,
        ),
        "destination_match_count": sum(
            profile.destination_match_count for profile in profiles
        ),
        "destination_match_user_count": destination_match_user_count,
        "destination_match_user_rate": _safe_rate(
            destination_match_user_count,
            sample_size,
        ),
        "season_match_count": sum(profile.season_match_count for profile in profiles),
        "benefit_event_count": sum(
            profile.deal_event_count
            + profile.free_cancellation_count
            + profile.breakfast_included_count
            + profile.price_event_count
            for profile in profiles
        ),
        "promotion_click_rate": _safe_rate(
            sum(profile.promotion_click_count for profile in profiles),
            sum(profile.promotion_impression_count for profile in profiles),
        ),
    }


def _condition_match_score(
    *,
    compilation: RawEventIntentCompilation,
    matched_condition_keys: Sequence[str],
) -> float:
    total_weight = sum(condition.weight for condition in compilation.compiled_conditions)
    if total_weight <= 0:
        return 0.0
    matched_keys = set(matched_condition_keys)
    matched_weight = sum(
        condition.weight
        for condition in compilation.compiled_conditions
        if condition.key in matched_keys
    )
    return _clamp01(matched_weight / total_weight)


def _performance_features(
    *,
    candidate_type: str,
    profiles: Sequence[RawEventUserSignalRecord],
    signal_metrics: Mapping[str, Any],
    baseline: Mapping[str, float],
    promotion_condition_match: float,
    destination_context_required: bool,
    sample_reliability: float,
) -> SegmentPerformanceFeatures:
    destination_match_count = sum(
        profile.destination_match_count for profile in profiles
    )
    destination_event_denominator = max(
        sum(
            profile.hotel_search_count
            + profile.hotel_detail_view_count
            + profile.booking_start_count
            + profile.booking_complete_count
            for profile in profiles
        ),
        destination_match_count,
        1,
    )
    return SegmentPerformanceFeatures(
        candidate_type=candidate_type,
        promotion_condition_match=promotion_condition_match,
        destination_context_required=destination_context_required,
        destination_match_user_rate=float(
            signal_metrics.get("destination_match_user_rate", 0.0) or 0.0
        ),
        destination_match_event_rate=_clamp01(
            destination_match_count / destination_event_denominator
        ),
        eligible_destination_match_user_rate=float(
            baseline.get("destination_match_user_rate", 0.0) or 0.0
        ),
        hotel_detail_view_user_rate=_user_rate(
            profiles,
            lambda profile: profile.hotel_detail_view_count > 0,
        ),
        booking_start_user_rate=_user_rate(
            profiles,
            lambda profile: profile.booking_start_count > 0,
        ),
        booking_complete_user_rate=_user_rate(
            profiles,
            lambda profile: profile.booking_complete_count > 0,
        ),
        funnel_recovery_user_rate=_user_rate(
            profiles,
            lambda profile: profile.booking_start_count
            > profile.booking_complete_count,
        ),
        benefit_user_rate=_user_rate(
            profiles,
            lambda profile: (
                profile.deal_event_count
                + profile.free_cancellation_count
                + profile.breakfast_included_count
                + profile.price_event_count
            )
            > 0,
        ),
        promotion_response_user_rate=_user_rate(
            profiles,
            lambda profile: profile.promotion_click_count > 0
            or profile.campaign_landing_count > 0,
        ),
        sample_reliability=sample_reliability,
    )


def _expected_goal_performance(
    *,
    promotion: PromotionRecord,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    performance_features: SegmentPerformanceFeatures,
    performance_predictor: SegmentPerformancePredictor,
) -> tuple[float, Mapping[str, Any]]:
    if promotion.goal_metric == "booking_conversion_rate":
        prediction = predict_segment_performance(
            performance_predictor,
            performance_features,
            sample_size=len(profiles),
        )
        prediction_metadata = prediction.metadata()
        adjustment = prediction_metadata.get("prediction_adjustment")
        if isinstance(adjustment, Mapping) and adjustment.get("applied"):
            log.info(
                "segment_performance_prediction_adjusted",
                {
                    "candidateType": performance_features.candidate_type,
                    "goalMetric": promotion.goal_metric,
                    "modelVersion": performance_predictor.version,
                    "candidateSampleSize": len(profiles),
                    "rawModelValue": adjustment.get("raw_model_value"),
                    "adjustedValue": adjustment.get("adjusted_value"),
                    "outOfDistributionFeatureCount": adjustment.get(
                        "out_of_distribution_feature_count"
                    ),
                    "influentialOutOfDistributionFeatureCount": adjustment.get(
                        "influential_out_of_distribution_feature_count"
                    ),
                    "maxAbsStandardizedValue": adjustment.get(
                        "max_abs_standardized_value"
                    ),
                },
            )
        return _clamp01(prediction.value), prediction_metadata
    if promotion.goal_metric == "inflow_rate":
        return (
            _smoothed_user_rate(
                profiles,
                lambda profile: profile.campaign_landing_count > 0,
                baseline_rate=baseline.get("campaign_landing_user_rate", 0.0),
            ),
            {},
        )
    return (
        _smoothed_user_rate(
            profiles,
            lambda profile: profile.booking_start_count > 0,
            baseline_rate=baseline.get("booking_start_user_rate", 0.0),
        ),
        {},
    )


def _user_count(
    profiles: Sequence[RawEventUserSignalRecord],
    predicate: Callable[[RawEventUserSignalRecord], bool],
) -> int:
    return sum(1 for profile in profiles if predicate(profile))


def _user_rate(
    profiles: Sequence[RawEventUserSignalRecord],
    predicate: Callable[[RawEventUserSignalRecord], bool],
) -> float:
    if not profiles:
        return 0.0
    return _safe_rate(_user_count(profiles, predicate), len(profiles))


def _smoothed_user_rate(
    profiles: Sequence[RawEventUserSignalRecord],
    predicate: Callable[[RawEventUserSignalRecord], bool],
    *,
    baseline_rate: float,
) -> float:
    sample_size = len(profiles)
    if sample_size <= 0:
        return _clamp01(baseline_rate)
    success_count = _user_count(profiles, predicate)
    prior_success = 0.5 + EXPECTED_RATE_PRIOR_USER_COUNT * _clamp01(baseline_rate)
    prior_failure = 0.5 + EXPECTED_RATE_PRIOR_USER_COUNT * (
        1.0 - _clamp01(baseline_rate)
    )
    return _clamp01(
        (success_count + prior_success)
        / (sample_size + prior_success + prior_failure)
    )


def _behavior_lift(
    *,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    candidate_type: str,
) -> float:
    total = max(len(profiles), 1)
    metric_key = {
        "intent_matched": "hotel_detail_view",
        "funnel_recovery": "booking_start",
        "promotion_responsive": "promotion_click",
        "target_destination_affinity": "destination_match",
        "general_destination_explorer": "hotel_search",
        "benefit_value_seeker": "benefit",
    }.get(candidate_type, "hotel_search")
    candidate_metric = {
        "hotel_detail_view": sum(profile.hotel_detail_view_count for profile in profiles),
        "booking_start": sum(profile.booking_start_count for profile in profiles),
        "promotion_click": sum(profile.promotion_click_count for profile in profiles),
        "hotel_search": sum(profile.hotel_search_count for profile in profiles),
        "destination_match": sum(
            profile.destination_match_count for profile in profiles
        ),
        "benefit": sum(
            profile.deal_event_count
            + profile.free_cancellation_count
            + profile.breakfast_included_count
            + profile.price_event_count
            for profile in profiles
        ),
    }[metric_key] / total
    baseline_metric = baseline.get(metric_key, 0.0)
    if baseline_metric <= 0:
        return 1.0 if candidate_metric > 0 else 0.5
    return _clamp01(candidate_metric / (baseline_metric * 2.0))


def _sample_reliability(*, sample_size: int, min_sample_size: int) -> float:
    if min_sample_size <= 0:
        return 1.0
    return _clamp01(sample_size / max(min_sample_size * 3, 1))


def _prediction_prior_user_count(candidate: _RawEventCandidate) -> int | None:
    adjustment = candidate.performance_model_metadata.get("prediction_adjustment")
    if not isinstance(adjustment, Mapping):
        return None
    try:
        prior_user_count = float(adjustment.get("prior_user_count", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if prior_user_count <= 0:
        return None
    return max(1, math.ceil(prior_user_count))


def _is_small_sample_candidate(candidate: _RawEventCandidate) -> bool:
    prior_user_count = _prediction_prior_user_count(candidate)
    if prior_user_count is not None:
        return candidate.sample_size < prior_user_count
    return candidate.sample_reliability < PRIMARY_RECOMMENDATION_MIN_RELIABILITY


def _recommendation_tier(candidate: _RawEventCandidate) -> dict[str, Any]:
    prior_user_count = _prediction_prior_user_count(candidate)
    if _is_small_sample_candidate(candidate):
        if prior_user_count is not None:
            reason = (
                f"행동 신호는 확인됐지만 추천 대상 {candidate.sample_size}명이 "
                f"예측 기준 표본 {prior_user_count}명보다 적어 별도 후보로 분류했습니다."
            )
        else:
            reason = (
                "행동 신호는 확인됐지만 표본 신뢰도가 주요 추천 기준에 "
                "미치지 않아 별도 후보로 분류했습니다."
            )
        return {
            "recommendation_tier": "small_high_intent",
            "recommendation_tier_label": "소규모 고의도 후보",
            "recommendation_tier_reason": reason,
            "rank_eligible": False,
            "minimum_primary_sample_size": prior_user_count,
        }
    return {
        "recommendation_tier": "primary",
        "recommendation_tier_label": "주요 추천",
        "recommendation_tier_reason": (
            "주요 추천에 필요한 표본 수준을 충족해 캠페인 집행 후보로 분류했습니다."
        ),
        "rank_eligible": True,
        "minimum_primary_sample_size": prior_user_count,
    }


def _recommendation_tier_priority(candidate: _RawEventCandidate) -> int:
    return 1 if _recommendation_tier(candidate)["rank_eligible"] else 0


def _expected_goal_achievement_count(candidate: _RawEventCandidate) -> float:
    return max(candidate.predicted_goal_rate, 0.0) * candidate.sample_size


def _score_components(candidate: _RawEventCandidate) -> dict[str, Any]:
    final_score = _final_score(candidate)
    weights = _score_weights(candidate)
    recommendation_tier = _recommendation_tier(candidate)
    return {
        "promotion_condition_match": round(candidate.promotion_condition_match, 6),
        "predicted_goal_rate": round(candidate.predicted_goal_rate, 6),
        "expected_goal_performance": round(candidate.expected_goal_performance, 6),
        "behavior_lift_vs_baseline": round(candidate.behavior_lift_vs_baseline, 6),
        "sample_reliability": round(candidate.sample_reliability, 6),
        "rank_distinctiveness": round(candidate.rank_distinctiveness, 6),
        "final_score": round(final_score, 6),
        "expected_goal_achievement_count": round(
            _expected_goal_achievement_count(candidate),
            6,
        ),
        "recommendation_tier": recommendation_tier["recommendation_tier"],
        "rank_eligible": recommendation_tier["rank_eligible"],
        "weights": dict(weights),
        "destination_context_required": candidate.destination_context_required,
        "primary_component": "expected_goal_performance",
    }


def _performance_estimate(
    *,
    promotion: PromotionRecord,
    candidate: _RawEventCandidate,
) -> dict[str, Any]:
    value = _clamp01(candidate.predicted_goal_rate)
    observed_value = _observed_goal_rate(
        goal_metric=promotion.goal_metric,
        signal_metrics=candidate.signal_metrics,
    )
    model_metadata = dict(candidate.performance_model_metadata)
    if promotion.goal_metric != "booking_conversion_rate":
        model_metadata = {
            "model_version": "dec.empirical-bayes-user-rate.v1",
            "method": "empirical_bayes_user_rate",
            "calibration_status": "historical_signal_estimate",
        }
    confidence_label, confidence_reason = _performance_confidence(
        candidate=candidate,
        model_metadata=model_metadata,
    )
    window_days = _positive_int(model_metadata.get("outcome_days"))
    expected_count = _expected_goal_achievement_count(candidate)
    estimate = {
        "metric": promotion.goal_metric,
        "label": _performance_estimate_label(promotion.goal_metric),
        "availability": "available",
        "unit": "rate",
        "value": round(value, 6),
        "formatted": _format_percent(value),
        "expected_count": round(expected_count, 6),
        "expected_count_formatted": _format_expected_count(expected_count),
        "expected_count_label": _performance_expected_count_label(
            promotion.goal_metric
        ),
        "observed_value": round(observed_value, 6),
        "basis_label": _performance_basis_label(promotion.goal_metric),
        "window_days": window_days,
        "window_label": _performance_window_label(
            promotion.goal_metric,
            window_days=window_days,
        ),
        "confidence_label": confidence_label,
        "confidence_reason": confidence_reason,
        "method": model_metadata.get("method"),
        "model_version": model_metadata.get("model_version"),
        "calibration_status": model_metadata.get("calibration_status"),
    }
    prediction_adjustment = model_metadata.get("prediction_adjustment")
    if isinstance(prediction_adjustment, Mapping):
        estimate["prediction_adjustment"] = dict(prediction_adjustment)
    return estimate


def _observed_goal_rate(
    *,
    goal_metric: str,
    signal_metrics: Mapping[str, Any],
) -> float:
    metric_key = {
        "booking_conversion_rate": "booking_complete_user_rate",
        "inflow_rate": "campaign_landing_user_rate",
        "funnel_step_rate": "booking_start_user_rate",
    }.get(goal_metric, "booking_start_user_rate")
    return _clamp01(float(signal_metrics.get(metric_key, 0.0) or 0.0))


def _performance_estimate_label(goal_metric: str) -> str:
    if goal_metric == "booking_conversion_rate":
        return "예상 예약 전환율"
    if goal_metric == "inflow_rate":
        return "예상 유입률"
    if goal_metric == "funnel_step_rate":
        return "예상 예약 시작 전환율"
    return "예상 성과"


def _performance_expected_count_label(goal_metric: str) -> str:
    if goal_metric == "booking_conversion_rate":
        return "예상 예약 인원"
    if goal_metric == "inflow_rate":
        return "예상 유입 인원"
    if goal_metric == "funnel_step_rate":
        return "예상 다음 단계 진입 인원"
    return "예상 목표 달성 인원"


def _performance_basis_label(goal_metric: str) -> str:
    if goal_metric == "booking_conversion_rate":
        return "과거 행동과 프로모션 조건을 반영한 예약 가능성"
    if goal_metric == "inflow_rate":
        return "최근 클릭·랜딩 행동을 전체 고객 기준으로 보정한 추정치"
    if goal_metric == "funnel_step_rate":
        return "최근 예약 시작 행동을 전체 고객 기준으로 보정한 추정치"
    return "최근 행동과 프로모션 조건을 반영한 추정치"


def _performance_window_label(goal_metric: str, *, window_days: int | None) -> str:
    if goal_metric == "booking_conversion_rate":
        if window_days is not None:
            return f"향후 {window_days}일 내 프로모션 조건 일치 예약"
        return "향후 프로모션 조건 일치 예약"
    if goal_metric == "inflow_rate":
        return "최근 행동 관찰 구간의 캠페인 랜딩"
    if goal_metric == "funnel_step_rate":
        return "최근 행동 관찰 구간의 예약 시작"
    return "최근 행동 관찰 구간"


def _performance_confidence(
    *,
    candidate: _RawEventCandidate,
    model_metadata: Mapping[str, Any],
) -> tuple[str, str]:
    calibration_status = str(model_metadata.get("calibration_status", ""))
    adjustment = model_metadata.get("prediction_adjustment")
    if calibration_status == "calibrated" and isinstance(adjustment, Mapping):
        sample_size = int(adjustment.get("candidate_sample_size", 0) or 0)
        prior_user_count = float(adjustment.get("prior_user_count", 0.0) or 0.0)
        is_small_sample = prior_user_count > 0 and sample_size < prior_user_count
        if is_small_sample:
            return (
                "low",
                "대표 표본이 제한적이어서 전체 고객의 행동 기준률을 함께 "
                "반영했습니다.",
            )
    if calibration_status == "calibrated":
        if candidate.sample_reliability >= 0.75:
            return "high", "충분한 행동 표본과 검증된 예약 예측 모델을 사용했습니다."
        return (
            "medium",
            "검증된 예약 예측 모델을 사용했으며 대표 표본 규모를 함께 "
            "고려했습니다.",
        )
    if calibration_status == "historical_signal_estimate":
        if candidate.sample_reliability >= 0.75:
            return "medium", "충분한 표본의 최근 행동 신호를 전체 고객 기준으로 보정했습니다."
        return "low", "최근 행동 신호를 사용했지만 표본 규모가 제한적입니다."
    return "low", "현재 데이터에서는 최근 행동 신호를 중심으로 추정했습니다."


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _format_percent(value: float) -> str:
    return f"{_clamp01(value) * 100:.1f}%"


def _format_expected_count(value: float) -> str:
    return f"약 {max(value, 0.0):.1f}명"


def _final_score(candidate: _RawEventCandidate) -> float:
    weights = _score_weights(candidate)
    return (
        weights["promotion_condition_match"] * candidate.promotion_condition_match
        + weights["expected_goal_performance"]
        * candidate.expected_goal_performance
        + weights["behavior_lift_vs_baseline"]
        * candidate.behavior_lift_vs_baseline
        + weights["sample_reliability"] * candidate.sample_reliability
        + weights["rank_distinctiveness"] * candidate.rank_distinctiveness
    )


def _score_weights(candidate: _RawEventCandidate) -> Mapping[str, float]:
    if candidate.destination_context_required:
        return DESTINATION_SCORE_WEIGHTS
    return DEFAULT_SCORE_WEIGHTS


def _signal_chips(
    matched_condition_keys: Sequence[str],
    *,
    candidate_type: str,
) -> tuple[str, ...]:
    chips = [
        CONDITION_LABELS.get(condition_key, (condition_key, condition_key))[1]
        for condition_key in matched_condition_keys
    ]
    if candidate_type == "funnel_recovery":
        chips = ["예약 시작", "예약 미완료", "호텔 상세 조회"]
    if candidate_type == "promotion_responsive":
        chips = ["프로모션 반응", "캠페인 랜딩", "클릭 행동"]
    if candidate_type == "target_destination_affinity":
        chips = ["이번 목적지 반복", "목적지 검색", "호텔 상세 조회"]
    if candidate_type == "general_destination_explorer":
        chips = ["다목적지 탐색", "숙소 비교", "여행지 확장"]
    if candidate_type == "benefit_value_seeker":
        chips = ["할인 관심", "가격 비교", "혜택 탐색"]
    return tuple(dict.fromkeys(chips))[:3]


def _missing_condition_keys(
    compilation: RawEventIntentCompilation,
    matched_condition_keys: Sequence[str],
) -> tuple[str, ...]:
    matched = set(matched_condition_keys)
    return tuple(
        condition.key
        for condition in compilation.compiled_conditions
        if condition.key not in matched
    )


def _strategy_difference_summary(candidate: _RawEventCandidate) -> str:
    if candidate.candidate_type == "intent_matched":
        return "프로모션 조건과 직접 맞는 숙소 관심 행동을 우선한 전략입니다."
    if candidate.candidate_type == "funnel_recovery":
        return "예약 시작 후 완료하지 않은 고객을 회수하는 전략입니다."
    if candidate.candidate_type == "promotion_responsive":
        return "캠페인 메시지에 반응한 고객을 확장하는 전략입니다."
    if candidate.candidate_type == "target_destination_affinity":
        return "이번 프로모션 목적지를 반복 탐색한 고객을 우선한 전략입니다."
    if candidate.candidate_type == "general_destination_explorer":
        return "여러 여행지를 비교하는 고객까지 넓히는 전략입니다."
    if candidate.candidate_type == "benefit_value_seeker":
        return "가격과 혜택에 민감한 고객을 우선한 전략입니다."
    return "다른 후보와 겹치지 않는 행동 조건을 우선한 전략입니다."


def _selection_consideration_summary(candidate: _RawEventCandidate) -> str:
    if _is_small_sample_candidate(candidate):
        return (
            "행동 의도는 뚜렷하지만 대표 표본이 작아, 좁은 고객군을 정밀하게 "
            "공략할 때 적합합니다."
        )
    if candidate.candidate_type == "intent_matched":
        return (
            "프로모션 조건과 직접 맞는 고객을 우선하지만 예약 퍼널 깊이는 "
            "다른 행동 근거와 함께 확인해야 합니다."
        )
    if candidate.candidate_type == "funnel_recovery":
        return (
            "예약 의도는 깊지만 프로모션 목적지와 혜택 조건의 직접 일치 정도를 "
            "함께 고려해야 합니다."
        )
    if candidate.candidate_type == "promotion_responsive":
        return (
            "캠페인 반응은 확인됐지만 클릭 행동만으로 예약 의도를 단정하지 않는 "
            "확장 전략입니다."
        )
    if candidate.candidate_type == "target_destination_affinity":
        return (
            "목적지 관심은 뚜렷하지만 가격이나 혜택에 대한 반응은 별도 행동 "
            "근거와 함께 확인해야 합니다."
        )
    if candidate.candidate_type == "general_destination_explorer":
        return (
            "도달 범위를 넓힐 수 있지만 특정 목적지에 대한 의도는 상대적으로 "
            "넓게 해석한 후보입니다."
        )
    if candidate.candidate_type == "benefit_value_seeker":
        return (
            "혜택 메시지와 잘 맞지만 목적지 선호와 예약 단계는 다른 행동 "
            "근거와 함께 고려해야 합니다."
        )
    return "후보의 행동 근거와 대표 표본 규모를 함께 확인해 선택하세요."


def _sample_ratio(*, sample_size: int, total_eligible_user_count: int) -> Decimal:
    if total_eligible_user_count <= 0:
        return Decimal("0")
    return Decimal(sample_size / total_eligible_user_count).quantize(Decimal("0.000001"))


def _raw_event_segment_id(
    *,
    promotion_id: str,
    candidate_type: str,
    candidate_user_ids: Sequence[str],
) -> str:
    stable_user_ids = sorted(set(candidate_user_ids))
    digest = hashlib.sha1(  # noqa: S324 - stable non-security identifier.
        ":".join(
            [promotion_id, candidate_type, ",".join(stable_user_ids)]
        ).encode("utf-8")
    ).hexdigest()[:10]
    return (
        f"seg_ai_raw_{_safe_identifier_part(promotion_id)[:32]}_"
        f"{candidate_type}_{digest}"
    )


def _safe_identifier_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in value
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_text_list(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [text for item in value if (text := _safe_text(item))]


def _is_placeholder_api_key(api_key: str) -> bool:
    normalized = api_key.strip().lower()
    return (
        not normalized
        or normalized.startswith("replace-with")
        or normalized in {"changeme", "placeholder"}
    )
