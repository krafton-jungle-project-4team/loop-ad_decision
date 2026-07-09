from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

from app.analysis.repositories import (
    PromotionRecord,
    RawEventUserSignalRecord,
    SegmentDefinitionRecord,
)
from app.config import Settings
from app.generation.adapters import (
    DEFAULT_OPENAI_CONTENT_MODEL,
    OPENAI_RESPONSES_URL,
    JsonTransport,
    _parse_output_json,
    _post_json,
)
from app.logging import duration_ms, log


RAW_EVENT_SEGMENT_VERSION = "raw-event-segment.v1"
RAW_EVENT_INTENT_COMPILER_VERSION = "raw-event-intent.v1"
INTENT_EXTRACTOR_VERSION = "dec.segment-intent.v1"
RAW_EVENT_CANDIDATE_USER_LIMIT = 160
EXPECTED_RATE_PRIOR_USER_COUNT = 30.0

CANDIDATE_TYPE_ORDER = (
    "intent_matched",
    "funnel_recovery",
    "promotion_responsive",
    "destination_affinity",
    "benefit_value_seeker",
)

CANDIDATE_TYPE_LABELS: Mapping[str, Mapping[str, str]] = {
    "intent_matched": {
        "rank_role": "프로모션 조건 정합형",
        "fallback_title": "프로모션 조건과 맞는 숙소 관심 고객",
    },
    "funnel_recovery": {
        "rank_role": "예약 이탈 회수형",
        "fallback_title": "예약 직전 이탈 고객",
    },
    "promotion_responsive": {
        "rank_role": "프로모션 반응 확장형",
        "fallback_title": "프로모션 반응이 확인된 고객",
    },
    "destination_affinity": {
        "rank_role": "목적지 반복 관심형",
        "fallback_title": "목적지 관심이 반복된 고객",
    },
    "benefit_value_seeker": {
        "rank_role": "혜택 민감형",
        "fallback_title": "할인과 혜택에 반응할 고객",
    },
}

# Stable ontology labels used for deterministic fallbacks and UI chips.
# These labels do not select users or decide ranking.
CONDITION_LABELS: Mapping[str, tuple[str, str]] = {
    "hotel_product_interest": ("숙소 관심 행동", "숙소 관심"),
    "recent_destination_search": ("목적지 숙소 검색", "목적지 검색"),
    "summer_checkin_search": ("여름 체크인 숙소 검색", "여름 체크인"),
    "winter_checkin_search": ("겨울 체크인 숙소 검색", "겨울 체크인"),
    "hotel_detail_view": ("호텔 상세 조회", "호텔 상세 조회"),
    "promotion_response": ("프로모션 반응", "프로모션 반응"),
    "campaign_landing": ("캠페인 랜딩", "캠페인 랜딩"),
    "booking_start_without_complete": ("예약 시작 후 미완료", "예약 미완료"),
    "destination_affinity": ("목적지 반복 관심", "목적지 반복"),
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
    def extract(self, promotion: PromotionRecord) -> "PromotionIntent":
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
            "explicit_conditions": list(self.explicit_conditions),
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
class _RawEventCandidate:
    candidate_type: str
    rank_role: str
    title: str
    reason: str
    action_hint: str
    candidate_user_ids: tuple[str, ...]
    matched_condition_keys: tuple[str, ...]
    missing_condition_keys: tuple[str, ...]
    signal_chips: tuple[str, ...]
    signal_metrics: Mapping[str, Any]
    promotion_condition_match: float
    expected_goal_performance: float
    behavior_lift_vs_baseline: float
    sample_reliability: float
    rank_distinctiveness: float = 1.0

    @property
    def sample_size(self) -> int:
        return len(self.candidate_user_ids)


class DeterministicPromotionIntentExtractor:
    def extract(self, promotion: PromotionRecord) -> PromotionIntent:
        return _fallback_intent(promotion=promotion, source="deterministic")


class OpenAIPromotionIntentExtractor:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENAI_CONTENT_MODEL,
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

    def extract(self, promotion: PromotionRecord) -> PromotionIntent:
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
                            "text": _intent_user_instruction(promotion),
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
        try:
            response_payload = self._transport(
                self._endpoint,
                headers,
                payload,
                self._timeout_seconds,
            )
            return _intent_from_payload(
                _parse_output_json(response_payload),
                promotion=promotion,
                source="openai",
            )
        except Exception as exc:
            log.warn(
                "promotion_intent_provider_request_failed",
                {
                    "provider": "openai",
                    "endpoint": self._endpoint,
                    "model": self._model,
                    "promotionId": promotion.promotion_id,
                    "err": exc,
                    "durationMs": duration_ms(started_at),
                },
            )
            return self._fallback_extractor.extract(promotion)


def build_promotion_intent_extractor(settings: Settings) -> PromotionIntentExtractor:
    if settings.env == "test" or _is_placeholder_api_key(settings.openai_api_key):
        return DeterministicPromotionIntentExtractor()
    return OpenAIPromotionIntentExtractor(
        api_key=settings.openai_api_key,
        model=settings.openai_content_model or DEFAULT_OPENAI_CONTENT_MODEL,
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
                weight=0.2,
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
) -> list[SegmentDefinitionRecord]:
    if len(profiles) < min_sample_size:
        return []
    total_eligible_user_count = len(profiles)
    baseline = _baseline_metrics(profiles)
    raw_candidates = [
        _intent_matched_candidate(
            promotion=promotion,
            intent=intent,
            compilation=compilation,
            profiles=profiles,
            baseline=baseline,
            min_sample_size=min_sample_size,
        ),
        _funnel_recovery_candidate(
            promotion=promotion,
            compilation=compilation,
            profiles=profiles,
            baseline=baseline,
            min_sample_size=min_sample_size,
        ),
        _promotion_responsive_candidate(
            promotion=promotion,
            compilation=compilation,
            profiles=profiles,
            baseline=baseline,
            min_sample_size=min_sample_size,
        ),
        _destination_affinity_candidate(
            promotion=promotion,
            compilation=compilation,
            profiles=profiles,
            baseline=baseline,
            min_sample_size=min_sample_size,
        ),
        _benefit_value_seeker_candidate(
            promotion=promotion,
            compilation=compilation,
            profiles=profiles,
            baseline=baseline,
            min_sample_size=min_sample_size,
        ),
    ]
    candidates = [candidate for candidate in raw_candidates if candidate is not None]
    ranked = _rank_candidates(candidates, max_suggested_segments=max_suggested_segments)
    return [
        _segment_definition_from_candidate(
            promotion=promotion,
            intent=intent,
            compilation=compilation,
            candidate=candidate,
            rank=rank,
            total_eligible_user_count=total_eligible_user_count,
        )
        for rank, candidate in enumerate(ranked)
    ]


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


def _intent_matched_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
) -> _RawEventCandidate | None:
    requires_destination = bool(intent.destinations)
    requires_season = bool(intent.season)
    matched_profiles = [
        profile
        for profile in profiles
        if (profile.hotel_search_count + profile.hotel_detail_view_count) > 0
        and (not requires_destination or profile.destination_match_count > 0)
        and (not requires_season or profile.season_match_count > 0)
    ]
    if len(matched_profiles) < min_sample_size:
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
    return _candidate_from_profiles(
        candidate_type="intent_matched",
        promotion=promotion,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=tuple(dict.fromkeys(matched_condition_keys)),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            matched_condition_keys,
        ),
    )


def _funnel_recovery_candidate(
    *,
    promotion: PromotionRecord,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
) -> _RawEventCandidate | None:
    matched_profiles = [
        profile
        for profile in profiles
        if profile.booking_start_count > profile.booking_complete_count
        or (profile.hotel_detail_view_count >= 2 and profile.booking_complete_count == 0)
    ]
    return _candidate_from_profiles(
        candidate_type="funnel_recovery",
        promotion=promotion,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=("booking_start_without_complete", "hotel_detail_view"),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            ("booking_start_without_complete", "hotel_detail_view"),
        ),
    )


def _promotion_responsive_candidate(
    *,
    promotion: PromotionRecord,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
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
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=("promotion_response", "campaign_landing"),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            ("promotion_response", "campaign_landing"),
        ),
    )


def _destination_affinity_candidate(
    *,
    promotion: PromotionRecord,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
) -> _RawEventCandidate | None:
    matched_profiles = [
        profile
        for profile in profiles
        if (
            profile.destination_match_count > 0
            or len(profile.destination_values) >= 2
            or len(profile.hotel_market_values) >= 2
            or len(profile.hotel_cluster_values) >= 2
        )
        and (profile.hotel_search_count + profile.hotel_detail_view_count) > 0
    ]
    return _candidate_from_profiles(
        candidate_type="destination_affinity",
        promotion=promotion,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=("destination_affinity", "recent_destination_search"),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            ("destination_affinity", "recent_destination_search"),
        ),
    )


def _benefit_value_seeker_candidate(
    *,
    promotion: PromotionRecord,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
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
        or profile.promotion_click_count > 0
    ]
    return _candidate_from_profiles(
        candidate_type="benefit_value_seeker",
        promotion=promotion,
        compilation=compilation,
        profiles=matched_profiles,
        baseline=baseline,
        min_sample_size=min_sample_size,
        matched_condition_keys=(
            "benefit_interest",
            "price_sensitive",
            "free_cancellation_interest",
            "breakfast_interest",
        ),
        missing_condition_keys=_missing_condition_keys(
            compilation,
            ("benefit_interest", "price_sensitive"),
        ),
    )


def _candidate_from_profiles(
    *,
    candidate_type: str,
    promotion: PromotionRecord,
    compilation: RawEventIntentCompilation,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
    min_sample_size: int,
    matched_condition_keys: Sequence[str],
    missing_condition_keys: Sequence[str],
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
    selected_profiles = ordered_profiles[:RAW_EVENT_CANDIDATE_USER_LIMIT]
    candidate_user_ids = tuple(profile.user_id for profile in selected_profiles)
    signal_metrics = _signal_metrics(
        selected_profiles,
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
    return _RawEventCandidate(
        candidate_type=candidate_type,
        rank_role=type_labels["rank_role"],
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
        expected_goal_performance=_expected_goal_performance(
            promotion=promotion,
            profiles=selected_profiles,
            baseline=baseline,
        ),
        behavior_lift_vs_baseline=_behavior_lift(
            profiles=selected_profiles,
            baseline=baseline,
            candidate_type=candidate_type,
        ),
        sample_reliability=_sample_reliability(
            sample_size=len(selected_profiles),
            min_sample_size=min_sample_size,
        ),
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


def _rank_candidates(
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
        next_candidate = max(
            scored,
            key=lambda candidate: (
                _final_score(candidate),
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
    distinctiveness = 1.0
    candidate_users = set(candidate.candidate_user_ids)
    candidate_chips = set(candidate.signal_chips)
    for selected in selected_candidates:
        if selected.candidate_type == candidate.candidate_type:
            distinctiveness -= 0.35
        selected_users = set(selected.candidate_user_ids)
        if candidate_users and selected_users:
            overlap = len(candidate_users & selected_users) / max(
                len(candidate_users),
                1,
            )
            if overlap >= 0.7:
                distinctiveness -= 0.45
            elif overlap >= 0.4:
                distinctiveness -= 0.2
        if candidate_chips == set(selected.signal_chips):
            distinctiveness -= 0.25
    return _RawEventCandidate(
        candidate_type=candidate.candidate_type,
        rank_role=candidate.rank_role,
        title=candidate.title,
        reason=candidate.reason,
        action_hint=candidate.action_hint,
        candidate_user_ids=candidate.candidate_user_ids,
        matched_condition_keys=candidate.matched_condition_keys,
        missing_condition_keys=candidate.missing_condition_keys,
        signal_chips=candidate.signal_chips,
        signal_metrics=candidate.signal_metrics,
        promotion_condition_match=candidate.promotion_condition_match,
        expected_goal_performance=candidate.expected_goal_performance,
        behavior_lift_vs_baseline=candidate.behavior_lift_vs_baseline,
        sample_reliability=candidate.sample_reliability,
        rank_distinctiveness=max(0.0, min(1.0, distinctiveness)),
    )


def _segment_definition_from_candidate(
    *,
    promotion: PromotionRecord,
    intent: PromotionIntent,
    compilation: RawEventIntentCompilation,
    candidate: _RawEventCandidate,
    rank: int,
    total_eligible_user_count: int,
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
    audience_summary = (
        f"분석 대상 {total_eligible_user_count}명 중 {candidate.sample_size}명 · "
        f"{float(sample_ratio) * 100:g}%"
    )
    performance_estimate = _performance_estimate(
        promotion=promotion,
        candidate=candidate,
    )
    display_copy = {
        "title": candidate.title,
        "rank_role": candidate.rank_role,
        "audience_summary": audience_summary,
        "performance_estimate": performance_estimate,
        "signal_chips": list(candidate.signal_chips),
        "reason": candidate.reason,
        "difference_summary": _difference_summary(candidate, rank=rank),
        "action_hint": candidate.action_hint,
    }
    segment_id = _raw_event_segment_id(
        promotion_id=promotion.promotion_id,
        candidate_type=candidate.candidate_type,
        rank=rank,
        candidate_user_ids=candidate.candidate_user_ids,
    )
    return SegmentDefinitionRecord(
        segment_id=segment_id,
        project_id=promotion.project_id,
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        segment_name=candidate.title,
        source="ai_suggested",
        query_preview_id=None,
        natural_language_query=(
            f"{candidate.rank_role}: {', '.join(matched_conditions[:3])} 조건을 "
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
        },
        profile_json={
            "primary_segment": segment_id,
            "source": "raw_event_intent",
            "rank_role": candidate.rank_role,
            "candidate_type": candidate.candidate_type,
            "score_components": score_components,
            "matched_conditions": matched_conditions,
            "missing_conditions": missing_conditions,
            "signal_chips": list(candidate.signal_chips),
            "performance_estimate": performance_estimate,
            "signal_metrics": {
                **dict(candidate.signal_metrics),
                "sample_size": candidate.sample_size,
                "total_eligible_user_count": total_eligible_user_count,
            },
            "promotion_intent": intent.to_json(),
            "compiled_intent": compilation.to_json(),
            "display_copy": display_copy,
            "recommendation_score": score_components["final_score"],
        },
        sample_size=candidate.sample_size,
        total_eligible_user_count=total_eligible_user_count,
        sample_ratio=sample_ratio,
        status="active",
    )


def _intent_system_instruction() -> str:
    return (
        "당신은 숙박/여행 프로모션을 세그먼트 추천 조건으로 구조화하는 분석기입니다. "
        "반드시 입력에 포함된 정보만 사용하고, 추정이 필요한 경우 넓은 의도 표현으로 남기세요. "
        "최종 고객 선정이나 순위 결정은 하지 말고 조건 추출만 수행하세요."
    )


def _intent_user_instruction(promotion: PromotionRecord) -> str:
    return "\n".join(
        [
            "프로모션 입력을 SDK raw_events 기반 세그먼트 추천 의도로 구조화하세요.",
            f"- channel: {promotion.channel}",
            f"- goal_metric: {promotion.goal_metric}",
            f"- goal_basis: {promotion.goal_basis}",
            f"- goal_target_value: {promotion.goal_target_value}",
            f"- landing_url: {promotion.landing_url or '-'}",
            f"- message_brief: {promotion.message_brief or '-'}",
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
            "explicit_conditions",
        ],
        "properties": {
            "summary": {"type": "string"},
            "product": {"type": "string"},
            "season": array_schema,
            "destinations": array_schema,
            "benefits": array_schema,
            "audience_hints": array_schema,
            "channel": {"type": "string"},
            "goal_metric": {"type": "string"},
            "funnel_goal": {"type": "string"},
            "desired_behaviors": array_schema,
            "explicit_conditions": array_schema,
        },
    }


def _intent_from_payload(
    payload: Mapping[str, Any],
    *,
    promotion: PromotionRecord,
    source: str,
) -> PromotionIntent:
    fallback = _fallback_intent(promotion=promotion, source=source)
    return PromotionIntent(
        summary=_safe_text(payload.get("summary")) or fallback.summary,
        product=_safe_text(payload.get("product")) or fallback.product,
        season=tuple(_safe_text_list(payload.get("season"))) or fallback.season,
        destinations=tuple(_safe_text_list(payload.get("destinations")))
        or fallback.destinations,
        benefits=tuple(_safe_text_list(payload.get("benefits"))) or fallback.benefits,
        audience_hints=tuple(_safe_text_list(payload.get("audience_hints")))
        or fallback.audience_hints,
        channel=_safe_text(payload.get("channel")) or promotion.channel,
        goal_metric=_safe_text(payload.get("goal_metric")) or promotion.goal_metric,
        funnel_goal=_safe_text(payload.get("funnel_goal")) or fallback.funnel_goal,
        desired_behaviors=tuple(_safe_text_list(payload.get("desired_behaviors")))
        or fallback.desired_behaviors,
        explicit_conditions=tuple(_safe_text_list(payload.get("explicit_conditions")))
        or fallback.explicit_conditions,
        source=source,
    )


def _fallback_intent(*, promotion: PromotionRecord, source: str) -> PromotionIntent:
    searchable = _promotion_searchable_text(promotion)
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
        explicit_conditions=tuple(dict.fromkeys(explicit_conditions)),
        source=source,
    )


def _promotion_searchable_text(promotion: PromotionRecord) -> str:
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
        ]
    ).lower()


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
    if any(term in searchable for term in ("discount", "deal", "sale", "할인", "특가", "혜택")):
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
            2.0 * max(profile.booking_start_count - profile.booking_complete_count, 0)
            + profile.hotel_detail_view_count
        )
    if candidate_type == "promotion_responsive":
        return (
            2.0 * profile.promotion_click_count
            + profile.campaign_landing_count
            + 0.25 * profile.promotion_impression_count
        )
    if candidate_type == "destination_affinity":
        return (
            profile.destination_match_count
            + len(profile.destination_values)
            + len(profile.hotel_market_values)
            + profile.hotel_search_count
        )
    if candidate_type == "benefit_value_seeker":
        return (
            profile.deal_event_count
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
        "booking_start_count": sum(profile.booking_start_count for profile in profiles),
        "booking_start_user_count": booking_start_user_count,
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


def _expected_goal_performance(
    *,
    promotion: PromotionRecord,
    profiles: Sequence[RawEventUserSignalRecord],
    baseline: Mapping[str, float],
) -> float:
    if promotion.goal_metric == "booking_conversion_rate":
        complete_rate = _smoothed_user_rate(
            profiles,
            lambda profile: profile.booking_complete_count > 0,
            baseline_rate=baseline.get("booking_complete_user_rate", 0.0),
        )
        start_rate = _user_rate(
            profiles,
            lambda profile: profile.booking_start_count > 0,
        )
        detail_rate = _user_rate(
            profiles,
            lambda profile: profile.hotel_detail_view_count > 0,
        )
        intent_support = 0.35 * start_rate + 0.15 * detail_rate
        return _clamp01(0.75 * complete_rate + 0.25 * intent_support)
    if promotion.goal_metric == "inflow_rate":
        landing_rate = _smoothed_user_rate(
            profiles,
            lambda profile: profile.campaign_landing_count > 0,
            baseline_rate=baseline.get("campaign_landing_user_rate", 0.0),
        )
        click_rate = _user_rate(
            profiles,
            lambda profile: profile.promotion_click_count > 0,
        )
        search_rate = _user_rate(
            profiles,
            lambda profile: profile.hotel_search_count > 0,
        )
        return _clamp01(0.70 * landing_rate + 0.20 * click_rate + 0.10 * search_rate)
    search_rate = _user_rate(
        profiles,
        lambda profile: profile.hotel_search_count > 0,
    )
    detail_rate = _user_rate(
        profiles,
        lambda profile: profile.hotel_detail_view_count > 0,
    )
    return _clamp01(0.35 * search_rate + 0.65 * detail_rate)


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
    return _clamp01(
        (success_count + EXPECTED_RATE_PRIOR_USER_COUNT * _clamp01(baseline_rate))
        / (sample_size + EXPECTED_RATE_PRIOR_USER_COUNT)
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
        "destination_affinity": "hotel_search",
        "benefit_value_seeker": "benefit",
    }.get(candidate_type, "hotel_search")
    candidate_metric = {
        "hotel_detail_view": sum(profile.hotel_detail_view_count for profile in profiles),
        "booking_start": sum(profile.booking_start_count for profile in profiles),
        "promotion_click": sum(profile.promotion_click_count for profile in profiles),
        "hotel_search": sum(profile.hotel_search_count for profile in profiles),
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


def _score_components(candidate: _RawEventCandidate) -> dict[str, float]:
    final_score = _final_score(candidate)
    return {
        "promotion_condition_match": round(candidate.promotion_condition_match, 6),
        "expected_goal_performance": round(candidate.expected_goal_performance, 6),
        "behavior_lift_vs_baseline": round(candidate.behavior_lift_vs_baseline, 6),
        "sample_reliability": round(candidate.sample_reliability, 6),
        "rank_distinctiveness": round(candidate.rank_distinctiveness, 6),
        "final_score": round(final_score, 6),
        "weights": {
            "promotion_condition_match": 0.30,
            "expected_goal_performance": 0.25,
            "behavior_lift_vs_baseline": 0.20,
            "sample_reliability": 0.15,
            "rank_distinctiveness": 0.10,
        },
    }


def _performance_estimate(
    *,
    promotion: PromotionRecord,
    candidate: _RawEventCandidate,
) -> dict[str, Any]:
    value = _clamp01(candidate.expected_goal_performance)
    return {
        "metric": promotion.goal_metric,
        "label": _performance_estimate_label(promotion.goal_metric),
        "value": round(value, 6),
        "formatted": _format_percent(value),
    }


def _performance_estimate_label(goal_metric: str) -> str:
    if goal_metric == "booking_conversion_rate":
        return "예상 전환율"
    if goal_metric == "inflow_rate":
        return "예상 유입률"
    if goal_metric == "funnel_progression_rate":
        return "예상 퍼널 이동률"
    return "예상 성과"


def _format_percent(value: float) -> str:
    return f"{_clamp01(value) * 100:.1f}%"


def _final_score(candidate: _RawEventCandidate) -> float:
    return (
        0.30 * candidate.promotion_condition_match
        + 0.25 * candidate.expected_goal_performance
        + 0.20 * candidate.behavior_lift_vs_baseline
        + 0.15 * candidate.sample_reliability
        + 0.10 * candidate.rank_distinctiveness
    )


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


def _difference_summary(candidate: _RawEventCandidate, *, rank: int) -> str:
    if rank == 0:
        return "프로모션 조건과 가장 직접적으로 연결되는 행동 신호를 우선한 후보입니다."
    if candidate.candidate_type == "funnel_recovery":
        return "상위 후보보다 목적지 조건은 약할 수 있지만 예약 의도 깊이가 더 강합니다."
    if candidate.candidate_type == "promotion_responsive":
        return "상위 후보보다 구매 단계는 넓지만 캠페인 메시지 반응 가능성이 더 높습니다."
    if candidate.candidate_type == "destination_affinity":
        return "상위 후보보다 퍼널 깊이는 낮아도 특정 목적지 관심이 반복적으로 확인됩니다."
    if candidate.candidate_type == "benefit_value_seeker":
        return "상위 후보보다 넓은 타겟이지만 할인과 혜택 메시지에 반응할 가능성이 큽니다."
    return "다른 Rank와 다른 행동 조건을 기준으로 분리한 후보입니다."


def _sample_ratio(*, sample_size: int, total_eligible_user_count: int) -> Decimal:
    if total_eligible_user_count <= 0:
        return Decimal("0")
    return Decimal(sample_size / total_eligible_user_count).quantize(Decimal("0.000001"))


def _raw_event_segment_id(
    *,
    promotion_id: str,
    candidate_type: str,
    rank: int,
    candidate_user_ids: Sequence[str],
) -> str:
    digest = hashlib.sha1(  # noqa: S324 - stable non-security identifier.
        ":".join([promotion_id, candidate_type, ",".join(candidate_user_ids[:20])]).encode(
            "utf-8"
        )
    ).hexdigest()[:10]
    return f"seg_ai_raw_{_safe_identifier_part(promotion_id)[:32]}_{rank + 1}_{candidate_type}_{digest}"


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
