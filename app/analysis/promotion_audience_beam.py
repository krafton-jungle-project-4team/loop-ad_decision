"""Bounded, executable predicate search for promotion audience candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.analysis.promotion_audience_ast import (
    DESTINATION_ANY_OF,
    build_promotion_audience_ast,
    compile_promotion_audience_ast,
)
from app.analysis.repositories import RawEventUserSignalRecord
from app.analysis.segment_audience_templates import (
    canonical_benefit_keys,
    canonical_destination_ids,
    canonical_season_months,
)


PROMOTION_AUDIENCE_BEAM_POLICY_VERSION = "promotion-audience-beam.v1"


@dataclass(frozen=True, slots=True)
class PromotionAudienceBeamPolicy:
    policy_version: str = PROMOTION_AUDIENCE_BEAM_POLICY_VERSION
    beam_width: int = 8
    maximum_depth: int = 3
    maximum_generated_candidates: int = 50
    maximum_final_candidates: int = 3
    maximum_jaccard_similarity: float = 0.85


DEFAULT_PROMOTION_AUDIENCE_BEAM_POLICY = PromotionAudienceBeamPolicy()


@dataclass(frozen=True, slots=True)
class ExecutableAudiencePredicate:
    predicate_key: str
    dimension: str
    event_name: str
    required_parameters: tuple[str, ...]
    minimum_count_options: tuple[int, ...]
    conflicts_with: tuple[str, ...]
    requires: tuple[str, ...]
    compiler_target: str
    display_label: str


PREDICATE_REGISTRY: tuple[ExecutableAudiencePredicate, ...] = (
    ExecutableAudiencePredicate(
        "destination_repeat_search",
        "destination_affinity",
        "hotel_search",
        ("destination_ids",),
        (2, 3),
        (),
        ("promotion_destination_search",),
        "custom_structured_condition.v1",
        "목적지 숙소 반복 검색",
    ),
    ExecutableAudiencePredicate(
        "hotel_detail_view",
        "consideration",
        "hotel_detail_view",
        (),
        (1, 2),
        (),
        (),
        "custom_structured_condition.v1",
        "호텔 상세 조회",
    ),
    ExecutableAudiencePredicate(
        "booking_start",
        "funnel",
        "booking_start",
        (),
        (1,),
        ("booking_start_without_complete",),
        (),
        "custom_structured_condition.v1",
        "예약 시작",
    ),
    ExecutableAudiencePredicate(
        "booking_start_without_complete",
        "funnel",
        "booking_start",
        (),
        (1,),
        ("booking_start",),
        (),
        "custom_structured_condition.v1",
        "예약 시작 후 미완료",
    ),
    ExecutableAudiencePredicate(
        "price_compare",
        "value",
        "hotel_search",
        ("price",),
        (1, 2),
        (),
        (),
        "custom_structured_condition.v1",
        "가격 비교",
    ),
    ExecutableAudiencePredicate(
        "discount_interest",
        "value",
        "hotel_search",
        ("deal",),
        (1,),
        (),
        (),
        "custom_structured_condition.v1",
        "할인·특가 관심",
    ),
    ExecutableAudiencePredicate(
        "free_cancellation_interest",
        "benefit",
        "hotel_search",
        ("free_cancellation",),
        (1,),
        (),
        (),
        "custom_structured_condition.v1",
        "무료 취소 관심",
    ),
    ExecutableAudiencePredicate(
        "breakfast_interest",
        "benefit",
        "hotel_search",
        ("breakfast_included",),
        (1,),
        (),
        (),
        "custom_structured_condition.v1",
        "조식 포함 관심",
    ),
    ExecutableAudiencePredicate(
        "promotion_click",
        "promotion_response",
        "promotion_click",
        (),
        (1,),
        (),
        (),
        "custom_structured_condition.v1",
        "프로모션 클릭",
    ),
    ExecutableAudiencePredicate(
        "campaign_landing",
        "promotion_response",
        "campaign_landing",
        (),
        (1,),
        (),
        (),
        "custom_structured_condition.v1",
        "캠페인 랜딩",
    ),
)


@dataclass(frozen=True, slots=True)
class BeamPredicateChoice:
    predicate_key: str
    minimum_count: int


@dataclass(frozen=True, slots=True)
class BeamAudienceCandidate:
    choices: tuple[BeamPredicateChoice, ...]
    structured_conditions: tuple[Mapping[str, Any], ...]
    user_ids: tuple[str, ...]
    candidate_type: str
    strategy_key: str
    score: float
    score_components: Mapping[str, float]

    @property
    def depth(self) -> int:
        return len(self.choices)


@dataclass(frozen=True, slots=True)
class BeamSearchResult:
    candidates: tuple[BeamAudienceCandidate, ...]
    generated_candidate_count: int
    pruned_candidate_counts: Mapping[str, int]
    policy: PromotionAudienceBeamPolicy


def search_promotion_audience_candidates(
    *,
    promotion_id: str,
    destination_ids: Sequence[str],
    season_months: Sequence[int],
    benefit_keys: Sequence[str],
    desired_behavior_keys: Sequence[str],
    profiles: Sequence[RawEventUserSignalRecord],
    min_sample_size: int,
    policy: PromotionAudienceBeamPolicy = DEFAULT_PROMOTION_AUDIENCE_BEAM_POLICY,
) -> BeamSearchResult:
    ordered_profiles = tuple(sorted(profiles, key=lambda value: value.user_id))
    destinations = canonical_destination_ids(destination_ids)
    seasons = canonical_season_months(season_months)
    benefits = canonical_benefit_keys(benefit_keys)
    mandatory_keys = _mandatory_predicate_keys(
        destination_ids=destinations,
        season_months=seasons,
        benefit_keys=benefits,
    )
    mandatory_conditions = _mandatory_conditions(
        destination_ids=destinations,
        season_months=seasons,
        benefit_keys=benefits,
    )
    mandatory_profiles = tuple(
        profile
        for profile in ordered_profiles
        if _matches_mandatory(
            profile,
            destination_ids=destinations,
            season_months=seasons,
            benefit_keys=benefits,
        )
    )
    pruned: dict[str, int] = {}
    if len(mandatory_profiles) < min_sample_size:
        return BeamSearchResult((), 0, {"minimum_sample": 1}, policy)

    registry = tuple(
        predicate
        for predicate in PREDICATE_REGISTRY
        if _predicate_is_available(
            predicate,
            mandatory_keys=mandatory_keys,
            destination_ids=destinations,
        )
    )
    beam: tuple[tuple[BeamPredicateChoice, ...], ...] = ((),)
    evaluated: list[BeamAudienceCandidate] = []
    generated = 0
    seen_choices: set[tuple[tuple[str, int], ...]] = set()
    seen_member_sets: dict[tuple[str, ...], BeamAudienceCandidate] = {}

    for _depth in range(1, policy.maximum_depth + 1):
        depth_candidates: list[BeamAudienceCandidate] = []
        for choices in beam:
            used_keys = {choice.predicate_key for choice in choices}
            for predicate in registry:
                if generated >= policy.maximum_generated_candidates:
                    break
                if predicate.predicate_key in used_keys:
                    _increment(pruned, "duplicate_predicate")
                    continue
                if not _compatible(predicate, used_keys=used_keys):
                    _increment(pruned, "conflict")
                    continue
                for minimum_count in predicate.minimum_count_options:
                    if generated >= policy.maximum_generated_candidates:
                        break
                    next_choices = tuple(
                        sorted(
                            (*choices, BeamPredicateChoice(
                                predicate.predicate_key,
                                minimum_count,
                            )),
                            key=lambda choice: (
                                _registry_index(choice.predicate_key),
                                choice.minimum_count,
                            ),
                        )
                    )
                    choice_key = tuple(
                        (choice.predicate_key, choice.minimum_count)
                        for choice in next_choices
                    )
                    if choice_key in seen_choices:
                        continue
                    seen_choices.add(choice_key)
                    generated += 1
                    candidate = _evaluate_candidate(
                        promotion_id=promotion_id,
                        choices=next_choices,
                        mandatory_conditions=mandatory_conditions,
                        mandatory_profiles=mandatory_profiles,
                        total_profile_count=len(ordered_profiles),
                        destination_ids=destinations,
                        season_months=seasons,
                        benefit_keys=benefits,
                        desired_behavior_keys=desired_behavior_keys,
                        min_sample_size=min_sample_size,
                        policy=policy,
                    )
                    if candidate is None:
                        _increment(pruned, "unusable")
                        continue
                    if len(candidate.user_ids) < min_sample_size:
                        _increment(pruned, "minimum_sample")
                        continue
                    if len(candidate.user_ids) == len(ordered_profiles):
                        _increment(pruned, "same_as_all_users")
                        continue
                    existing = seen_member_sets.get(candidate.user_ids)
                    if existing is not None:
                        _increment(pruned, "duplicate_members")
                        if candidate.score <= existing.score:
                            continue
                        evaluated.remove(existing)
                        if existing in depth_candidates:
                            depth_candidates.remove(existing)
                    seen_member_sets[candidate.user_ids] = candidate
                    evaluated.append(candidate)
                    depth_candidates.append(candidate)
        if not depth_candidates or generated >= policy.maximum_generated_candidates:
            break
        beam = tuple(
            candidate.choices
            for candidate in sorted(
                depth_candidates,
                key=_candidate_sort_key,
            )[: policy.beam_width]
        )

    selected: list[BeamAudienceCandidate] = []
    for candidate in sorted(evaluated, key=_candidate_sort_key):
        overlap = max(
            (_jaccard(candidate.user_ids, value.user_ids) for value in selected),
            default=0.0,
        )
        if overlap >= policy.maximum_jaccard_similarity:
            _increment(pruned, "candidate_overlap")
            continue
        adjusted_components = dict(candidate.score_components)
        adjusted_components["candidate_overlap_penalty"] = overlap * 0.12
        adjusted = BeamAudienceCandidate(
            choices=candidate.choices,
            structured_conditions=candidate.structured_conditions,
            user_ids=candidate.user_ids,
            candidate_type=candidate.candidate_type,
            strategy_key=candidate.strategy_key,
            score=candidate.score - overlap * 0.12,
            score_components=adjusted_components,
        )
        selected.append(adjusted)
        if len(selected) >= policy.maximum_final_candidates:
            break
    return BeamSearchResult(tuple(selected), generated, dict(sorted(pruned.items())), policy)


def _evaluate_candidate(
    *,
    promotion_id: str,
    choices: tuple[BeamPredicateChoice, ...],
    mandatory_conditions: tuple[Mapping[str, Any], ...],
    mandatory_profiles: Sequence[RawEventUserSignalRecord],
    total_profile_count: int,
    destination_ids: tuple[str, ...],
    season_months: tuple[int, ...],
    benefit_keys: tuple[str, ...],
    desired_behavior_keys: Sequence[str],
    min_sample_size: int,
    policy: PromotionAudienceBeamPolicy,
) -> BeamAudienceCandidate | None:
    predicates = tuple(_predicate(choice.predicate_key) for choice in choices)
    matched_profiles = tuple(
        profile
        for profile in mandatory_profiles
        if all(
            _matches_predicate(profile, predicate, choice.minimum_count)
            for predicate, choice in zip(predicates, choices, strict=True)
        )
    )
    conditions = _canonical_conditions(
        (
            *mandatory_conditions,
            *(
                condition
                for predicate, choice in zip(predicates, choices, strict=True)
                for condition in _predicate_conditions(
                    predicate,
                    minimum_count=choice.minimum_count,
                    destination_ids=destination_ids,
                )
            ),
        )
    )
    if not conditions or len(conditions) > 8:
        return None
    candidate_type = _candidate_type(predicates)
    strategy_key = "beam_" + "__".join(
        f"{choice.predicate_key}_{choice.minimum_count}" for choice in choices
    )
    try:
        ast = build_promotion_audience_ast(
            promotion_id=promotion_id,
            candidate_type=candidate_type,
            strategy_key=strategy_key,
            matched_condition_keys=tuple(
                choice.predicate_key for choice in choices
            ),
            destination_ids=destination_ids,
            season_months=season_months,
            benefit_keys=benefit_keys,
            destination_operator=DESTINATION_ANY_OF,
            structured_conditions=conditions,
            beam_policy_version=policy.policy_version,
        )
        compile_promotion_audience_ast(ast)
    except (TypeError, ValueError):
        return None

    count = len(matched_profiles)
    reach = count / max(total_profile_count, 1)
    desired = set(desired_behavior_keys)
    aligned = sum(
        1
        for choice in choices
        if choice.predicate_key in desired
        or _predicate(choice.predicate_key).dimension in {"funnel", "value", "benefit"}
    )
    promotion_alignment = min(1.0, 0.7 + aligned / max(10.0, len(choices) * 3.0))
    behavior_intent = sum(
        _dimension_intent_score(predicate.dimension) for predicate in predicates
    ) / max(len(predicates), 1)
    if any(
        predicate.predicate_key == "booking_start_without_complete"
        for predicate in predicates
    ):
        behavior_intent = min(1.0, behavior_intent + 0.05)
    sample_reliability = min(1.0, count / max(min_sample_size * 3.0, 1.0))
    expected_booking = _expected_booking_score(matched_profiles)
    audience_reach = min(1.0, math.sqrt(max(reach, 0.0)))
    distinctiveness = max(0.0, 1.0 - reach)
    complexity_penalty = max(0, len(choices) - 1) * 0.04
    small_sample_penalty = max(
        0.0,
        (min_sample_size * 2 - count) / max(min_sample_size * 2, 1),
    ) * 0.15
    score = (
        0.22 * promotion_alignment
        + 0.18 * behavior_intent
        + 0.14 * sample_reliability
        + 0.24 * expected_booking
        + 0.10 * audience_reach
        + 0.12 * distinctiveness
        - complexity_penalty
        - small_sample_penalty
    )
    user_ids = tuple(sorted(profile.user_id for profile in matched_profiles))
    return BeamAudienceCandidate(
        choices=choices,
        structured_conditions=conditions,
        user_ids=user_ids,
        candidate_type=candidate_type,
        strategy_key=strategy_key,
        score=score,
        score_components={
            "promotion_alignment_score": promotion_alignment,
            "behavior_intent_score": behavior_intent,
            "sample_reliability_score": sample_reliability,
            "expected_booking_score": expected_booking,
            "audience_reach_score": audience_reach,
            "distinctiveness_score": distinctiveness,
            "complexity_penalty": complexity_penalty,
            "candidate_overlap_penalty": 0.0,
            "small_sample_penalty": small_sample_penalty,
        },
    )


def _mandatory_predicate_keys(
    *,
    destination_ids: Sequence[str],
    season_months: Sequence[int],
    benefit_keys: Sequence[str],
) -> set[str]:
    result: set[str] = set()
    if destination_ids:
        result.add("promotion_destination_search")
    if season_months:
        result.add("promotion_season_search")
    result.update(_benefit_predicate_keys(benefit_keys))
    return result


def _mandatory_conditions(
    *,
    destination_ids: tuple[str, ...],
    season_months: tuple[int, ...],
    benefit_keys: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    conditions: list[Mapping[str, Any]] = []
    if destination_ids or season_months:
        destination_label = "·".join(destination_ids)
        season_label = "시즌 일치 " if season_months else ""
        conditions.append(
            _condition(
                label=f"{destination_label} {season_label}숙소 검색".strip(),
                event_name="hotel_search",
                minimum_count=1,
                destination=",".join(destination_ids) or None,
                checkin_months=season_months,
            )
        )
    for predicate_key in _benefit_predicate_keys(benefit_keys):
        predicate = _predicate(predicate_key)
        conditions.extend(
            _predicate_conditions(
                predicate,
                minimum_count=1,
                destination_ids=destination_ids,
            )
        )
    return _canonical_conditions(conditions)


def _matches_mandatory(
    profile: RawEventUserSignalRecord,
    *,
    destination_ids: Sequence[str],
    season_months: Sequence[int],
    benefit_keys: Sequence[str],
) -> bool:
    if destination_ids and profile.destination_match_count <= 0:
        return False
    if season_months and profile.season_match_count <= 0:
        return False
    return all(
        _matches_predicate(profile, _predicate(key), 1)
        for key in _benefit_predicate_keys(benefit_keys)
    )


def _benefit_predicate_keys(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        key = {
            "discount": "discount_interest",
            "early_booking": "discount_interest",
            "free_cancellation": "free_cancellation_interest",
            "breakfast_included": "breakfast_interest",
        }.get(value)
        if key and key not in result:
            result.append(key)
    return tuple(result)


def _predicate_is_available(
    predicate: ExecutableAudiencePredicate,
    *,
    mandatory_keys: set[str],
    destination_ids: Sequence[str],
) -> bool:
    if predicate.compiler_target != "custom_structured_condition.v1":
        return False
    if predicate.predicate_key in mandatory_keys:
        return False
    if "destination_ids" in predicate.required_parameters and not destination_ids:
        return False
    return all(required in mandatory_keys for required in predicate.requires)


def _compatible(
    predicate: ExecutableAudiencePredicate,
    *,
    used_keys: set[str],
) -> bool:
    return not used_keys.intersection(predicate.conflicts_with) and not any(
        predicate.predicate_key in _predicate(key).conflicts_with for key in used_keys
    )


def _matches_predicate(
    profile: RawEventUserSignalRecord,
    predicate: ExecutableAudiencePredicate,
    minimum_count: int,
) -> bool:
    key = predicate.predicate_key
    if key == "destination_repeat_search":
        return profile.destination_match_count >= minimum_count
    if key == "hotel_detail_view":
        return profile.hotel_detail_view_count >= minimum_count
    if key == "booking_start":
        return profile.booking_start_count >= minimum_count
    if key == "booking_start_without_complete":
        return (
            profile.booking_start_count >= minimum_count
            and profile.booking_complete_count == 0
        )
    if key == "price_compare":
        return profile.price_event_count >= minimum_count
    if key == "discount_interest":
        return profile.deal_event_count >= minimum_count
    if key == "free_cancellation_interest":
        return profile.free_cancellation_count >= minimum_count
    if key == "breakfast_interest":
        return profile.breakfast_included_count >= minimum_count
    if key == "promotion_click":
        return profile.promotion_click_count >= minimum_count
    if key == "campaign_landing":
        return profile.campaign_landing_count >= minimum_count
    return False


def _predicate_conditions(
    predicate: ExecutableAudiencePredicate,
    *,
    minimum_count: int,
    destination_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    destination = (
        ",".join(destination_ids)
        if "destination_ids" in predicate.required_parameters
        else None
    )
    if predicate.predicate_key == "booking_start_without_complete":
        return (
            _condition(
                label=predicate.display_label,
                event_name="booking_start",
                minimum_count=minimum_count,
            ),
            _condition(
                label="예약 미완료",
                event_name="booking_complete",
                minimum_count=0,
                maximum_count=0,
            ),
        )
    property_filter = {
        "price_compare": ("price", "exists", "true"),
        "discount_interest": ("deal", "equals", "true"),
        "free_cancellation_interest": (
            "free_cancellation",
            "equals",
            "true",
        ),
        "breakfast_interest": ("breakfast_included", "equals", "true"),
    }.get(predicate.predicate_key)
    filters = (
        ({"key": property_filter[0], "operator": property_filter[1], "value": property_filter[2]},)
        if property_filter
        else ()
    )
    label = (
        f"{predicate.display_label} {minimum_count}회 이상"
        if minimum_count > 1
        else predicate.display_label
    )
    return (
        _condition(
            label=label,
            event_name=predicate.event_name,
            minimum_count=minimum_count,
            destination=destination,
            property_filters=filters,
        ),
    )


def _condition(
    *,
    label: str,
    event_name: str,
    minimum_count: int,
    maximum_count: int | None = None,
    destination: str | None = None,
    checkin_months: Sequence[int] = (),
    property_filters: Sequence[Mapping[str, str]] = (),
) -> Mapping[str, Any]:
    return {
        "label": label,
        "event_name": event_name,
        "minimum_count": minimum_count,
        "maximum_count": maximum_count,
        "destination": destination,
        "checkin_months": list(checkin_months),
        "property_filters": [dict(value) for value in property_filters],
    }


def _canonical_conditions(
    values: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    keyed: dict[str, Mapping[str, Any]] = {}
    for value in values:
        semantic = {key: item for key, item in value.items() if key != "label"}
        key = json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        keyed.setdefault(key, value)
    return tuple(dict(keyed[key]) for key in sorted(keyed))


def _candidate_type(
    predicates: Sequence[ExecutableAudiencePredicate],
) -> str:
    keys = {predicate.predicate_key for predicate in predicates}
    if "booking_start_without_complete" in keys:
        return "funnel_recovery"
    if keys.intersection(
        {"price_compare", "discount_interest", "free_cancellation_interest", "breakfast_interest"}
    ):
        return "benefit_value_seeker"
    if keys.intersection({"promotion_click", "campaign_landing"}):
        return "promotion_responsive"
    if "destination_repeat_search" in keys:
        return "target_destination_affinity"
    return "intent_matched"


def _expected_booking_score(
    profiles: Sequence[RawEventUserSignalRecord],
) -> float:
    if not profiles:
        return 0.0
    count = len(profiles)
    detail_rate = sum(value.hotel_detail_view_count > 0 for value in profiles) / count
    start_rate = sum(value.booking_start_count > 0 for value in profiles) / count
    historical_complete_rate = (
        sum(value.booking_complete_count > 0 for value in profiles) / count
    )
    return min(
        1.0,
        0.35 * detail_rate + 0.45 * start_rate + 0.20 * historical_complete_rate,
    )


def _dimension_intent_score(dimension: str) -> float:
    return {
        "funnel": 0.95,
        "destination_affinity": 0.92,
        "consideration": 0.82,
        "value": 0.78,
        "benefit": 0.76,
        "promotion_response": 0.62,
    }.get(dimension, 0.5)


def _candidate_sort_key(candidate: BeamAudienceCandidate) -> tuple[Any, ...]:
    return (
        -candidate.score,
        -len(candidate.user_ids),
        tuple(
            (choice.predicate_key, choice.minimum_count)
            for choice in candidate.choices
        ),
    )


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_values = set(left)
    right_values = set(right)
    union = left_values | right_values
    return len(left_values & right_values) / len(union) if union else 0.0


def _predicate(predicate_key: str) -> ExecutableAudiencePredicate:
    return next(
        value for value in PREDICATE_REGISTRY if value.predicate_key == predicate_key
    )


def _registry_index(predicate_key: str) -> int:
    return next(
        index
        for index, value in enumerate(PREDICATE_REGISTRY)
        if value.predicate_key == predicate_key
    )


def _increment(values: dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1
