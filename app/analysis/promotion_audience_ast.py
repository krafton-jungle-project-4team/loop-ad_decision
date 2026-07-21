from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.audience_contract import (
    CUSTOM_STRUCTURED_ANCHOR_POLICY_ID,
    CUSTOM_STRUCTURED_CANDIDATE_TYPE,
    CUSTOM_STRUCTURED_CONDITION_KEY,
    CUSTOM_STRUCTURED_PARAMETER_POLICY_ID,
    CUSTOM_STRUCTURED_SELECTION_POLICY_ID,
    CUSTOM_STRUCTURED_TEMPLATE_HASH,
    CUSTOM_STRUCTURED_TEMPLATE_ID,
    CUSTOM_STRUCTURED_TEMPLATE_VERSION,
    CUSTOM_STRUCTURED_WINDOW_DAYS,
    SEGMENT_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_SCHEMA_VERSION,
    SegmentDefinitionAudienceAdapter,
)
from app.analysis.behavior_manifest import destination_alias_groups
from app.analysis.segment_audience_templates import (
    RegisteredSegmentAudienceBinder,
    TEMPLATE_ID_BY_CANDIDATE_TYPE,
    canonical_benefit_keys,
    canonical_destination_ids,
    canonical_season_months,
    require_registered_template,
)


PROMOTION_AUDIENCE_AST_VERSION = "promotion_audience_ast.v1"
PROMOTION_AUDIENCE_COMPILER_VERSION = "promotion-audience-compiler.v1"
PROMOTION_AUDIENCE_CONTRACT_VERSION = SEGMENT_AUDIENCE_CONTRACT
PROMOTION_AUDIENCE_LOOKBACK_DAYS = CUSTOM_STRUCTURED_WINDOW_DAYS
PROMOTION_AUDIENCE_WINDOW_POLICY = "relative_lookback_end_exclusive.v1"

DESTINATION_ANY_OF = "any_of"
DESTINATION_CONTAINS_ALL = "contains_all"

_STRATEGY_ROLES: Mapping[str, str] = {
    "intent_matched": "프로모션 조건 정합형",
    "target_destination_affinity": "목적지 반복 관심형",
    "funnel_recovery": "예약 이탈 회수형",
    "benefit_value_seeker": "혜택 민감형",
    "promotion_responsive": "프로모션 반응형",
    "general_destination_explorer": "다목적지 탐색형",
    "destination_comparison": "목적지 비교 탐색형",
}

_BENEFIT_LABELS: Mapping[str, str] = {
    "discount": "할인",
    "early_booking": "조기 예약",
    "free_cancellation": "무료 취소",
    "breakfast_included": "조식 포함",
}

_BEHAVIOR_CHIPS: Mapping[str, str] = {
    "hotel_product_interest": "숙소 관심",
    "recent_destination_search": "목적지 검색",
    "summer_checkin_search": "여름 체크인",
    "winter_checkin_search": "겨울 체크인",
    "hotel_detail_view": "호텔 상세 조회",
    "promotion_response": "프로모션 반응",
    "campaign_landing": "캠페인 랜딩",
    "booking_start_without_complete": "예약 시작 후 미완료",
    "target_destination_affinity": "목적지 반복 탐색",
    "general_destination_exploration": "여러 목적지 비교",
    "benefit_interest": "혜택 관심",
    "price_sensitive": "가격 비교",
    "free_cancellation_interest": "무료 취소 관심",
    "breakfast_interest": "조식 포함 관심",
    "profile_hint": "고객 프로필 조건",
    "hotel_search": "숙소 검색",
    "season_match": "체크인 시즌 일치",
}


@dataclass(frozen=True, slots=True)
class PromotionAudienceAst:
    promotion_id: str
    strategy_key: str
    execution_candidate_type: str
    behavior_condition_keys: tuple[str, ...]
    destination_operator: str | None = None
    destination_ids: tuple[str, ...] = ()
    season_months: tuple[int, ...] = ()
    benefit_keys: tuple[str, ...] = ()
    lookback_days: int = PROMOTION_AUDIENCE_LOOKBACK_DAYS
    creative_only: tuple[str, ...] = ()
    unsupported_conditions: tuple[str, ...] = ()
    reference_signal_keys: tuple[str, ...] = ()

    def semantic_payload(self) -> dict[str, Any]:
        destination: dict[str, Any] | None = None
        if self.destination_ids:
            destination = {
                "operator": self.destination_operator or DESTINATION_ANY_OF,
                "values": list(self.destination_ids),
            }
        return {
            "schema_version": PROMOTION_AUDIENCE_AST_VERSION,
            "strategy_key": self.strategy_key,
            "execution_candidate_type": self.execution_candidate_type,
            "logical_operator": "and",
            "behavior_condition_keys": list(self.behavior_condition_keys),
            "destination": destination,
            "season_months": list(self.season_months),
            "benefit_keys": list(self.benefit_keys),
        }

    def evaluation_window(self) -> dict[str, Any]:
        return {
            "lookback_days": self.lookback_days,
            "policy": PROMOTION_AUDIENCE_WINDOW_POLICY,
            "end_exclusive": True,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "evaluation_window": self.evaluation_window(),
            "creative_only": list(self.creative_only),
            "unsupported_conditions": list(self.unsupported_conditions),
            "reference_signal_keys": list(self.reference_signal_keys),
        }


@dataclass(frozen=True, slots=True)
class PromotionAudienceCompilation:
    ast: PromotionAudienceAst
    ast_hash: str
    segment_id: str
    segment_audience_spec: Mapping[str, Any]
    segment_audience_spec_hash: str
    display_model: Mapping[str, Any]


def build_promotion_audience_ast(
    *,
    promotion_id: str,
    candidate_type: str,
    matched_condition_keys: Sequence[str],
    destination_ids: Sequence[str] = (),
    season_months: Sequence[int] = (),
    benefit_keys: Sequence[str] = (),
    destination_operator: str | None = None,
    strategy_key: str | None = None,
    lookback_days: int = PROMOTION_AUDIENCE_LOOKBACK_DAYS,
    unsupported_conditions: Sequence[str] = (),
) -> PromotionAudienceAst:
    destinations = canonical_destination_ids(destination_ids)
    operator = destination_operator if destinations else None
    if operator not in {None, DESTINATION_ANY_OF, DESTINATION_CONTAINS_ALL}:
        raise ValueError(f"unsupported destination operator: {operator}")
    if operator == DESTINATION_CONTAINS_ALL and len(destinations) < 2:
        raise ValueError("contains_all requires at least two destinations")
    creative_only, unsupported = _partition_non_executable_conditions(
        unsupported_conditions
    )
    seasons = canonical_season_months(season_months)
    executable_conditions = _executable_condition_keys(
        candidate_type=candidate_type,
        destination_operator=operator,
        destination_ids=destinations,
        season_months=seasons,
    )
    return PromotionAudienceAst(
        promotion_id=promotion_id,
        strategy_key=strategy_key or candidate_type,
        execution_candidate_type=candidate_type,
        behavior_condition_keys=executable_conditions,
        destination_operator=operator or (DESTINATION_ANY_OF if destinations else None),
        destination_ids=destinations,
        season_months=seasons,
        benefit_keys=canonical_benefit_keys(benefit_keys),
        lookback_days=int(lookback_days),
        creative_only=creative_only,
        unsupported_conditions=unsupported,
        reference_signal_keys=_canonical_condition_keys(matched_condition_keys),
    )


def compile_promotion_audience_ast(
    ast: PromotionAudienceAst,
) -> PromotionAudienceCompilation:
    if ast.lookback_days != PROMOTION_AUDIENCE_LOOKBACK_DAYS:
        raise ValueError(
            "the current Segment Audience V2 contract supports a 30-day lookback"
        )
    if ast.destination_operator == DESTINATION_CONTAINS_ALL:
        audience_spec = _custom_contains_all_spec(ast)
    else:
        audience_spec = RegisteredSegmentAudienceBinder().bind(
            candidate_type=ast.execution_candidate_type,
            destination_ids=ast.destination_ids,
            season_months=ast.season_months,
            benefit_keys=ast.benefit_keys,
        )

    ast_hash = promotion_audience_ast_hash(ast)
    segment_id = promotion_audience_segment_id(ast, ast_hash=ast_hash)
    resolution = SegmentDefinitionAudienceAdapter().resolve(
        segment_id=segment_id,
        rule_json={
            "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
            "segment_audience_spec": dict(audience_spec),
        },
    )
    if resolution.spec is None:
        raise ValueError("compiled promotion audience AST did not produce a V2 spec")
    return PromotionAudienceCompilation(
        ast=ast,
        ast_hash=ast_hash,
        segment_id=segment_id,
        segment_audience_spec=audience_spec,
        segment_audience_spec_hash=resolution.spec.spec_hash,
        display_model=_display_model(ast),
    )


def promotion_audience_ast_hash(ast: PromotionAudienceAst) -> str:
    return _sha256_json(ast.semantic_payload())


def promotion_audience_segment_id(
    ast: PromotionAudienceAst,
    *,
    ast_hash: str | None = None,
) -> str:
    fingerprint_payload = {
        "promotion_id": ast.promotion_id,
        "normalized_condition_ast": ast.semantic_payload(),
        "evaluation_window": ast.evaluation_window(),
        "condition_compiler_version": PROMOTION_AUDIENCE_COMPILER_VERSION,
        "audience_contract_version": PROMOTION_AUDIENCE_CONTRACT_VERSION,
    }
    fingerprint = _sha256_json(fingerprint_payload)
    strategy = _safe_identifier_part(ast.strategy_key)[:36] or "dynamic"
    promotion = _safe_identifier_part(ast.promotion_id)[:32] or "promotion"
    return f"seg_ai_dynamic_{promotion}_{strategy}_{fingerprint[:12]}"


def _custom_contains_all_spec(ast: PromotionAudienceAst) -> Mapping[str, Any]:
    if ast.execution_candidate_type != "general_destination_explorer":
        raise ValueError("contains_all is only supported by destination comparison")
    conditions = [
        {
            "event_name": "hotel_search",
            "minimum_count": 1,
            "maximum_count": None,
            "destination": destination_id,
            "checkin_months": [],
            "property_filters": [],
            "label": f"{_destination_label(destination_id)} 숙소 검색",
        }
        for destination_id in ast.destination_ids
    ]
    return {
        "schema_version": SEGMENT_AUDIENCE_SCHEMA_VERSION,
        "template_id": CUSTOM_STRUCTURED_TEMPLATE_ID,
        "template_version": CUSTOM_STRUCTURED_TEMPLATE_VERSION,
        "template_semantic_hash": CUSTOM_STRUCTURED_TEMPLATE_HASH,
        "candidate_type": CUSTOM_STRUCTURED_CANDIDATE_TYPE,
        "condition_keys": [CUSTOM_STRUCTURED_CONDITION_KEY],
        "query_signal_keys": ["hotel_search_intensity"],
        "hard_predicate_keys": [CUSTOM_STRUCTURED_CONDITION_KEY],
        "parameters": {
            "lookback_days": ast.lookback_days,
            "conditions": conditions,
        },
        "parameter_policy_id": CUSTOM_STRUCTURED_PARAMETER_POLICY_ID,
        "semantic_selection_policy_id": CUSTOM_STRUCTURED_SELECTION_POLICY_ID,
        "semantic_anchor_policy_id": CUSTOM_STRUCTURED_ANCHOR_POLICY_ID,
        "observation_window_days": CUSTOM_STRUCTURED_WINDOW_DAYS,
    }


def _display_model(ast: PromotionAudienceAst) -> Mapping[str, Any]:
    destination_text = _destination_text(ast.destination_ids)
    destination_prefix = f"{destination_text} " if destination_text else ""
    benefit_text = "·".join(
        _BENEFIT_LABELS.get(key, key) for key in ast.benefit_keys
    )
    strategy_key = ast.strategy_key
    if strategy_key == "funnel_recovery":
        title = f"{destination_prefix}예약 직전 이탈 고객"
        reason = (
            f"{destination_text or '프로모션 목적지'} 숙소를 탐색하고 예약을 "
            "시작했지만 완료하지 않은 고객입니다."
        )
    elif strategy_key == "target_destination_affinity":
        title = f"{destination_prefix}숙소를 반복 탐색한 고객"
        reason = (
            f"{destination_text or '프로모션 목적지'} 숙소를 반복해서 탐색한 "
            "행동이 확인된 고객입니다."
        )
    elif strategy_key == "benefit_value_seeker":
        title = f"{destination_prefix}{benefit_text or '할인·혜택'} 관심 고객"
        reason = (
            f"{destination_text or '프로모션 목적지'} 숙소 관심과 "
            f"{benefit_text or '가격·혜택'} 탐색 행동이 함께 확인된 고객입니다."
        )
    elif strategy_key == "destination_comparison":
        title = f"{destination_text}를 비교 탐색한 고객"
        reason = (
            f"{destination_text} 숙소를 모두 검색해 여행지를 비교한 고객입니다."
        )
    elif strategy_key == "general_destination_explorer":
        title = "여러 여행지를 비교 탐색한 고객"
        reason = "두 곳 이상의 여행지 숙소를 비교 탐색한 고객입니다."
    elif strategy_key == "promotion_responsive":
        title = f"{destination_prefix}프로모션 반응 고객"
        reason = "프로모션 클릭이나 캠페인 랜딩 행동이 확인된 고객입니다."
    else:
        title = f"{destination_prefix}프로모션 조건 관심 고객"
        reason = (
            f"{destination_text or '프로모션'} 조건과 일치하는 숙소 탐색 행동이 "
            "확인된 고객입니다."
        )

    chips = list(_destination_chips(ast))
    chips.extend(
        _BEHAVIOR_CHIPS.get(key, key) for key in ast.behavior_condition_keys
    )
    chips.extend(_BENEFIT_LABELS.get(key, key) for key in ast.benefit_keys)
    return {
        "title": " ".join(title.split()),
        "strategy_role": _STRATEGY_ROLES.get(strategy_key, "동적 조건형"),
        "signal_chips": list(dict.fromkeys(chips))[:5],
        "reason": reason,
        "metric_label": "행동 기반 예상 예약 전환율",
        "metric_description": (
            "과거 행동을 바탕으로 추정한 향후 예약 가능성이며, "
            "광고로 인한 증가율은 아닙니다."
        ),
    }


def _destination_chips(ast: PromotionAudienceAst) -> tuple[str, ...]:
    if not ast.destination_ids:
        return ()
    destination_text = _destination_text(ast.destination_ids)
    if ast.destination_operator == DESTINATION_CONTAINS_ALL:
        return (f"{destination_text} 모두 검색",)
    if len(ast.destination_ids) > 1:
        return (f"{destination_text} 중 한 곳 탐색",)
    return (f"{destination_text} 숙소 탐색",)


def _destination_text(destination_ids: Sequence[str]) -> str:
    return "·".join(_destination_label(value) for value in destination_ids)


def _destination_label(destination_id: str) -> str:
    aliases = destination_alias_groups().get(destination_id, ())
    for alias in aliases:
        if any(ord(character) > 127 for character in alias):
            return alias
    return destination_id.replace("_", " ").replace("-", " ").title()


def _partition_non_executable_conditions(
    values: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    creative: list[str] = []
    unsupported: list[str] = []
    for raw_value in values:
        value = " ".join(str(raw_value).strip().split())
        if not value:
            continue
        if value.startswith(("destination:", "benefit:")):
            unsupported.append(value)
        else:
            creative.append(value)
    return tuple(dict.fromkeys(creative)), tuple(dict.fromkeys(unsupported))


def _canonical_condition_keys(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _executable_condition_keys(
    *,
    candidate_type: str,
    destination_operator: str | None,
    destination_ids: Sequence[str],
    season_months: Sequence[int],
) -> tuple[str, ...]:
    if destination_operator == DESTINATION_CONTAINS_ALL:
        return ("hotel_search",)
    template_id = TEMPLATE_ID_BY_CANDIDATE_TYPE.get(candidate_type)
    if template_id is None:
        raise ValueError(f"unregistered segment audience candidate: {candidate_type}")
    template = require_registered_template(template_id)
    return _canonical_condition_keys(
        template.hard_predicate_keys(
            destination_ids=destination_ids,
            season_months=season_months,
        )
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_identifier_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in value
    )
