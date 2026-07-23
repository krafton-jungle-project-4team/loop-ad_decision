"""Canonical property conditions shared by intent, search, and display layers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from app.audience_contract import (
    CUSTOM_STRUCTURED_EVENT_NAMES,
    CUSTOM_STRUCTURED_PROPERTY_KEYS,
    CUSTOM_STRUCTURED_PROPERTY_OPERATORS,
)


# The public structured audience contract accepts at most eight conditions.
# Keep two slots for a promotion anchor and a candidate-specific behavior.
MAX_SEGMENT_PROPERTY_CONDITIONS = 6

_PROPERTY_LABELS: Mapping[str, str] = {
    "deal": "할인·특가",
    "free_cancellation": "무료 취소",
    "breakfast_included": "조식 포함",
    "age_group": "연령대",
    "gender": "성별",
    "region": "지역",
    "preferred_category": "선호 카테고리",
    "user_segment": "고객 분류",
    "adult_count": "성인 인원",
    "child_count": "아동 인원",
    "rooms": "객실 수",
    "hotel_id": "숙소",
    "hotel_name": "숙소명",
    "hotel_city": "숙소 도시",
    "hotel_country": "숙소 국가",
    "hotel_market": "숙소 권역",
    "hotel_cluster": "숙소 군집",
    "hotel_star_rating": "호텔 등급",
    "hotel_guest_rating": "고객 평점",
    "price": "가격",
    "property_type": "숙소 유형",
    "room_type": "객실 유형",
    "revenue": "숙박 총액",
}

_TRUE_VALUES = {"1", "true", "yes", "y", "예", "있음"}
_FALSE_VALUES = {"0", "false", "no", "n", "아니오", "없음"}
_AGE_TWENTIES_THIRTIES = (
    "20",
    "20-29",
    "20대",
    "20s",
    "30",
    "30-39",
    "30대",
    "30s",
)
_GENDER_VALUES: Mapping[str, tuple[str, ...]] = {
    "male": ("m", "male", "남", "남성"),
    "female": ("f", "female", "여", "여성"),
}


@dataclass(frozen=True, slots=True)
class SegmentPropertyCondition:
    event_name: str
    property_key: str
    operator: str
    value: str
    minimum_count: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "property_key": self.property_key,
            "operator": self.operator,
            "value": self.value,
            "minimum_count": self.minimum_count,
        }

    def to_structured_condition(self) -> Mapping[str, Any]:
        property_filter = {
            "key": self.property_key,
            "operator": self.operator,
            "value": self.value,
        }
        return {
            "label": segment_property_filter_label(property_filter),
            "event_name": self.event_name,
            "minimum_count": self.minimum_count,
            "maximum_count": None,
            "destination": None,
            "checkin_months": [],
            "property_filters": [property_filter],
        }


def canonical_segment_property_conditions(
    payload: Any,
) -> tuple[tuple[SegmentPropertyCondition, ...], tuple[str, ...]]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return (), (() if payload is None else ("property_condition:invalid_array",))

    valid: dict[tuple[str, str, str, str, int], SegmentPropertyCondition] = {}
    unsupported: list[str] = []
    for item in payload[:MAX_SEGMENT_PROPERTY_CONDITIONS]:
        if not isinstance(item, Mapping):
            unsupported.append("property_condition:invalid_item")
            continue
        try:
            condition = _canonical_condition(item)
        except ValueError as exc:
            unsupported.append(f"property_condition:{exc}")
            continue
        key = (
            condition.event_name,
            condition.property_key,
            condition.operator,
            condition.value,
            condition.minimum_count,
        )
        valid.setdefault(key, condition)
    return (
        tuple(valid[key] for key in sorted(valid)),
        tuple(dict.fromkeys(unsupported)),
    )


def segment_property_conditions_from_hints(
    audience_hints: Sequence[str],
) -> tuple[SegmentPropertyCondition, ...]:
    hints = set(audience_hints)
    conditions: list[SegmentPropertyCondition] = []
    if "20s_30s" in hints:
        conditions.append(
            SegmentPropertyCondition(
                event_name="hotel_search",
                property_key="age_group",
                operator="in",
                value=",".join(_AGE_TWENTIES_THIRTIES),
            )
        )
    requested_genders = hints & _GENDER_VALUES.keys()
    if len(requested_genders) == 1:
        gender = next(iter(requested_genders))
        conditions.append(
            SegmentPropertyCondition(
                event_name="hotel_search",
                property_key="gender",
                operator="in",
                value=",".join(_GENDER_VALUES[gender]),
            )
        )
    return tuple(conditions)


def merge_segment_property_conditions(
    primary: Sequence[SegmentPropertyCondition],
    fallback: Sequence[SegmentPropertyCondition],
) -> tuple[SegmentPropertyCondition, ...]:
    merged: list[SegmentPropertyCondition] = list(primary)
    occupied_keys = {
        condition.property_key
        for condition in primary
        if condition.property_key in {"age_group", "gender"}
    }
    merged.extend(
        condition
        for condition in fallback
        if condition.property_key not in occupied_keys
    )
    keyed = {
        (
            condition.event_name,
            condition.property_key,
            condition.operator,
            condition.value,
            condition.minimum_count,
        ): condition
        for condition in merged
    }
    return tuple(keyed[key] for key in sorted(keyed))[
        :MAX_SEGMENT_PROPERTY_CONDITIONS
    ]


def structured_conditions_from_segment_properties(
    conditions: Sequence[SegmentPropertyCondition],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(condition.to_structured_condition() for condition in conditions)


def profile_matches_segment_properties(
    profile: Any,
    conditions: Sequence[SegmentPropertyCondition],
) -> bool:
    if not conditions:
        return True
    exact_count = getattr(profile, "segment_property_match_count", None)
    if exact_count is not None:
        return int(exact_count) >= len(conditions)
    return all(_profile_matches_condition(profile, condition) for condition in conditions)


def segment_property_filter_label(property_filter: Mapping[str, Any]) -> str:
    key = str(property_filter.get("key", "")).strip()
    operator = str(property_filter.get("operator", "")).strip()
    value = str(property_filter.get("value", "")).strip()
    label = _PROPERTY_LABELS.get(key, key)

    if key == "age_group" and operator == "in":
        values = set(_split_values(value))
        if values.intersection(_AGE_TWENTIES_THIRTIES[:4]) and values.intersection(
            _AGE_TWENTIES_THIRTIES[4:]
        ):
            return "20·30대"
    if key == "gender":
        values = set(_split_values(value))
        if values.intersection(_GENDER_VALUES["female"]):
            return "여성"
        if values.intersection(_GENDER_VALUES["male"]):
            return "남성"
    if operator == "exists":
        return f"{label} 있음"
    if operator == "gte":
        return f"{label} {value} 이상"
    if operator == "lte":
        return f"{label} {value} 이하"
    if operator == "contains":
        return f"{label} {value} 포함"
    if operator == "in":
        return f"{label} {'·'.join(_split_values(value))}"
    if key in {"deal", "free_cancellation", "breakfast_included"}:
        return f"{label} 관심" if value == "true" else f"{label} 제외"
    return f"{label} {value}".strip()


def property_parameter_value(condition: SegmentPropertyCondition) -> Any:
    if condition.operator == "in":
        return list(_split_values(condition.value))
    if condition.operator in {"gte", "lte"}:
        return float(condition.value)
    if (
        condition.operator == "equals"
        and condition.property_key
        in {"free_cancellation", "breakfast_included"}
    ):
        return "1" if condition.value == "true" else "0"
    return condition.value


def clickhouse_segment_property_match_expression(
    conditions: Sequence[SegmentPropertyCondition],
) -> tuple[str, Mapping[str, Any]]:
    if not conditions:
        return "toUInt16(0)", {}

    matches: list[str] = []
    parameters: dict[str, Any] = {}
    for index, condition in enumerate(conditions):
        prefix = f"segment_property_{index}"
        parameters[f"{prefix}_event_name"] = condition.event_name
        parameters[f"{prefix}_minimum_count"] = condition.minimum_count
        property_predicate = _clickhouse_property_predicate(
            condition,
            parameter_name=f"{prefix}_value",
        )
        if condition.operator != "exists":
            parameters[f"{prefix}_value"] = property_parameter_value(condition)
        matches.append(
            "if("
            "countIf("
            f"event_name = {{{prefix}_event_name:String}} "
            f"AND ({property_predicate})"
            f") >= {{{prefix}_minimum_count:UInt32}}, "
            "toUInt16(1), toUInt16(0)"
            ")"
        )
    return " + ".join(matches), parameters


def _canonical_condition(item: Mapping[str, Any]) -> SegmentPropertyCondition:
    event_name = str(item.get("event_name", "")).strip()
    property_key = str(item.get("property_key", "")).strip()
    operator = str(item.get("operator", "")).strip()
    if event_name not in CUSTOM_STRUCTURED_EVENT_NAMES:
        raise ValueError(f"unsupported_event:{event_name or 'missing'}")
    if property_key not in CUSTOM_STRUCTURED_PROPERTY_KEYS:
        raise ValueError(f"unsupported_property:{property_key or 'missing'}")
    if operator not in CUSTOM_STRUCTURED_PROPERTY_OPERATORS:
        raise ValueError(f"unsupported_operator:{operator or 'missing'}")
    try:
        minimum_count = int(item.get("minimum_count", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_minimum_count") from exc
    if not 1 <= minimum_count <= 100:
        raise ValueError("invalid_minimum_count")
    value = _canonical_value(
        property_key=property_key,
        operator=operator,
        raw_value=item.get("value"),
    )
    return SegmentPropertyCondition(
        event_name=event_name,
        property_key=property_key,
        operator=operator,
        value=value,
        minimum_count=minimum_count,
    )


def _clickhouse_property_predicate(
    condition: SegmentPropertyCondition,
    *,
    parameter_name: str,
) -> str:
    key = condition.property_key
    if key not in CUSTOM_STRUCTURED_PROPERTY_KEYS:
        raise ValueError(f"unsupported structured property key: {key}")
    extracted = f"ifNull(JSONExtractString(properties_json, '{key}'), '')"
    if condition.operator == "exists":
        return f"nullIf({extracted}, '') IS NOT NULL"
    if condition.operator == "contains":
        return (
            f"positionCaseInsensitiveUTF8({extracted}, "
            f"{{{parameter_name}:String}}) > 0"
        )
    if condition.operator == "in":
        return (
            f"has({{{parameter_name}:Array(String)}}, "
            f"lowerUTF8({extracted}))"
        )
    if condition.operator in {"gte", "lte"}:
        comparison = ">=" if condition.operator == "gte" else "<="
        return (
            f"toFloat64OrNull(nullIf({extracted}, '')) {comparison} "
            f"{{{parameter_name}:Float64}}"
        )
    if condition.operator != "equals":
        raise ValueError(
            f"unsupported structured property operator: {condition.operator}"
        )
    if key in {"free_cancellation", "breakfast_included"}:
        return (
            f"toUInt8OrZero({extracted}) = "
            f"toUInt8OrZero({{{parameter_name}:String}})"
        )
    return (
        f"lowerUTF8({extracted}) = "
        f"lowerUTF8({{{parameter_name}:String}})"
    )


def _canonical_value(
    *,
    property_key: str,
    operator: str,
    raw_value: Any,
) -> str:
    if operator == "exists":
        return "true"
    value = " ".join(str(raw_value or "").strip().split()).casefold()
    if not value:
        raise ValueError(f"missing_value:{property_key}")
    if operator in {"gte", "lte"}:
        try:
            decimal_value = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid_numeric_value:{property_key}") from exc
        return format(decimal_value.normalize(), "f")
    if property_key in {"free_cancellation", "breakfast_included"}:
        if value in _TRUE_VALUES:
            return "true"
        if value in _FALSE_VALUES:
            return "false"
        raise ValueError(f"invalid_boolean_value:{property_key}")
    if operator == "in":
        values = _split_values(value)
        if not values:
            raise ValueError(f"missing_value:{property_key}")
        if property_key == "age_group" and _has_twenty_thirty(values):
            values = _AGE_TWENTIES_THIRTIES
        elif property_key == "gender":
            values = _canonical_gender_values(values)
        return ",".join(values)
    return value


def _split_values(value: str) -> tuple[str, ...]:
    normalized = value.replace("，", ",").replace("/", ",").replace("·", ",")
    normalized = normalized.replace("또는", ",").replace("혹은", ",")
    return tuple(
        sorted(
            {
                item.strip().casefold()
                for item in normalized.split(",")
                if item.strip()
            }
        )
    )


def _has_twenty_thirty(values: Sequence[str]) -> bool:
    twenties = any(value in set(_AGE_TWENTIES_THIRTIES[:4]) for value in values)
    thirties = any(value in set(_AGE_TWENTIES_THIRTIES[4:]) for value in values)
    return twenties and thirties


def _canonical_gender_values(values: Sequence[str]) -> tuple[str, ...]:
    normalized = set(values)
    for aliases in _GENDER_VALUES.values():
        if normalized.intersection(aliases):
            return aliases
    return tuple(sorted(normalized))


def _profile_matches_condition(
    profile: Any,
    condition: SegmentPropertyCondition,
) -> bool:
    profile_values = {
        "age_group": getattr(profile, "age_group_values", ()),
        "gender": getattr(profile, "gender_values", ()),
        "preferred_category": getattr(profile, "preferred_category_values", ()),
        "hotel_market": getattr(profile, "hotel_market_values", ()),
        "hotel_cluster": getattr(profile, "hotel_cluster_values", ()),
    }.get(condition.property_key)
    if profile_values is None:
        return False
    values = tuple(str(value).strip().casefold() for value in profile_values if value)
    if condition.operator == "exists":
        return bool(values)
    expected = _split_values(condition.value)
    if condition.operator == "in":
        return any(value in expected for value in values)
    if condition.operator == "equals":
        return any(value == condition.value for value in values)
    if condition.operator == "contains":
        return any(condition.value in value for value in values)
    return False
