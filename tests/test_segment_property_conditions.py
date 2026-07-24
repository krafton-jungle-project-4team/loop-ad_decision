from __future__ import annotations

from app.audience_contract import CUSTOM_STRUCTURED_PROPERTY_KEYS
from app.analysis.segment_property_conditions import (
    SegmentPropertyCondition,
    canonical_segment_property_conditions,
    clickhouse_segment_property_match_expression,
    segment_property_conditions_from_hints,
    segment_property_filter_label,
    structured_conditions_from_segment_properties,
)


def test_all_registered_property_keys_are_available_to_segment_conditions() -> None:
    numeric_keys = {
        "adult_count",
        "child_count",
        "rooms",
        "hotel_star_rating",
        "hotel_guest_rating",
        "price",
        "revenue",
    }
    boolean_keys = {"deal", "free_cancellation", "breakfast_included"}

    for property_key in sorted(CUSTOM_STRUCTURED_PROPERTY_KEYS):
        operator = "gte" if property_key in numeric_keys else "equals"
        value = "1" if property_key in numeric_keys else (
            "true" if property_key in boolean_keys else "sample"
        )
        conditions, unsupported = canonical_segment_property_conditions(
            [
                {
                    "event_name": "hotel_search",
                    "property_key": property_key,
                    "operator": operator,
                    "value": value,
                    "minimum_count": 1,
                }
            ]
        )

        assert unsupported == (), property_key
        assert conditions[0].property_key == property_key
        assert segment_property_filter_label(
            conditions[0].to_structured_condition()["property_filters"][0]
        )


def test_canonicalizes_allowlisted_property_conditions_by_type() -> None:
    conditions, unsupported = canonical_segment_property_conditions(
        [
            {
                "event_name": "hotel_search",
                "property_key": "region",
                "operator": "equals",
                "value": "  Seoul  ",
                "minimum_count": 1,
            },
            {
                "event_name": "hotel_detail_view",
                "property_key": "hotel_star_rating",
                "operator": "gte",
                "value": "4.0",
                "minimum_count": 2,
            },
            {
                "event_name": "hotel_search",
                "property_key": "room_type",
                "operator": "in",
                "value": "Suite, Deluxe, suite",
                "minimum_count": 1,
            },
        ]
    )

    assert unsupported == ()
    assert [condition.to_json() for condition in conditions] == [
        {
            "event_name": "hotel_detail_view",
            "property_key": "hotel_star_rating",
            "operator": "gte",
            "value": "4",
            "minimum_count": 2,
        },
        {
            "event_name": "hotel_search",
            "property_key": "region",
            "operator": "equals",
            "value": "seoul",
            "minimum_count": 1,
        },
        {
            "event_name": "hotel_search",
            "property_key": "room_type",
            "operator": "in",
            "value": "deluxe,suite",
            "minimum_count": 1,
        },
    ]


def test_rejects_unregistered_property_and_operator() -> None:
    conditions, unsupported = canonical_segment_property_conditions(
        [
            {
                "event_name": "hotel_search",
                "property_key": "email",
                "operator": "equals",
                "value": "person@example.com",
            },
            {
                "event_name": "hotel_search",
                "property_key": "region",
                "operator": "regex",
                "value": "seoul.*",
            },
        ]
    )

    assert conditions == ()
    assert unsupported == (
        "property_condition:unsupported_property:email",
        "property_condition:unsupported_operator:regex",
    )


def test_demographic_hints_compile_to_existing_structured_contract() -> None:
    conditions = segment_property_conditions_from_hints(
        ("20s_30s", "female")
    )

    assert [condition.property_key for condition in conditions] == [
        "age_group",
        "gender",
    ]
    assert [condition.event_name for condition in conditions] == [
        "page_view",
        "page_view",
    ]
    assert [
        item["property_filters"][0]["key"]
        for item in structured_conditions_from_segment_properties(conditions)
    ] == ["age_group", "gender"]
    assert segment_property_filter_label(
        conditions[0].to_structured_condition()["property_filters"][0]
    ) == "20·30대"
    assert segment_property_filter_label(
        conditions[1].to_structured_condition()["property_filters"][0]
    ) == "여성"


def test_price_filter_label_preserves_strict_threshold_semantics() -> None:
    assert segment_property_filter_label(
        {"key": "price", "operator": "gte", "value": "200001"}
    ) == "가격 20만원 초과"
    assert segment_property_filter_label(
        {"key": "revenue", "operator": "gte", "value": "200000"}
    ) == "숙박 총액 20만원 이상"


def test_clickhouse_match_expression_parameterizes_values() -> None:
    expression, parameters = clickhouse_segment_property_match_expression(
        (
            SegmentPropertyCondition(
                event_name="hotel_search",
                property_key="user_segment",
                operator="contains",
                value="vip",
                minimum_count=2,
            ),
        )
    )

    assert "JSONExtractString(properties_json, 'user_segment')" in expression
    assert "vip" not in expression
    assert parameters == {
        "segment_property_0_event_name": "hotel_search",
        "segment_property_0_minimum_count": 2,
        "segment_property_0_value": "vip",
    }
