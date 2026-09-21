from __future__ import annotations

from dataclasses import replace

from app.analysis.promotion_audience_ast import (
    DESTINATION_CONTAINS_ALL,
    build_promotion_audience_ast,
    compile_promotion_audience_ast,
    promotion_audience_segment_id,
)


def test_any_of_compiles_to_registered_destination_ids() -> None:
    ast = build_promotion_audience_ast(
        promotion_id="promo_black_friday",
        candidate_type="funnel_recovery",
        matched_condition_keys=(
            "hotel_detail_view",
            "booking_start_without_complete",
        ),
        destination_ids=("오키나와", "jeju", "제주"),
        unsupported_conditions=("후기 기반 추천",),
    )

    compiled = compile_promotion_audience_ast(ast)

    assert compiled.segment_audience_spec["template_id"] == "hotel.funnel_recovery.v1"
    assert compiled.segment_audience_spec["template_version"] == 1
    assert compiled.segment_audience_spec["parameters"]["destination_ids"] == [
        "jeju",
        "okinawa",
    ]
    assert ast.behavior_condition_keys == (
        "booking_start_without_complete",
        "recent_destination_search",
    )
    assert ast.creative_only == ("후기 기반 추천",)
    assert compiled.display_model["title"] == "제주·오키나와 예약 직전 이탈 고객"
    assert "호텔 상세 조회" not in compiled.display_model["signal_chips"]
    assert compiled.display_model["metric_label"] == "행동 기반 예상 예약 전환율"


def test_contains_all_compiles_to_custom_v1_and_conditions() -> None:
    ast = build_promotion_audience_ast(
        promotion_id="promo_compare",
        candidate_type="general_destination_explorer",
        strategy_key="destination_comparison",
        matched_condition_keys=("general_destination_exploration",),
        destination_ids=("okinawa", "jeju"),
        destination_operator=DESTINATION_CONTAINS_ALL,
    )

    compiled = compile_promotion_audience_ast(ast)

    spec = compiled.segment_audience_spec
    assert spec["template_id"] == "custom_structured_condition"
    assert spec["template_version"] == 1
    assert [
        condition["destination"]
        for condition in spec["parameters"]["conditions"]
    ] == ["jeju", "okinawa"]
    assert all(
        condition["event_name"] == "hotel_search"
        for condition in spec["parameters"]["conditions"]
    )
    assert compiled.display_model["title"] == "제주·오키나와를 비교 탐색한 고객"


def test_identity_ignores_order_reference_signals_and_current_members() -> None:
    first = build_promotion_audience_ast(
        promotion_id="promo_same",
        candidate_type="target_destination_affinity",
        matched_condition_keys=("hotel_detail_view", "recent_destination_search"),
        destination_ids=("okinawa", "jeju"),
    )
    second = build_promotion_audience_ast(
        promotion_id="promo_same",
        candidate_type="target_destination_affinity",
        matched_condition_keys=("recent_destination_search",),
        destination_ids=("jeju", "okinawa", "jeju"),
    )

    first_compiled = compile_promotion_audience_ast(first)
    second_compiled = compile_promotion_audience_ast(second)

    assert first_compiled.segment_id == second_compiled.segment_id
    assert first_compiled.ast_hash == second_compiled.ast_hash
    assert (
        first_compiled.segment_audience_spec_hash
        == second_compiled.segment_audience_spec_hash
    )


def test_identity_changes_with_window_or_destination_operator() -> None:
    any_of = build_promotion_audience_ast(
        promotion_id="promo_window",
        candidate_type="general_destination_explorer",
        strategy_key="destination_comparison",
        matched_condition_keys=("general_destination_exploration",),
        destination_ids=("jeju", "okinawa"),
    )
    contains_all = build_promotion_audience_ast(
        promotion_id="promo_window",
        candidate_type="general_destination_explorer",
        strategy_key="destination_comparison",
        matched_condition_keys=("general_destination_exploration",),
        destination_ids=("jeju", "okinawa"),
        destination_operator=DESTINATION_CONTAINS_ALL,
    )
    seven_days = replace(any_of, lookback_days=7)

    assert promotion_audience_segment_id(any_of) != promotion_audience_segment_id(
        contains_all
    )
    assert promotion_audience_segment_id(any_of) != promotion_audience_segment_id(
        seven_days
    )


def test_beam_display_uses_only_compiled_execution_conditions() -> None:
    ast = build_promotion_audience_ast(
        promotion_id="promo_black_friday",
        candidate_type="benefit_value_seeker",
        strategy_key="beam_discount_interest_1",
        matched_condition_keys=("discount_interest",),
        destination_ids=("okinawa", "jeju"),
        season_months=(8, 6, 7),
        benefit_keys=("discount", "early_booking"),
        structured_conditions=(
            {
                "label": "jeju·okinawa 시즌 일치 숙소 검색",
                "event_name": "hotel_search",
                "minimum_count": 1,
                "maximum_count": None,
                "destination": "jeju,okinawa",
                "checkin_months": [6, 7, 8],
                "property_filters": [],
            },
            {
                "label": "할인·특가 관심",
                "event_name": "hotel_search",
                "minimum_count": 1,
                "maximum_count": None,
                "destination": None,
                "checkin_months": [],
                "property_filters": [
                    {"key": "deal", "operator": "equals", "value": "true"}
                ],
            },
        ),
        beam_policy_version="promotion-audience-beam.v2",
    )

    compiled = compile_promotion_audience_ast(ast)

    assert compiled.display_model["title"] == "제주·오키나와 할인·특가 관심 고객"
    assert compiled.display_model["signal_chips"] == [
        "제주·오키나와 체크인 6·7·8월 숙소 검색",
        "할인·특가 관심",
    ]
    assert "할인" not in compiled.display_model["signal_chips"]
    assert "조기 예약" not in compiled.display_model["signal_chips"]


def test_beam_display_keeps_title_compact_with_four_execution_conditions() -> None:
    ast = build_promotion_audience_ast(
        promotion_id="promo_black_friday",
        candidate_type="funnel_recovery",
        strategy_key="beam_booking_start_without_complete_1",
        matched_condition_keys=(
            "booking_start_without_complete",
            "hotel_detail_view",
        ),
        destination_ids=("jeju", "okinawa"),
        season_months=(6, 7, 8),
        structured_conditions=(
            {
                "label": "시즌 일치 숙소 검색",
                "event_name": "hotel_search",
                "minimum_count": 1,
                "maximum_count": None,
                "destination": "jeju,okinawa",
                "checkin_months": [6, 7, 8],
                "property_filters": [],
            },
            {
                "label": "예약 시작",
                "event_name": "booking_start",
                "minimum_count": 1,
                "maximum_count": None,
                "destination": None,
                "checkin_months": [],
                "property_filters": [],
            },
            {
                "label": "예약 미완료",
                "event_name": "booking_complete",
                "minimum_count": 0,
                "maximum_count": 0,
                "destination": None,
                "checkin_months": [],
                "property_filters": [],
            },
            {
                "label": "호텔 상세 조회",
                "event_name": "hotel_detail_view",
                "minimum_count": 1,
                "maximum_count": None,
                "destination": None,
                "checkin_months": [],
                "property_filters": [],
            },
        ),
        beam_policy_version="promotion-audience-beam.v2",
    )

    compiled = compile_promotion_audience_ast(ast)

    assert compiled.display_model["signal_chips"] == [
        "제주·오키나와 체크인 6·7·8월 숙소 검색",
        "예약 시작 후 미완료",
        "호텔 상세 조회",
    ]
    assert compiled.display_model["title"] == "제주·오키나와 예약 이탈 고객"
