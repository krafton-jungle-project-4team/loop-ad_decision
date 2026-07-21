from __future__ import annotations

from dataclasses import replace

from app.analysis.promotion_audience_ast import (
    build_promotion_audience_ast,
    compile_promotion_audience_ast,
    promotion_audience_segment_id,
)
from app.analysis.promotion_audience_beam import (
    PREDICATE_REGISTRY,
    PromotionAudienceBeamPolicy,
    search_promotion_audience_candidates,
)
from app.analysis.repositories import RawEventUserSignalRecord


def _profile(
    user_id: str,
    *,
    detail: int = 0,
    booking_start: int = 0,
    booking_complete: int = 0,
    price: int = 0,
    deal: int = 0,
    destination_match: int = 1,
    season_match: int = 1,
    promotion_condition_search: int | None = None,
    target_destination_search: int | None = None,
    deal_search: int | None = None,
) -> RawEventUserSignalRecord:
    return RawEventUserSignalRecord(
        project_id="demo_project",
        user_id=user_id,
        event_count=max(
            1,
            1 + detail + booking_start + booking_complete + price + deal,
        ),
        hotel_search_count=3,
        hotel_click_count=0,
        hotel_detail_view_count=detail,
        promotion_impression_count=0,
        promotion_click_count=0,
        campaign_redirect_click_count=0,
        campaign_landing_count=0,
        booking_start_count=booking_start,
        booking_complete_count=booking_complete,
        booking_cancel_count=0,
        deal_event_count=deal,
        free_cancellation_count=0,
        breakfast_included_count=0,
        price_event_count=price,
        avg_price=0.0,
        destination_values=("jeju", "okinawa"),
        checkin_dates=("2026-07-10",),
        hotel_market_values=(),
        hotel_cluster_values=(),
        age_group_values=(),
        gender_values=(),
        preferred_category_values=(),
        destination_match_count=destination_match,
        season_match_count=season_match,
        promotion_condition_search_count=promotion_condition_search,
        target_destination_search_count=target_destination_search,
        deal_search_count=deal_search,
    )


def _profiles() -> list[RawEventUserSignalRecord]:
    return [
        _profile(
            f"user-{index:02d}",
            detail=2 if index < 7 else 0,
            booking_start=1 if 2 <= index < 8 else 0,
            price=2 if index in {0, 2, 3, 4, 8} else 0,
            deal=1 if index in {1, 2, 3, 5, 9} else 0,
        )
        for index in range(12)
    ]


def _search(
    profiles: list[RawEventUserSignalRecord],
    *,
    min_sample_size: int = 2,
    policy: PromotionAudienceBeamPolicy = PromotionAudienceBeamPolicy(),
):
    return search_promotion_audience_candidates(
        promotion_id="promo-jeju-okinawa",
        destination_ids=("okinawa", "jeju"),
        season_months=(8, 6, 7),
        benefit_keys=(),
        desired_behavior_keys=(
            "booking_start_without_complete",
            "hotel_detail_view",
            "price_compare",
        ),
        profiles=profiles,
        min_sample_size=min_sample_size,
        policy=policy,
    )


def test_predicate_registry_is_executable_and_explicit() -> None:
    assert PREDICATE_REGISTRY
    assert len({value.predicate_key for value in PREDICATE_REGISTRY}) == len(
        PREDICATE_REGISTRY
    )
    for predicate in PREDICATE_REGISTRY:
        assert predicate.dimension
        assert predicate.event_name
        assert predicate.minimum_count_options
        assert predicate.compiler_target == "custom_structured_condition.v1"
        assert predicate.display_label


def test_beam_generates_compound_candidate_and_keeps_mandatory_conditions() -> None:
    result = _search(_profiles())

    assert result.candidates
    assert any(candidate.depth >= 2 for candidate in result.candidates)
    assert all(candidate.strategy_key.startswith("beam_") for candidate in result.candidates)
    for candidate in result.candidates:
        ast = build_promotion_audience_ast(
            promotion_id="promo-jeju-okinawa",
            candidate_type=candidate.candidate_type,
            strategy_key=candidate.strategy_key,
            matched_condition_keys=tuple(
                choice.predicate_key for choice in candidate.choices
            ),
            destination_ids=("jeju", "okinawa"),
            season_months=(6, 7, 8),
            structured_conditions=candidate.structured_conditions,
            beam_policy_version=result.policy.policy_version,
        )
        compiled = compile_promotion_audience_ast(ast)
        spec = compiled.segment_audience_spec
        assert spec["template_id"] == "custom_structured_condition"
        assert spec["template_version"] == 1
        assert all(
            condition["destination"] == "jeju,okinawa"
            and condition["checkin_months"] == [6, 7, 8]
            for condition in spec["parameters"]["conditions"]
            if "숙소 검색" in condition["label"]
            and not condition["property_filters"]
        )


def test_beam_prunes_small_and_duplicate_member_sets() -> None:
    result = _search(_profiles())
    too_small = _search(_profiles()[:2], min_sample_size=3)

    assert result.pruned_candidate_counts.get("duplicate_members", 0) > 0
    assert too_small.candidates == ()
    assert too_small.pruned_candidate_counts["minimum_sample"] > 0


def test_beam_limits_width_depth_and_generated_candidates() -> None:
    policy = PromotionAudienceBeamPolicy(
        beam_width=2,
        maximum_depth=2,
        maximum_generated_candidates=7,
        maximum_final_candidates=2,
    )
    result = _search(_profiles(), policy=policy)

    assert result.generated_candidate_count <= 7
    assert len(result.candidates) <= 2
    assert all(candidate.depth <= 2 for candidate in result.candidates)


def test_beam_is_independent_of_profile_input_order() -> None:
    forward = _search(_profiles())
    reverse = _search(list(reversed(_profiles())))

    assert [
        (candidate.strategy_key, candidate.user_ids)
        for candidate in forward.candidates
    ] == [
        (candidate.strategy_key, candidate.user_ids)
        for candidate in reverse.candidates
    ]


def test_beam_uses_same_event_promotion_and_benefit_conditions_as_compiler() -> None:
    profiles = [
        _profile(
            "coarse-only",
            detail=2,
            deal=1,
            promotion_condition_search=0,
            deal_search=0,
        ),
        _profile(
            "executable-match",
            detail=2,
            deal=1,
            promotion_condition_search=1,
            deal_search=1,
        ),
    ]

    result = search_promotion_audience_candidates(
        promotion_id="promo-jeju-okinawa",
        destination_ids=("jeju", "okinawa"),
        season_months=(6, 7, 8),
        benefit_keys=("discount",),
        desired_behavior_keys=("hotel_detail_view",),
        profiles=profiles,
        min_sample_size=1,
    )

    assert result.candidates
    assert all(
        candidate.user_ids == ("executable-match",)
        for candidate in result.candidates
    )


def test_beam_uses_destination_search_count_for_repeat_predicate() -> None:
    result = search_promotion_audience_candidates(
        promotion_id="promo-jeju-okinawa",
        destination_ids=("jeju", "okinawa"),
        season_months=(6, 7, 8),
        benefit_keys=(),
        desired_behavior_keys=("destination_repeat_search",),
        profiles=[
            _profile(
                "coarse-repeat-only",
                destination_match=3,
                promotion_condition_search=1,
                target_destination_search=1,
            ),
            _profile(
                "executable-repeat",
                destination_match=3,
                promotion_condition_search=1,
                target_destination_search=2,
            ),
        ],
        min_sample_size=1,
        policy=PromotionAudienceBeamPolicy(maximum_final_candidates=10),
    )

    repeat_candidate = next(
        candidate
        for candidate in result.candidates
        if any(
            choice.predicate_key == "destination_repeat_search"
            for choice in candidate.choices
        )
    )
    assert repeat_candidate.user_ids == ("executable-repeat",)


def test_beam_policy_version_changes_identity_without_public_v3() -> None:
    candidate = _search(_profiles()).candidates[0]
    ast = build_promotion_audience_ast(
        promotion_id="promo-jeju-okinawa",
        candidate_type=candidate.candidate_type,
        strategy_key=candidate.strategy_key,
        matched_condition_keys=tuple(
            choice.predicate_key for choice in candidate.choices
        ),
        destination_ids=("jeju", "okinawa"),
        season_months=(6, 7, 8),
        structured_conditions=candidate.structured_conditions,
        beam_policy_version="promotion-audience-beam.v1",
    )
    changed_policy = replace(ast, beam_policy_version="promotion-audience-beam.v2")

    assert promotion_audience_segment_id(ast) != promotion_audience_segment_id(
        changed_policy
    )
    assert compile_promotion_audience_ast(ast).segment_audience_spec[
        "template_version"
    ] == 1


def test_display_label_does_not_change_identity_or_execution_spec() -> None:
    candidate = _search(_profiles()).candidates[0]
    ast = build_promotion_audience_ast(
        promotion_id="promo-jeju-okinawa",
        candidate_type=candidate.candidate_type,
        strategy_key=candidate.strategy_key,
        matched_condition_keys=tuple(
            choice.predicate_key for choice in candidate.choices
        ),
        destination_ids=("jeju", "okinawa"),
        season_months=(6, 7, 8),
        structured_conditions=candidate.structured_conditions,
        beam_policy_version="promotion-audience-beam.v1",
    )
    relabeled = replace(
        ast,
        structured_conditions=tuple(
            {**condition, "label": f"표시 문구 {index}"}
            for index, condition in enumerate(ast.structured_conditions)
        ),
    )

    original = compile_promotion_audience_ast(ast)
    changed = compile_promotion_audience_ast(relabeled)
    assert original.segment_id == changed.segment_id
    assert original.segment_audience_spec_hash == changed.segment_audience_spec_hash
