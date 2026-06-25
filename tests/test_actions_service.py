import pytest

from app.actions.catalog import ACTION_CATALOG
from app.actions.schemas import ActionRecommendationRequest, CauseCandidate
from app.actions.service import calculate_priority_score, recommend_actions


def action_request(
    causes: list[dict[str, object]],
    *,
    top_n: int = 10,
    segment: dict[str, str | None] | None = None,
) -> ActionRecommendationRequest:
    return ActionRecommendationRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
        segment=segment or {},
        causes=causes,
        top_n=top_n,
    )


def cause_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "cause_id": "cause-1",
        "cause_type": "LOW_VIEW_TO_CART_RATE",
        "label": "상품 상세 전환 저하",
        "description": "상품 상세에서 장바구니 전환이 낮습니다.",
        "affected_step": "product_view_to_add_to_cart",
        "severity": 0.8,
        "confidence": 0.7,
        "evidence": [],
        "attributes": {},
    }
    values.update(overrides)
    return values


def root_cause_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "rank": 1,
        "cause_type": "channel_specific_drop",
        "dimension": "channel",
        "value": "kakao",
        "metric": "view_to_purchase_rate",
        "funnel_step": "product_view_to_purchase",
        "severity": "critical",
        "current_value": 0.10,
        "baseline_value": 0.30,
        "drop_point": 0.20,
        "relative_drop": 0.67,
        "current_denominator": 100,
        "baseline_denominator": 120,
        "support_share": 0.40,
        "excess_lost_sessions": 20.0,
        "score": 2.4,
        "message": "channel=kakao 전환율이 하락했습니다.",
    }
    values.update(overrides)
    return values


def recommendation_ids(request: ActionRecommendationRequest) -> list[str]:
    return [action.action_id for action in recommend_actions(request).recommendations]


def test_low_view_to_cart_rate_recommends_product_detail_actions() -> None:
    request = action_request([cause_payload()])

    ids = recommendation_ids(request)

    assert "emphasize_reviews" in ids
    assert "show_price_benefit" in ids
    assert "improve_product_detail" in ids


def test_out_of_stock_recommends_inventory_actions() -> None:
    request = action_request(
        [
            cause_payload(
                cause_id="stock-1",
                cause_type="OUT_OF_STOCK",
                affected_step=None,
                description="inventory status changed",
                attributes={"inventory_status": "out_of_stock"},
            )
        ]
    )

    ids = recommendation_ids(request)

    assert "recommend_alternative_product" in ids
    assert "restock_notification" in ids
    assert "pause_out_of_stock_ads" in ids


def test_high_checkout_dropoff_recommends_checkout_actions() -> None:
    request = action_request(
        [
            cause_payload(
                cause_id="checkout-1",
                cause_type="HIGH_CHECKOUT_DROPOFF",
                affected_step="checkout_start_to_purchase",
                attributes={"metric": "checkout_to_purchase_rate"},
            )
        ]
    )

    ids = recommendation_ids(request)

    assert "free_shipping_coupon" in ids
    assert "checkout_reminder_message" in ids


def test_priority_score_increases_with_severity_and_confidence() -> None:
    low_cause = CauseCandidate(
        cause_id="low",
        cause_type="LOW_VIEW_TO_CART_RATE",
        severity=0.2,
        confidence=0.2,
    )
    high_cause = CauseCandidate(
        cause_id="high",
        cause_type="LOW_VIEW_TO_CART_RATE",
        severity=0.9,
        confidence=0.9,
    )

    low_score = calculate_priority_score(low_cause, ACTION_CATALOG["emphasize_reviews"])
    high_score = calculate_priority_score(high_cause, ACTION_CATALOG["emphasize_reviews"])

    assert high_score > low_score


def test_duplicate_action_merges_triggered_by_without_duplicate_recommendations() -> None:
    request = action_request(
        [
            cause_payload(cause_id="cause-a"),
            cause_payload(
                cause_id="cause-b",
                severity=0.9,
                confidence=0.9,
                description="다른 세그먼트에서 장바구니 전환이 낮습니다.",
            ),
        ]
    )

    recommendations = recommend_actions(request).recommendations
    emphasize_reviews = [
        action for action in recommendations if action.action_id == "emphasize_reviews"
    ]

    assert len(emphasize_reviews) == 1
    assert emphasize_reviews[0].triggered_by == ["cause-a", "cause-b"]


def test_unknown_cause_type_returns_manual_review_fallback() -> None:
    request = action_request(
        [
            cause_payload(
                cause_id="unknown-1",
                cause_type="UNKNOWN_CAUSE",
                affected_step=None,
                description="known catalog terms are absent",
            )
        ]
    )

    ids = recommendation_ids(request)

    assert ids == ["manual_review"]


def test_top_n_limits_recommendations() -> None:
    request = action_request(
        [
            cause_payload(),
            cause_payload(
                cause_id="checkout-1",
                cause_type="HIGH_CHECKOUT_DROPOFF",
                affected_step="checkout_start_to_purchase",
            ),
        ],
        top_n=2,
    )

    response = recommend_actions(request)

    assert len(response.recommendations) == 2


def test_root_cause_candidate_payload_is_normalized() -> None:
    request = action_request([root_cause_payload()])

    cause = request.causes[0]

    assert cause.cause_id == "channel_specific_drop:channel:kakao:view_to_purchase_rate"
    assert cause.affected_step == "product_view_to_purchase"
    assert cause.description == "channel=kakao 전환율이 하락했습니다."
    assert cause.attributes["dimension"] == "channel"
    assert cause.attributes["value"] == "kakao"
    assert cause.attributes["metric"] == "view_to_purchase_rate"
    assert cause.attributes["funnel_step"] == "product_view_to_purchase"
    assert cause.attributes["support_share"] == 0.40
    assert cause.attributes["excess_lost_sessions"] == 20.0
    assert cause.evidence[0].metric_name == "view_to_purchase_rate"
    assert cause.evidence[0].delta == 0.20
    assert cause.evidence[0].relative_drop == 0.67
    assert cause.evidence[0].note == "channel=kakao 전환율이 하락했습니다."
    assert cause.severity == 1.0
    assert cause.confidence == pytest.approx(0.8)


def test_existing_root_cause_types_match_by_structured_fields() -> None:
    request = action_request(
        [
            root_cause_payload(
                cause_type="inventory_issue",
                dimension="inventory_status",
                value="out_of_stock",
                metric="view_to_purchase_rate",
                funnel_step="product_view_to_purchase",
                message="structured inventory candidate",
            ),
            root_cause_payload(
                cause_type="channel_specific_drop",
                dimension="channel",
                value="kakao",
                metric="view_to_purchase_rate",
                funnel_step="product_view_to_purchase",
                message="structured channel candidate",
            ),
            root_cause_payload(
                cause_type="product_specific_drop",
                dimension="product_id",
                value="sku-1",
                metric="view_to_cart_rate",
                funnel_step="product_view_to_add_to_cart",
                message="structured product candidate",
            ),
        ]
    )

    ids = recommendation_ids(request)

    assert "recommend_alternative_product" in ids
    assert "adjust_landing_page" in ids
    assert "emphasize_reviews" in ids


@pytest.mark.parametrize(
    ("cause_type", "dimension", "value"),
    [
        ("product_specific_drop", "product_id", "sku-1"),
        ("category_specific_drop", "category", "fresh_food"),
        ("customer_segment_drop", "age_group", "30s"),
    ],
)
def test_product_segment_view_to_purchase_does_not_fallback_to_manual_review(
    cause_type: str,
    dimension: str,
    value: str,
) -> None:
    request = action_request(
        [
            root_cause_payload(
                cause_type=cause_type,
                dimension=dimension,
                value=value,
                metric="view_to_purchase_rate",
                funnel_step="product_view_to_purchase",
                message="structured root cause candidate",
            )
        ]
    )

    ids = recommendation_ids(request)

    assert "manual_review" not in ids
    assert "adjust_landing_page" in ids
    assert "show_price_benefit" in ids
    assert "improve_product_detail" in ids
