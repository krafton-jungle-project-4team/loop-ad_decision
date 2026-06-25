from app.actions.catalog import (
    ACTION_CATALOG,
    CART_TO_CHECKOUT_ACTION_IDS,
    CHANNEL_CONVERSION_ACTION_IDS,
    CHECKOUT_TO_PURCHASE_ACTION_IDS,
    HIGH_PURCHASE_INTENT_ACTION_IDS,
    OUT_OF_STOCK_ACTION_IDS,
    VIEW_TO_CART_ACTION_IDS,
    ActionCatalogItem,
)
from app.actions.schemas import (
    ActionExperiment,
    ActionRecommendationRequest,
    ActionRecommendationResponse,
    CauseCandidate,
    RecommendedAction,
)

VIEW_TO_CART_CAUSE_TYPES = {"LOW_VIEW_TO_CART_RATE"}
CART_TO_CHECKOUT_CAUSE_TYPES = {"LOW_CART_TO_CHECKOUT_RATE"}
CHECKOUT_CAUSE_TYPES = {"HIGH_CHECKOUT_DROPOFF", "LOW_CHECKOUT_TO_PURCHASE_RATE"}
OUT_OF_STOCK_CAUSE_TYPES = {"OUT_OF_STOCK"}
CHANNEL_CAUSE_TYPES = {
    "LOW_CHANNEL_CONVERSION",
    "campaign_specific_drop",
    "channel_specific_drop",
}
PRODUCT_SEGMENT_CAUSE_TYPES = {
    "product_specific_drop",
    "category_specific_drop",
    "customer_segment_drop",
    "device_specific_drop",
}
HIGH_PURCHASE_INTENT_CAUSE_TYPES = {"HIGH_PURCHASE_INTENT", "COUPON_WASTE_RISK"}

VIEW_TO_CART_METRICS = {"view_to_cart_rate"}
CART_TO_CHECKOUT_METRICS = {"cart_to_checkout_rate"}
CHECKOUT_TO_PURCHASE_METRICS = {"checkout_to_purchase_rate"}
VIEW_TO_PURCHASE_METRICS = {"view_to_purchase_rate"}

VIEW_TO_CART_STEPS = {"product_view_to_add_to_cart"}
CART_TO_CHECKOUT_STEPS = {"add_to_cart_to_checkout_start"}
CHECKOUT_TO_PURCHASE_STEPS = {"checkout_start_to_purchase"}
VIEW_TO_PURCHASE_STEPS = {"product_view_to_purchase"}

CHANNEL_DIMENSIONS = {"channel", "campaign_id"}
PRODUCT_SEGMENT_DIMENSIONS = {"product_id", "category", "age_group", "gender", "device"}
OUT_OF_STOCK_VALUES = {"out_of_stock", "품절"}
VIEW_TO_PURCHASE_ACTION_IDS = [
    "adjust_landing_page",
    "show_price_benefit",
    "improve_product_detail",
]


def recommend_actions(request: ActionRecommendationRequest) -> ActionRecommendationResponse:
    recommendations_by_id: dict[str, RecommendedAction] = {}

    for cause in request.causes:
        matched_actions = match_actions_for_cause(cause, request.segment)
        if not matched_actions:
            matched_actions = [ACTION_CATALOG["manual_review"]]

        for action in matched_actions:
            recommendation = build_recommendation(action, cause, request.segment)
            existing = recommendations_by_id.get(action.action_id)
            if existing is None:
                recommendations_by_id[action.action_id] = recommendation
                continue

            merged_triggers = merge_triggered_by(existing.triggered_by, cause.cause_id)
            if recommendation.priority_score > existing.priority_score:
                recommendations_by_id[action.action_id] = recommendation.model_copy(
                    update={"triggered_by": merged_triggers}
                )
            else:
                recommendations_by_id[action.action_id] = existing.model_copy(
                    update={"triggered_by": merged_triggers}
                )

    sorted_recommendations = sorted(
        recommendations_by_id.values(),
        key=lambda recommendation: (-recommendation.priority_score, recommendation.action_id),
    )

    return ActionRecommendationResponse(
        project_id=request.project_id,
        window_start=request.window_start,
        window_end=request.window_end,
        segment=request.segment,
        recommendations=sorted_recommendations[: request.top_n],
    )


def match_actions_for_cause(
    cause: CauseCandidate,
    segment: dict[str, str | None],
) -> list[ActionCatalogItem]:
    action_ids: list[str] = []

    if is_out_of_stock_cause(cause):
        action_ids.extend(OUT_OF_STOCK_ACTION_IDS)
    if is_channel_conversion_cause(cause, segment):
        action_ids.extend(CHANNEL_CONVERSION_ACTION_IDS)
    if is_high_purchase_intent_cause(cause):
        action_ids.extend(HIGH_PURCHASE_INTENT_ACTION_IDS)
    if matches_view_to_cart(cause):
        action_ids.extend(VIEW_TO_CART_ACTION_IDS)
    if matches_cart_to_checkout(cause):
        action_ids.extend(CART_TO_CHECKOUT_ACTION_IDS)
    if matches_checkout_to_purchase(cause):
        action_ids.extend(CHECKOUT_TO_PURCHASE_ACTION_IDS)
    if matches_product_segment_view_to_purchase(cause):
        action_ids.extend(VIEW_TO_PURCHASE_ACTION_IDS)

    if not action_ids:
        action_ids.extend(match_actions_by_text(cause))

    return [ACTION_CATALOG[action_id] for action_id in dedupe(action_ids)]


def is_out_of_stock_cause(cause: CauseCandidate) -> bool:
    if cause.cause_type in OUT_OF_STOCK_CAUSE_TYPES:
        return True
    if cause.cause_type == "inventory_issue":
        dimension = normalized_attribute(cause, "dimension")
        value = normalized_attribute(cause, "value")
        inventory_status = normalized_attribute(cause, "inventory_status")
        return (
            dimension == "inventory_status"
            and value in OUT_OF_STOCK_VALUES
        ) or inventory_status in OUT_OF_STOCK_VALUES
    return False


def is_channel_conversion_cause(
    cause: CauseCandidate,
    segment: dict[str, str | None],
) -> bool:
    dimension = normalized_attribute(cause, "dimension")
    has_channel_segment = bool(segment.get("channel"))
    has_channel_attribute = bool(cause.attributes.get("channel"))

    return (
        cause.cause_type in CHANNEL_CAUSE_TYPES
        or dimension in CHANNEL_DIMENSIONS
        or (
            cause.cause_type == "LOW_CHANNEL_CONVERSION"
            and (has_channel_segment or has_channel_attribute)
        )
    )


def is_high_purchase_intent_cause(cause: CauseCandidate) -> bool:
    return cause.cause_type in HIGH_PURCHASE_INTENT_CAUSE_TYPES


def matches_view_to_cart(cause: CauseCandidate) -> bool:
    return (
        cause.cause_type in VIEW_TO_CART_CAUSE_TYPES
        or normalized_attribute(cause, "metric") in VIEW_TO_CART_METRICS
        or normalized_affected_step(cause) in VIEW_TO_CART_STEPS
    )


def matches_cart_to_checkout(cause: CauseCandidate) -> bool:
    return (
        cause.cause_type in CART_TO_CHECKOUT_CAUSE_TYPES
        or normalized_attribute(cause, "metric") in CART_TO_CHECKOUT_METRICS
        or normalized_affected_step(cause) in CART_TO_CHECKOUT_STEPS
    )


def matches_checkout_to_purchase(cause: CauseCandidate) -> bool:
    return (
        cause.cause_type in CHECKOUT_CAUSE_TYPES
        or normalized_attribute(cause, "metric") in CHECKOUT_TO_PURCHASE_METRICS
        or normalized_affected_step(cause) in CHECKOUT_TO_PURCHASE_STEPS
    )


def matches_product_segment_view_to_purchase(cause: CauseCandidate) -> bool:
    dimension = normalized_attribute(cause, "dimension")
    matches_purchase_path = (
        normalized_attribute(cause, "metric") in VIEW_TO_PURCHASE_METRICS
        or normalized_affected_step(cause) in VIEW_TO_PURCHASE_STEPS
    )
    matches_product_segment = (
        cause.cause_type in PRODUCT_SEGMENT_CAUSE_TYPES
        or dimension in PRODUCT_SEGMENT_DIMENSIONS
    )
    return matches_purchase_path and matches_product_segment


def match_actions_by_text(cause: CauseCandidate) -> list[str]:
    text = " ".join(
        part.lower()
        for part in [
            cause.cause_id,
            cause.cause_type,
            cause.label,
            cause.description,
            cause.affected_step,
        ]
        if part
    )

    if contains_any(text, ["view_to_cart", "add_to_cart", "상품상세", "장바구니"]):
        return VIEW_TO_CART_ACTION_IDS
    if contains_any(text, ["cart_to_checkout", "checkout_start", "결제 시작"]):
        return CART_TO_CHECKOUT_ACTION_IDS
    if contains_any(text, ["checkout_to_purchase", "checkout_start_to_purchase", "결제 직전"]):
        return CHECKOUT_TO_PURCHASE_ACTION_IDS
    if contains_any(text, ["out_of_stock", "품절", "재고"]):
        return OUT_OF_STOCK_ACTION_IDS
    if contains_any(text, ["channel", "campaign", "채널", "캠페인"]):
        return CHANNEL_CONVERSION_ACTION_IDS
    return []


def build_recommendation(
    action: ActionCatalogItem,
    cause: CauseCandidate,
    segment: dict[str, str | None],
) -> RecommendedAction:
    return RecommendedAction(
        action_id=action.action_id,
        action_type=action.action_type,
        title=action.title,
        description=action.description,
        target_step=action.target_step or cause.affected_step,
        priority_score=calculate_priority_score(cause, action),
        expected_impact=action.expected_impact,
        rationale=build_rationale(action, cause, segment),
        triggered_by=[cause.cause_id],
        execution_hint=build_execution_hint(action, cause, segment),
        experiment=build_experiment(action),
    )


def calculate_priority_score(cause: CauseCandidate, action: ActionCatalogItem) -> float:
    score = 0.45 * cause.severity + 0.35 * cause.confidence + 0.20 * action.base_weight
    return round(clamp(score), 4)


def build_rationale(
    action: ActionCatalogItem,
    cause: CauseCandidate,
    segment: dict[str, str | None],
) -> str:
    segment_label = format_segment(segment)
    segment_prefix = f"{segment_label}에서 " if segment_label else ""
    cause_label = build_cause_label(cause)

    if action.action_id in OUT_OF_STOCK_ACTION_IDS:
        return (
            f"{segment_prefix}{cause_label} 품절 또는 재고 이슈가 전환 저하 원인으로 "
            f"감지되어, {action.title} 액션을 추천합니다."
        )
    if action.action_id in CHANNEL_CONVERSION_ACTION_IDS:
        return (
            f"{segment_prefix}{cause_label} 유입 전환 효율이 낮게 나타나 "
            f"{action.title} 액션을 추천합니다."
        )
    if action.action_id in HIGH_PURCHASE_INTENT_ACTION_IDS:
        return (
            f"{segment_prefix}{cause_label} 사용자는 구매 가능성이 높거나 쿠폰 비용 낭비 "
            f"위험이 있어, {action.title} 액션을 추천합니다."
        )
    if action.action_id == "manual_review":
        return (
            f"{segment_prefix}{cause_label} 원인 후보에 직접 매칭되는 액션이 없어 "
            f"운영자 수동 검토를 추천합니다."
        )
    return (
        f"{segment_prefix}{cause_label} 단계의 핵심 지표가 낮게 나타났기 때문에, "
        f"{action.title} 액션을 추천합니다."
    )


def build_execution_hint(
    action: ActionCatalogItem,
    cause: CauseCandidate,
    segment: dict[str, str | None],
) -> dict[str, str | int | float | bool | None]:
    hint: dict[str, str | int | float | bool | None] = {
        **action.execution_hint,
        "requires_admin_approval": True,
        "target_segment": format_target_segment(segment, cause),
    }

    channel = segment.get("channel") or get_attribute_value(cause, "channel")
    if channel is not None:
        hint["channel"] = channel

    campaign_id = segment.get("campaign_id") or get_attribute_value(cause, "campaign_id")
    if campaign_id is not None:
        hint["campaign_id"] = campaign_id

    product_id = segment.get("product_id") or get_product_id(cause)
    if product_id is not None:
        hint["product_id"] = product_id

    if cause.affected_step is not None:
        hint["target_step"] = cause.affected_step

    return hint


def build_experiment(action: ActionCatalogItem) -> ActionExperiment | None:
    if action.action_type == "REVIEW":
        return None
    return ActionExperiment(
        enabled=True,
        primary_metric=action.primary_metric,
        guardrail_metrics=guardrail_metrics_for_action(action),
        variants=["control", "treatment"],
    )


def guardrail_metrics_for_action(action: ActionCatalogItem) -> list[str]:
    if action.action_type == "INCENTIVE":
        return ["coupon_cost", "margin"]
    if action.action_type == "AD":
        return ["ad_spend", "roas"]
    if action.action_type == "MESSAGE":
        return ["unsubscribe_rate", "message_fatigue"]
    if action.action_type == "COST_CONTROL":
        return ["purchase_rate", "coupon_cost"]
    return ["purchase_rate"]


def normalized_attribute(cause: CauseCandidate, key: str) -> str | None:
    value = cause.attributes.get(key)
    if value is None:
        return None
    return str(value).lower()


def normalized_affected_step(cause: CauseCandidate) -> str | None:
    if cause.affected_step is not None:
        return cause.affected_step.lower()
    return normalized_attribute(cause, "funnel_step")


def get_attribute_value(cause: CauseCandidate, key: str) -> str | int | float | bool | None:
    value = cause.attributes.get(key)
    if value is not None:
        return value
    dimension = cause.attributes.get("dimension")
    if dimension == key:
        return cause.attributes.get("value")
    return None


def get_product_id(cause: CauseCandidate) -> str | int | float | bool | None:
    product_id = get_attribute_value(cause, "product_id")
    if product_id is not None:
        return product_id
    if cause.attributes.get("dimension") == "product_id":
        return cause.attributes.get("value")
    return None


def format_target_segment(
    segment: dict[str, str | None],
    cause: CauseCandidate,
) -> str:
    values = {key: value for key, value in segment.items() if value is not None}
    dimension = cause.attributes.get("dimension")
    value = cause.attributes.get("value")
    if isinstance(dimension, str) and value is not None and dimension not in values:
        values[dimension] = str(value)
    return ", ".join(f"{key}={value}" for key, value in values.items())


def format_segment(segment: dict[str, str | None]) -> str:
    values = [f"{key}={value}" for key, value in segment.items() if value is not None]
    return " / ".join(values)


def build_cause_label(cause: CauseCandidate) -> str:
    if cause.label:
        return cause.label
    dimension = cause.attributes.get("dimension")
    value = cause.attributes.get("value")
    metric = cause.attributes.get("metric")
    if dimension is not None and value is not None and metric is not None:
        return f"{dimension}={value}의 {metric}"
    if cause.affected_step:
        return cause.affected_step
    if cause.description:
        return cause.description
    return cause.cause_type


def merge_triggered_by(existing: list[str], cause_id: str) -> list[str]:
    if cause_id in existing:
        return existing
    return [*existing, cause_id]


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


def clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))
