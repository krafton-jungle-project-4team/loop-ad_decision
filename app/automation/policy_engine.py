from typing import Any

from app.automation.schemas import ActionPolicyDecision, PolicyDecision, TrafficSplit

POLICY_MISSING = "policy_missing"
POLICY_DISABLED = "policy_disabled"
AUTO_EXECUTE_DISABLED = "auto_execute_disabled"
MANUAL_REVIEW_ACTION = "manual_review_action"
ACTION_BLOCKED = "action_blocked"
NO_ALLOWED_ACTIONS_CONFIGURED = "no_allowed_actions_configured"
ACTION_NOT_ALLOWED = "action_not_allowed"
PRIORITY_SCORE_TOO_LOW = "priority_score_too_low"
DISCOUNT_RATE_TOO_HIGH = "discount_rate_too_high"

MANUAL_REVIEW_ACTION_IDS = {"manual_review"}
MANUAL_REVIEW_ACTION_TYPES = {"REVIEW"}


def evaluate_auto_execution(
    policy: Any,
    recommendation: Any,
    segment: dict[str, str | None] | None = None,
) -> PolicyDecision:
    return evaluate_recommendations(policy, [recommendation], segment)


def evaluate_recommendations(
    policy: Any,
    recommendations: list[Any],
    segment: dict[str, str | None] | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        policy_id=get_value(policy, "id"),
        auto_execute_enabled=bool(get_value(policy, "auto_execute_enabled", False)),
        actions=[
            evaluate_action(policy, recommendation, segment or {})
            for recommendation in recommendations
        ],
    )


def evaluate_action(
    policy: Any,
    recommendation: Any,
    segment: dict[str, str | None],
) -> ActionPolicyDecision:
    action_id = str(get_value(recommendation, "action_id", ""))
    action_type = str(get_value(recommendation, "action_type", ""))
    priority_score = to_float(get_value(recommendation, "priority_score"), default=0.0)
    execution_hint = normalize_mapping(get_value(recommendation, "execution_hint", {}))
    reasons: list[str] = []
    metadata: dict[str, Any] = {}

    if policy is None:
        reasons.append(POLICY_MISSING)
    else:
        if not bool(get_value(policy, "enabled", False)):
            reasons.append(POLICY_DISABLED)
        if not bool(get_value(policy, "auto_execute_enabled", False)):
            reasons.append(AUTO_EXECUTE_DISABLED)
        if is_manual_review_action(action_id, action_type):
            reasons.append(MANUAL_REVIEW_ACTION)
        reasons.extend(evaluate_action_allowlist(policy, action_id, action_type))
        reasons.extend(evaluate_priority(policy, priority_score))
        reasons.extend(evaluate_discount_cap(policy, action_type, execution_hint, metadata))
        apply_message_metadata(policy, action_type, metadata)

    auto_executed = not reasons
    return ActionPolicyDecision(
        action_id=action_id,
        action_type=action_type,
        status="auto_executed" if auto_executed else "blocked",
        allowed=auto_executed,
        blocked=not auto_executed,
        auto_executed=auto_executed,
        reasons=reasons,
        traffic_split=build_traffic_split(policy) if auto_executed else None,
        metadata=metadata,
    )


def evaluate_action_allowlist(policy: Any, action_id: str, action_type: str) -> list[str]:
    blocked_action_ids = string_set(get_value(policy, "blocked_action_ids", []))
    if action_id in blocked_action_ids:
        return [ACTION_BLOCKED]

    allowed_action_ids = string_set(get_value(policy, "allowed_action_ids", []))
    allowed_action_types = string_set(get_value(policy, "allowed_action_types", []))
    if not allowed_action_ids and not allowed_action_types:
        return [NO_ALLOWED_ACTIONS_CONFIGURED]
    if action_id in allowed_action_ids or action_type in allowed_action_types:
        return []
    return [ACTION_NOT_ALLOWED]


def evaluate_priority(policy: Any, priority_score: float) -> list[str]:
    min_priority_score = to_float(get_value(policy, "min_priority_score"), default=0.0)
    if priority_score < min_priority_score:
        return [PRIORITY_SCORE_TOO_LOW]
    return []


def evaluate_discount_cap(
    policy: Any,
    action_type: str,
    execution_hint: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    if action_type != "INCENTIVE":
        return []

    max_discount_rate = get_value(policy, "max_discount_rate")
    if max_discount_rate is None:
        return []

    metadata["max_discount_rate"] = max_discount_rate
    discount_rate = execution_hint.get("discount_rate")
    if discount_rate is None:
        return []
    if to_float(discount_rate, default=0.0) > to_float(max_discount_rate, default=1.0):
        return [DISCOUNT_RATE_TOO_HIGH]
    return []


def apply_message_metadata(policy: Any, action_type: str, metadata: dict[str, Any]) -> None:
    if action_type != "MESSAGE":
        return
    max_message_per_user_per_day = get_value(policy, "max_message_per_user_per_day")
    if max_message_per_user_per_day is not None:
        metadata["max_message_per_user_per_day"] = max_message_per_user_per_day


def build_traffic_split(policy: Any) -> TrafficSplit:
    treatment = clamp(
        to_float(get_value(policy, "max_experiment_traffic_ratio"), default=0.2)
    )
    return TrafficSplit(control=round(1.0 - treatment, 4), treatment=round(treatment, 4))


def is_manual_review_action(action_id: str, action_type: str) -> bool:
    return action_id in MANUAL_REVIEW_ACTION_IDS or action_type in MANUAL_REVIEW_ACTION_TYPES


def normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def get_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))
