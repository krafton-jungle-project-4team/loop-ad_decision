from types import SimpleNamespace

from app.automation.policy_engine import (
    ACTION_BLOCKED,
    ACTION_NOT_ALLOWED,
    AUTO_EXECUTE_DISABLED,
    DISCOUNT_RATE_TOO_HIGH,
    MANUAL_REVIEW_ACTION,
    NO_ALLOWED_ACTIONS_CONFIGURED,
    POLICY_DISABLED,
    POLICY_MISSING,
    PRIORITY_SCORE_TOO_LOW,
    evaluate_auto_execution,
    evaluate_recommendations,
)


def policy(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": 1,
        "enabled": True,
        "auto_execute_enabled": True,
        "allowed_action_ids": [],
        "allowed_action_types": ["PRODUCT"],
        "blocked_action_ids": [],
        "max_experiment_traffic_ratio": 0.2,
        "min_priority_score": 0.5,
        "max_discount_rate": None,
        "max_daily_coupon_budget": None,
        "max_message_per_user_per_day": None,
        "stop_loss_relative_drop": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def recommendation(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "action_id": "recommend_alternative_product",
        "action_type": "PRODUCT",
        "priority_score": 0.8,
        "execution_hint": {},
    }
    values.update(overrides)
    return values


def first_action(decision):
    return decision.actions[0]


def test_policy_missing_blocks_auto_execution() -> None:
    action = first_action(evaluate_auto_execution(None, recommendation()))

    assert action.status == "blocked"
    assert action.allowed is False
    assert action.blocked is True
    assert action.auto_executed is False
    assert action.reasons == [POLICY_MISSING]


def test_policy_disabled_blocks_auto_execution() -> None:
    action = first_action(evaluate_auto_execution(policy(enabled=False), recommendation()))

    assert action.status == "blocked"
    assert POLICY_DISABLED in action.reasons


def test_auto_execute_disabled_blocks_auto_execution() -> None:
    action = first_action(
        evaluate_auto_execution(policy(auto_execute_enabled=False), recommendation())
    )

    assert action.status == "blocked"
    assert AUTO_EXECUTE_DISABLED in action.reasons


def test_allowed_action_type_auto_executes_with_traffic_split() -> None:
    action = first_action(evaluate_auto_execution(policy(), recommendation()))

    assert action.status == "auto_executed"
    assert action.allowed is True
    assert action.blocked is False
    assert action.auto_executed is True
    assert action.reasons == []
    assert action.traffic_split is not None
    assert action.traffic_split.control == 0.8
    assert action.traffic_split.treatment == 0.2


def test_allowed_action_id_auto_executes() -> None:
    action = first_action(
        evaluate_auto_execution(
            policy(allowed_action_ids=["limited_time_coupon"], allowed_action_types=[]),
            recommendation(action_id="limited_time_coupon", action_type="INCENTIVE"),
        )
    )

    assert action.status == "auto_executed"


def test_empty_allowed_lists_block_auto_execution() -> None:
    action = first_action(
        evaluate_auto_execution(
            policy(allowed_action_ids=[], allowed_action_types=[]),
            recommendation(),
        )
    )

    assert action.status == "blocked"
    assert NO_ALLOWED_ACTIONS_CONFIGURED in action.reasons


def test_action_not_in_allowed_ids_or_types_is_blocked() -> None:
    action = first_action(
        evaluate_auto_execution(
            policy(allowed_action_ids=["pause_out_of_stock_ads"], allowed_action_types=["AD"]),
            recommendation(action_id="recommend_alternative_product", action_type="PRODUCT"),
        )
    )

    assert action.status == "blocked"
    assert ACTION_NOT_ALLOWED in action.reasons


def test_manual_review_action_is_never_auto_executed() -> None:
    action = first_action(
        evaluate_auto_execution(
            policy(allowed_action_ids=["manual_review"], allowed_action_types=["REVIEW"]),
            recommendation(action_id="manual_review", action_type="REVIEW"),
        )
    )

    assert action.status == "blocked"
    assert MANUAL_REVIEW_ACTION in action.reasons


def test_blocked_action_id_overrides_allowed_action_type() -> None:
    action = first_action(
        evaluate_auto_execution(
            policy(blocked_action_ids=["recommend_alternative_product"]),
            recommendation(),
        )
    )

    assert action.status == "blocked"
    assert action.reasons[0] == ACTION_BLOCKED


def test_low_priority_score_blocks_auto_execution() -> None:
    action = first_action(
        evaluate_auto_execution(policy(min_priority_score=0.9), recommendation())
    )

    assert action.status == "blocked"
    assert PRIORITY_SCORE_TOO_LOW in action.reasons


def test_discount_rate_above_policy_cap_blocks_incentive() -> None:
    action = first_action(
        evaluate_auto_execution(
            policy(allowed_action_types=["INCENTIVE"], max_discount_rate=0.1),
            recommendation(
                action_id="limited_time_coupon",
                action_type="INCENTIVE",
                execution_hint={"discount_rate": 0.2},
            ),
        )
    )

    assert action.status == "blocked"
    assert DISCOUNT_RATE_TOO_HIGH in action.reasons
    assert action.metadata["max_discount_rate"] == 0.1


def test_message_policy_metadata_is_included() -> None:
    action = first_action(
        evaluate_auto_execution(
            policy(
                allowed_action_types=["MESSAGE"],
                max_message_per_user_per_day=1,
            ),
            recommendation(
                action_id="cart_reminder_message",
                action_type="MESSAGE",
            ),
        )
    )

    assert action.status == "auto_executed"
    assert action.metadata["max_message_per_user_per_day"] == 1


def test_policy_decision_json_shape_is_fixed() -> None:
    decision = evaluate_recommendations(policy(), [recommendation()], {"channel": "kakao"})
    payload = decision.model_dump(mode="json")

    assert payload["policy_id"] == 1
    assert payload["auto_execute_enabled"] is True
    assert set(payload["actions"][0]) == {
        "action_id",
        "action_type",
        "status",
        "allowed",
        "blocked",
        "auto_executed",
        "reasons",
        "traffic_split",
        "metadata",
    }
    assert payload["actions"][0]["status"] == "auto_executed"
