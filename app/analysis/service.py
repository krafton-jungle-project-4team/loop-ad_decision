from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from app.actions.schemas import (
    ActionExperiment,
    ActionRecommendationRequest,
    ActionRecommendationResponse,
    CauseCandidate,
    RecommendedAction,
)
from app.actions.service import recommend_actions
from app.analysis.schemas import (
    BlockedActionSummary,
    FunnelRecommendationAnalysisRequest,
    FunnelRecommendationAnalysisResponse,
)
from app.anomalies.schemas import (
    FunnelAnomalyEvaluation,
    FunnelAnomalyRequest,
    FunnelAnomalyResponse,
    VolumeAnomalyEvaluation,
)
from app.anomalies.service import calculate_funnel_anomalies
from app.automation.policy_engine import evaluate_recommendations
from app.automation.schemas import ActionPolicyDecision, PolicyDecision
from app.persistence.repository import PostgresRepository
from app.recommendations.content_linking import resolve_mapping_content_ids
from app.root_causes.schemas import (
    RootCauseAnalysisRequest,
    RootCauseAnalysisResponse,
    RootCauseCandidate,
)
from app.root_causes.service import calculate_root_causes

AnalysisCalculator = Callable[[FunnelAnomalyRequest, Any], FunnelAnomalyResponse]
RootCauseCalculator = Callable[[RootCauseAnalysisRequest, Any], RootCauseAnalysisResponse]
ActionRecommender = Callable[[ActionRecommendationRequest], ActionRecommendationResponse]
PolicyEvaluator = Callable[[Any, list[RecommendedAction], dict[str, str | None]], PolicyDecision]

NO_ACTION = "no_action"
PENDING_ACTIONS = "pending_actions"
PARTIALLY_EXECUTED = "partially_executed"
DISMISSED = "dismissed"
ACTION_PENDING_REVIEW = "pending_review"
ACTION_POLICY_BLOCKED = "policy_blocked"
ACTION_EXPERIMENT_RUNNING = "experiment_running"
EXPERIMENT_RUNNING = "experiment_running"


def run_funnel_recommendation_analysis(
    *,
    request: FunnelRecommendationAnalysisRequest,
    metrics_repository: Any,
    root_cause_repository: Any,
    persistence_repository: PostgresRepository,
    anomaly_calculator: AnalysisCalculator = calculate_funnel_anomalies,
    root_cause_calculator: RootCauseCalculator = calculate_root_causes,
    action_recommender: ActionRecommender = recommend_actions,
    policy_evaluator: PolicyEvaluator = evaluate_recommendations,
) -> FunnelRecommendationAnalysisResponse:
    try:
        response = execute_funnel_recommendation_analysis(
            request=request,
            metrics_repository=metrics_repository,
            root_cause_repository=root_cause_repository,
            persistence_repository=persistence_repository,
            anomaly_calculator=anomaly_calculator,
            root_cause_calculator=root_cause_calculator,
            action_recommender=action_recommender,
            policy_evaluator=policy_evaluator,
        )
        persistence_repository.commit()
        return response
    except Exception:
        persistence_repository.rollback()
        raise


def execute_funnel_recommendation_analysis(
    *,
    request: FunnelRecommendationAnalysisRequest,
    metrics_repository: Any,
    root_cause_repository: Any,
    persistence_repository: PostgresRepository,
    anomaly_calculator: AnalysisCalculator,
    root_cause_calculator: RootCauseCalculator,
    action_recommender: ActionRecommender,
    policy_evaluator: PolicyEvaluator,
) -> FunnelRecommendationAnalysisResponse:
    anomaly_response = anomaly_calculator(
        build_anomaly_request(request),
        metrics_repository,
    )

    if not has_anomaly(anomaly_response):
        result = persistence_repository.create_recommendation_result(
            project_id=request.project_id,
            window_start=anomaly_response.window_start,
            window_end=anomaly_response.window_end,
            baseline_start=anomaly_response.baseline_start,
            baseline_end=anomaly_response.baseline_end,
            segment_json=anomaly_response.segment,
            status=NO_ACTION,
            anomaly_json=serialize_model(anomaly_response),
            root_causes_json={},
            recommendations_json={},
            policy_decision_json={"policy_id": None, "auto_execute_enabled": False, "actions": []},
        )
        return FunnelRecommendationAnalysisResponse(
            recommendation_result_id=result.id,
            status=NO_ACTION,
            anomaly_summary=anomaly_response.summary_message,
            root_cause_candidates=[],
            recommended_actions=[],
            auto_executed_action_ids=[],
            blocked_actions=[],
            created_experiment_ids=[],
            created_segment_ad_mapping_ids=[],
            policy_decision=None,
        )

    root_cause_response = root_cause_calculator(
        build_root_cause_request(request),
        root_cause_repository,
    )
    has_synthetic_cause = not root_cause_response.candidates
    causes = (
        [build_unexplained_funnel_anomaly_cause(anomaly_response.primary_anomaly)]
        if has_synthetic_cause
        else [candidate.model_dump(mode="json") for candidate in root_cause_response.candidates]
    )
    action_response = action_recommender(
        ActionRecommendationRequest(
            project_id=request.project_id,
            window_start=anomaly_response.window_start,
            window_end=anomaly_response.window_end,
            segment=anomaly_response.segment,
            causes=causes,
            top_n=request.top_n,
        )
    )

    policy = persistence_repository.get_automation_policy(request.project_id)
    policy_decision = policy_evaluator(
        policy,
        action_response.recommendations,
        action_response.segment,
    )
    result = persistence_repository.create_recommendation_result(
        project_id=request.project_id,
        window_start=anomaly_response.window_start,
        window_end=anomaly_response.window_end,
        baseline_start=anomaly_response.baseline_start,
        baseline_end=anomaly_response.baseline_end,
        segment_json=anomaly_response.segment,
        status=PENDING_ACTIONS,
        anomaly_json=serialize_model(anomaly_response),
        root_causes_json=build_root_causes_payload(root_cause_response, causes, has_synthetic_cause),
        recommendations_json=serialize_model(action_response),
        policy_decision_json=policy_decision.model_dump(mode="json"),
        summary_message=anomaly_response.summary_message,
    )

    created_experiment_ids: list[int] = []
    created_mapping_ids: list[int] = []
    recommendations_by_id = {
        recommendation.action_id: recommendation
        for recommendation in action_response.recommendations
    }
    action_decisions_by_id = {
        action.action_id: action
        for action in policy_decision.actions
    }
    stored_actions = []
    now = datetime.now(UTC)
    for recommendation in action_response.recommendations:
        action_decision = action_decisions_by_id.get(recommendation.action_id)
        action_status = resolve_recommendation_action_status(
            policy=policy,
            action_decision=action_decision,
        )
        stored_action = persistence_repository.create_recommendation_action(
            project_id=request.project_id,
            recommendation_result_id=result.id,
            action_id=recommendation.action_id,
            action_type=recommendation.action_type,
            title=recommendation.title,
            description=recommendation.description,
            target_step=recommendation.target_step,
            priority_score=recommendation.priority_score,
            expected_impact=recommendation.expected_impact,
            rationale=recommendation.rationale,
            triggered_by_json=recommendation.triggered_by,
            execution_hint_json=recommendation.execution_hint,
            experiment_json=serialize_model(recommendation.experiment) or {},
            policy_status=get_value(action_decision, "status"),
            policy_reasons_json=get_value(action_decision, "reasons", []),
            policy_decision_json=serialize_model(action_decision) or {},
            status=action_status,
            auto_executed_at=now if action_decision is not None and action_decision.auto_executed else None,
        )
        stored_actions.append(stored_action)

    for action_decision in policy_decision.actions:
        if not action_decision.auto_executed:
            continue
        recommendation = recommendations_by_id[action_decision.action_id]
        stored_action = next(
            action
            for action in stored_actions
            if action.action_id == recommendation.action_id
        )
        experiment = get_or_create_experiment(
            persistence_repository=persistence_repository,
            recommendation_result_id=result.id,
            recommendation_action_id=stored_action.id,
            project_id=request.project_id,
            recommendation=recommendation,
            recommendation_action=stored_action,
            action_decision=action_decision,
            segment=anomaly_response.segment,
        )
        mapping = get_or_create_segment_ad_mapping(
            persistence_repository=persistence_repository,
            recommendation_result_id=result.id,
            recommendation_action_id=stored_action.id,
            project_id=request.project_id,
            recommendation=recommendation,
            recommendation_action=stored_action,
            experiment_id=experiment.id,
            segment=anomaly_response.segment,
        )
        persistence_repository.update_recommendation_action(
            stored_action.id,
            {"status": ACTION_EXPERIMENT_RUNNING},
        )
        stored_action.status = ACTION_EXPERIMENT_RUNNING
        created_experiment_ids.append(experiment.id)
        created_mapping_ids.append(mapping.id)

    status = resolve_recommendation_result_status([action.status for action in stored_actions])
    persistence_repository.update_recommendation_result(result.id, {"status": status})
    auto_executed_action_ids = [
        action.action_id
        for action in stored_actions
        if action.status == ACTION_EXPERIMENT_RUNNING
    ]

    return FunnelRecommendationAnalysisResponse(
        recommendation_result_id=result.id,
        status=status,
        anomaly_summary=anomaly_response.summary_message,
        root_cause_candidates=root_cause_response.candidates,
        recommended_actions=action_response.recommendations,
        auto_executed_action_ids=auto_executed_action_ids,
        blocked_actions=[
            BlockedActionSummary(action_id=action.action_id, reasons=action.reasons)
            for action in policy_decision.actions
            if action.blocked
        ],
        created_experiment_ids=created_experiment_ids,
        created_segment_ad_mapping_ids=created_mapping_ids,
        policy_decision=policy_decision,
    )


def build_anomaly_request(request: FunnelRecommendationAnalysisRequest) -> FunnelAnomalyRequest:
    return FunnelAnomalyRequest(
        project_id=request.project_id,
        window_start=request.window_start,
        window_end=request.window_end,
        baseline_start=request.baseline_start,
        baseline_end=request.baseline_end,
        filters=request.filters,
        min_sample_size=request.min_sample_size,
        warning_abs_drop=request.warning_abs_drop,
        critical_abs_drop=request.critical_abs_drop,
        warning_relative_drop=request.warning_relative_drop,
        critical_relative_drop=request.critical_relative_drop,
        include_volume_anomalies=request.include_volume_anomalies,
        min_volume_count=request.min_volume_count,
        warning_volume_relative_drop=request.warning_volume_relative_drop,
        critical_volume_relative_drop=request.critical_volume_relative_drop,
    )


def build_root_cause_request(
    request: FunnelRecommendationAnalysisRequest,
) -> RootCauseAnalysisRequest:
    values = request.model_dump(exclude={"top_n"})
    return RootCauseAnalysisRequest(**values)


def has_anomaly(anomaly_response: FunnelAnomalyResponse) -> bool:
    return anomaly_response.primary_anomaly is not None and anomaly_response.status != "normal"


def build_unexplained_funnel_anomaly_cause(
    anomaly: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation | None,
) -> CauseCandidate:
    severity = 1.0 if getattr(anomaly, "severity", None) == "critical" else 0.7
    return CauseCandidate(
        cause_id="UNEXPLAINED_FUNNEL_ANOMALY",
        cause_type="UNEXPLAINED_FUNNEL_ANOMALY",
        label="원인 미확인 퍼널 이상징후",
        description="이상징후는 감지되었지만 자동 원인 후보가 발견되지 않았습니다.",
        affected_step=None,
        severity=severity,
        confidence=0.3,
        evidence=[],
        attributes={"requires_manual_review": True},
    )


def resolve_recommendation_action_status(
    *,
    policy: Any,
    action_decision: ActionPolicyDecision | None,
) -> str:
    if action_decision is None:
        return ACTION_PENDING_REVIEW
    if "manual_review_action" in action_decision.reasons:
        return ACTION_PENDING_REVIEW
    if action_decision.auto_executed:
        return ACTION_EXPERIMENT_RUNNING
    if policy is None:
        return ACTION_PENDING_REVIEW
    if not bool(get_value(policy, "enabled", False)):
        return ACTION_PENDING_REVIEW
    if not bool(get_value(policy, "auto_execute_enabled", False)):
        return ACTION_PENDING_REVIEW
    return ACTION_POLICY_BLOCKED


def resolve_recommendation_result_status(action_statuses: list[str]) -> str:
    if not action_statuses:
        return NO_ACTION
    executable_statuses = {ACTION_EXPERIMENT_RUNNING, "auto_executed", "approved"}
    if all(status == "rejected" for status in action_statuses):
        return DISMISSED
    executed_count = sum(1 for status in action_statuses if status in executable_statuses)
    if executed_count == len(action_statuses):
        return EXPERIMENT_RUNNING
    if executed_count > 0:
        return PARTIALLY_EXECUTED
    return PENDING_ACTIONS


def get_or_create_experiment(
    *,
    persistence_repository: PostgresRepository,
    recommendation_result_id: int,
    recommendation_action_id: int,
    project_id: str,
    recommendation: RecommendedAction,
    recommendation_action: Any,
    action_decision: ActionPolicyDecision,
    segment: dict[str, str | None],
) -> Any:
    existing = persistence_repository.get_experiment_by_recommendation_action(
        recommendation_action_id=recommendation_action_id,
    )
    if existing is not None:
        return existing

    experiment_config = recommendation.experiment
    return persistence_repository.create_experiment(
        project_id=project_id,
        recommendation_result_id=recommendation_result_id,
        recommendation_action_id=recommendation_action_id,
        bandit_policy_id=get_value(recommendation_action, "bandit_policy_id"),
        bandit_arm_id=get_value(recommendation_action, "bandit_arm_id"),
        segment_json=segment,
        action_id=recommendation.action_id,
        action_type=recommendation.action_type,
        status="running",
        traffic_split_json=serialize_model(action_decision.traffic_split) or {},
        primary_metric=get_primary_metric(experiment_config),
        guardrail_metrics_json=get_guardrail_metrics(experiment_config),
        started_at=datetime.now(UTC),
    )


def get_or_create_segment_ad_mapping(
    *,
    persistence_repository: PostgresRepository,
    recommendation_result_id: int,
    recommendation_action_id: int,
    project_id: str,
    recommendation: RecommendedAction,
    recommendation_action: Any,
    experiment_id: int,
    segment: dict[str, str | None],
) -> Any:
    existing = persistence_repository.get_segment_ad_mapping_by_recommendation_action(
        recommendation_action_id=recommendation_action_id,
    )
    if existing is not None:
        return existing

    content_ids = resolve_mapping_content_ids(
        repository=persistence_repository,
        project_id=project_id,
        action_id=recommendation.action_id,
        execution_hint_json=recommendation.execution_hint,
    )
    return persistence_repository.create_segment_ad_mapping(
        project_id=project_id,
        recommendation_result_id=recommendation_result_id,
        recommendation_action_id=recommendation_action_id,
        segment_json=segment,
        experiment_id=experiment_id,
        bandit_policy_id=get_value(recommendation_action, "bandit_policy_id"),
        bandit_arm_id=get_value(recommendation_action, "bandit_arm_id"),
        campaign_id=content_ids.campaign_id,
        creative_id=content_ids.creative_id,
        coupon_id=content_ids.coupon_id,
        action_id=recommendation.action_id,
        action_type=recommendation.action_type,
        execution_hint_json=recommendation.execution_hint,
        status="active",
        source="auto_policy",
    )


def get_primary_metric(experiment: ActionExperiment | None) -> str | None:
    if experiment is None:
        return None
    return experiment.primary_metric


def get_guardrail_metrics(experiment: ActionExperiment | None) -> list[str]:
    if experiment is None:
        return []
    return experiment.guardrail_metrics


def build_root_causes_payload(
    root_cause_response: RootCauseAnalysisResponse,
    causes: list[CauseCandidate | dict[str, Any]],
    has_synthetic_cause: bool,
) -> dict[str, Any]:
    payload = serialize_model(root_cause_response)
    if has_synthetic_cause:
        payload["synthetic_causes"] = [serialize_model(cause) for cause in causes]
    return payload


def serialize_model(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [serialize_model(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_model(item) for key, item in value.items()}
    return value


def get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)
