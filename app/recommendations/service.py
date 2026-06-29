from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status

from app.analysis.service import serialize_model
from app.persistence.models import RecommendationAction
from app.persistence.repository import PostgresRepository
from app.recommendations.content_linking import resolve_mapping_content_ids
from app.recommendations.schemas import (
    ActiveSegmentAdMappingResponse,
    RecommendationApprovalResponse,
    RecommendationApproveRequest,
    RecommendationRejectionResponse,
    RecommendationRejectRequest,
    RecommendationResultResponse,
    RecommendationActionResponse,
)

APPROVABLE_ACTION_STATUSES = {"pending_review", "policy_blocked", "approved"}
REJECTABLE_ACTION_STATUSES = {
    "pending_review",
    "policy_blocked",
    "approved",
    "auto_executed",
    "experiment_running",
}
MANUAL_REVIEW_ACTION_IDS = {"manual_review"}
MANUAL_REVIEW_ACTION_TYPES = {"REVIEW"}


def list_recommendation_results(
    *,
    repository: PostgresRepository,
    project_id: str | None,
    status_filter: str | None,
    limit: int,
) -> list[RecommendationResultResponse]:
    results = repository.list_recommendation_results(
        project_id=project_id,
        status=status_filter,
        limit=limit,
    )
    return [
        build_recommendation_result_response(repository=repository, result=result)
        for result in results
    ]


def get_recommendation_result_response(
    *,
    repository: PostgresRepository,
    recommendation_result_id: int,
) -> RecommendationResultResponse:
    result = repository.get_recommendation_result(recommendation_result_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="recommendation result not found",
        )
    return build_recommendation_result_response(repository=repository, result=result)


def build_recommendation_result_response(
    *,
    repository: PostgresRepository,
    result: Any,
) -> RecommendationResultResponse:
    response = RecommendationResultResponse.model_validate(result)
    response.actions = [
        RecommendationActionResponse.model_validate(action)
        for action in repository.list_recommendation_actions(
            recommendation_result_id=result.id,
        )
    ]
    return response


def approve_recommendation_result(
    *,
    repository: PostgresRepository,
    recommendation_result_id: int,
    request: RecommendationApproveRequest,
) -> RecommendationApprovalResponse:
    try:
        result = repository.get_recommendation_result(recommendation_result_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="recommendation result not found",
            )
        stored_actions = repository.list_recommendation_actions(
            recommendation_result_id=recommendation_result_id
        )
        if not stored_actions:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="legacy recommendation result has no action rows",
            )
        actions_by_id = {action.action_id: action for action in stored_actions}
        invalid_action_ids = [
            action_id
            for action_id in request.action_ids
            if action_id not in actions_by_id
        ]
        if invalid_action_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "unknown action_ids",
                    "action_ids": invalid_action_ids,
                },
            )

        approved_action_ids: list[str] = []
        experiment_ids: list[int] = []
        mapping_ids: list[int] = []
        for action_id in dict.fromkeys(request.action_ids):
            recommendation_action = actions_by_id[action_id]
            validate_manual_approval_action(recommendation_action)
            if recommendation_action.status not in APPROVABLE_ACTION_STATUSES:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"action {action_id} is not approvable",
                )
            repository.update_recommendation_action(
                recommendation_action.id,
                {
                    "status": "approved",
                    "reviewed_by": request.approved_by,
                    "reviewed_at": datetime.now(UTC),
                    "review_reason": request.reason,
                },
            )
            experiment = get_or_create_manual_experiment(
                repository=repository,
                result=result,
                recommendation_action=recommendation_action,
            )
            mapping = get_or_create_manual_mapping(
                repository=repository,
                result=result,
                recommendation_action=recommendation_action,
                experiment_id=experiment.id,
            )
            repository.update_recommendation_action(
                recommendation_action.id,
                {"status": "experiment_running"},
            )
            recommendation_action.status = "experiment_running"
            approved_action_ids.append(action_id)
            experiment_ids.append(experiment.id)
            mapping_ids.append(mapping.id)

        latest_actions = repository.list_recommendation_actions(
            recommendation_result_id=recommendation_result_id
        )
        result_status = resolve_result_status_from_actions(latest_actions)

        repository.update_recommendation_result(
            recommendation_result_id,
            {
                "status": result_status,
                "policy_decision_json": build_manual_approval_policy_decision(
                    result.policy_decision_json,
                    approved_action_ids,
                    request,
                ),
            },
        )
        repository.commit()
        return RecommendationApprovalResponse(
            recommendation_result_id=recommendation_result_id,
            status=result_status,
            approved_action_ids=approved_action_ids,
            created_experiment_ids=experiment_ids,
            created_segment_ad_mapping_ids=mapping_ids,
        )
    except Exception:
        repository.rollback()
        raise


def reject_recommendation_result(
    *,
    repository: PostgresRepository,
    recommendation_result_id: int,
    request: RecommendationRejectRequest,
) -> RecommendationRejectionResponse:
    try:
        result = repository.get_recommendation_result(recommendation_result_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="recommendation result not found",
            )
        stored_actions = repository.list_recommendation_actions(
            recommendation_result_id=recommendation_result_id
        )
        if not stored_actions:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="legacy recommendation result has no action rows",
            )
        action_ids = list(dict.fromkeys(request.action_ids or [
            action.action_id
            for action in stored_actions
            if action.status in REJECTABLE_ACTION_STATUSES
        ]))
        actions_by_id = {action.action_id: action for action in stored_actions}
        missing_action_ids = [action_id for action_id in action_ids if action_id not in actions_by_id]
        if missing_action_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "unknown action_ids", "action_ids": missing_action_ids},
            )

        selected_action_row_ids = {
            actions_by_id[action_id].id
            for action_id in action_ids
        }

        inactive_mapping_ids: list[int] = []
        for mapping in repository.list_segment_ad_mappings(
            recommendation_result_id=recommendation_result_id,
            status="active",
        ):
            if mapping.recommendation_action_id in selected_action_row_ids:
                repository.update_segment_ad_mapping(mapping.id, {"status": "inactive"})
                inactive_mapping_ids.append(mapping.id)

        stopped_experiment_ids: list[int] = []
        for experiment in repository.list_experiments(
            recommendation_result_id=recommendation_result_id
        ):
            if (
                experiment.recommendation_action_id in selected_action_row_ids
                and experiment.status != "stopped"
            ):
                repository.update_experiment(
                    experiment.id,
                    {"status": "stopped", "ended_at": datetime.now(UTC)},
                )
                stopped_experiment_ids.append(experiment.id)

        for action_id in action_ids:
            action = actions_by_id[action_id]
            if action.status not in REJECTABLE_ACTION_STATUSES and request.action_ids:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"action {action_id} is not rejectable",
                )
            repository.update_recommendation_action(
                action.id,
                {
                    "status": "rejected",
                    "reviewed_by": request.rejected_by,
                    "reviewed_at": datetime.now(UTC),
                    "review_reason": request.reason,
                },
            )
            action.status = "rejected"

        latest_actions = repository.list_recommendation_actions(
            recommendation_result_id=recommendation_result_id
        )
        result_status = resolve_result_status_from_actions(latest_actions)

        repository.update_recommendation_result(
            recommendation_result_id,
            {
                "status": result_status,
                "policy_decision_json": build_manual_rejection_policy_decision(
                    result.policy_decision_json,
                    request,
                ),
            },
        )
        repository.commit()
        return RecommendationRejectionResponse(
            recommendation_result_id=recommendation_result_id,
            status=result_status,
            rejected_action_ids=action_ids,
            inactivated_segment_ad_mapping_ids=inactive_mapping_ids,
            stopped_experiment_ids=stopped_experiment_ids,
        )
    except Exception:
        repository.rollback()
        raise


def list_active_segment_ad_mappings(
    *,
    repository: PostgresRepository,
    project_id: str,
) -> list[ActiveSegmentAdMappingResponse]:
    mappings = repository.list_active_segment_ad_mappings_with_creatives(project_id)
    responses: list[ActiveSegmentAdMappingResponse] = []
    for row in mappings:
        content_url = getattr(row.creative, "image_url", None)
        if not content_url:
            continue
        responses.append(
            ActiveSegmentAdMappingResponse(
                mapping_id=row.mapping.id,
                project_id=row.mapping.project_id,
                segment_json=row.mapping.segment_json,
                segment_hash=row.mapping.segment_hash,
                action_id=row.mapping.action_id,
                action_type=row.mapping.action_type,
                execution_hint_json=row.mapping.execution_hint_json,
                experiment_id=row.mapping.experiment_id,
                recommendation_result_id=row.mapping.recommendation_result_id,
                recommendation_action_id=row.mapping.recommendation_action_id,
                bandit_policy_id=row.mapping.bandit_policy_id,
                bandit_arm_id=row.mapping.bandit_arm_id,
                bandit_decision_id=row.mapping.bandit_decision_id,
                campaign_id=row.mapping.campaign_id,
                creative_id=row.mapping.creative_id,
                coupon_id=row.mapping.coupon_id,
                content_url=content_url,
                title=getattr(row.creative, "title", None),
                message=getattr(row.creative, "message", None),
                landing_url=getattr(row.creative, "landing_url", None),
                serving_weight=get_serving_weight(row.mapping.execution_hint_json),
                source=row.mapping.source,
                status=row.mapping.status,
            )
        )
    return responses


def validate_manual_approval_action(recommendation: RecommendationAction) -> None:
    if (
        recommendation.action_id in MANUAL_REVIEW_ACTION_IDS
        or recommendation.action_type in MANUAL_REVIEW_ACTION_TYPES
        or not recommendation.experiment_json
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="action cannot create an experiment or segment mapping",
        )


def get_or_create_manual_experiment(
    *,
    repository: PostgresRepository,
    result: Any,
    recommendation_action: RecommendationAction,
) -> Any:
    existing = repository.get_experiment_by_recommendation_action(
        recommendation_action_id=recommendation_action.id,
    )
    if existing is not None:
        return existing

    experiment_config = recommendation_action.experiment_json
    return repository.create_experiment(
        project_id=result.project_id,
        recommendation_result_id=result.id,
        recommendation_action_id=recommendation_action.id,
        bandit_policy_id=recommendation_action.bandit_policy_id,
        bandit_arm_id=recommendation_action.bandit_arm_id,
        segment_json=result.segment_json,
        action_id=recommendation_action.action_id,
        action_type=recommendation_action.action_type,
        status="running",
        traffic_split_json=default_manual_traffic_split(),
        primary_metric=get_primary_metric(experiment_config),
        guardrail_metrics_json=get_guardrail_metrics(experiment_config),
        started_at=datetime.now(UTC),
    )


def get_or_create_manual_mapping(
    *,
    repository: PostgresRepository,
    result: Any,
    recommendation_action: RecommendationAction,
    experiment_id: int,
) -> Any:
    existing = repository.get_segment_ad_mapping_by_recommendation_action(
        recommendation_action_id=recommendation_action.id,
    )
    if existing is not None:
        return existing

    content_ids = resolve_mapping_content_ids(
        repository=repository,
        project_id=result.project_id,
        action_id=recommendation_action.action_id,
        execution_hint_json=recommendation_action.execution_hint_json,
    )
    return repository.create_segment_ad_mapping(
        project_id=result.project_id,
        recommendation_result_id=result.id,
        recommendation_action_id=recommendation_action.id,
        segment_json=result.segment_json,
        experiment_id=experiment_id,
        bandit_policy_id=recommendation_action.bandit_policy_id,
        bandit_arm_id=recommendation_action.bandit_arm_id,
        campaign_id=content_ids.campaign_id,
        creative_id=content_ids.creative_id,
        coupon_id=content_ids.coupon_id,
        action_id=recommendation_action.action_id,
        action_type=recommendation_action.action_type,
        execution_hint_json=recommendation_action.execution_hint_json,
        status="active",
        source="manual_approval",
    )


def default_manual_traffic_split() -> dict[str, float]:
    return {"control": 0.8, "treatment": 0.2}


def build_manual_approval_policy_decision(
    existing_decision: dict[str, Any],
    approved_action_ids: list[str],
    request: RecommendationApproveRequest,
) -> dict[str, Any]:
    decision = dict(existing_decision or {})
    manual_review = dict(decision.get("manual_review", {}))
    manual_review["approval"] = {
        "approved_by": request.approved_by,
        "action_ids": approved_action_ids,
        "reason": request.reason,
        "approved_at": datetime.now(UTC).isoformat(),
    }
    decision["manual_review"] = manual_review
    return serialize_model(decision)


def build_manual_rejection_policy_decision(
    existing_decision: dict[str, Any],
    request: RecommendationRejectRequest,
) -> dict[str, Any]:
    decision = dict(existing_decision or {})
    manual_review = dict(decision.get("manual_review", {}))
    manual_review["rejection"] = {
        "rejected_by": request.rejected_by,
        "reason": request.reason,
        "rejected_at": datetime.now(UTC).isoformat(),
    }
    decision["manual_review"] = manual_review
    return serialize_model(decision)


def get_primary_metric(experiment_json: dict[str, Any] | None) -> str | None:
    if not experiment_json:
        return None
    return experiment_json.get("primary_metric")


def get_guardrail_metrics(experiment_json: dict[str, Any] | None) -> list[str]:
    if not experiment_json:
        return []
    metrics = experiment_json.get("guardrail_metrics")
    return metrics if isinstance(metrics, list) else []


def resolve_result_status_from_actions(actions: list[Any]) -> str:
    if not actions:
        return "no_action"
    statuses = [action.status for action in actions]
    if all(status == "rejected" for status in statuses):
        return "dismissed"
    executed = {"approved", "auto_executed", "experiment_running"}
    executed_count = sum(1 for status in statuses if status in executed)
    if executed_count == len(statuses):
        return "experiment_running"
    if executed_count > 0:
        return "partially_executed"
    return "pending_actions"


def get_serving_weight(execution_hint_json: dict[str, Any]) -> float | None:
    value = (execution_hint_json or {}).get("serving_weight")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
