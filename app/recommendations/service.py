from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from pydantic import ValidationError

from app.actions.schemas import ActionExperiment, RecommendedAction
from app.analysis.service import get_guardrail_metrics, get_primary_metric, serialize_model
from app.persistence.repository import PostgresRepository
from app.recommendations.schemas import (
    ActiveSegmentAdMappingResponse,
    RecommendationApprovalResponse,
    RecommendationApproveRequest,
    RecommendationRejectionResponse,
    RecommendationRejectRequest,
    RecommendationResultResponse,
)

APPROVABLE_STATUSES = {"pending_review", "policy_blocked"}
REJECTABLE_STATUSES = {"pending_review", "policy_blocked", "experiment_running"}
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
    return [RecommendationResultResponse.model_validate(result) for result in results]


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
    return RecommendationResultResponse.model_validate(result)


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
        if result.status not in APPROVABLE_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="recommendation result is not approvable",
            )

        recommendations = parse_recommendations(result.recommendations_json)
        recommendations_by_id = {
            recommendation.action_id: recommendation
            for recommendation in recommendations
        }
        invalid_action_ids = [
            action_id
            for action_id in request.action_ids
            if action_id not in recommendations_by_id
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
            recommendation = recommendations_by_id[action_id]
            validate_manual_approval_action(recommendation)
            experiment = get_or_create_manual_experiment(
                repository=repository,
                result=result,
                recommendation=recommendation,
            )
            mapping = get_or_create_manual_mapping(
                repository=repository,
                result=result,
                recommendation=recommendation,
                experiment_id=experiment.id,
            )
            approved_action_ids.append(action_id)
            experiment_ids.append(experiment.id)
            mapping_ids.append(mapping.id)

        repository.update_recommendation_result(
            recommendation_result_id,
            {
                "status": "experiment_running",
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
            status="experiment_running",
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
        if result.status not in REJECTABLE_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="recommendation result is not rejectable",
            )

        inactive_mapping_ids: list[int] = []
        for mapping in repository.list_segment_ad_mappings(
            recommendation_result_id=recommendation_result_id,
            status="active",
        ):
            repository.update_segment_ad_mapping(mapping.id, {"status": "inactive"})
            inactive_mapping_ids.append(mapping.id)

        stopped_experiment_ids: list[int] = []
        for experiment in repository.list_experiments(
            recommendation_result_id=recommendation_result_id
        ):
            if experiment.status != "stopped":
                repository.update_experiment(
                    experiment.id,
                    {"status": "stopped", "ended_at": datetime.now(UTC)},
                )
                stopped_experiment_ids.append(experiment.id)

        repository.update_recommendation_result(
            recommendation_result_id,
            {
                "status": "rejected",
                "policy_decision_json": build_manual_rejection_policy_decision(
                    result.policy_decision_json,
                    request,
                ),
            },
        )
        repository.commit()
        return RecommendationRejectionResponse(
            recommendation_result_id=recommendation_result_id,
            status="rejected",
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
    mappings = repository.list_active_segment_ad_mappings(project_id)
    return [
        ActiveSegmentAdMappingResponse(
            mapping_id=mapping.id,
            project_id=mapping.project_id,
            segment_json=mapping.segment_json,
            segment_hash=mapping.segment_hash,
            action_id=mapping.action_id,
            action_type=mapping.action_type,
            execution_hint_json=mapping.execution_hint_json,
            experiment_id=mapping.experiment_id,
            recommendation_result_id=mapping.recommendation_result_id,
            source=mapping.source,
            status=mapping.status,
        )
        for mapping in mappings
    ]


def parse_recommendations(recommendations_json: dict[str, Any]) -> list[RecommendedAction]:
    raw_recommendations = recommendations_json.get("recommendations", [])
    try:
        return [
            RecommendedAction.model_validate(recommendation)
            for recommendation in raw_recommendations
        ]
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="stored recommendations_json is invalid",
        ) from exc


def validate_manual_approval_action(recommendation: RecommendedAction) -> None:
    if (
        recommendation.action_id in MANUAL_REVIEW_ACTION_IDS
        or recommendation.action_type in MANUAL_REVIEW_ACTION_TYPES
        or recommendation.experiment is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="action cannot create an experiment or segment mapping",
        )


def get_or_create_manual_experiment(
    *,
    repository: PostgresRepository,
    result: Any,
    recommendation: RecommendedAction,
) -> Any:
    existing = repository.get_experiment_by_recommendation_action(
        recommendation_result_id=result.id,
        action_id=recommendation.action_id,
    )
    if existing is not None:
        return existing

    experiment_config = recommendation.experiment
    return repository.create_experiment(
        project_id=result.project_id,
        recommendation_result_id=result.id,
        segment_json=result.segment_json,
        action_id=recommendation.action_id,
        action_type=recommendation.action_type,
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
    recommendation: RecommendedAction,
    experiment_id: int,
) -> Any:
    existing = repository.get_segment_ad_mapping_by_recommendation_action(
        recommendation_result_id=result.id,
        action_id=recommendation.action_id,
    )
    if existing is not None:
        return existing

    return repository.create_segment_ad_mapping(
        project_id=result.project_id,
        recommendation_result_id=result.id,
        segment_json=result.segment_json,
        experiment_id=experiment_id,
        action_id=recommendation.action_id,
        action_type=recommendation.action_type,
        execution_hint_json=recommendation.execution_hint,
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
