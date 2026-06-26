import hashlib
import json
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.models import (
    AutomationPolicy,
    Experiment,
    RecommendationResult,
    SegmentAdMapping,
)

JsonObject = dict[str, Any]


def canonical_segment_json(segment: JsonObject | None) -> str:
    cleaned_segment = {
        key: value
        for key, value in (segment or {}).items()
        if value is not None
    }
    return json.dumps(
        cleaned_segment,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_segment_hash(segment: JsonObject | None) -> str:
    return hashlib.sha256(canonical_segment_json(segment).encode("utf-8")).hexdigest()


class PostgresRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def get_automation_policy(self, project_id: str) -> AutomationPolicy | None:
        return self.session.scalar(
            select(AutomationPolicy).where(AutomationPolicy.project_id == project_id)
        )

    def upsert_automation_policy(
        self,
        project_id: str,
        values: JsonObject,
    ) -> AutomationPolicy:
        policy = self.get_automation_policy(project_id)
        if policy is None:
            policy = AutomationPolicy(project_id=project_id, **values)
            self.session.add(policy)
        else:
            for key, value in values.items():
                setattr(policy, key, value)
        self.session.flush()
        return policy

    def create_recommendation_result(
        self,
        *,
        project_id: str,
        window_start: datetime,
        window_end: datetime,
        status: str,
        segment_json: JsonObject | None = None,
        baseline_start: datetime | None = None,
        baseline_end: datetime | None = None,
        anomaly_json: JsonObject | None = None,
        root_causes_json: JsonObject | None = None,
        recommendations_json: JsonObject | None = None,
        policy_decision_json: JsonObject | None = None,
    ) -> RecommendationResult:
        result = RecommendationResult(
            project_id=project_id,
            window_start=window_start,
            window_end=window_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            segment_json=segment_json or {},
            segment_hash=build_segment_hash(segment_json),
            status=status,
            anomaly_json=anomaly_json or {},
            root_causes_json=root_causes_json or {},
            recommendations_json=recommendations_json or {},
            policy_decision_json=policy_decision_json or {},
        )
        self.session.add(result)
        self.session.flush()
        return result

    def get_recommendation_result(self, recommendation_result_id: int) -> RecommendationResult | None:
        return self.session.get(RecommendationResult, recommendation_result_id)

    def list_recommendation_results(
        self,
        *,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RecommendationResult]:
        statement = select(RecommendationResult).order_by(RecommendationResult.created_at.desc())
        if project_id is not None:
            statement = statement.where(RecommendationResult.project_id == project_id)
        if status is not None:
            statement = statement.where(RecommendationResult.status == status)
        return list(self.session.scalars(statement.limit(limit)))

    def update_recommendation_result(
        self,
        recommendation_result_id: int,
        values: JsonObject,
    ) -> RecommendationResult | None:
        result = self.get_recommendation_result(recommendation_result_id)
        if result is None:
            return None
        for key, value in values.items():
            setattr(result, key, value)
        self.session.flush()
        return result

    def create_experiment(
        self,
        *,
        project_id: str,
        recommendation_result_id: int,
        action_id: str,
        action_type: str,
        status: str,
        segment_json: JsonObject | None = None,
        traffic_split_json: JsonObject | None = None,
        primary_metric: str | None = None,
        guardrail_metrics_json: list[Any] | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> Experiment:
        experiment = Experiment(
            project_id=project_id,
            recommendation_result_id=recommendation_result_id,
            segment_json=segment_json or {},
            segment_hash=build_segment_hash(segment_json),
            action_id=action_id,
            action_type=action_type,
            status=status,
            traffic_split_json=traffic_split_json or {},
            primary_metric=primary_metric,
            guardrail_metrics_json=guardrail_metrics_json or [],
            started_at=started_at,
            ended_at=ended_at,
        )
        self.session.add(experiment)
        self.session.flush()
        return experiment

    def get_experiment(self, experiment_id: int) -> Experiment | None:
        return self.session.get(Experiment, experiment_id)

    def get_experiment_by_recommendation_action(
        self,
        *,
        recommendation_result_id: int,
        action_id: str,
    ) -> Experiment | None:
        return self.session.scalar(
            select(Experiment)
            .where(Experiment.recommendation_result_id == recommendation_result_id)
            .where(Experiment.action_id == action_id)
        )

    def list_experiments(
        self,
        *,
        project_id: str | None = None,
        recommendation_result_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Experiment]:
        statement = select(Experiment).order_by(Experiment.created_at.desc())
        if project_id is not None:
            statement = statement.where(Experiment.project_id == project_id)
        if recommendation_result_id is not None:
            statement = statement.where(
                Experiment.recommendation_result_id == recommendation_result_id
            )
        if status is not None:
            statement = statement.where(Experiment.status == status)
        return list(self.session.scalars(statement.limit(limit)))

    def update_experiment(self, experiment_id: int, values: JsonObject) -> Experiment | None:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return None
        for key, value in values.items():
            setattr(experiment, key, value)
        self.session.flush()
        return experiment

    def create_segment_ad_mapping(
        self,
        *,
        project_id: str,
        recommendation_result_id: int,
        action_id: str,
        action_type: str,
        status: str,
        source: str,
        segment_json: JsonObject | None = None,
        experiment_id: int | None = None,
        execution_hint_json: JsonObject | None = None,
        expires_at: datetime | None = None,
    ) -> SegmentAdMapping:
        mapping = SegmentAdMapping(
            project_id=project_id,
            segment_json=segment_json or {},
            segment_hash=build_segment_hash(segment_json),
            recommendation_result_id=recommendation_result_id,
            experiment_id=experiment_id,
            action_id=action_id,
            action_type=action_type,
            execution_hint_json=execution_hint_json or {},
            status=status,
            source=source,
            expires_at=expires_at,
        )
        self.session.add(mapping)
        self.session.flush()
        return mapping

    def get_segment_ad_mapping(self, mapping_id: int) -> SegmentAdMapping | None:
        return self.session.get(SegmentAdMapping, mapping_id)

    def get_segment_ad_mapping_by_recommendation_action(
        self,
        *,
        recommendation_result_id: int,
        action_id: str,
    ) -> SegmentAdMapping | None:
        return self.session.scalar(
            select(SegmentAdMapping)
            .where(SegmentAdMapping.recommendation_result_id == recommendation_result_id)
            .where(SegmentAdMapping.action_id == action_id)
        )

    def list_segment_ad_mappings(
        self,
        *,
        project_id: str | None = None,
        recommendation_result_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SegmentAdMapping]:
        statement = select(SegmentAdMapping).order_by(SegmentAdMapping.created_at.desc())
        if project_id is not None:
            statement = statement.where(SegmentAdMapping.project_id == project_id)
        if recommendation_result_id is not None:
            statement = statement.where(
                SegmentAdMapping.recommendation_result_id == recommendation_result_id
            )
        if status is not None:
            statement = statement.where(SegmentAdMapping.status == status)
        return list(self.session.scalars(statement.limit(limit)))

    def list_active_segment_ad_mappings(self, project_id: str) -> list[SegmentAdMapping]:
        return list(
            self.session.scalars(
                select(SegmentAdMapping)
                .where(SegmentAdMapping.project_id == project_id)
                .where(SegmentAdMapping.status == "active")
                .order_by(SegmentAdMapping.created_at.desc())
            )
        )

    def update_segment_ad_mapping(
        self,
        mapping_id: int,
        values: JsonObject,
    ) -> SegmentAdMapping | None:
        mapping = self.get_segment_ad_mapping(mapping_id)
        if mapping is None:
            return None
        for key, value in values.items():
            setattr(mapping, key, value)
        self.session.flush()
        return mapping
