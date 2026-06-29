import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.job_statuses import (
    ANALYSIS_JOB_STATUS_DONE,
    ANALYSIS_JOB_STATUS_FAILED,
    ANALYSIS_JOB_STATUS_QUEUED,
    ANALYSIS_JOB_STATUS_RUNNING,
)
from app.persistence.models import (
    AdCreative,
    AnalysisJob,
    AutomationPolicy,
    BanditArm,
    BanditDecision,
    BanditPolicy,
    Experiment,
    RecommendationAction,
    RecommendationResult,
    SegmentAdMapping,
)

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ActiveSegmentAdMappingRow:
    mapping: SegmentAdMapping
    creative: AdCreative | None = None


@dataclass(frozen=True)
class BanditArmServingRow:
    arm: BanditArm
    mapping: SegmentAdMapping | None = None
    creative: AdCreative | None = None


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
        bandit_decision_summary_json: JsonObject | None = None,
        summary_message: str | None = None,
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
            bandit_decision_summary_json=bandit_decision_summary_json or {},
            summary_message=summary_message,
        )
        self.session.add(result)
        self.session.flush()
        return result

    def create_recommendation_action(
        self,
        *,
        project_id: str,
        recommendation_result_id: int,
        action_id: str,
        action_type: str,
        title: str | None = None,
        description: str | None = None,
        target_step: str | None = None,
        priority_score: float | None = None,
        expected_impact: str | None = None,
        rationale: str | None = None,
        triggered_by_json: list[Any] | None = None,
        execution_hint_json: JsonObject | None = None,
        experiment_json: JsonObject | None = None,
        policy_status: str | None = None,
        policy_reasons_json: list[Any] | None = None,
        policy_decision_json: JsonObject | None = None,
        selected_by_strategy: str = "rule_based",
        bandit_policy_id: int | None = None,
        bandit_arm_id: int | None = None,
        sampled_value: float | None = None,
        status: str = "pending_review",
        auto_executed_at: datetime | None = None,
    ) -> RecommendationAction:
        action = RecommendationAction(
            project_id=project_id,
            recommendation_result_id=recommendation_result_id,
            action_id=action_id,
            action_type=action_type,
            title=title,
            description=description,
            target_step=target_step,
            priority_score=priority_score,
            expected_impact=expected_impact,
            rationale=rationale,
            triggered_by_json=triggered_by_json or [],
            execution_hint_json=execution_hint_json or {},
            experiment_json=experiment_json or {},
            policy_status=policy_status,
            policy_reasons_json=policy_reasons_json or [],
            policy_decision_json=policy_decision_json or {},
            selected_by_strategy=selected_by_strategy,
            bandit_policy_id=bandit_policy_id,
            bandit_arm_id=bandit_arm_id,
            sampled_value=sampled_value,
            status=status,
            auto_executed_at=auto_executed_at,
        )
        self.session.add(action)
        self.session.flush()
        return action

    def get_recommendation_action(self, recommendation_action_id: int) -> RecommendationAction | None:
        return self.session.get(RecommendationAction, recommendation_action_id)

    def list_recommendation_actions(
        self,
        *,
        recommendation_result_id: int | None = None,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[RecommendationAction]:
        statement = select(RecommendationAction).order_by(RecommendationAction.created_at)
        if recommendation_result_id is not None:
            statement = statement.where(
                RecommendationAction.recommendation_result_id == recommendation_result_id
            )
        if project_id is not None:
            statement = statement.where(RecommendationAction.project_id == project_id)
        if status is not None:
            statement = statement.where(RecommendationAction.status == status)
        return list(self.session.scalars(statement.limit(limit)))

    def get_recommendation_action_by_result_action(
        self,
        *,
        recommendation_result_id: int,
        action_id: str,
    ) -> RecommendationAction | None:
        return self.session.scalar(
            select(RecommendationAction)
            .where(RecommendationAction.recommendation_result_id == recommendation_result_id)
            .where(RecommendationAction.action_id == action_id)
        )

    def update_recommendation_action(
        self,
        recommendation_action_id: int,
        values: JsonObject,
    ) -> RecommendationAction | None:
        action = self.get_recommendation_action(recommendation_action_id)
        if action is None:
            return None
        for key, value in values.items():
            setattr(action, key, value)
        self.session.flush()
        return action

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

    def create_analysis_job(
        self,
        project_id: str,
        request_json: JsonObject,
        status: str = ANALYSIS_JOB_STATUS_QUEUED,
    ) -> AnalysisJob:
        job = AnalysisJob(
            project_id=project_id,
            status=status,
            request_json=request_json,
        )
        self.session.add(job)
        self.session.flush()
        return job

    def get_analysis_job(self, job_id: int) -> AnalysisJob | None:
        return self.session.get(AnalysisJob, job_id)

    def claim_next_analysis_job(self) -> AnalysisJob | None:
        statement = (
            select(AnalysisJob)
            .where(AnalysisJob.status == ANALYSIS_JOB_STATUS_QUEUED)
            .order_by(AnalysisJob.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        job = self.session.scalar(statement)
        if job is None:
            return None

        now = datetime.now(UTC)
        job.status = ANALYSIS_JOB_STATUS_RUNNING
        job.attempts = (job.attempts or 0) + 1
        job.locked_at = now
        job.started_at = now
        job.error_message = None
        self.session.flush()
        return job

    def mark_analysis_job_done(
        self,
        job_id: int,
        recommendation_result_id: int,
    ) -> AnalysisJob | None:
        job = self.get_analysis_job(job_id)
        if job is None:
            return None

        job.status = ANALYSIS_JOB_STATUS_DONE
        job.recommendation_result_id = recommendation_result_id
        job.error_message = None
        job.finished_at = datetime.now(UTC)
        self.session.flush()
        return job

    def mark_analysis_job_failed(
        self,
        job_id: int,
        error_message: str,
    ) -> AnalysisJob | None:
        job = self.get_analysis_job(job_id)
        if job is None:
            return None

        job.status = ANALYSIS_JOB_STATUS_FAILED
        job.error_message = error_message
        job.finished_at = datetime.now(UTC)
        self.session.flush()
        return job

    def create_experiment(
        self,
        *,
        project_id: str,
        recommendation_result_id: int,
        recommendation_action_id: int,
        action_id: str,
        action_type: str,
        status: str,
        bandit_policy_id: int | None = None,
        bandit_arm_id: int | None = None,
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
            recommendation_action_id=recommendation_action_id,
            bandit_policy_id=bandit_policy_id,
            bandit_arm_id=bandit_arm_id,
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
        recommendation_action_id: int,
    ) -> Experiment | None:
        return self.session.scalar(
            select(Experiment)
            .where(Experiment.recommendation_action_id == recommendation_action_id)
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
        recommendation_action_id: int,
        action_id: str,
        action_type: str,
        status: str,
        source: str,
        segment_json: JsonObject | None = None,
        experiment_id: int | None = None,
        bandit_policy_id: int | None = None,
        bandit_arm_id: int | None = None,
        bandit_decision_id: int | None = None,
        campaign_id: int | None = None,
        creative_id: int | None = None,
        coupon_id: int | None = None,
        execution_hint_json: JsonObject | None = None,
        expires_at: datetime | None = None,
    ) -> SegmentAdMapping:
        mapping = SegmentAdMapping(
            project_id=project_id,
            segment_json=segment_json or {},
            segment_hash=build_segment_hash(segment_json),
            recommendation_result_id=recommendation_result_id,
            recommendation_action_id=recommendation_action_id,
            experiment_id=experiment_id,
            bandit_policy_id=bandit_policy_id,
            bandit_arm_id=bandit_arm_id,
            bandit_decision_id=bandit_decision_id,
            campaign_id=campaign_id,
            creative_id=creative_id,
            coupon_id=coupon_id,
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
        recommendation_action_id: int,
    ) -> SegmentAdMapping | None:
        return self.session.scalar(
            select(SegmentAdMapping)
            .where(SegmentAdMapping.recommendation_action_id == recommendation_action_id)
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

    def list_active_segment_ad_mappings_with_creatives(
        self,
        project_id: str,
    ) -> list[ActiveSegmentAdMappingRow]:
        rows = self.session.execute(
            select(SegmentAdMapping, AdCreative)
            .outerjoin(AdCreative, SegmentAdMapping.creative_id == AdCreative.id)
            .where(SegmentAdMapping.project_id == project_id)
            .where(SegmentAdMapping.status == "active")
            .order_by(SegmentAdMapping.created_at.desc())
        )
        return [
            ActiveSegmentAdMappingRow(mapping=mapping, creative=creative)
            for mapping, creative in rows
        ]

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

    def get_bandit_policy(self, bandit_policy_id: int) -> BanditPolicy | None:
        return self.session.get(BanditPolicy, bandit_policy_id)

    def get_bandit_arm(self, bandit_arm_id: int) -> BanditArm | None:
        return self.session.get(BanditArm, bandit_arm_id)

    def list_bandit_arms_by_policy(
        self,
        bandit_policy_id: int,
        *,
        status: str | None = "active",
    ) -> list[BanditArm]:
        statement = (
            select(BanditArm)
            .where(BanditArm.bandit_policy_id == bandit_policy_id)
            .order_by(BanditArm.id)
        )
        if status is not None:
            statement = statement.where(BanditArm.status == status)
        return list(self.session.scalars(statement))

    def list_bandit_arm_serving_rows(
        self,
        bandit_policy_id: int,
    ) -> list[BanditArmServingRow]:
        arms = self.list_bandit_arms_by_policy(bandit_policy_id)
        rows: list[BanditArmServingRow] = []
        for arm in arms:
            result = self.session.execute(
                select(SegmentAdMapping, AdCreative)
                .outerjoin(AdCreative, SegmentAdMapping.creative_id == AdCreative.id)
                .where(SegmentAdMapping.bandit_policy_id == bandit_policy_id)
                .where(SegmentAdMapping.bandit_arm_id == arm.id)
                .where(SegmentAdMapping.status == "active")
                .order_by(SegmentAdMapping.created_at.desc())
                .limit(1)
            ).first()
            if result is None:
                rows.append(BanditArmServingRow(arm=arm))
            else:
                mapping, creative = result
                rows.append(BanditArmServingRow(arm=arm, mapping=mapping, creative=creative))
        return rows

    def get_ad_creative(self, creative_id: int) -> AdCreative | None:
        return self.session.get(AdCreative, creative_id)

    def create_bandit_decision(
        self,
        *,
        project_id: str,
        bandit_policy_id: int,
        selected_arm_id: int,
        selected_action_id: str,
        segment_hash: str,
        segment_json: JsonObject | None = None,
        sampled_values_json: JsonObject | None = None,
        recommendation_result_id: int | None = None,
        recommendation_action_id: int | None = None,
        experiment_id: int | None = None,
        selected_sampled_value: float | None = None,
    ) -> BanditDecision:
        decision = BanditDecision(
            project_id=project_id,
            bandit_policy_id=bandit_policy_id,
            selected_arm_id=selected_arm_id,
            recommendation_result_id=recommendation_result_id,
            recommendation_action_id=recommendation_action_id,
            experiment_id=experiment_id,
            segment_json=segment_json or {},
            segment_hash=segment_hash,
            sampled_values_json=sampled_values_json or {},
            selected_action_id=selected_action_id,
            selected_sampled_value=selected_sampled_value,
        )
        self.session.add(decision)
        self.session.flush()
        return decision
