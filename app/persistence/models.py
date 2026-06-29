from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.persistence.job_statuses import ANALYSIS_JOB_STATUS_QUEUED


class Base(DeclarativeBase):
    pass


JsonObject = dict[str, Any]
JsonArray = list[Any]


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sdk_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class DashboardUser(TimestampMixin, Base):
    __tablename__ = "dashboard_users"
    __table_args__ = (
        UniqueConstraint("project_id", "email", name="dashboard_users_project_id_email_key"),
        Index("idx_dashboard_users_project", "project_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="admin")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class UserProfile(TimestampMixin, Base):
    __tablename__ = "user_profiles"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "external_user_id",
            name="user_profiles_project_id_external_user_id_key",
        ),
        Index("idx_user_profiles_project_user", "project_id", "external_user_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    external_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    age_group: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    membership_level: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attributes_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)


class Segment(TimestampMixin, Base):
    __tablename__ = "segments"
    __table_args__ = (
        UniqueConstraint("project_id", "segment_hash", name="segments_project_id_segment_hash_key"),
        Index("idx_segments_project_status", "project_id", "status"),
        Index("gin_segments_conditions_json", "conditions_json", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    conditions_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class Campaign(TimestampMixin, Base):
    __tablename__ = "campaigns"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "external_campaign_id",
            name="campaigns_project_id_external_campaign_id_key",
        ),
        Index("idx_campaigns_project_status", "project_id", "status"),
        Index("idx_campaigns_project_channel", "project_id", "channel"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    external_campaign_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    channel: Mapped[str | None] = mapped_column(String(64), nullable=True)
    goal: Mapped[str | None] = mapped_column(String(64), nullable=True)
    budget: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)


class Coupon(TimestampMixin, Base):
    __tablename__ = "coupons"
    __table_args__ = (
        UniqueConstraint("project_id", "code", name="coupons_project_id_code_key"),
        Index("idx_coupons_project_status", "project_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    code: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    discount_type: Mapped[str] = mapped_column(String(64), nullable=False)
    discount_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    discount_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    max_discount_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    budget: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    starts_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)


class AdCreative(TimestampMixin, Base):
    __tablename__ = "ad_creatives"
    __table_args__ = (
        Index("idx_ad_creatives_project_status", "project_id", "status"),
        Index("idx_ad_creatives_project_action_status", "project_id", "action_id", "status"),
        Index("idx_ad_creatives_campaign", "campaign_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    campaign_id: Mapped[int | None] = mapped_column(
        ForeignKey("campaigns.id", ondelete="SET NULL"),
        nullable=True,
    )
    coupon_id: Mapped[int | None] = mapped_column(
        ForeignKey("coupons.id", ondelete="SET NULL"),
        nullable=True,
    )
    action_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    creative_type: Mapped[str] = mapped_column(String(64), nullable=False, default="banner")
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    landing_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class ActionCatalog(TimestampMixin, Base):
    __tablename__ = "action_catalog"
    __table_args__ = (
        Index("idx_action_catalog_type_status", "action_type", "status"),
    )

    action_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    base_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    primary_metric: Mapped[str | None] = mapped_column(String(128), nullable=True)
    expected_impact: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_hint_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class AutomationPolicy(TimestampMixin, Base):
    __tablename__ = "automation_policies"
    __table_args__ = (
        Index("idx_automation_policies_project", "project_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    auto_execute_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    allowed_action_ids: Mapped[JsonArray] = mapped_column(JSONB, nullable=False, default=list)
    allowed_action_types: Mapped[JsonArray] = mapped_column(JSONB, nullable=False, default=list)
    blocked_action_ids: Mapped[JsonArray] = mapped_column(JSONB, nullable=False, default=list)
    max_experiment_traffic_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    min_priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_discount_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_daily_coupon_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_message_per_user_per_day: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    stop_loss_relative_drop: Mapped[float | None] = mapped_column(Float, nullable=True)


class BanditPolicy(TimestampMixin, Base):
    __tablename__ = "bandit_policies"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "segment_hash",
            "objective_metric",
            name="bandit_policies_project_id_segment_hash_objective_metric_key",
        ),
        Index("idx_bandit_policies_project_status", "project_id", "status"),
        Index("idx_bandit_policies_segment_hash", "segment_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    objective_metric: Mapped[str] = mapped_column(String(128), nullable=False, default="purchase_rate")
    reward_event_name: Mapped[str] = mapped_column(String(128), nullable=False, default="purchase")
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False, default="thompson_sampling")
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active")
    min_samples_per_arm: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    exploration_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    config_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)


class BanditArm(TimestampMixin, Base):
    __tablename__ = "bandit_arms"
    __table_args__ = (
        UniqueConstraint("bandit_policy_id", "action_id", name="bandit_arms_bandit_policy_id_action_id_key"),
        CheckConstraint("alpha > 0", name="bandit_arms_alpha_check"),
        CheckConstraint("beta > 0", name="bandit_arms_beta_check"),
        CheckConstraint("impressions >= 0", name="bandit_arms_impressions_check"),
        CheckConstraint("conversions >= 0", name="bandit_arms_conversions_check"),
        CheckConstraint("failures >= 0", name="bandit_arms_failures_check"),
        Index("idx_bandit_arms_policy_status", "bandit_policy_id", "status"),
        Index("idx_bandit_arms_project_action", "project_id", "action_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    bandit_policy_id: Mapped[int] = mapped_column(
        ForeignKey("bandit_policies.id", ondelete="CASCADE"),
        nullable=False,
    )
    action_id: Mapped[str] = mapped_column(
        ForeignKey("action_catalog.action_id", ondelete="RESTRICT"),
        nullable=False,
    )
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active")
    alpha: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    beta: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    impressions: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    conversions: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    failures: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    last_sampled_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_selected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_reward_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)


class RecommendationResult(TimestampMixin, Base):
    __tablename__ = "recommendation_results"
    __table_args__ = (
        Index("idx_recommendation_results_project_status", "project_id", "status"),
        Index("idx_recommendation_results_project_created", "project_id", "created_at"),
        Index("idx_recommendation_results_segment_hash", "segment_hash"),
        Index("gin_recommendation_results_segment_json", "segment_json", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    baseline_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    baseline_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    anomaly_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    root_causes_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    recommendations_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    policy_decision_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    bandit_decision_summary_json: Mapped[JsonObject] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    summary_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class RecommendationAction(TimestampMixin, Base):
    __tablename__ = "recommendation_actions"
    __table_args__ = (
        UniqueConstraint(
            "recommendation_result_id",
            "action_id",
            name="recommendation_actions_recommendation_result_id_action_id_key",
        ),
        Index("idx_recommendation_actions_project_status", "project_id", "status"),
        Index("idx_recommendation_actions_result", "recommendation_result_id"),
        Index("idx_recommendation_actions_result_status", "recommendation_result_id", "status"),
        Index("idx_recommendation_actions_action", "action_id"),
        Index("idx_recommendation_actions_policy_status", "policy_status"),
        Index("idx_recommendation_actions_bandit_arm", "bandit_arm_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    recommendation_result_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    action_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    priority_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_impact: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    triggered_by_json: Mapped[JsonArray] = mapped_column(JSONB, nullable=False, default=list)
    execution_hint_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    experiment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    policy_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    policy_reasons_json: Mapped[JsonArray] = mapped_column(JSONB, nullable=False, default=list)
    policy_decision_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    selected_by_strategy: Mapped[str] = mapped_column(String(64), nullable=False, default="rule_based")
    bandit_policy_id: Mapped[int | None] = mapped_column(
        ForeignKey("bandit_policies.id", ondelete="SET NULL"),
        nullable=True,
    )
    bandit_arm_id: Mapped[int | None] = mapped_column(
        ForeignKey("bandit_arms.id", ondelete="SET NULL"),
        nullable=True,
    )
    sampled_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending_review")
    auto_executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class AnalysisJob(TimestampMixin, Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        Index("idx_analysis_jobs_project_id", "project_id"),
        Index("idx_analysis_jobs_status", "status"),
        Index("idx_analysis_jobs_recommendation_result", "recommendation_result_id"),
        Index("idx_analysis_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ANALYSIS_JOB_STATUS_QUEUED,
        server_default=ANALYSIS_JOB_STATUS_QUEUED,
    )
    request_json: Mapped[JsonObject] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    recommendation_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendation_results.id", ondelete="SET NULL"),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    max_attempts: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Experiment(TimestampMixin, Base):
    __tablename__ = "experiments"
    __table_args__ = (
        UniqueConstraint("recommendation_action_id", name="uq_experiments_recommendation_action"),
        Index("idx_experiments_project_status", "project_id", "status"),
        Index("idx_experiments_recommendation_result", "recommendation_result_id"),
        Index("idx_experiments_recommendation_action", "recommendation_action_id"),
        Index("idx_experiments_segment_hash", "segment_hash"),
        Index("idx_experiments_bandit_arm", "bandit_arm_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    recommendation_result_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    recommendation_action_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_actions.id", ondelete="CASCADE"),
        nullable=False,
    )
    bandit_policy_id: Mapped[int | None] = mapped_column(
        ForeignKey("bandit_policies.id", ondelete="SET NULL"),
        nullable=True,
    )
    bandit_arm_id: Mapped[int | None] = mapped_column(
        ForeignKey("bandit_arms.id", ondelete="SET NULL"),
        nullable=True,
    )
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    action_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    traffic_split_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    primary_metric: Mapped[str | None] = mapped_column(String(128), nullable=True)
    guardrail_metrics_json: Mapped[JsonArray] = mapped_column(JSONB, nullable=False, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BanditDecision(TimestampMixin, Base):
    __tablename__ = "bandit_decisions"
    __table_args__ = (
        Index("idx_bandit_decisions_policy_created", "bandit_policy_id", "created_at"),
        Index("idx_bandit_decisions_project_created", "project_id", "created_at"),
        Index("idx_bandit_decisions_recommendation_action", "recommendation_action_id"),
        Index("idx_bandit_decisions_experiment", "experiment_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    bandit_policy_id: Mapped[int] = mapped_column(
        ForeignKey("bandit_policies.id", ondelete="CASCADE"),
        nullable=False,
    )
    selected_arm_id: Mapped[int] = mapped_column(
        ForeignKey("bandit_arms.id", ondelete="CASCADE"),
        nullable=False,
    )
    recommendation_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendation_results.id", ondelete="SET NULL"),
        nullable=True,
    )
    recommendation_action_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendation_actions.id", ondelete="SET NULL"),
        nullable=True,
    )
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True,
    )
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    sampled_values_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    selected_action_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_sampled_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    reward_observed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reward_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    reward_event_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reward_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SegmentAdMapping(TimestampMixin, Base):
    __tablename__ = "segment_ad_mappings"
    __table_args__ = (
        UniqueConstraint("recommendation_action_id", name="uq_segment_ad_mappings_recommendation_action"),
        Index("idx_segment_ad_mappings_project_status", "project_id", "status"),
        Index("idx_segment_ad_mappings_project_segment_status", "project_id", "segment_hash", "status"),
        Index("idx_segment_ad_mappings_recommendation_result", "recommendation_result_id"),
        Index("idx_segment_ad_mappings_recommendation_action", "recommendation_action_id"),
        Index("idx_segment_ad_mappings_experiment", "experiment_id"),
        Index("idx_segment_ad_mappings_bandit_arm", "bandit_arm_id"),
        Index("idx_segment_ad_mappings_campaign", "campaign_id"),
        Index("idx_segment_ad_mappings_creative", "creative_id"),
        Index("gin_segment_ad_mappings_segment_json", "segment_json", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    recommendation_result_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    recommendation_action_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_actions.id", ondelete="CASCADE"),
        nullable=False,
    )
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True,
    )
    bandit_policy_id: Mapped[int | None] = mapped_column(
        ForeignKey("bandit_policies.id", ondelete="SET NULL"),
        nullable=True,
    )
    bandit_arm_id: Mapped[int | None] = mapped_column(
        ForeignKey("bandit_arms.id", ondelete="SET NULL"),
        nullable=True,
    )
    bandit_decision_id: Mapped[int | None] = mapped_column(
        ForeignKey("bandit_decisions.id", ondelete="SET NULL"),
        nullable=True,
    )
    campaign_id: Mapped[int | None] = mapped_column(
        ForeignKey("campaigns.id", ondelete="SET NULL"),
        nullable=True,
    )
    creative_id: Mapped[int | None] = mapped_column(
        ForeignKey("ad_creatives.id", ondelete="SET NULL"),
        nullable=True,
    )
    coupon_id: Mapped[int | None] = mapped_column(
        ForeignKey("coupons.id", ondelete="SET NULL"),
        nullable=True,
    )
    action_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_hint_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
