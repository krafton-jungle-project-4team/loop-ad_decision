from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
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


class AutomationPolicy(TimestampMixin, Base):
    __tablename__ = "automation_policies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
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


class RecommendationResult(TimestampMixin, Base):
    __tablename__ = "recommendation_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    baseline_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    baseline_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    anomaly_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    root_causes_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    recommendations_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    policy_decision_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)


class AnalysisJob(TimestampMixin, Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        Index("idx_analysis_jobs_project_id", "project_id"),
        Index("idx_analysis_jobs_status", "status"),
        Index("idx_analysis_jobs_recommendation_result", "recommendation_result_id"),
        Index("idx_analysis_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), nullable=False)
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
        UniqueConstraint(
            "recommendation_result_id",
            "action_id",
            name="uq_experiments_recommendation_action",
        ),
        Index("ix_experiments_project_status", "project_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    recommendation_result_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    traffic_split_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    primary_metric: Mapped[str | None] = mapped_column(String(128), nullable=True)
    guardrail_metrics_json: Mapped[JsonArray] = mapped_column(JSONB, nullable=False, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SegmentAdMapping(TimestampMixin, Base):
    __tablename__ = "segment_ad_mappings"
    __table_args__ = (
        UniqueConstraint(
            "recommendation_result_id",
            "action_id",
            name="uq_segment_ad_mappings_recommendation_action",
        ),
        Index("ix_segment_ad_mappings_project_status", "project_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    segment_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    segment_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    recommendation_result_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_hint_json: Mapped[JsonObject] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
