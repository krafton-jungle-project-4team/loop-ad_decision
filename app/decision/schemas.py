from enum import StrEnum
from decimal import Decimal
from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)


class Channel(StrEnum):
    EMAIL = "email"
    SMS = "sms"
    ONSITE_BANNER = "onsite_banner"


class GoalMetric(StrEnum):
    INFLOW_RATE = "inflow_rate"
    BOOKING_CONVERSION_RATE = "booking_conversion_rate"
    FUNNEL_STEP_RATE = "funnel_step_rate"


class GoalBasis(StrEnum):
    PROMOTION_AVERAGE = "promotion_average"
    ALL_SEGMENTS = "all_segments"


class PromotionRunStatus(StrEnum):
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    EVALUATING = "evaluating"
    PARTIAL_GOAL_MET = "partial_goal_met"
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    INSUFFICIENT_DATA = "insufficient_data"
    STOPPED = "stopped"


class AdExperimentStatus(StrEnum):
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    EVALUATING = "evaluating"
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    INSUFFICIENT_DATA = "insufficient_data"
    STOPPED = "stopped"


class PromotionEvaluationStatus(StrEnum):
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    PARTIAL_GOAL_MET = "partial_goal_met"
    INSUFFICIENT_DATA = "insufficient_data"


class AssignmentSource(StrEnum):
    DECISION_BATCH = "decision_batch"
    FALLBACK = "fallback"
    MANUAL = "manual"
    FIXTURE = "fixture"


class ContentApprovalMode(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class NextLoopPreparationStatus(StrEnum):
    AWAITING_CONTENT_APPROVAL = "awaiting_content_approval"
    ACTIVATED = "activated"
    REJECTED = "rejected"


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    analysis_id: str | None = Field(default=None, min_length=1)
    generation_id: str | None = Field(default=None, min_length=1)
    segment_ids: list[str] | None = None
    loop_count: int = Field(default=1, ge=1)
    next_loop_preparation_id: str | None = Field(default=None, min_length=1)

    @field_validator("segment_ids")
    @classmethod
    def validate_segment_ids(cls, segment_ids: list[str] | None) -> list[str] | None:
        return segment_ids


class AdExperimentCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ad_experiment_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    segment_name: str | None = None
    content_id: str = Field(min_length=1)
    content_option_id: str = Field(min_length=1)
    channel: Channel
    loop_count: int = Field(ge=1)
    status: AdExperimentStatus
    is_fallback: bool


class RunCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    promotion_run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    analysis_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    loop_count: int = Field(ge=1)
    status: PromotionRunStatus
    goal_snapshot_json: dict[str, Any]
    segment_ids: list[str]
    ad_experiments: list[AdExperimentCreateResponse]


class SegmentAssignmentBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_ids: list[str] | None = None
    eligible_user_limit: int | None = Field(default=None, ge=1)
    vector_version: str = Field(default="v1", min_length=1)
    expires_in_days: int | None = Field(default=None, ge=1)


class SegmentAssignmentBuildResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    promotion_run_id: str = Field(min_length=1)
    matching_mode: Literal["pgvector_hnsw_rerank"] = "pgvector_hnsw_rerank"
    vector_version: str = Field(min_length=1)
    ann_candidate_limit: int = Field(ge=1)
    ann_candidate_count: int = Field(ge=0)
    exact_reranked_pair_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    processed_user_count: int = Field(ge=0)
    assignment_count: int = Field(ge=0)
    run_assignment_count: int = Field(ge=0)
    run_has_fallback: bool
    run_fallback_count: int = Field(ge=0)
    insert_conflict_count: int = Field(ge=0)
    segment_assignment_counts: dict[str, int]
    batch_has_fallback: bool
    fallback_count: int = Field(ge=0)
    fallback_rate: float | None = Field(default=None, ge=0, le=1)
    fallback_reason_counts: dict[str, int]
    below_threshold_fallback_count: int = Field(ge=0)
    no_candidate_fallback_count: int = Field(ge=0)
    invalid_user_vector_fallback_count: int = Field(ge=0)
    similarity_score_buckets: dict[str, int] = Field(
        description=(
            "Buckets of persisted similarity scores after raw cosine similarity "
            "is clamped to the Data Contract range [0, 1]."
        )
    )
    ann_underfilled_user_count: int = Field(ge=0)
    exact_rescue_user_count: int = Field(ge=0)
    ann_applied: bool
    ann_not_applied_reason: Literal[
        "no_users_to_match",
        "no_valid_user_vectors",
    ] | None = None
    skipped_existing_count: int = Field(ge=0)
    insufficient_segment_count: Literal[0] = Field(
        default=0,
        deprecated=True,
        description=(
            "Deprecated: assignment no longer determines insufficient_data. "
            "This field is always zero."
        ),
    )
    completion_scope: Literal["current_request"] = "current_request"
    assignment_mode: Literal["live_keyset", "explicit_user_ids"]
    input_stability: Literal["not_snapshotted"] = "not_snapshotted"
    status: Literal["completed"] = "completed"


class AdExperimentEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AdExperimentEvaluateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evaluation_id: str = Field(min_length=1)
    ad_experiment_id: str = Field(min_length=1)
    promotion_run_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    metric: GoalMetric
    target_value: Decimal
    actual_value: Decimal
    numerator_count: int = Field(ge=0)
    denominator_count: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    basis: GoalBasis
    status: PromotionEvaluationStatus
    next_loop_required: bool
    feedback: str | None = None


class PromotionRunEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PromotionRunAdExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ad_experiment_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    actual_value: Decimal
    status: PromotionEvaluationStatus


class PromotionRunEvaluateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    promotion_run_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    status: PromotionRunStatus
    ad_experiment_results: list[PromotionRunAdExperimentResult]
    next_loop_required: bool
    failed_segment_ids: list[str]
    failed_ad_experiment_ids: list[str]


class NextLoopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    failed_segment_ids: list[str] = Field(default_factory=list)
    failed_ad_experiment_ids: list[str] = Field(default_factory=list)
    operator_instruction: str | None = None
    content_approval_mode: ContentApprovalMode = ContentApprovalMode.AUTOMATIC


class NextLoopResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    _ADDITIVE_MANUAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "status",
            "content_approval_required",
            "next_loop_preparation_id",
            "pending_content_ids",
        }
    )

    status: NextLoopPreparationStatus | None = None
    content_approval_required: bool = False
    next_loop_preparation_id: str | None = Field(default=None, min_length=1)
    previous_promotion_run_id: str = Field(min_length=1)
    next_promotion_run_id: str | None = None
    promotion_id: str = Field(min_length=1)
    loop_count: int = Field(ge=1)
    segment_ids: list[str]
    next_analysis_id: str | None = None
    next_generation_id: str | None = None
    pending_content_ids: list[str] = Field(default_factory=list)
    next_ad_experiments: list[AdExperimentCreateResponse]

    @model_serializer(mode="wrap")
    def _serialize_explicit_manual_fields(
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        serialized = handler(self)
        for field_name in self._ADDITIVE_MANUAL_FIELDS - self.model_fields_set:
            serialized.pop(field_name, None)
        return serialized
