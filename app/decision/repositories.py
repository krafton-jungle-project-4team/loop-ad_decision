from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
import math
from typing import (
    Any,
    ClassVar,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    TypeVar,
    cast,
)

from psycopg import errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.decision.matcher import (
    HNSW_EF_SEARCH as DEFAULT_HNSW_EF_SEARCH,
    HNSW_MAX_SCAN_TUPLES as DEFAULT_HNSW_MAX_SCAN_TUPLES,
)

EMAIL_CHANNEL = "email"


_T = TypeVar("_T")

NextLoopPreparationStatusValue = Literal[
    "awaiting_content_approval",
    "rejected",
    "activated",
]


class NextLoopPreparationConflictError(Exception):
    def __init__(self, constraint_name: str | None) -> None:
        self.constraint_name = constraint_name
        detail = constraint_name or "unknown constraint"
        super().__init__(f"next-loop preparation unique conflict: {detail}")


class PostgresExecutor(Protocol):
    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        ...

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        ...

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        ...


class ClickHouseClient(Protocol):
    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class PsycopgPostgresExecutor:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, _adapt_params(params))
            return cursor.fetchone()

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, _adapt_params(params))
            return list(cursor.fetchall())

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, _adapt_params(params))


def _adapt_params(
    params: Sequence[Any] | Mapping[str, Any],
) -> Sequence[Any] | Mapping[str, Any]:
    if isinstance(params, Mapping):
        return {key: _adapt_param(value) for key, value in params.items()}
    return tuple(_adapt_param(value) for value in params)


def _adapt_param(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Jsonb(value)
    return value


@dataclass(frozen=True)
class PromotionRecord:
    project_id: str
    campaign_id: str
    promotion_id: str
    channel: str
    goal_metric: str
    goal_target_value: Decimal
    goal_basis: str
    min_sample_size: int
    max_loop_count: int


@dataclass(frozen=True)
class PromotionAnalysisRecord:
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    focus_segment_ids_json: Sequence[str] | None
    operator_instruction: str | None
    input_snapshot_json: Mapping[str, Any]
    profile_summary_json: Mapping[str, Any]
    output_json: Mapping[str, Any] | None
    status: str


@dataclass(frozen=True)
class PromotionTargetSegmentRecord:
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    segment_id: str
    segment_name: str
    segment_vector_id: str | None
    rule_json: Mapping[str, Any]
    profile_json: Mapping[str, Any]
    content_brief_json: Mapping[str, Any]
    data_evidence_json: Mapping[str, Any]
    estimated_size: int
    priority: str | None
    status: str


@dataclass(frozen=True)
class GenerationRunRecord:
    generation_id: str
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    content_option_count: int
    operator_instruction: str | None
    input_json: Mapping[str, Any]
    output_json: Mapping[str, Any] | None
    generation_report_json: Mapping[str, Any]
    status: str


@dataclass(frozen=True)
class NextLoopGenerationAttemptRecord:
    generation_id: str
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    content_option_count: int
    operator_instruction: str | None
    input_json: Mapping[str, Any]
    status: str
    analysis_input_snapshot_json: Mapping[str, Any] | None
    preparation_analysis_id: str | None = None
    preparation_attempt_no: int | None = None
    preparation_status: str | None = None


@dataclass(frozen=True)
class ContentCandidateRecord:
    content_id: str
    content_option_id: str
    generation_id: str
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    segment_id: str
    channel: str
    status: str


@dataclass(frozen=True)
class PromotionRunRecord:
    promotion_run_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    analysis_id: str
    generation_id: str
    loop_count: int
    status: str
    goal_snapshot_json: Mapping[str, Any]
    segment_scope_json: Sequence[str]
    segment_scope_fingerprint: str


@dataclass(frozen=True)
class PromotionRunWrite:
    promotion_run_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    analysis_id: str
    generation_id: str
    loop_count: int
    status: str
    goal_snapshot_json: Mapping[str, Any]
    segment_scope_json: Sequence[str]
    segment_scope_fingerprint: str


@dataclass(frozen=True)
class AdExperimentRecord:
    ad_experiment_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    promotion_run_id: str
    analysis_id: str
    generation_id: str
    segment_id: str
    segment_name: str | None
    content_id: str
    content_option_id: str
    channel: str
    loop_count: int
    status: str
    goal_metric: str
    goal_target_value: Decimal
    goal_basis: str
    parent_ad_experiment_id: str | None = None
    source_evaluation_id: str | None = None


@dataclass(frozen=True)
class AdExperimentWrite:
    ad_experiment_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    promotion_run_id: str
    analysis_id: str
    generation_id: str
    segment_id: str
    segment_name: str | None
    content_id: str
    content_option_id: str
    channel: str
    loop_count: int
    status: str
    goal_metric: str
    goal_target_value: Decimal
    goal_basis: str
    parent_ad_experiment_id: str | None = None
    source_evaluation_id: str | None = None


@dataclass(frozen=True)
class SegmentVectorRecord:
    segment_vector_id: str
    project_id: str
    promotion_id: str | None
    promotion_run_id: str | None
    analysis_id: str | None
    segment_id: str
    vector_dim: int
    vector_values: Any
    vector_version: str
    source: str
    embedding: Any | None = None


@dataclass(frozen=True)
class UserBehaviorVectorRecord:
    project_id: str
    user_id: str
    vector_dim: int
    vector_values: list[float]
    vector_version: str
    source: str


class MetricCountRecord(NamedTuple):
    numerator_count: int
    denominator_count: int


@dataclass(frozen=True)
class UserSegmentAssignmentWrite:
    project_id: str
    promotion_run_id: str
    user_id: str
    segment_id: str
    ad_experiment_id: str
    content_id: str
    content_option_id: str
    similarity_score: Decimal | None
    fallback: bool
    fallback_reason: str | None
    assignment_source: str
    assigned_at: datetime
    expires_at: datetime | None


@dataclass(frozen=True)
class UserSegmentAssignmentInsertRecord:
    user_id: str
    segment_id: str
    fallback: bool
    fallback_reason: str | None
    similarity_score: Decimal | None


@dataclass(frozen=True)
class PromotionEvaluationWrite:
    evaluation_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    promotion_run_id: str
    ad_experiment_id: str | None
    segment_id: str | None
    content_id: str | None
    content_option_id: str | None
    metric: str
    target_value: Decimal
    actual_value: Decimal
    numerator_count: int
    denominator_count: int
    sample_size: int
    basis: str
    status: str
    feedback: str | None
    next_loop_required: bool
    result_json: Mapping[str, Any]


@dataclass(frozen=True)
class PromotionEvaluationRecord:
    evaluation_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    promotion_run_id: str
    ad_experiment_id: str | None
    segment_id: str | None
    content_id: str | None
    content_option_id: str | None
    metric: str
    target_value: Decimal
    actual_value: Decimal
    numerator_count: int
    denominator_count: int
    sample_size: int
    basis: str
    status: str
    feedback: str | None
    next_loop_required: bool
    result_json: Mapping[str, Any]


@dataclass(frozen=True)
class NextLoopPreparationRecord:
    next_loop_preparation_id: str
    source_promotion_run_id: str
    analysis_id: str
    generation_id: str
    attempt_no: int
    failed_segment_ids_json: tuple[str, ...]
    failed_ad_experiment_ids_json: tuple[str, ...]
    source_evaluation_ids_json: tuple[str, ...]
    status: NextLoopPreparationStatusValue
    activated_promotion_run_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class NextLoopPreparationWrite:
    status: ClassVar[Literal["awaiting_content_approval"]] = (
        "awaiting_content_approval"
    )
    next_loop_preparation_id: str
    source_promotion_run_id: str
    analysis_id: str
    generation_id: str
    attempt_no: int
    failed_segment_ids_json: tuple[str, ...]
    failed_ad_experiment_ids_json: tuple[str, ...]
    source_evaluation_ids_json: tuple[str, ...]


class PromotionReader(Protocol):
    def get_by_id(self, promotion_id: str) -> PromotionRecord | None:
        ...


class PromotionAnalysisReader(Protocol):
    def get_by_id(self, analysis_id: str) -> PromotionAnalysisRecord | None:
        ...

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> PromotionAnalysisRecord | None:
        ...


class PromotionTargetSegmentReader(Protocol):
    def list_for_analysis(
        self,
        analysis_id: str,
    ) -> list[PromotionTargetSegmentRecord]:
        ...

    def list_approved_for_analysis(
        self,
        analysis_id: str,
        segment_ids: Sequence[str] | None = None,
    ) -> list[PromotionTargetSegmentRecord]:
        ...

    def update_status(
        self,
        *,
        analysis_id: str,
        segment_id: str,
        status: str,
    ) -> None:
        ...


class GenerationRunReader(Protocol):
    def get_by_id(self, generation_id: str) -> GenerationRunRecord | None:
        ...

    def list_next_loop_generation_attempts(
        self,
        source_promotion_run_id: str,
    ) -> list[NextLoopGenerationAttemptRecord]:
        ...

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> GenerationRunRecord | None:
        ...


class ContentCandidateReader(Protocol):
    def list_approved_or_active_for_generation(
        self,
        generation_id: str,
    ) -> list[ContentCandidateRecord]:
        ...


class PromotionRunWriter(Protocol):
    def lock_activation_scope(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        generation_id: str,
        segment_scope_fingerprint: str,
        loop_count: int,
    ) -> None:
        ...

    def insert_if_absent(self, run: PromotionRunWrite) -> bool:
        ...

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        ...

    def get_by_scope(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        generation_id: str,
        segment_scope_fingerprint: str,
        loop_count: int,
    ) -> PromotionRunRecord | None:
        ...

    def update_status(self, *, promotion_run_id: str, status: str) -> None:
        ...


class AdExperimentReader(Protocol):
    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        ...


class AdExperimentWriter(AdExperimentReader, Protocol):
    def get_by_id(self, ad_experiment_id: str) -> AdExperimentRecord | None:
        ...

    def insert_many(self, experiments: Sequence[AdExperimentWrite]) -> None:
        ...

    def exists_for_run_segment(
        self,
        *,
        promotion_run_id: str,
        segment_id: str,
    ) -> bool:
        ...

    def update_status(self, *, ad_experiment_id: str, status: str) -> None:
        ...


class SegmentVectorReader(Protocol):
    def list_for_run_segments(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_ids: Sequence[str],
        vector_version: str,
    ) -> list[SegmentVectorRecord]:
        ...

    def configure_ann_search(self) -> None:
        ...

    def list_ann_candidates(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_vector_ids: Sequence[str],
        vector_version: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[SegmentVectorRecord]:
        ...

    def list_ann_candidates_for_users(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_vector_ids: Sequence[str],
        vector_version: str,
        user_ids: Sequence[str],
        query_vectors: Sequence[Sequence[float]],
        limit: int,
    ) -> dict[str, list[SegmentVectorRecord]]:
        ...


class UserBehaviorVectorReader(Protocol):
    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: Sequence[str],
        vector_version: str,
        source: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        ...

    def list_for_project(
        self,
        *,
        project_id: str,
        vector_version: str,
        limit: int,
        source: str | None = None,
        after_user_id: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        ...


class UserSegmentAssignmentWriter(Protocol):
    def list_existing_user_ids(
        self,
        *,
        promotion_run_id: str,
        user_ids: Sequence[str],
    ) -> set[str]:
        ...

    def insert_many(
        self,
        assignments: Sequence[UserSegmentAssignmentWrite],
    ) -> list[UserSegmentAssignmentInsertRecord]:
        ...


class PromotionEvaluationWriter(Protocol):
    def insert(self, evaluation: PromotionEvaluationWrite) -> None:
        ...

    def list_latest_by_run_ad_experiments(
        self,
        promotion_run_id: str,
    ) -> list[PromotionEvaluationRecord]:
        ...


class NextLoopPreparationWriter(Protocol):
    def get_active_by_source_run(
        self,
        source_promotion_run_id: str,
    ) -> NextLoopPreparationRecord | None:
        ...

    def get_by_id(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        ...

    def get_by_id_for_update(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        ...

    def get_next_attempt_no(self, source_promotion_run_id: str) -> int:
        ...

    def insert(
        self,
        preparation: NextLoopPreparationWrite,
    ) -> NextLoopPreparationRecord:
        ...

    def mark_rejected(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        ...

    def mark_activated(
        self,
        *,
        next_loop_preparation_id: str,
        activated_promotion_run_id: str,
    ) -> NextLoopPreparationRecord | None:
        ...


class EvaluationMetricReader(Protocol):
    def count_inflow_rate(
        self,
        experiment: AdExperimentRecord,
        *,
        evaluation_cutoff_at: datetime,
    ) -> MetricCountRecord:
        ...

    def count_booking_conversion_rate(
        self,
        experiment: AdExperimentRecord,
        *,
        evaluation_cutoff_at: datetime,
    ) -> MetricCountRecord:
        ...


class PromotionRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_by_id(self, promotion_id: str) -> PromotionRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                project_id,
                campaign_id,
                promotion_id,
                channel,
                goal_metric,
                goal_target_value,
                goal_basis,
                min_sample_size,
                max_loop_count
            FROM promotions
            WHERE promotion_id = %s
            """,
            (promotion_id,),
        )
        if row is None:
            return None
        return PromotionRecord(**row)


class PromotionAnalysisRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_by_id(self, analysis_id: str) -> PromotionAnalysisRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                focus_segment_ids_json,
                operator_instruction,
                input_snapshot_json,
                profile_summary_json,
                output_json,
                status
            FROM promotion_analyses
            WHERE analysis_id = %s
            """,
            (analysis_id,),
        )
        if row is None:
            return None
        return PromotionAnalysisRecord(**row)

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> PromotionAnalysisRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                focus_segment_ids_json,
                operator_instruction,
                input_snapshot_json,
                profile_summary_json,
                output_json,
                status
            FROM promotion_analyses
            WHERE promotion_id = %s
              AND status = 'completed'
            ORDER BY updated_at DESC, created_at DESC, analysis_id DESC
            LIMIT 1
            """,
            (promotion_id,),
        )
        if row is None:
            return None
        return PromotionAnalysisRecord(**row)


class PromotionTargetSegmentRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def list_for_analysis(
        self,
        analysis_id: str,
    ) -> list[PromotionTargetSegmentRecord]:
        rows = self._db.fetchall(
            """
            SELECT
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                segment_id,
                segment_name,
                segment_vector_id,
                rule_json,
                profile_json,
                content_brief_json,
                data_evidence_json,
                estimated_size,
                priority,
                status
            FROM promotion_target_segments
            WHERE analysis_id = %s
            ORDER BY id ASC
            """,
            (analysis_id,),
        )
        return [PromotionTargetSegmentRecord(**row) for row in rows]

    def list_approved_for_analysis(
        self,
        analysis_id: str,
        segment_ids: Sequence[str] | None = None,
    ) -> list[PromotionTargetSegmentRecord]:
        segment_filter = ""
        params: Sequence[Any] = (analysis_id,)
        if segment_ids is not None:
            segment_filter = "AND segment_id = ANY(%s)"
            params = (analysis_id, list(segment_ids))
        rows = self._db.fetchall(
            f"""
            SELECT
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                segment_id,
                segment_name,
                segment_vector_id,
                rule_json,
                profile_json,
                content_brief_json,
                data_evidence_json,
                estimated_size,
                priority,
                status
            FROM promotion_target_segments
            WHERE analysis_id = %s
              AND status = 'approved'
              {segment_filter}
            ORDER BY id ASC
            """,
            params,
        )
        return [PromotionTargetSegmentRecord(**row) for row in rows]

    def update_status(
        self,
        *,
        analysis_id: str,
        segment_id: str,
        status: str,
    ) -> None:
        self._db.execute(
            """
            UPDATE promotion_target_segments
            SET status = %s
            WHERE analysis_id = %s
              AND segment_id = %s
            """,
            (status, analysis_id, segment_id),
        )


class GenerationRunRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_by_id(self, generation_id: str) -> GenerationRunRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                generation_id,
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                content_option_count,
                operator_instruction,
                input_json,
                output_json,
                generation_report_json,
                status
            FROM generation_runs
            WHERE generation_id = %s
            """,
            (generation_id,),
        )
        if row is None:
            return None
        return GenerationRunRecord(**row)

    def list_next_loop_generation_attempts(
        self,
        source_promotion_run_id: str,
    ) -> list[NextLoopGenerationAttemptRecord]:
        _require_non_blank_string(
            source_promotion_run_id,
            field_name="source_promotion_run_id",
        )
        rows = self._db.fetchall(
            """
            WITH attempt_lock AS MATERIALIZED (
                SELECT pg_advisory_xact_lock(
                    hashtextextended(%s, 0)
                )
            )
            SELECT
                generation_run.generation_id,
                generation_run.analysis_id,
                generation_run.project_id,
                generation_run.campaign_id,
                generation_run.promotion_id,
                generation_run.content_option_count,
                generation_run.operator_instruction,
                generation_run.input_json,
                generation_run.status,
                analysis.input_snapshot_json
                    AS analysis_input_snapshot_json,
                preparation.analysis_id AS preparation_analysis_id,
                preparation.attempt_no AS preparation_attempt_no,
                preparation.status AS preparation_status
            FROM attempt_lock
            LEFT JOIN generation_runs AS generation_run
              ON generation_run.input_json
                    #>> '{next_loop,source_promotion_run_id}' = %s
            LEFT JOIN promotion_analyses AS analysis
              ON analysis.analysis_id = generation_run.analysis_id
            LEFT JOIN next_loop_preparations AS preparation
              ON preparation.source_promotion_run_id = %s
             AND preparation.generation_id = generation_run.generation_id
            ORDER BY generation_run.created_at ASC,
                     generation_run.generation_id ASC
            """,
            (
                source_promotion_run_id,
                source_promotion_run_id,
                source_promotion_run_id,
            ),
        )
        return [
            NextLoopGenerationAttemptRecord(
                generation_id=str(row["generation_id"]),
                analysis_id=str(row["analysis_id"]),
                project_id=str(row["project_id"]),
                campaign_id=str(row["campaign_id"]),
                promotion_id=str(row["promotion_id"]),
                content_option_count=int(row["content_option_count"]),
                operator_instruction=(
                    str(row["operator_instruction"])
                    if row.get("operator_instruction") is not None
                    else None
                ),
                input_json=dict(row.get("input_json") or {}),
                status=str(row["status"]),
                analysis_input_snapshot_json=(
                    dict(row["analysis_input_snapshot_json"])
                    if row.get("analysis_input_snapshot_json") is not None
                    else None
                ),
                preparation_analysis_id=(
                    str(row["preparation_analysis_id"])
                    if row.get("preparation_analysis_id") is not None
                    else None
                ),
                preparation_attempt_no=(
                    int(row["preparation_attempt_no"])
                    if row.get("preparation_attempt_no") is not None
                    else None
                ),
                preparation_status=(
                    str(row["preparation_status"])
                    if row.get("preparation_status") is not None
                    else None
                ),
            )
            for row in rows
            if row.get("generation_id") is not None
        ]

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> GenerationRunRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                generation_id,
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                content_option_count,
                operator_instruction,
                input_json,
                output_json,
                generation_report_json,
                status
            FROM generation_runs
            WHERE promotion_id = %s
              AND status = 'completed'
            ORDER BY updated_at DESC, created_at DESC, generation_id DESC
            LIMIT 1
            """,
            (promotion_id,),
        )
        if row is None:
            return None
        return GenerationRunRecord(**row)


class ContentCandidateRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def list_approved_or_active_for_generation(
        self,
        generation_id: str,
    ) -> list[ContentCandidateRecord]:
        rows = self._db.fetchall(
            """
            SELECT
                content_id,
                content_option_id,
                generation_id,
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                segment_id,
                channel,
                status
            FROM content_candidates
            WHERE generation_id = %s
              AND status IN ('approved', 'active')
            ORDER BY segment_id ASC, content_option_id ASC
            """,
            (generation_id,),
        )
        return [ContentCandidateRecord(**row) for row in rows]

class PromotionRunRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def lock_activation_scope(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        generation_id: str,
        segment_scope_fingerprint: str,
        loop_count: int,
    ) -> None:
        identity_parts = (
            _require_non_blank_string(project_id, field_name="project_id"),
            _require_non_blank_string(promotion_id, field_name="promotion_id"),
            _require_non_blank_string(analysis_id, field_name="analysis_id"),
            _require_non_blank_string(generation_id, field_name="generation_id"),
            _require_non_blank_string(
                segment_scope_fingerprint,
                field_name="segment_scope_fingerprint",
            ),
        )
        if loop_count < 1:
            raise ValueError("loop_count must be at least 1")
        identity_key = json.dumps(
            [*identity_parts, loop_count],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._db.execute(
            """
            SELECT pg_advisory_xact_lock(
                hashtext('promotion-run-activation-v1'),
                hashtext(%s)
            )
            """,
            (identity_key,),
        )

    def insert_if_absent(self, run: PromotionRunWrite) -> bool:
        row = self._db.fetchone(
            """
            INSERT INTO promotion_runs (
                promotion_run_id,
                project_id,
                campaign_id,
                promotion_id,
                analysis_id,
                generation_id,
                loop_count,
                status,
                goal_snapshot_json,
                segment_scope_json,
                segment_scope_fingerprint
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING promotion_run_id
            """,
            (
                run.promotion_run_id,
                run.project_id,
                run.campaign_id,
                run.promotion_id,
                run.analysis_id,
                run.generation_id,
                run.loop_count,
                run.status,
                run.goal_snapshot_json,
                Jsonb(list(run.segment_scope_json)),
                run.segment_scope_fingerprint,
            ),
        )
        return row is not None

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                promotion_run_id,
                project_id,
                campaign_id,
                promotion_id,
                analysis_id,
                generation_id,
                loop_count,
                status,
                goal_snapshot_json,
                segment_scope_json,
                segment_scope_fingerprint
            FROM promotion_runs
            WHERE promotion_run_id = %s
            """,
            (promotion_run_id,),
        )
        if row is None:
            return None
        return PromotionRunRecord(**row)

    def get_by_scope(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        generation_id: str,
        segment_scope_fingerprint: str,
        loop_count: int,
    ) -> PromotionRunRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                promotion_run_id,
                project_id,
                campaign_id,
                promotion_id,
                analysis_id,
                generation_id,
                loop_count,
                status,
                goal_snapshot_json,
                segment_scope_json,
                segment_scope_fingerprint
            FROM promotion_runs
            WHERE project_id = %s
              AND promotion_id = %s
              AND analysis_id = %s
              AND generation_id = %s
              AND segment_scope_fingerprint = %s
              AND loop_count = %s
            LIMIT 1
            """,
            (
                project_id,
                promotion_id,
                analysis_id,
                generation_id,
                segment_scope_fingerprint,
                loop_count,
            ),
        )
        if row is None:
            return None
        return PromotionRunRecord(**row)

    def update_status(self, *, promotion_run_id: str, status: str) -> None:
        self._db.execute(
            """
            UPDATE promotion_runs
            SET status = %s,
                updated_at = now()
            WHERE promotion_run_id = %s
            """,
            (status, promotion_run_id),
        )


class AdExperimentRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_by_id(self, ad_experiment_id: str) -> AdExperimentRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                ad_experiment_id,
                project_id,
                campaign_id,
                promotion_id,
                promotion_run_id,
                analysis_id,
                generation_id,
                segment_id,
                segment_name,
                content_id,
                content_option_id,
                parent_ad_experiment_id,
                source_evaluation_id,
                channel,
                loop_count,
                status,
                goal_metric,
                goal_target_value,
                goal_basis
            FROM ad_experiments
            WHERE ad_experiment_id = %s
            """,
            (ad_experiment_id,),
        )
        if row is None:
            return None
        return AdExperimentRecord(**row)

    def insert_many(self, experiments: Sequence[AdExperimentWrite]) -> None:
        for experiment in experiments:
            self._db.execute(
                """
                INSERT INTO ad_experiments (
                    ad_experiment_id,
                    project_id,
                    campaign_id,
                    promotion_id,
                    promotion_run_id,
                    analysis_id,
                    generation_id,
                    segment_id,
                    segment_name,
                    content_id,
                    content_option_id,
                    parent_ad_experiment_id,
                    source_evaluation_id,
                    channel,
                    loop_count,
                    status,
                    goal_metric,
                    goal_target_value,
                    goal_basis
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    experiment.ad_experiment_id,
                    experiment.project_id,
                    experiment.campaign_id,
                    experiment.promotion_id,
                    experiment.promotion_run_id,
                    experiment.analysis_id,
                    experiment.generation_id,
                    experiment.segment_id,
                    experiment.segment_name,
                    experiment.content_id,
                    experiment.content_option_id,
                    experiment.parent_ad_experiment_id,
                    experiment.source_evaluation_id,
                    experiment.channel,
                    experiment.loop_count,
                    experiment.status,
                    experiment.goal_metric,
                    experiment.goal_target_value,
                    experiment.goal_basis,
                ),
            )

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        rows = self._db.fetchall(
            """
            SELECT
                ad_experiment_id,
                project_id,
                campaign_id,
                promotion_id,
                promotion_run_id,
                analysis_id,
                generation_id,
                segment_id,
                segment_name,
                content_id,
                content_option_id,
                parent_ad_experiment_id,
                source_evaluation_id,
                channel,
                loop_count,
                status,
                goal_metric,
                goal_target_value,
                goal_basis
            FROM ad_experiments
            WHERE promotion_run_id = %s
            ORDER BY segment_id ASC
            """,
            (promotion_run_id,),
        )
        return [AdExperimentRecord(**row) for row in rows]

    def exists_for_run_segment(
        self,
        *,
        promotion_run_id: str,
        segment_id: str,
    ) -> bool:
        row = self._db.fetchone(
            """
            SELECT 1
            FROM ad_experiments
            WHERE promotion_run_id = %s
              AND segment_id = %s
            LIMIT 1
            """,
            (promotion_run_id, segment_id),
        )
        return row is not None

    def update_status(self, *, ad_experiment_id: str, status: str) -> None:
        self._db.execute(
            """
            UPDATE ad_experiments
            SET status = %s,
                updated_at = now()
            WHERE ad_experiment_id = %s
            """,
            (status, ad_experiment_id),
        )


class SegmentVectorRepository:
    VECTOR_DIM = 64
    HNSW_EF_SEARCH = DEFAULT_HNSW_EF_SEARCH
    HNSW_MAX_SCAN_TUPLES = DEFAULT_HNSW_MAX_SCAN_TUPLES

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def configure_ann_search(self) -> None:
        self._db.execute(
            "SELECT set_config('hnsw.ef_search', %s, true)",
            (str(self.HNSW_EF_SEARCH),),
        )
        self._db.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        self._db.execute(
            "SELECT set_config('hnsw.max_scan_tuples', %s, true)",
            (str(self.HNSW_MAX_SCAN_TUPLES),),
        )

    def list_for_run_segments(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_ids: Sequence[str],
        vector_version: str,
    ) -> list[SegmentVectorRecord]:
        if not segment_ids:
            return []

        rows = self._db.fetchall(
            """
            SELECT
                segment_vector_id,
                project_id,
                promotion_id,
                promotion_run_id,
                analysis_id,
                segment_id,
                vector_dim,
                vector_values,
                vector_version,
                source,
                embedding::text AS embedding
            FROM segment_vectors
            WHERE project_id = %s
              AND promotion_id = %s
              AND analysis_id = %s
              AND segment_id = ANY(%s)
              AND vector_version = %s
              AND vector_dim = %s
            ORDER BY segment_id ASC, created_at DESC, segment_vector_id DESC
            """,
            (
                project_id,
                promotion_id,
                analysis_id,
                list(segment_ids),
                vector_version,
                self.VECTOR_DIM,
            ),
        )
        return [SegmentVectorRecord(**row) for row in rows]

    def list_ann_candidates(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_vector_ids: Sequence[str],
        vector_version: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[SegmentVectorRecord]:
        if not segment_vector_ids:
            return []

        vector_literal = _vector_literal(query_vector, self.VECTOR_DIM)
        rows = self._db.fetchall(
            """
            SELECT
                segment_vector_id,
                project_id,
                promotion_id,
                promotion_run_id,
                analysis_id,
                segment_id,
                vector_dim,
                vector_values,
                vector_version,
                source,
                embedding::text AS embedding
            FROM segment_vectors
            WHERE project_id = %s
              AND promotion_id = %s
              AND analysis_id = %s
              AND segment_vector_id = ANY(%s)
              AND vector_version = %s
              AND vector_dim = %s
            ORDER BY embedding <=> %s::vector, segment_id ASC
            LIMIT %s
            """,
            (
                project_id,
                promotion_id,
                analysis_id,
                list(segment_vector_ids),
                vector_version,
                self.VECTOR_DIM,
                vector_literal,
                limit,
            ),
        )
        return [SegmentVectorRecord(**row) for row in rows]

    def list_ann_candidates_for_users(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_vector_ids: Sequence[str],
        vector_version: str,
        user_ids: Sequence[str],
        query_vectors: Sequence[Sequence[float]],
        limit: int,
    ) -> dict[str, list[SegmentVectorRecord]]:
        if len(user_ids) != len(query_vectors):
            raise ValueError("user_ids and query_vectors must have the same length")
        if len(set(user_ids)) != len(user_ids):
            raise ValueError("user_ids must not contain duplicates")

        candidates_by_user = {user_id: [] for user_id in user_ids}
        if not user_ids or not segment_vector_ids:
            return candidates_by_user

        query_vector_literals = [
            _vector_literal(query_vector, self.VECTOR_DIM)
            for query_vector in query_vectors
        ]
        rows = self._db.fetchall(
            """
            WITH query_users AS (
                SELECT
                    user_id,
                    query_vector,
                    query_ordinal
                FROM unnest(%s::text[], %s::text[]) WITH ORDINALITY
                    AS q(user_id, query_vector, query_ordinal)
            )
            SELECT
                q.user_id AS query_user_id,
                q.query_ordinal AS query_ordinal,
                sv.segment_vector_id,
                sv.project_id,
                sv.promotion_id,
                sv.promotion_run_id,
                sv.analysis_id,
                sv.segment_id,
                sv.vector_dim,
                sv.vector_values,
                sv.vector_version,
                sv.source,
                sv.embedding::text AS embedding
            FROM query_users q
            CROSS JOIN LATERAL (
                SELECT
                    segment_vector_id,
                    project_id,
                    promotion_id,
                    promotion_run_id,
                    analysis_id,
                    segment_id,
                    vector_dim,
                    vector_values,
                    vector_version,
                    source,
                    embedding
                FROM segment_vectors
                WHERE project_id = %s
                  AND promotion_id = %s
                  AND analysis_id = %s
                  AND segment_vector_id = ANY(%s)
                  AND vector_version = %s
                  AND vector_dim = %s
                ORDER BY embedding <=> q.query_vector::vector, segment_id ASC
                LIMIT %s
            ) sv
            ORDER BY q.query_ordinal ASC, sv.segment_id ASC
            """,
            (
                list(user_ids),
                query_vector_literals,
                project_id,
                promotion_id,
                analysis_id,
                list(segment_vector_ids),
                vector_version,
                self.VECTOR_DIM,
                limit,
            ),
        )
        for row in rows:
            query_user_id = str(row["query_user_id"])
            if query_user_id not in candidates_by_user:
                raise ValueError("unexpected query_user_id returned by ANN query")
            candidates_by_user[query_user_id].append(
                SegmentVectorRecord(
                    segment_vector_id=row["segment_vector_id"],
                    project_id=row["project_id"],
                    promotion_id=row["promotion_id"],
                    promotion_run_id=row["promotion_run_id"],
                    analysis_id=row["analysis_id"],
                    segment_id=row["segment_id"],
                    vector_dim=row["vector_dim"],
                    vector_values=row["vector_values"],
                    vector_version=row["vector_version"],
                    source=row["source"],
                    embedding=row["embedding"],
                )
            )
        return candidates_by_user


class UserSegmentAssignmentRepository:
    INSERT_BATCH_SIZE = 1000

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def list_existing_user_ids(
        self,
        *,
        promotion_run_id: str,
        user_ids: Sequence[str],
    ) -> set[str]:
        if not user_ids:
            return set()

        rows = self._db.fetchall(
            """
            SELECT user_id
            FROM user_segment_assignments
            WHERE promotion_run_id = %s
              AND user_id = ANY(%s)
            """,
            (promotion_run_id, list(user_ids)),
        )
        return {str(row["user_id"]) for row in rows}

    def insert_many(
        self,
        assignments: Sequence[UserSegmentAssignmentWrite],
    ) -> list[UserSegmentAssignmentInsertRecord]:
        inserted_records: list[UserSegmentAssignmentInsertRecord] = []
        for chunk in _chunks(assignments, self.INSERT_BATCH_SIZE):
            rows = self._db.fetchall(
                """
                WITH assignment_rows AS (
                    SELECT
                        project_id,
                        promotion_run_id,
                        user_id,
                        segment_id,
                        ad_experiment_id,
                        content_id,
                        content_option_id,
                        similarity_score,
                        fallback_value,
                        fallback_reason,
                        assignment_source,
                        assigned_at,
                        expires_at,
                        row_ordinal
                    FROM unnest(
                        %s::text[],
                        %s::text[],
                        %s::text[],
                        %s::text[],
                        %s::text[],
                        %s::text[],
                        %s::text[],
                        %s::numeric[],
                        %s::boolean[],
                        %s::text[],
                        %s::text[],
                        %s::timestamptz[],
                        %s::timestamptz[]
                    ) WITH ORDINALITY AS assignment_input(
                        project_id,
                        promotion_run_id,
                        user_id,
                        segment_id,
                        ad_experiment_id,
                        content_id,
                        content_option_id,
                        similarity_score,
                        fallback_value,
                        fallback_reason,
                        assignment_source,
                        assigned_at,
                        expires_at,
                        row_ordinal
                    )
                )
                INSERT INTO user_segment_assignments (
                    project_id,
                    promotion_run_id,
                    user_id,
                    segment_id,
                    ad_experiment_id,
                    content_id,
                    content_option_id,
                    similarity_score,
                    fallback,
                    fallback_reason,
                    assignment_source,
                    assigned_at,
                    expires_at
                )
                SELECT
                    project_id,
                    promotion_run_id,
                    user_id,
                    segment_id,
                    ad_experiment_id,
                    content_id,
                    content_option_id,
                    similarity_score,
                    fallback_value,
                    fallback_reason,
                    assignment_source,
                    assigned_at,
                    expires_at
                FROM assignment_rows
                ORDER BY row_ordinal ASC
                ON CONFLICT (promotion_run_id, user_id) DO NOTHING
                RETURNING
                    user_id,
                    segment_id,
                    fallback,
                    fallback_reason,
                    similarity_score
                """,
                (
                    [assignment.project_id for assignment in chunk],
                    [assignment.promotion_run_id for assignment in chunk],
                    [assignment.user_id for assignment in chunk],
                    [assignment.segment_id for assignment in chunk],
                    [assignment.ad_experiment_id for assignment in chunk],
                    [assignment.content_id for assignment in chunk],
                    [assignment.content_option_id for assignment in chunk],
                    [assignment.similarity_score for assignment in chunk],
                    [assignment.fallback for assignment in chunk],
                    [assignment.fallback_reason for assignment in chunk],
                    [assignment.assignment_source for assignment in chunk],
                    [assignment.assigned_at for assignment in chunk],
                    [assignment.expires_at for assignment in chunk],
                ),
            )
            inserted_records.extend(
                UserSegmentAssignmentInsertRecord(
                    user_id=str(row["user_id"]),
                    segment_id=str(row["segment_id"]),
                    fallback=bool(row["fallback"]),
                    fallback_reason=(
                        str(row["fallback_reason"])
                        if row["fallback_reason"] is not None
                        else None
                    ),
                    similarity_score=row["similarity_score"],
                )
                for row in rows
            )
        return inserted_records


class UserBehaviorVectorRepository:
    VECTOR_DIM = 64

    def __init__(self, client: ClickHouseClient) -> None:
        self._client = client

    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: Sequence[str],
        vector_version: str,
        source: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        if not user_ids:
            return []

        source_filter = (
            "                  AND source = {source:String}\n"
            if source is not None
            else ""
        )
        parameters: dict[str, Any] = {
            "project_id": project_id,
            "vector_version": vector_version,
            "vector_dim": self.VECTOR_DIM,
            "user_ids": list(user_ids),
        }
        if source is not None:
            parameters["source"] = source

        query = (
            """
            SELECT
                project_id,
                user_id,
                argMax(vector_dim, updated_at) AS vector_dim,
                argMax(vector_values, updated_at) AS vector_values,
                vector_version,
                argMax(source, updated_at) AS source
            FROM (
                SELECT
                    project_id,
                    user_id,
                    vector_dim,
                    vector_values,
                    vector_version,
                    source,
                    updated_at
                FROM user_behavior_vectors
                WHERE project_id = {project_id:String}
                  AND vector_version = {vector_version:String}
                  AND vector_dim = {vector_dim:UInt16}
            """
            + source_filter
            + """
                  AND user_id IN {user_ids:Array(String)}
            )
            GROUP BY project_id, user_id, vector_version
            ORDER BY user_id ASC
            """
        )
        result = self._client.query(
            query,
            parameters=parameters,
        )
        return self._records_from_result(result)

    def list_for_project(
        self,
        *,
        project_id: str,
        vector_version: str,
        limit: int,
        source: str | None = None,
        after_user_id: str | None = None,
    ) -> list[UserBehaviorVectorRecord]:
        source_filter = (
            "                  AND source = {source:String}\n"
            if source is not None
            else ""
        )
        cursor_filter = (
            "                  AND user_id > {after_user_id:String}\n"
            if after_user_id is not None
            else ""
        )
        parameters: dict[str, Any] = {
            "project_id": project_id,
            "vector_version": vector_version,
            "vector_dim": self.VECTOR_DIM,
            "limit": limit,
        }
        if source is not None:
            parameters["source"] = source
        if after_user_id is not None:
            parameters["after_user_id"] = after_user_id

        query = (
            """
            SELECT
                project_id,
                user_id,
                argMax(vector_dim, updated_at) AS vector_dim,
                argMax(vector_values, updated_at) AS vector_values,
                vector_version,
                argMax(source, updated_at) AS source
            FROM (
                SELECT
                    project_id,
                    user_id,
                    vector_dim,
                    vector_values,
                    vector_version,
                    source,
                    updated_at
                FROM user_behavior_vectors
                WHERE project_id = {project_id:String}
                  AND vector_version = {vector_version:String}
                  AND vector_dim = {vector_dim:UInt16}
            """
            + source_filter
            + cursor_filter
            + """
            )
            GROUP BY project_id, user_id, vector_version
            ORDER BY user_id ASC
            LIMIT {limit:UInt32}
            """
        )
        result = self._client.query(
            query,
            parameters=parameters,
        )
        return self._records_from_result(result)

    def _records_from_result(self, result: Any) -> list[UserBehaviorVectorRecord]:
        return [
            UserBehaviorVectorRecord(
                project_id=_clickhouse_value(row, "project_id", 0),
                user_id=_clickhouse_value(row, "user_id", 1),
                vector_dim=int(_clickhouse_value(row, "vector_dim", 2)),
                vector_values=[
                    float(value)
                    for value in _clickhouse_value(row, "vector_values", 3)
                ],
                vector_version=_clickhouse_value(row, "vector_version", 4),
                source=_clickhouse_value(row, "source", 5),
            )
            for row in _clickhouse_rows(result)
        ]


class PromotionEvaluationRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def insert(self, evaluation: PromotionEvaluationWrite) -> None:
        self._db.execute(
            """
            INSERT INTO promotion_evaluations (
                evaluation_id,
                project_id,
                campaign_id,
                promotion_id,
                promotion_run_id,
                ad_experiment_id,
                segment_id,
                content_id,
                content_option_id,
                metric,
                target_value,
                actual_value,
                numerator_count,
                denominator_count,
                sample_size,
                basis,
                status,
                feedback,
                next_loop_required,
                result_json
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                evaluation.evaluation_id,
                evaluation.project_id,
                evaluation.campaign_id,
                evaluation.promotion_id,
                evaluation.promotion_run_id,
                evaluation.ad_experiment_id,
                evaluation.segment_id,
                evaluation.content_id,
                evaluation.content_option_id,
                evaluation.metric,
                evaluation.target_value,
                evaluation.actual_value,
                evaluation.numerator_count,
                evaluation.denominator_count,
                evaluation.sample_size,
                evaluation.basis,
                evaluation.status,
                evaluation.feedback,
                evaluation.next_loop_required,
                evaluation.result_json,
            ),
        )

    def list_latest_by_run_ad_experiments(
        self,
        promotion_run_id: str,
    ) -> list[PromotionEvaluationRecord]:
        rows = self._db.fetchall(
            """
            SELECT DISTINCT ON (ad_experiment_id)
                evaluation_id,
                project_id,
                campaign_id,
                promotion_id,
                promotion_run_id,
                ad_experiment_id,
                segment_id,
                content_id,
                content_option_id,
                metric,
                target_value,
                actual_value,
                numerator_count,
                denominator_count,
                sample_size,
                basis,
                status,
                feedback,
                next_loop_required,
                result_json
            FROM promotion_evaluations
            WHERE promotion_run_id = %s
              AND ad_experiment_id IS NOT NULL
            ORDER BY ad_experiment_id ASC, created_at DESC, evaluation_id DESC
            """,
            (promotion_run_id,),
        )
        return [PromotionEvaluationRecord(**row) for row in rows]


class NextLoopPreparationRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_active_by_source_run(
        self,
        source_promotion_run_id: str,
    ) -> NextLoopPreparationRecord | None:
        _require_non_blank_string(
            source_promotion_run_id,
            field_name="source_promotion_run_id",
        )
        row = self._db.fetchone(
            """
            SELECT
                next_loop_preparation_id,
                source_promotion_run_id,
                analysis_id,
                generation_id,
                attempt_no,
                failed_segment_ids_json,
                failed_ad_experiment_ids_json,
                source_evaluation_ids_json,
                status,
                activated_promotion_run_id,
                created_at,
                updated_at
            FROM next_loop_preparations
            WHERE source_promotion_run_id = %s
              AND status = 'awaiting_content_approval'
            ORDER BY attempt_no DESC
            LIMIT 1
            FOR UPDATE
            """,
            (source_promotion_run_id,),
        )
        return _next_loop_preparation_record_or_none(row)

    def get_by_id(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        _require_non_blank_string(
            next_loop_preparation_id,
            field_name="next_loop_preparation_id",
        )
        row = self._db.fetchone(
            """
            SELECT
                next_loop_preparation_id,
                source_promotion_run_id,
                analysis_id,
                generation_id,
                attempt_no,
                failed_segment_ids_json,
                failed_ad_experiment_ids_json,
                source_evaluation_ids_json,
                status,
                activated_promotion_run_id,
                created_at,
                updated_at
            FROM next_loop_preparations
            WHERE next_loop_preparation_id = %s
            """,
            (next_loop_preparation_id,),
        )
        return _next_loop_preparation_record_or_none(row)

    def get_by_id_for_update(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        _require_non_blank_string(
            next_loop_preparation_id,
            field_name="next_loop_preparation_id",
        )
        row = self._db.fetchone(
            """
            SELECT
                next_loop_preparation_id,
                source_promotion_run_id,
                analysis_id,
                generation_id,
                attempt_no,
                failed_segment_ids_json,
                failed_ad_experiment_ids_json,
                source_evaluation_ids_json,
                status,
                activated_promotion_run_id,
                created_at,
                updated_at
            FROM next_loop_preparations
            WHERE next_loop_preparation_id = %s
            FOR UPDATE
            """,
            (next_loop_preparation_id,),
        )
        return _next_loop_preparation_record_or_none(row)

    def get_next_attempt_no(self, source_promotion_run_id: str) -> int:
        _require_non_blank_string(
            source_promotion_run_id,
            field_name="source_promotion_run_id",
        )
        row = self._db.fetchone(
            """
            SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_attempt_no
            FROM next_loop_preparations
            WHERE source_promotion_run_id = %s
            """,
            (source_promotion_run_id,),
        )
        if row is None:
            return 1
        return int(row["next_attempt_no"])

    def insert(
        self,
        preparation: NextLoopPreparationWrite,
    ) -> NextLoopPreparationRecord:
        _require_non_blank_string(
            preparation.next_loop_preparation_id,
            field_name="next_loop_preparation_id",
        )
        _require_non_blank_string(
            preparation.source_promotion_run_id,
            field_name="source_promotion_run_id",
        )
        _require_non_blank_string(
            preparation.analysis_id,
            field_name="analysis_id",
        )
        _require_non_blank_string(
            preparation.generation_id,
            field_name="generation_id",
        )
        if preparation.attempt_no < 1:
            raise ValueError("attempt_no must be at least 1")
        failed_segment_ids = _normalize_id_set(
            preparation.failed_segment_ids_json,
            field_name="failed_segment_ids_json",
        )
        failed_ad_experiment_ids = _normalize_id_set(
            preparation.failed_ad_experiment_ids_json,
            field_name="failed_ad_experiment_ids_json",
        )
        source_evaluation_ids = _normalize_id_set(
            preparation.source_evaluation_ids_json,
            field_name="source_evaluation_ids_json",
        )
        try:
            row = self._db.fetchone(
                """
                INSERT INTO next_loop_preparations (
                    next_loop_preparation_id,
                    source_promotion_run_id,
                    analysis_id,
                    generation_id,
                    attempt_no,
                    failed_segment_ids_json,
                    failed_ad_experiment_ids_json,
                    source_evaluation_ids_json,
                    status,
                    activated_promotion_run_id
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    'awaiting_content_approval', NULL
                )
                RETURNING
                    next_loop_preparation_id,
                    source_promotion_run_id,
                    analysis_id,
                    generation_id,
                    attempt_no,
                    failed_segment_ids_json,
                    failed_ad_experiment_ids_json,
                    source_evaluation_ids_json,
                    status,
                    activated_promotion_run_id,
                    created_at,
                    updated_at
                """,
                (
                    preparation.next_loop_preparation_id,
                    preparation.source_promotion_run_id,
                    preparation.analysis_id,
                    preparation.generation_id,
                    preparation.attempt_no,
                    Jsonb(list(failed_segment_ids)),
                    Jsonb(list(failed_ad_experiment_ids)),
                    Jsonb(list(source_evaluation_ids)),
                ),
            )
        except errors.UniqueViolation as exc:
            raise _next_loop_preparation_conflict(exc) from exc
        if row is None:
            raise RuntimeError("next-loop preparation insert returned no row")
        return _next_loop_preparation_record(row)

    def mark_rejected(
        self,
        next_loop_preparation_id: str,
    ) -> NextLoopPreparationRecord | None:
        _require_non_blank_string(
            next_loop_preparation_id,
            field_name="next_loop_preparation_id",
        )
        row = self._db.fetchone(
            """
            UPDATE next_loop_preparations
            SET status = 'rejected',
                updated_at = now()
            WHERE next_loop_preparation_id = %s
              AND status = 'awaiting_content_approval'
            RETURNING
                next_loop_preparation_id,
                source_promotion_run_id,
                analysis_id,
                generation_id,
                attempt_no,
                failed_segment_ids_json,
                failed_ad_experiment_ids_json,
                source_evaluation_ids_json,
                status,
                activated_promotion_run_id,
                created_at,
                updated_at
            """,
            (next_loop_preparation_id,),
        )
        return _next_loop_preparation_record_or_none(row)

    def mark_activated(
        self,
        *,
        next_loop_preparation_id: str,
        activated_promotion_run_id: str,
    ) -> NextLoopPreparationRecord | None:
        _require_non_blank_string(
            next_loop_preparation_id,
            field_name="next_loop_preparation_id",
        )
        _require_non_blank_string(
            activated_promotion_run_id,
            field_name="activated_promotion_run_id",
        )
        try:
            row = self._db.fetchone(
                """
                UPDATE next_loop_preparations
                SET status = 'activated',
                    activated_promotion_run_id = %s,
                    updated_at = now()
                WHERE next_loop_preparation_id = %s
                  AND status = 'awaiting_content_approval'
                RETURNING
                    next_loop_preparation_id,
                    source_promotion_run_id,
                    analysis_id,
                    generation_id,
                    attempt_no,
                    failed_segment_ids_json,
                    failed_ad_experiment_ids_json,
                    source_evaluation_ids_json,
                    status,
                    activated_promotion_run_id,
                    created_at,
                    updated_at
                """,
                (activated_promotion_run_id, next_loop_preparation_id),
            )
        except errors.UniqueViolation as exc:
            conflict = _next_loop_preparation_conflict(exc)
            if (
                conflict.constraint_name
                == "uq_next_loop_preparations_activated_run"
            ):
                raise conflict from exc
            raise
        return _next_loop_preparation_record_or_none(row)


class EvaluationMetricRepository:
    def __init__(self, client: ClickHouseClient) -> None:
        self._client = client

    def count_inflow_rate(
        self,
        experiment: AdExperimentRecord,
        *,
        evaluation_cutoff_at: datetime,
    ) -> MetricCountRecord:
        result = self._client.query(
            """
            WITH
                if(
                    notEmpty(ifNull(redirect_id, '')),
                    concat('redirect:', ifNull(redirect_id, '')),
                    concat('user:', user_id)
                ) AS attribution_key
            SELECT
                countDistinctIf(attribution_key, event_name = 'campaign_landing') AS numerator_count,
                countDistinctIf(attribution_key, event_name = 'campaign_redirect_click') AS denominator_count
            FROM promotion_touch_events
            WHERE project_id = {project_id:String}
              AND promotion_run_id = {promotion_run_id:String}
              AND ad_experiment_id = {ad_experiment_id:String}
              AND event_name IN ('campaign_redirect_click', 'campaign_landing')
              AND event_time <= {evaluation_cutoff_at:DateTime64(3, 'UTC')}
              AND (notEmpty(ifNull(redirect_id, '')) OR notEmpty(user_id))
            """,
            parameters={
                "project_id": experiment.project_id,
                "promotion_run_id": experiment.promotion_run_id,
                "ad_experiment_id": experiment.ad_experiment_id,
                "evaluation_cutoff_at": evaluation_cutoff_at,
            },
        )
        return _metric_count_from_result(result)

    def count_booking_conversion_rate(
        self,
        experiment: AdExperimentRecord,
        *,
        evaluation_cutoff_at: datetime,
    ) -> MetricCountRecord:
        denominator_event_name = _booking_conversion_denominator_event(experiment)
        result = self._client.query(
            """
            SELECT
                (
                    SELECT countDistinct(user_id)
                    FROM booking_outcome_events
                    WHERE project_id = {project_id:String}
                      AND promotion_run_id IS NOT NULL
                      AND ad_experiment_id IS NOT NULL
                      AND promotion_run_id = {promotion_run_id:String}
                      AND ad_experiment_id = {ad_experiment_id:String}
                      AND event_name = 'booking_complete'
                      AND event_time <= {evaluation_cutoff_at:DateTime64(3, 'UTC')}
                ) AS numerator_count,
                (
                    SELECT countDistinct(user_id)
                    FROM promotion_touch_events
                    WHERE project_id = {project_id:String}
                      AND promotion_run_id = {promotion_run_id:String}
                      AND ad_experiment_id = {ad_experiment_id:String}
                      AND event_name = {denominator_event_name:String}
                      AND event_time <= {evaluation_cutoff_at:DateTime64(3, 'UTC')}
                ) AS denominator_count
            """,
            parameters={
                "project_id": experiment.project_id,
                "promotion_run_id": experiment.promotion_run_id,
                "ad_experiment_id": experiment.ad_experiment_id,
                "denominator_event_name": denominator_event_name,
                "evaluation_cutoff_at": evaluation_cutoff_at,
            },
        )
        return _metric_count_from_result(result)


def _next_loop_preparation_record_or_none(
    row: Mapping[str, Any] | None,
) -> NextLoopPreparationRecord | None:
    if row is None:
        return None
    return _next_loop_preparation_record(row)


def _next_loop_preparation_record(
    row: Mapping[str, Any],
) -> NextLoopPreparationRecord:
    return NextLoopPreparationRecord(
        next_loop_preparation_id=str(row["next_loop_preparation_id"]),
        source_promotion_run_id=str(row["source_promotion_run_id"]),
        analysis_id=str(row["analysis_id"]),
        generation_id=str(row["generation_id"]),
        attempt_no=int(row["attempt_no"]),
        failed_segment_ids_json=_normalize_id_set(
            row["failed_segment_ids_json"],
            field_name="failed_segment_ids_json",
        ),
        failed_ad_experiment_ids_json=_normalize_id_set(
            row["failed_ad_experiment_ids_json"],
            field_name="failed_ad_experiment_ids_json",
        ),
        source_evaluation_ids_json=_normalize_id_set(
            row["source_evaluation_ids_json"],
            field_name="source_evaluation_ids_json",
        ),
        status=_next_loop_preparation_status(row["status"]),
        activated_promotion_run_id=row["activated_promotion_run_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _require_non_blank_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _next_loop_preparation_status(value: Any) -> NextLoopPreparationStatusValue:
    allowed_statuses = {
        "awaiting_content_approval",
        "rejected",
        "activated",
    }
    if not isinstance(value, str) or value not in allowed_statuses:
        raise ValueError("status must be a valid next-loop preparation status")
    return cast(NextLoopPreparationStatusValue, value)


def _normalize_id_set(value: Any, *, field_name: str) -> tuple[str, ...]:
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be a JSON array") from exc
    if not isinstance(decoded, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    if not decoded:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in decoded):
        raise ValueError(f"{field_name} must contain non-empty string IDs")
    return tuple(sorted(set(decoded)))


def _next_loop_preparation_conflict(
    exc: errors.UniqueViolation,
) -> NextLoopPreparationConflictError:
    constraint_name = getattr(getattr(exc, "diag", None), "constraint_name", None)
    return NextLoopPreparationConflictError(constraint_name)


def _booking_conversion_denominator_event(experiment: AdExperimentRecord) -> str:
    if experiment.channel == EMAIL_CHANNEL:
        return "campaign_landing"
    return "promotion_click"


def _clickhouse_rows(result: Any) -> list[Any]:
    if hasattr(result, "named_results"):
        return list(result.named_results())
    return list(result.result_rows)


def _clickhouse_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return row[index]


def _metric_count_from_result(result: Any) -> MetricCountRecord:
    rows = _clickhouse_rows(result)
    if not rows:
        return MetricCountRecord(numerator_count=0, denominator_count=0)
    row = rows[0]
    return MetricCountRecord(
        numerator_count=int(_clickhouse_value(row, "numerator_count", 0)),
        denominator_count=int(_clickhouse_value(row, "denominator_count", 1)),
    )


def _vector_literal(values: Sequence[float], vector_dim: int) -> str:
    if len(values) != vector_dim:
        raise ValueError("vector literal must contain 64 values")
    numeric_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("vector literal values must be finite")
    return "[" + ",".join(str(value) for value in numeric_values) + "]"


def _chunks(items: Sequence[_T], size: int) -> list[Sequence[_T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
