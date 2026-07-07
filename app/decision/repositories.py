from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import math
from typing import Any, Mapping, NamedTuple, Protocol, Sequence, TypeVar

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.decision.matcher import (
    HNSW_EF_SEARCH as DEFAULT_HNSW_EF_SEARCH,
    HNSW_MAX_SCAN_TUPLES as DEFAULT_HNSW_MAX_SCAN_TUPLES,
)

EMAIL_CHANNEL = "email"


_T = TypeVar("_T")


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
    def insert(self, run: PromotionRunWrite) -> None:
        ...

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        ...

    def exists_for_promotion_loop(self, *, promotion_id: str, loop_count: int) -> bool:
        ...

    def update_status(self, *, promotion_run_id: str, status: str) -> None:
        ...


class AdExperimentWriter(Protocol):
    def get_by_id(self, ad_experiment_id: str) -> AdExperimentRecord | None:
        ...

    def insert_many(self, experiments: Sequence[AdExperimentWrite]) -> None:
        ...

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
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

    def insert_many(self, assignments: Sequence[UserSegmentAssignmentWrite]) -> int:
        ...

    def count_by_run_segments(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
    ) -> dict[str, int]:
        ...


class PromotionEvaluationWriter(Protocol):
    def insert(self, evaluation: PromotionEvaluationWrite) -> None:
        ...

    def list_latest_by_run_ad_experiments(
        self,
        promotion_run_id: str,
    ) -> list[PromotionEvaluationRecord]:
        ...


class EvaluationMetricReader(Protocol):
    def count_inflow_rate(self, experiment: AdExperimentRecord) -> MetricCountRecord:
        ...

    def count_booking_conversion_rate(
        self,
        experiment: AdExperimentRecord,
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

    def insert(self, run: PromotionRunWrite) -> None:
        self._db.execute(
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
                goal_snapshot_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            ),
        )

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
                goal_snapshot_json
            FROM promotion_runs
            WHERE promotion_run_id = %s
            """,
            (promotion_run_id,),
        )
        if row is None:
            return None
        return PromotionRunRecord(**row)

    def exists_for_promotion_loop(self, *, promotion_id: str, loop_count: int) -> bool:
        row = self._db.fetchone(
            """
            SELECT 1
            FROM promotion_runs
            WHERE promotion_id = %s
              AND loop_count = %s
            LIMIT 1
            """,
            (promotion_id, loop_count),
        )
        return row is not None

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
                    channel,
                    loop_count,
                    status,
                    goal_metric,
                    goal_target_value,
                    goal_basis
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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

    def insert_many(self, assignments: Sequence[UserSegmentAssignmentWrite]) -> int:
        inserted_count = 0
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
                RETURNING id
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
            inserted_count += len(rows)
        return inserted_count

    def count_by_run_segments(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
    ) -> dict[str, int]:
        if not segment_ids:
            return {}

        rows = self._db.fetchall(
            """
            SELECT
                segment_id,
                count(*) AS assigned_user_count
            FROM user_segment_assignments
            WHERE promotion_run_id = %s
              AND segment_id = ANY(%s)
            GROUP BY segment_id
            """,
            (promotion_run_id, list(segment_ids)),
        )
        return {
            str(row["segment_id"]): int(row["assigned_user_count"])
            for row in rows
        }


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
    ) -> list[UserBehaviorVectorRecord]:
        source_filter = (
            "                  AND source = {source:String}\n"
            if source is not None
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


class EvaluationMetricRepository:
    def __init__(self, client: ClickHouseClient) -> None:
        self._client = client

    def count_inflow_rate(self, experiment: AdExperimentRecord) -> MetricCountRecord:
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
              AND (notEmpty(ifNull(redirect_id, '')) OR notEmpty(user_id))
            """,
            parameters={
                "project_id": experiment.project_id,
                "promotion_run_id": experiment.promotion_run_id,
                "ad_experiment_id": experiment.ad_experiment_id,
            },
        )
        return _metric_count_from_result(result)

    def count_booking_conversion_rate(
        self,
        experiment: AdExperimentRecord,
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
                ) AS numerator_count,
                (
                    SELECT countDistinct(user_id)
                    FROM promotion_touch_events
                    WHERE project_id = {project_id:String}
                      AND promotion_run_id = {promotion_run_id:String}
                      AND ad_experiment_id = {ad_experiment_id:String}
                      AND event_name = {denominator_event_name:String}
                ) AS denominator_count
            """,
            parameters={
                "project_id": experiment.project_id,
                "promotion_run_id": experiment.promotion_run_id,
                "ad_experiment_id": experiment.ad_experiment_id,
                "denominator_event_name": denominator_event_name,
            },
        )
        return _metric_count_from_result(result)


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
