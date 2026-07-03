from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence


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


class AdExperimentWriter(Protocol):
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


class AdExperimentRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

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
