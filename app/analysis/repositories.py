from __future__ import annotations

from dataclasses import dataclass
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


class ClickHouseClient(Protocol):
    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
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
    landing_url: str | None
    message_brief: str | None


@dataclass(frozen=True)
class SegmentDefinitionRecord:
    segment_id: str
    project_id: str
    segment_name: str
    source: str
    query_preview_id: str | None
    natural_language_query: str | None
    generated_sql: str | None
    rule_json: Mapping[str, Any]
    sample_size: int
    sample_ratio: Decimal
    status: str


@dataclass(frozen=True)
class PromotionAnalysisWrite:
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    status: str
    input_snapshot_json: Mapping[str, Any]
    profile_summary_json: Mapping[str, Any]


@dataclass(frozen=True)
class PromotionTargetSegmentWrite:
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    segment_id: str
    segment_name: str
    rule_json: Mapping[str, Any]
    profile_json: Mapping[str, Any]
    content_brief_json: Mapping[str, Any]
    segment_vector_id: str | None
    estimated_size: int
    priority: int


@dataclass(frozen=True)
class SegmentVectorRecord:
    segment_vector_id: str
    project_id: str
    promotion_id: str
    promotion_run_id: str | None
    segment_id: str
    vector_dim: int
    vector_values: list[float]
    vector_version: str


@dataclass(frozen=True)
class HotelMarketingProfileRecord:
    project_id: str
    profile_name: str
    profile_json: Mapping[str, Any]


class PromotionRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_for_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionRecord | None:
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
                landing_url,
                message_brief
            FROM promotions
            WHERE project_id = %s
              AND campaign_id = %s
              AND promotion_id = %s
            """,
            (project_id, campaign_id, promotion_id),
        )
        if row is None:
            return None
        return PromotionRecord(**row)


class SegmentDefinitionRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def list_active(
        self,
        *,
        project_id: str,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        params: list[Any] = [project_id]
        source_filter = ""
        if sources:
            placeholders = ", ".join(["%s"] * len(sources))
            source_filter = f" AND source IN ({placeholders})"
            params.extend(sources)

        rows = self._db.fetchall(
            f"""
            SELECT
                segment_id,
                project_id,
                segment_name,
                source,
                query_preview_id,
                natural_language_query,
                generated_sql,
                rule_json,
                sample_size,
                sample_ratio,
                status
            FROM segment_definitions
            WHERE project_id = %s
              AND status = 'active'
              {source_filter}
            ORDER BY sample_size DESC, segment_id ASC
            """,
            tuple(params),
        )
        return [SegmentDefinitionRecord(**row) for row in rows]


class PromotionAnalysisRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def save_analysis(self, analysis: PromotionAnalysisWrite) -> None:
        self._db.execute(
            """
            INSERT INTO promotion_analyses (
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                status,
                input_snapshot_json,
                profile_summary_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                analysis.analysis_id,
                analysis.project_id,
                analysis.campaign_id,
                analysis.promotion_id,
                analysis.status,
                analysis.input_snapshot_json,
                analysis.profile_summary_json,
            ),
        )

    def save_target_segments(
        self,
        target_segments: Sequence[PromotionTargetSegmentWrite],
    ) -> None:
        for segment in target_segments:
            self._db.execute(
                """
                INSERT INTO promotion_target_segments (
                    analysis_id,
                    project_id,
                    campaign_id,
                    promotion_id,
                    segment_id,
                    segment_name,
                    rule_json,
                    profile_json,
                    content_brief_json,
                    segment_vector_id,
                    estimated_size,
                    priority
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    segment.analysis_id,
                    segment.project_id,
                    segment.campaign_id,
                    segment.promotion_id,
                    segment.segment_id,
                    segment.segment_name,
                    segment.rule_json,
                    segment.profile_json,
                    segment.content_brief_json,
                    segment.segment_vector_id,
                    segment.estimated_size,
                    segment.priority,
                ),
            )


class SegmentVectorRepository:
    VECTOR_DIM = 64

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_by_segment(
        self,
        *,
        project_id: str,
        promotion_id: str,
        segment_id: str,
    ) -> SegmentVectorRecord | None:
        row = self._db.fetchone(
            """
            SELECT
                segment_vector_id,
                project_id,
                promotion_id,
                promotion_run_id,
                segment_id,
                vector_dim,
                vector_values,
                vector_version
            FROM segment_vectors
            WHERE project_id = %s
              AND promotion_id = %s
              AND segment_id = %s
            """,
            (project_id, promotion_id, segment_id),
        )
        if row is None:
            return None
        return SegmentVectorRecord(**row)

    def save(self, vector: SegmentVectorRecord) -> None:
        self._validate_vector(vector)
        self._db.execute(
            """
            INSERT INTO segment_vectors (
                segment_vector_id,
                project_id,
                promotion_id,
                promotion_run_id,
                segment_id,
                vector_dim,
                vector_values,
                vector_version
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                vector.segment_vector_id,
                vector.project_id,
                vector.promotion_id,
                vector.promotion_run_id,
                vector.segment_id,
                vector.vector_dim,
                vector.vector_values,
                vector.vector_version,
            ),
        )

    def _validate_vector(self, vector: SegmentVectorRecord) -> None:
        if vector.vector_dim != self.VECTOR_DIM:
            raise ValueError("segment vector_dim must be 64")
        if len(vector.vector_values) != self.VECTOR_DIM:
            raise ValueError("segment vector_values must contain 64 values")


class HotelProfileRepository:
    def __init__(self, client: ClickHouseClient) -> None:
        self._client = client

    def list_marketing_profiles(
        self,
        *,
        project_id: str,
    ) -> list[HotelMarketingProfileRecord]:
        result = self._client.query(
            """
            SELECT
                project_id,
                profile_name,
                profile_json
            FROM hotel_marketing_profiles
            WHERE project_id = {project_id:String}
            """,
            parameters={"project_id": project_id},
        )
        return [
            HotelMarketingProfileRecord(
                project_id=row[0],
                profile_name=row[1],
                profile_json=row[2],
            )
            for row in result.result_rows
        ]

    def summarize_expedia_hotel_events(
        self,
        *,
        project_id: str,
        limit: int = 20,
    ) -> list[Mapping[str, Any]]:
        result = self._client.query(
            """
            SELECT
                hotel_cluster,
                count() AS event_count
            FROM expedia_hotel_events
            WHERE project_id = {project_id:String}
            GROUP BY hotel_cluster
            ORDER BY event_count DESC
            LIMIT {limit:UInt32}
            """,
            parameters={"project_id": project_id, "limit": limit},
        )
        return [
            {"hotel_cluster": row[0], "event_count": row[1]}
            for row in result.result_rows
        ]
