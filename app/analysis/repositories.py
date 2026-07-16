from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Any, Mapping, Protocol, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


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
    if isinstance(value, list):
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
    profile_json: Mapping[str, Any]
    sample_size: int
    total_eligible_user_count: int
    sample_ratio: Decimal
    status: str
    campaign_id: str | None = None
    promotion_id: str | None = None


@dataclass(frozen=True)
class PromotionAnalysisWrite:
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    status: str
    focus_segment_ids_json: Sequence[str] | None
    operator_instruction: str | None
    input_snapshot_json: Mapping[str, Any]
    profile_summary_json: Mapping[str, Any]
    output_json: Mapping[str, Any] | None


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
    data_evidence_json: Mapping[str, Any]
    segment_vector_id: str | None
    estimated_size: int
    priority: str | None
    status: str
    audience_snapshot_id: str | None = None


@dataclass(frozen=True)
class PromotionSegmentSuggestionWrite:
    suggestion_id: str
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    segment_id: str
    suggested_rank: int
    suggestion_source: str
    status: str
    score_json: Mapping[str, Any]
    reason_json: Mapping[str, Any]
    metadata_json: Mapping[str, Any]
    audience_snapshot_id: str | None = None


@dataclass(frozen=True)
class SegmentSuggestionAudienceBindingRecord:
    suggestion_id: str
    analysis_id: str
    segment_id: str
    audience_snapshot_id: str | None


@dataclass(frozen=True)
class SegmentVectorRecord:
    segment_vector_id: str
    project_id: str
    promotion_id: str | None
    promotion_run_id: str | None
    analysis_id: str | None
    segment_id: str
    vector_dim: int
    vector_values: list[float]
    vector_version: str
    source: str
    embedding: str | None = None


@dataclass(frozen=True)
class UserBehaviorVectorRecord:
    project_id: str
    user_id: str
    vector_dim: int
    vector_values: list[float]
    vector_version: str
    source: str


@dataclass(frozen=True)
class RawEventUserSignalRecord:
    project_id: str
    user_id: str
    event_count: int
    hotel_search_count: int
    hotel_click_count: int
    hotel_detail_view_count: int
    promotion_impression_count: int
    promotion_click_count: int
    campaign_redirect_click_count: int
    campaign_landing_count: int
    booking_start_count: int
    booking_complete_count: int
    booking_cancel_count: int
    deal_event_count: int
    free_cancellation_count: int
    breakfast_included_count: int
    price_event_count: int
    avg_price: float
    destination_values: tuple[str, ...]
    checkin_dates: tuple[str, ...]
    hotel_market_values: tuple[str, ...]
    hotel_cluster_values: tuple[str, ...]
    age_group_values: tuple[str, ...]
    gender_values: tuple[str, ...]
    preferred_category_values: tuple[str, ...]
    destination_match_count: int
    season_match_count: int
    page_view_count: int = 0
    hotel_search_recency_days: int | None = None
    hotel_detail_recency_days: int | None = None
    booking_start_recency_days: int | None = None
    deal_recency_days: int | None = None
    promotion_response_recency_days: int | None = None
    lead_time_0_7_count: int = 0
    lead_time_8_30_count: int = 0
    lead_time_gt_30_count: int = 0
    weekend_checkin_count: int = 0
    budget_price_count: int = 0
    premium_price_count: int = 0


@dataclass(frozen=True)
class HotelMarketingProfileRecord:
    project_id: str
    profile_name: str
    profile_json: Mapping[str, Any]


@dataclass(frozen=True)
class BookingTrainingRecord:
    is_mobile: float
    is_package: float
    stay_nights: float
    days_until_checkin: float
    event_count: int
    booking_count: int


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
                min_sample_size,
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
        campaign_id: str | None = None,
        promotion_id: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        params: list[Any] = [project_id, campaign_id, promotion_id]
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
                campaign_id,
                promotion_id,
                segment_name,
                source,
                query_preview_id,
                natural_language_query,
                generated_sql,
                rule_json,
                profile_json,
                sample_size,
                total_eligible_user_count,
                sample_ratio,
                status
            FROM segment_definitions
            WHERE project_id = %s
              AND status = 'active'
              AND (campaign_id IS NULL OR campaign_id = %s)
              AND (promotion_id IS NULL OR promotion_id = %s)
              {source_filter}
            ORDER BY sample_size DESC, segment_id ASC
            """,
            tuple(params),
        )
        return [SegmentDefinitionRecord(**row) for row in rows]

    def save_ai_suggested(
        self,
        segments: Sequence[SegmentDefinitionRecord],
    ) -> None:
        for segment in segments:
            if segment.source != "ai_suggested":
                raise ValueError("only ai_suggested segment definitions can be saved")
            if segment.campaign_id is None or segment.promotion_id is None:
                raise ValueError("ai_suggested segment definitions must be promotion scoped")
            self._db.execute(
                """
                INSERT INTO segment_definitions (
                    segment_id,
                    project_id,
                    campaign_id,
                    promotion_id,
                    segment_name,
                    source,
                    query_preview_id,
                    natural_language_query,
                    generated_sql,
                    rule_json,
                    profile_json,
                    sample_size,
                    total_eligible_user_count,
                    sample_ratio,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (segment_id) DO UPDATE SET
                    campaign_id = EXCLUDED.campaign_id,
                    promotion_id = EXCLUDED.promotion_id,
                    segment_name = EXCLUDED.segment_name,
                    natural_language_query = EXCLUDED.natural_language_query,
                    generated_sql = EXCLUDED.generated_sql,
                    rule_json = EXCLUDED.rule_json,
                    profile_json = EXCLUDED.profile_json,
                    sample_size = EXCLUDED.sample_size,
                    total_eligible_user_count = EXCLUDED.total_eligible_user_count,
                    sample_ratio = EXCLUDED.sample_ratio,
                    status = EXCLUDED.status,
                    updated_at = now()
                WHERE segment_definitions.source = 'ai_suggested'
                """,
                (
                    segment.segment_id,
                    segment.project_id,
                    segment.campaign_id,
                    segment.promotion_id,
                    segment.segment_name,
                    segment.source,
                    segment.query_preview_id,
                    segment.natural_language_query,
                    segment.generated_sql,
                    segment.rule_json,
                    segment.profile_json,
                    segment.sample_size,
                    segment.total_eligible_user_count,
                    segment.sample_ratio,
                    segment.status,
                ),
            )


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
                focus_segment_ids_json,
                operator_instruction,
                input_snapshot_json,
                profile_summary_json,
                output_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                analysis.analysis_id,
                analysis.project_id,
                analysis.campaign_id,
                analysis.promotion_id,
                analysis.status,
                analysis.focus_segment_ids_json,
                analysis.operator_instruction,
                analysis.input_snapshot_json,
                analysis.profile_summary_json,
                analysis.output_json,
            ),
        )

    def get_latest_audience_bindings(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        segment_ids: Sequence[str],
    ) -> list[SegmentSuggestionAudienceBindingRecord]:
        if not segment_ids:
            return []
        rows = self._db.fetchall(
            """
            WITH ranked AS (
                SELECT
                    suggestion_id,
                    analysis_id,
                    segment_id,
                    audience_snapshot_id,
                    row_number() OVER (
                        PARTITION BY segment_id
                        ORDER BY created_at DESC, suggestion_id DESC
                    ) AS row_rank
                FROM promotion_segment_suggestions
                WHERE project_id = %s
                  AND campaign_id = %s
                  AND promotion_id = %s
                  AND segment_id = ANY(%s)
                  AND status = 'suggested'
            )
            SELECT
                suggestion_id,
                analysis_id,
                segment_id,
                audience_snapshot_id
            FROM ranked
            WHERE row_rank = 1
            ORDER BY segment_id ASC
            """,
            (project_id, campaign_id, promotion_id, list(segment_ids)),
        )
        return [SegmentSuggestionAudienceBindingRecord(**row) for row in rows]

    def save_segment_suggestions(
        self,
        suggestions: Sequence[PromotionSegmentSuggestionWrite],
    ) -> None:
        for suggestion in suggestions:
            self._db.execute(
                """
                INSERT INTO promotion_segment_suggestions (
                    suggestion_id,
                    analysis_id,
                    project_id,
                    campaign_id,
                    promotion_id,
                    segment_id,
                    suggested_rank,
                    suggestion_source,
                    status,
                    score_json,
                    reason_json,
                    metadata_json,
                    audience_snapshot_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    suggestion.suggestion_id,
                    suggestion.analysis_id,
                    suggestion.project_id,
                    suggestion.campaign_id,
                    suggestion.promotion_id,
                    suggestion.segment_id,
                    suggestion.suggested_rank,
                    suggestion.suggestion_source,
                    suggestion.status,
                    suggestion.score_json,
                    suggestion.reason_json,
                    suggestion.metadata_json,
                    suggestion.audience_snapshot_id,
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
                    data_evidence_json,
                    segment_vector_id,
                    estimated_size,
                    priority,
                    status,
                    audience_snapshot_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    segment.data_evidence_json,
                    segment.segment_vector_id,
                    segment.estimated_size,
                    segment.priority,
                    segment.status,
                    segment.audience_snapshot_id,
                ),
            )


class SegmentVectorRepository:
    VECTOR_DIM = 64

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def get_by_segment_snapshot(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_id: str,
        vector_version: str,
    ) -> SegmentVectorRecord | None:
        row = self._db.fetchone(
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
              AND segment_id = %s
              AND vector_version = %s
              AND vector_dim = %s
            ORDER BY created_at DESC, segment_vector_id DESC
            """,
            (
                project_id,
                promotion_id,
                analysis_id,
                segment_id,
                vector_version,
                self.VECTOR_DIM,
            ),
        )
        if row is None:
            return None
        return SegmentVectorRecord(**row)

    def get_latest_by_segment(
        self,
        *,
        project_id: str,
        promotion_id: str,
        segment_id: str,
        vector_version: str,
    ) -> SegmentVectorRecord | None:
        row = self._db.fetchone(
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
              AND segment_id = %s
              AND vector_version = %s
              AND vector_dim = %s
            ORDER BY created_at DESC, segment_vector_id DESC
            """,
            (
                project_id,
                promotion_id,
                segment_id,
                vector_version,
                self.VECTOR_DIM,
            ),
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
                analysis_id,
                segment_id,
                vector_dim,
                vector_values,
                embedding,
                vector_version,
                source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
            """,
            (
                vector.segment_vector_id,
                vector.project_id,
                vector.promotion_id,
                vector.promotion_run_id,
                vector.analysis_id,
                vector.segment_id,
                vector.vector_dim,
                vector.vector_values,
                _vector_literal(vector.vector_values, self.VECTOR_DIM),
                vector.vector_version,
                vector.source,
            ),
        )

    def _validate_vector(self, vector: SegmentVectorRecord) -> None:
        if vector.vector_dim != self.VECTOR_DIM:
            raise ValueError("segment vector_dim must be 64")
        if len(vector.vector_values) != self.VECTOR_DIM:
            raise ValueError("segment vector_values must contain 64 values")
        numeric_values = [float(value) for value in vector.vector_values]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("segment vector_values must be finite")
        if math.sqrt(sum(value * value for value in numeric_values)) == 0:
            raise ValueError("segment vector_values must not be a zero vector")


class UserBehaviorVectorRepository:
    VECTOR_DIM = 64
    RAW_EVENTS_SOURCE = "raw_events"

    def __init__(self, client: ClickHouseClient) -> None:
        self._client = client

    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: Sequence[str],
        vector_version: str = "v1",
    ) -> list[UserBehaviorVectorRecord]:
        if not user_ids:
            return []

        result = self._client.query(
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
                  AND vector_dim = {vector_dim:UInt16}
                  AND vector_version = {vector_version:String}
                  AND user_id IN {user_ids:Array(String)}
            )
            GROUP BY project_id, user_id, vector_version
            ORDER BY user_id ASC
            """,
            parameters={
                "project_id": project_id,
                "vector_dim": self.VECTOR_DIM,
                "vector_version": vector_version,
                "user_ids": list(user_ids),
            },
        )
        return [
            UserBehaviorVectorRecord(
                project_id=_clickhouse_value(row, "project_id", 0),
                user_id=_clickhouse_value(row, "user_id", 1),
                vector_dim=_clickhouse_value(row, "vector_dim", 2),
                vector_values=[
                    float(value)
                    for value in _clickhouse_value(row, "vector_values", 3)
                ],
                vector_version=_clickhouse_value(row, "vector_version", 4),
                source=_clickhouse_value(row, "source", 5),
            )
            for row in _clickhouse_rows(result)
        ]

    def list_recent(
        self,
        *,
        project_id: str,
        limit: int = 200,
        vector_version: str = "v1",
    ) -> list[UserBehaviorVectorRecord]:
        result = self._client.query(
            """
            SELECT
                project_id,
                user_id,
                argMax(vector_dim, updated_at) AS vector_dim,
                argMax(vector_values, updated_at) AS vector_values,
                vector_version,
                argMax(source, updated_at) AS source,
                max(updated_at) AS last_updated_at
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
                  AND vector_dim = {vector_dim:UInt16}
                  AND vector_version = {vector_version:String}
            )
            GROUP BY project_id, user_id, vector_version
            ORDER BY last_updated_at DESC, user_id ASC
            LIMIT {limit:UInt32}
            """,
            parameters={
                "project_id": project_id,
                "vector_dim": self.VECTOR_DIM,
                "vector_version": vector_version,
                "limit": limit,
            },
        )
        return [
            UserBehaviorVectorRecord(
                project_id=_clickhouse_value(row, "project_id", 0),
                user_id=_clickhouse_value(row, "user_id", 1),
                vector_dim=_clickhouse_value(row, "vector_dim", 2),
                vector_values=[
                    float(value)
                    for value in _clickhouse_value(row, "vector_values", 3)
                ],
                vector_version=_clickhouse_value(row, "vector_version", 4),
                source=_clickhouse_value(row, "source", 5),
            )
            for row in _clickhouse_rows(result)
        ]

    def list_raw_event_user_signals(
        self,
        *,
        project_id: str,
        vector_version: str = "v1",
        destination_terms: Sequence[str] = (),
        season_months: Sequence[int] = (),
        limit: int = 1000,
    ) -> list[RawEventUserSignalRecord]:
        # clickhouse-connect serializes Array(String) parameters from lists.
        # Tuples are rendered as SQL tuples and fail at runtime for this placeholder.
        cleaned_destination_terms = [
            str(term).strip().lower()
            for term in destination_terms
            if str(term).strip()
        ]
        result = self._client.query(
            """
            WITH
                (
                    SELECT argMax(window_start, updated_at)
                    FROM user_behavior_vectors
                    WHERE project_id = {project_id:String}
                      AND vector_dim = {vector_dim:UInt16}
                      AND vector_version = {vector_version:String}
                      AND source = {vector_source:String}
                ) AS vector_window_start,
                (
                    SELECT argMax(window_end, updated_at)
                    FROM user_behavior_vectors
                    WHERE project_id = {project_id:String}
                      AND vector_dim = {vector_dim:UInt16}
                      AND vector_version = {vector_version:String}
                      AND source = {vector_source:String}
                ) AS vector_window_end
            SELECT
                project_id,
                user_id,
                count() AS event_count,
                countIf(event_name = 'hotel_search') AS hotel_search_count,
                countIf(event_name = 'hotel_click') AS hotel_click_count,
                countIf(event_name = 'hotel_detail_view') AS hotel_detail_view_count,
                countIf(event_name = 'promotion_impression') AS promotion_impression_count,
                countIf(event_name = 'promotion_click') AS promotion_click_count,
                countIf(event_name = 'campaign_redirect_click') AS campaign_redirect_click_count,
                countIf(event_name = 'campaign_landing') AS campaign_landing_count,
                countIf(event_name = 'booking_start') AS booking_start_count,
                countIf(event_name = 'booking_complete') AS booking_complete_count,
                countIf(event_name = 'booking_cancel') AS booking_cancel_count,
                countIf(nullIf(JSONExtractString(properties_json, 'deal'), '') != '') AS deal_event_count,
                countIf(toUInt8OrZero(JSONExtractString(properties_json, 'free_cancellation')) = 1) AS free_cancellation_count,
                countIf(toUInt8OrZero(JSONExtractString(properties_json, 'breakfast_included')) = 1) AS breakfast_included_count,
                countIf(nullIf(JSONExtractString(properties_json, 'price'), '') != '') AS price_event_count,
                avgIf(
                    toFloat64OrZero(JSONExtractString(properties_json, 'price')),
                    nullIf(JSONExtractString(properties_json, 'price'), '') != ''
                ) AS avg_price,
                groupUniqArray(20)(
                    concat(
                        ifNull(JSONExtractString(properties_json, 'destination_id'), ''),
                        ' ',
                        ifNull(JSONExtractString(properties_json, 'destination_name'), ''),
                        ' ',
                        ifNull(JSONExtractString(properties_json, 'hotel_city'), ''),
                        ' ',
                        ifNull(JSONExtractString(properties_json, 'hotel_country'), '')
                    )
                ) AS destination_values,
                groupUniqArray(20)(ifNull(JSONExtractString(properties_json, 'checkin_date'), '')) AS checkin_dates,
                groupUniqArray(20)(ifNull(JSONExtractString(properties_json, 'hotel_market'), '')) AS hotel_market_values,
                groupUniqArray(20)(ifNull(JSONExtractString(properties_json, 'hotel_cluster'), '')) AS hotel_cluster_values,
                groupUniqArray(10)(ifNull(JSONExtractString(properties_json, 'age_group'), '')) AS age_group_values,
                groupUniqArray(10)(ifNull(JSONExtractString(properties_json, 'gender'), '')) AS gender_values,
                groupUniqArray(10)(ifNull(JSONExtractString(properties_json, 'preferred_category'), '')) AS preferred_category_values,
                countIf(
                    arrayExists(
                        term -> positionCaseInsensitiveUTF8(
                            concat(
                                ifNull(JSONExtractString(properties_json, 'destination_id'), ''),
                                ' ',
                                ifNull(JSONExtractString(properties_json, 'destination_name'), ''),
                                ' ',
                                ifNull(JSONExtractString(properties_json, 'hotel_city'), ''),
                                ' ',
                                ifNull(JSONExtractString(properties_json, 'hotel_country'), '')
                            ),
                            term
                        ) > 0,
                        {destination_terms:Array(String)}
                    )
                ) AS destination_match_count
            FROM raw_events
            WHERE project_id = {project_id:String}
              AND validation_status = 'valid'
              AND notEmpty(user_id)
              AND event_time >= vector_window_start
              AND event_time < vector_window_end
              AND user_id IN (
                  SELECT user_id
                  FROM user_behavior_vectors
                  WHERE project_id = {project_id:String}
                    AND vector_dim = {vector_dim:UInt16}
                    AND vector_version = {vector_version:String}
                    AND source = {vector_source:String}
                    AND window_start = vector_window_start
                    AND window_end = vector_window_end
                  GROUP BY user_id
              )
            GROUP BY project_id, user_id
            ORDER BY max(event_time) DESC, user_id ASC
            LIMIT {limit:UInt32}
            """,
            parameters={
                "project_id": project_id,
                "vector_dim": self.VECTOR_DIM,
                "vector_version": vector_version,
                "vector_source": self.RAW_EVENTS_SOURCE,
                "destination_terms": cleaned_destination_terms,
                "limit": limit,
            },
        )
        cleaned_season_months = {
            int(month)
            for month in season_months
            if 1 <= int(month) <= 12
        }
        records: list[RawEventUserSignalRecord] = []
        for row in _clickhouse_rows(result):
            destination_values = _clean_string_tuple(
                _clickhouse_value(row, "destination_values", 18)
            )
            checkin_dates = _clean_string_tuple(
                _clickhouse_value(row, "checkin_dates", 19)
            )
            records.append(
                RawEventUserSignalRecord(
                    project_id=_clickhouse_value(row, "project_id", 0),
                    user_id=_clickhouse_value(row, "user_id", 1),
                    event_count=int(_clickhouse_value(row, "event_count", 2) or 0),
                    hotel_search_count=int(
                        _clickhouse_value(row, "hotel_search_count", 3) or 0
                    ),
                    hotel_click_count=int(
                        _clickhouse_value(row, "hotel_click_count", 4) or 0
                    ),
                    hotel_detail_view_count=int(
                        _clickhouse_value(row, "hotel_detail_view_count", 5) or 0
                    ),
                    promotion_impression_count=int(
                        _clickhouse_value(row, "promotion_impression_count", 6) or 0
                    ),
                    promotion_click_count=int(
                        _clickhouse_value(row, "promotion_click_count", 7) or 0
                    ),
                    campaign_redirect_click_count=int(
                        _clickhouse_value(row, "campaign_redirect_click_count", 8) or 0
                    ),
                    campaign_landing_count=int(
                        _clickhouse_value(row, "campaign_landing_count", 9) or 0
                    ),
                    booking_start_count=int(
                        _clickhouse_value(row, "booking_start_count", 10) or 0
                    ),
                    booking_complete_count=int(
                        _clickhouse_value(row, "booking_complete_count", 11) or 0
                    ),
                    booking_cancel_count=int(
                        _clickhouse_value(row, "booking_cancel_count", 12) or 0
                    ),
                    deal_event_count=int(
                        _clickhouse_value(row, "deal_event_count", 13) or 0
                    ),
                    free_cancellation_count=int(
                        _clickhouse_value(row, "free_cancellation_count", 14) or 0
                    ),
                    breakfast_included_count=int(
                        _clickhouse_value(row, "breakfast_included_count", 15) or 0
                    ),
                    price_event_count=int(
                        _clickhouse_value(row, "price_event_count", 16) or 0
                    ),
                    avg_price=float(_clickhouse_value(row, "avg_price", 17) or 0.0),
                    destination_values=destination_values,
                    checkin_dates=checkin_dates,
                    hotel_market_values=_clean_string_tuple(
                        _clickhouse_value(row, "hotel_market_values", 20)
                    ),
                    hotel_cluster_values=_clean_string_tuple(
                        _clickhouse_value(row, "hotel_cluster_values", 21)
                    ),
                    age_group_values=_clean_string_tuple(
                        _clickhouse_value(row, "age_group_values", 22)
                    ),
                    gender_values=_clean_string_tuple(
                        _clickhouse_value(row, "gender_values", 23)
                    ),
                    preferred_category_values=_clean_string_tuple(
                        _clickhouse_value(row, "preferred_category_values", 24)
                    ),
                    destination_match_count=int(
                        _clickhouse_value(row, "destination_match_count", 25) or 0
                    ),
                    season_match_count=_season_match_count(
                        values=checkin_dates,
                        season_months=cleaned_season_months,
                    ),
                )
            )
        return records


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
                primary_segment,
                count() AS event_count,
                countIf(is_booking = 1) AS booking_count,
                avg(is_mobile) AS mobile_ratio,
                avg(is_package) AS package_ratio,
                avg(stay_nights) AS avg_stay_nights,
                avg(days_until_checkin) AS avg_days_until_checkin
            FROM hotel_marketing_profiles
            GROUP BY primary_segment
            ORDER BY event_count DESC
            """,
        )
        return [
            HotelMarketingProfileRecord(
                project_id=project_id,
                profile_name=_clickhouse_value(row, "primary_segment", 0),
                profile_json={
                    "event_count": _clickhouse_value(row, "event_count", 1),
                    "booking_count": _clickhouse_value(row, "booking_count", 2),
                    "mobile_ratio": _clickhouse_value(row, "mobile_ratio", 3),
                    "package_ratio": _clickhouse_value(row, "package_ratio", 4),
                    "avg_stay_nights": _clickhouse_value(row, "avg_stay_nights", 5),
                    "avg_days_until_checkin": _clickhouse_value(
                        row,
                        "avg_days_until_checkin",
                        6,
                    ),
                },
            )
            for row in _clickhouse_rows(result)
        ]

    def summarize_user_ids(
        self,
        *,
        project_id: str,
        profile_name: str,
        user_ids: Sequence[str],
    ) -> HotelMarketingProfileRecord | None:
        if not user_ids:
            return None

        result = self._client.query(
            """
            SELECT
                count() AS event_count,
                countIf(is_booking = 1) AS booking_count,
                avg(is_mobile) AS mobile_ratio,
                avg(is_package) AS package_ratio,
                avg(stay_nights) AS avg_stay_nights,
                avg(days_until_checkin) AS avg_days_until_checkin
            FROM hotel_marketing_profiles
            WHERE user_id IN {user_ids:Array(String)}
            """,
            parameters={"user_ids": list(user_ids)},
        )
        rows = _clickhouse_rows(result)
        if not rows:
            return None

        row = rows[0]
        event_count = int(_clickhouse_value(row, "event_count", 0) or 0)
        if event_count <= 0:
            return None

        return HotelMarketingProfileRecord(
            project_id=project_id,
            profile_name=profile_name,
            profile_json={
                "event_count": event_count,
                "booking_count": _clickhouse_value(row, "booking_count", 1),
                "mobile_ratio": _clickhouse_value(row, "mobile_ratio", 2),
                "package_ratio": _clickhouse_value(row, "package_ratio", 3),
                "avg_stay_nights": _clickhouse_value(row, "avg_stay_nights", 4),
                "avg_days_until_checkin": _clickhouse_value(
                    row,
                    "avg_days_until_checkin",
                    5,
                ),
            },
        )

    def list_booking_training_records(
        self,
        *,
        limit: int = 500,
    ) -> list[BookingTrainingRecord]:
        result = self._client.query(
            """
            SELECT
                is_mobile,
                is_package,
                if(
                    isNull(srch_ci) OR isNull(srch_co),
                    0,
                    least(greatest(dateDiff('day', srch_ci, srch_co), 0), 14)
                ) AS stay_nights,
                if(
                    isNull(srch_ci),
                    30,
                    least(greatest(dateDiff('day', toDate(date_time), srch_ci), 0), 60)
                ) AS days_until_checkin,
                count() AS event_count,
                countIf(is_booking = 1) AS booking_count
            FROM expedia_hotel_events
            GROUP BY
                is_mobile,
                is_package,
                stay_nights,
                days_until_checkin
            ORDER BY event_count DESC
            LIMIT {limit:UInt32}
            """,
            parameters={"limit": limit},
        )
        return [
            BookingTrainingRecord(
                is_mobile=float(_clickhouse_value(row, "is_mobile", 0)),
                is_package=float(_clickhouse_value(row, "is_package", 1)),
                stay_nights=float(_clickhouse_value(row, "stay_nights", 2)),
                days_until_checkin=float(
                    _clickhouse_value(row, "days_until_checkin", 3),
                ),
                event_count=int(_clickhouse_value(row, "event_count", 4)),
                booking_count=int(_clickhouse_value(row, "booking_count", 5)),
            )
            for row in _clickhouse_rows(result)
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
            GROUP BY hotel_cluster
            ORDER BY event_count DESC
            LIMIT {limit:UInt32}
            """,
            parameters={"limit": limit},
        )
        return [
            {
                "hotel_cluster": _clickhouse_value(row, "hotel_cluster", 0),
                "event_count": _clickhouse_value(row, "event_count", 1),
            }
            for row in _clickhouse_rows(result)
        ]


def _clickhouse_rows(result: Any) -> list[Any]:
    if hasattr(result, "named_results"):
        return list(result.named_results())
    return list(result.result_rows)


def _clickhouse_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return row[index]


def _clean_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        raw_values = ()
    cleaned: list[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def _season_match_count(*, values: Sequence[str], season_months: set[int]) -> int:
    if not season_months:
        return 0
    matched = 0
    for value in values:
        parts = value.replace("/", "-").split("-")
        if len(parts) < 2:
            continue
        try:
            month = int(parts[1])
        except ValueError:
            continue
        if month in season_months:
            matched += 1
    return matched


def _vector_literal(values: Sequence[float], vector_dim: int) -> str:
    if len(values) != vector_dim:
        raise ValueError("vector literal must contain 64 values")
    numeric_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("vector literal values must be finite")
    return "[" + ",".join(str(value) for value in numeric_values) + "]"
