from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
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


@dataclass(frozen=True)
class UserBehaviorVectorRecord:
    project_id: str
    user_id: str
    vector_dim: int
    vector_values: list[float]
    vector_version: str
    source: str


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
                    metadata_json
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
                analysis_id,
                segment_id,
                vector_dim,
                vector_values,
                vector_version,
                source
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
                analysis_id,
                segment_id,
                vector_dim,
                vector_values,
                vector_version,
                source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                vector.vector_version,
                vector.source,
            ),
        )

    def _validate_vector(self, vector: SegmentVectorRecord) -> None:
        if vector.vector_dim != self.VECTOR_DIM:
            raise ValueError("segment vector_dim must be 64")
        if len(vector.vector_values) != self.VECTOR_DIM:
            raise ValueError("segment vector_values must contain 64 values")


class UserBehaviorVectorRepository:
    VECTOR_DIM = 64

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
                vector_dim,
                vector_values,
                vector_version,
                source
            FROM user_behavior_vectors
            WHERE project_id = {project_id:String}
              AND vector_dim = {vector_dim:UInt16}
              AND vector_version = {vector_version:String}
              AND user_id IN {user_ids:Array(String)}
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
                vector_dim,
                vector_values,
                vector_version,
                source
            FROM user_behavior_vectors
            WHERE project_id = {project_id:String}
              AND vector_dim = {vector_dim:UInt16}
              AND vector_version = {vector_version:String}
            ORDER BY updated_at DESC, user_id ASC
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
