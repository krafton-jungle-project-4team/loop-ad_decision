from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Protocol

from app.internal.schemas import UserBehaviorVectorBuildRequest
from app.logging import log, log_context_scope, now_ms, duration_ms


RAW_EVENTS_SOURCE = "raw_events"


class ClickHouseBatchClient(Protocol):
    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class UserBehaviorVectorBuildResult:
    project_id: str
    vector_version: str
    source: str
    vector_dim: int
    processed_user_count: int
    window_start: datetime
    window_end: datetime
    status: str


class UserBehaviorVectorBatchService:
    VECTOR_DIM = 64

    def __init__(
        self,
        repository: "UserBehaviorVectorBuildRepository",
        *,
        now: datetime | None = None,
    ) -> None:
        self._repository = repository
        self._now = now

    @log_context_scope
    def build(
        self,
        request: UserBehaviorVectorBuildRequest,
    ) -> UserBehaviorVectorBuildResult:
        started_at = now_ms()
        log.assign_context({"projectId": request.project_id})
        log.info("started", {"request": request})
        window_end = self._resolve_now()
        window_start = window_end - timedelta(days=request.window_days)
        processed_user_count = self._repository.count_raw_event_users(
            project_id=request.project_id,
            window_start=window_start,
            window_end=window_end,
        )

        if processed_user_count > 0:
            self._repository.insert_raw_event_user_vectors(
                project_id=request.project_id,
                vector_version=request.vector_version,
                source=RAW_EVENTS_SOURCE,
                window_start=window_start,
                window_end=window_end,
            )
            log.info(
                "user_behavior_vectors_created",
                {
                    "processedUserCount": processed_user_count,
                    "vectorVersion": request.vector_version,
                },
            )
        else:
            log.info("raw_event_users_empty", {"processedUserCount": 0})

        response = UserBehaviorVectorBuildResult(
            project_id=request.project_id,
            vector_version=request.vector_version,
            source=RAW_EVENTS_SOURCE,
            vector_dim=self.VECTOR_DIM,
            processed_user_count=processed_user_count,
            window_start=window_start,
            window_end=window_end,
            status="completed",
        )
        log.info(
            "completed",
            {"response": response, "durationMs": duration_ms(started_at)},
        )
        return response

    def _resolve_now(self) -> datetime:
        value = self._now or datetime.now(UTC)
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class UserBehaviorVectorBuildRepository:
    VECTOR_DIM = 64

    def __init__(self, client: ClickHouseBatchClient) -> None:
        self._client = client

    def count_raw_event_users(
        self,
        *,
        project_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> int:
        result = self._client.query(
            """
            SELECT countDistinct(user_id) AS processed_user_count
            FROM raw_events
            WHERE project_id = {project_id:String}
              AND validation_status = 'valid'
              AND event_time >= toDateTime64(
                  parseDateTimeBestEffort({window_start:String}),
                  3,
                  'UTC'
              )
              AND event_time < toDateTime64(
                  parseDateTimeBestEffort({window_end:String}),
                  3,
                  'UTC'
              )
            """,
            parameters={
                "project_id": project_id,
                "window_start": _clickhouse_datetime(window_start),
                "window_end": _clickhouse_datetime(window_end),
            },
        )
        return int(_first_scalar(result, "processed_user_count"))

    def insert_raw_event_user_vectors(
        self,
        *,
        project_id: str,
        vector_version: str,
        source: str,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        parameters = {
            "project_id": project_id,
            "vector_dim": self.VECTOR_DIM,
            "vector_version": vector_version,
            "source": source,
            "window_start": _clickhouse_datetime(window_start),
            "window_end": _clickhouse_datetime(window_end),
        }
        self._execute_insert(_build_raw_events_insert_sql(), parameters)

    def _execute_insert(self, query: str, parameters: Mapping[str, Any]) -> None:
        command = getattr(self._client, "command", None)
        if callable(command):
            command(query, parameters=parameters)
            return
        self._client.query(query, parameters=parameters)


def _build_raw_events_insert_sql() -> str:
    event_names = [
        "page_view",
        "hotel_search",
        "hotel_click",
        "hotel_detail_view",
        "promotion_impression",
        "promotion_click",
        "campaign_redirect_click",
        "campaign_landing",
        "booking_start",
        "booking_complete",
        "booking_cancel",
    ]
    event_counts = [
        f"countIf(event_name = '{event_name}') AS {event_name}_count"
        for event_name in event_names
    ]
    event_counts.append(
        "countIf(event_name NOT IN ("
        + ", ".join(f"'{event_name}'" for event_name in event_names)
        + ")) AS other_event_count"
    )
    source_counts = [
        f"countIf(modulo(cityHash64(source), 4) = {index}) "
        f"AS source_bucket_{index}_count"
        for index in range(4)
    ]
    hotel_cluster_counts = [
        "countIf("
        "nullIf(JSONExtractString(properties_json, 'hotel_cluster'), '') != '' "
        f"AND modulo(cityHash64(JSONExtractString(properties_json, 'hotel_cluster')), 16) = {index}"
        f") AS hotel_cluster_bucket_{index}_count"
        for index in range(16)
    ]
    hotel_market_counts = [
        "countIf("
        "nullIf(JSONExtractString(properties_json, 'hotel_market'), '') != '' "
        f"AND modulo(cityHash64(JSONExtractString(properties_json, 'hotel_market')), 16) = {index}"
        f") AS hotel_market_bucket_{index}_count"
        for index in range(16)
    ]
    page_path_counts = [
        "countIf("
        "nullIf(JSONExtractString(properties_json, 'page_path'), '') != '' "
        f"AND modulo(cityHash64(JSONExtractString(properties_json, 'page_path')), 8) = {index}"
        f") AS page_path_bucket_{index}_count"
        for index in range(8)
    ]
    aggregate_terms = ",\n                    ".join(
        [
            *event_counts,
            *source_counts,
            *hotel_cluster_counts,
            *hotel_market_counts,
            *page_path_counts,
            "countIf(nullIf(JSONExtractString(properties_json, 'promotion_id'), '') != '') AS promotion_event_count",
            "countIf(nullIf(JSONExtractString(properties_json, 'ad_experiment_id'), '') != '') AS ad_experiment_event_count",
            "countIf(nullIf(JSONExtractString(properties_json, 'segment_id'), '') != '') AS segment_event_count",
            "countIf(toUInt8OrZero(JSONExtractString(properties_json, 'free_cancellation')) = 1) AS free_cancellation_count",
            "countIf(toUInt8OrZero(JSONExtractString(properties_json, 'breakfast_included')) = 1) AS breakfast_included_count",
            """
            if(
                countIf(nullIf(JSONExtractString(properties_json, 'price'), '') != '') = 0,
                0.0,
                avgIf(
                    least(
                        greatest(
                            toFloat64OrZero(JSONExtractString(properties_json, 'price')) / 1000000.0,
                            0.0
                        ),
                        1.0
                    ),
                    nullIf(JSONExtractString(properties_json, 'price'), '') != ''
                )
            ) AS price_score
            """,
        ]
    )
    vector_terms = [
        *[_ratio_term(f"{event_name}_count") for event_name in event_names],
        _ratio_term("other_event_count"),
        *[_ratio_term(f"source_bucket_{index}_count") for index in range(4)],
        *[
            _ratio_term(f"hotel_cluster_bucket_{index}_count")
            for index in range(16)
        ],
        *[
            _ratio_term(f"hotel_market_bucket_{index}_count")
            for index in range(16)
        ],
        *[_ratio_term(f"page_path_bucket_{index}_count") for index in range(8)],
        _ratio_term("promotion_event_count"),
        _ratio_term("ad_experiment_event_count"),
        _ratio_term("segment_event_count"),
        _ratio_term("free_cancellation_count"),
        _ratio_term("breakfast_included_count"),
        "price_score",
        "if(booking_start_count = 0, 0.0, toFloat64(booking_complete_count) / toFloat64(booking_start_count))",
        "if(promotion_impression_count = 0, 0.0, toFloat64(promotion_click_count) / toFloat64(promotion_impression_count))",
    ]
    if len(vector_terms) != 64:
        raise RuntimeError("raw event user behavior vector must contain 64 features")
    vector_sql = ",\n                ".join(vector_terms)

    return f"""
        INSERT INTO user_behavior_vectors (
            project_id,
            user_id,
            vector_dim,
            vector_values,
            vector_version,
            source,
            window_start,
            window_end,
            updated_at
        )
        SELECT
            {{project_id:String}} AS project_id,
            user_id,
            toUInt16({{vector_dim:UInt16}}) AS vector_dim,
            arrayMap(
                value -> toFloat32(value),
                [
                {vector_sql}
                ]
            ) AS vector_values,
            {{vector_version:String}} AS vector_version,
            {{source:String}} AS source,
            toDateTime64(parseDateTimeBestEffort({{window_start:String}}), 3, 'UTC')
                AS window_start,
            toDateTime64(parseDateTimeBestEffort({{window_end:String}}), 3, 'UTC')
                AS window_end,
            now64(3) AS updated_at
        FROM (
            SELECT
                user_id,
                count() AS event_count,
                    {aggregate_terms}
            FROM raw_events
            WHERE project_id = {{project_id:String}}
              AND validation_status = 'valid'
              AND event_time >= toDateTime64(
                  parseDateTimeBestEffort({{window_start:String}}),
                  3,
                  'UTC'
              )
              AND event_time < toDateTime64(
                  parseDateTimeBestEffort({{window_end:String}}),
                  3,
                  'UTC'
              )
            GROUP BY user_id
        ) AS user_profiles
        WHERE event_count > 0
        """


def _ratio_term(count_column: str) -> str:
    return (
        f"if(event_count = 0, 0.0, toFloat64({count_column}) / "
        "toFloat64(event_count))"
    )


def _first_scalar(result: Any, key: str) -> Any:
    rows = _clickhouse_rows(result)
    if not rows:
        return 0
    first_row = rows[0]
    if isinstance(first_row, Mapping):
        return first_row[key]
    return first_row[0]


def _clickhouse_rows(result: Any) -> list[Any]:
    if hasattr(result, "named_results"):
        return list(result.named_results())
    return list(result.result_rows)


def _clickhouse_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
