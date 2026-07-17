from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import uuid
from typing import Any, Mapping, Protocol

from app.analysis.behavior_manifest import (
    behavior_manifest_hash,
    clickhouse_canonical_destination_sql,
    order_vector_terms_by_manifest,
)
from app.internal.schemas import UserBehaviorVectorBuildRequest
from app.logging import log, log_context_scope, now_ms, duration_ms


RAW_EVENTS_SOURCE = "raw_events"
HOTEL_BEHAVIOR_V2 = "hotel_behavior.v2"


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
    vector_generation_id: str
    expected_user_count: int
    manifest_hash: str
    source_revision_cutoff: datetime
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
        manifest_hash = (
            behavior_manifest_hash()
            if request.vector_version == HOTEL_BEHAVIOR_V2
            else hashlib.sha256(
                _build_raw_events_insert_sql().encode("utf-8")
            ).hexdigest()
        )
        vector_generation_id = deterministic_vector_generation_id(
            project_id=request.project_id,
            vector_version=request.vector_version,
            window_start=window_start,
            window_end=window_end,
        )
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
            log.info("user_behavior_vectors_created", {"processedUserCount": processed_user_count, "vectorVersion": request.vector_version})
        else:
            log.info("raw_event_users_empty", {"processedUserCount": 0})

        source_revision_cutoff = window_end
        if request.vector_version == HOTEL_BEHAVIOR_V2 and processed_user_count > 0:
            source_revision_cutoff = self._repository.get_revision_cutoff(
                project_id=request.project_id,
                vector_version=request.vector_version,
                window_start=window_start,
                window_end=window_end,
            )
            if source_revision_cutoff is None:
                raise RuntimeError(
                    "ClickHouse vector revision was not materialized for the build"
                )

        response = UserBehaviorVectorBuildResult(
            project_id=request.project_id,
            vector_version=request.vector_version,
            source=RAW_EVENTS_SOURCE,
            vector_dim=self.VECTOR_DIM,
            processed_user_count=processed_user_count,
            vector_generation_id=vector_generation_id,
            expected_user_count=processed_user_count,
            manifest_hash=manifest_hash,
            source_revision_cutoff=source_revision_cutoff,
            window_start=window_start,
            window_end=window_end,
            status="completed",
        )
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response

    def _resolve_now(self) -> datetime:
        value = self._now or datetime.now(UTC)
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        # ClickHouse request parameters are serialized at second precision.
        # Freeze the generation identity at the same precision so PostgreSQL
        # generation metadata always selects the exact ClickHouse window.
        return value.replace(microsecond=0)


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
        query = (
            _build_hotel_behavior_v2_insert_sql()
            if vector_version == HOTEL_BEHAVIOR_V2
            else _build_raw_events_insert_sql()
        )
        self._execute_insert(query, parameters)

    def get_revision_cutoff(
        self,
        *,
        project_id: str,
        vector_version: str,
        window_start: datetime,
        window_end: datetime,
    ) -> datetime | None:
        result = self._client.query(
            """
            SELECT max(ingested_at) AS source_revision_cutoff
            FROM user_behavior_vector_revisions
            WHERE project_id = {project_id:String}
              AND vector_version = {vector_version:String}
              AND window_start = toDateTime64(
                  parseDateTimeBestEffort({window_start:String}), 3, 'UTC'
              )
              AND window_end = toDateTime64(
                  parseDateTimeBestEffort({window_end:String}), 3, 'UTC'
              )
            """,
            parameters={
                "project_id": project_id,
                "vector_version": vector_version,
                "window_start": _clickhouse_datetime(window_start),
                "window_end": _clickhouse_datetime(window_end),
            },
        )
        value = _first_scalar(result, "source_revision_cutoff")
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise RuntimeError("ClickHouse revision cutoff must be a datetime")
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _execute_insert(self, query: str, parameters: Mapping[str, Any]) -> None:
        command = getattr(self._client, "command", None)
        if callable(command):
            command(query, parameters=parameters)
            return
        self._client.query(query, parameters=parameters)


def deterministic_vector_generation_id(
    *,
    project_id: str,
    vector_version: str,
    window_start: datetime,
    window_end: datetime,
) -> str:
    value = ":".join(
        (
            project_id,
            vector_version,
            window_start.astimezone(UTC).isoformat(),
            window_end.astimezone(UTC).isoformat(),
        )
    )
    return "uvgen_" + uuid.uuid5(uuid.NAMESPACE_URL, value).hex


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


def _build_hotel_behavior_v2_insert_sql() -> str:
    destination_source = """
        coalesce(
            nullIf(JSONExtractString(properties_json, 'destination_id'), ''),
            nullIf(JSONExtractString(properties_json, 'destination_name'), ''),
            nullIf(JSONExtractString(properties_json, 'hotel_city'), ''),
            nullIf(JSONExtractString(properties_json, 'hotel_market'), ''),
            nullIf(JSONExtractString(properties_json, 'hotel_cluster'), ''),
            ''
        )
    """.strip()
    destination_value = clickhouse_canonical_destination_sql(destination_source)
    destination_hash = f"SHA256({destination_value})"
    destination_bucket = (
        "modulo(reinterpretAsUInt64(reverse(substring("
        f"{destination_hash}, 1, 8))), 16)"
    )
    destination_sign = (
        "if(bitAnd(reinterpretAsUInt8(substring("
        f"{destination_hash}, 9, 1)), 1) = 0, 1.0, -1.0)"
    )
    destination_terms = [
        "sumIf("
        f"{destination_sign}, {destination_value} != '' "
        f"AND {destination_bucket} = {index}) AS destination_bucket_{index}"
        for index in range(16)
    ]
    aggregates = [
        "countIf(event_name = 'page_view') AS page_view_count",
        "countIf(event_name = 'hotel_search') AS hotel_search_count",
        "countIf(event_name = 'hotel_click') AS hotel_click_count",
        "countIf(event_name = 'hotel_detail_view') AS hotel_detail_view_count",
        "countIf(event_name = 'booking_start') AS booking_start_count",
        "countIf(event_name = 'booking_complete') AS booking_complete_count",
        "countIf(event_name = 'booking_cancel') AS booking_cancel_count",
        "countIf(event_name = 'promotion_impression') AS promotion_impression_count",
        "countIf(event_name = 'promotion_click') AS promotion_click_count",
        "countIf(event_name = 'campaign_redirect_click') AS campaign_redirect_count",
        "countIf(event_name = 'campaign_landing') AS campaign_landing_count",
        "maxIf(event_time, event_name = 'hotel_search') AS last_search_at",
        "maxIf(event_time, event_name = 'hotel_detail_view') AS last_detail_at",
        "maxIf(event_time, event_name = 'booking_start') AS last_booking_start_at",
        "maxIf(event_time, event_name IN ('promotion_click','campaign_landing')) AS last_response_at",
        "countIf(toUInt8OrZero(JSONExtractString(properties_json, 'deal')) = 1) AS deal_count",
        "maxIf(event_time, toUInt8OrZero(JSONExtractString(properties_json, 'deal')) = 1) AS last_deal_at",
        "countIf(nullIf(JSONExtractString(properties_json, 'price'), '') IS NOT NULL) AS price_count",
        "countIf(toFloat64OrZero(JSONExtractString(properties_json, 'price')) > 0 AND toFloat64OrZero(JSONExtractString(properties_json, 'price')) < 150000) AS budget_price_count",
        "countIf(toFloat64OrZero(JSONExtractString(properties_json, 'price')) >= 300000) AS premium_price_count",
        "countIf(toUInt8OrZero(JSONExtractString(properties_json, 'free_cancellation')) = 1) AS free_cancellation_count",
        "countIf(toUInt8OrZero(JSONExtractString(properties_json, 'breakfast_included')) = 1) AS breakfast_count",
        f"uniqExactIf({destination_value}, {destination_value} != '') AS destination_count",
        "countIf(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date')) IS NOT NULL) AS checkin_date_count",
        "countIf(toMonth(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date'))) IN (3,4,5)) AS spring_count",
        "countIf(toMonth(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date'))) IN (6,7,8)) AS summer_count",
        "countIf(toMonth(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date'))) IN (9,10,11)) AS fall_count",
        "countIf(toMonth(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date'))) IN (12,1,2)) AS winter_count",
        "countIf(dateDiff('day', toDate(event_time), toDate(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date')))) BETWEEN 0 AND 7) AS lead_0_7_count",
        "countIf(dateDiff('day', toDate(event_time), toDate(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date')))) BETWEEN 8 AND 30) AS lead_8_30_count",
        "countIf(dateDiff('day', toDate(event_time), toDate(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date')))) > 30) AS lead_gt_30_count",
        "countIf(toDayOfWeek(parseDateTimeBestEffortOrNull(JSONExtractString(properties_json, 'checkin_date'))) IN (6,7)) AS weekend_count",
        *destination_terms,
    ]
    intensity = lambda name: (
        f"if(event_count = 0, 0.0, least(1.0, log1p(toFloat64({name})) / "
        "log1p(toFloat64(greatest(event_count, 1)))))"
    )
    recency = lambda name: (
        f"if(toUnixTimestamp64Milli({name}) <= 0, 0.0, "
        f"exp(-toFloat64(dateDiff('day', {name}, window_end_value)) / 30.0))"
    )
    smooth = lambda numerator, denominator: (
        f"(toFloat64({numerator}) + 1.0) / (toFloat64({denominator}) + 10.0)"
    )
    vector_terms_by_name = {
        "page_view_intensity": intensity("page_view_count"),
        "hotel_search_intensity": intensity("hotel_search_count"),
        "hotel_click_intensity": intensity("hotel_click_count"),
        "hotel_detail_view_intensity": intensity("hotel_detail_view_count"),
        "booking_start_intensity": intensity("booking_start_count"),
        "booking_complete_intensity": intensity("booking_complete_count"),
        "booking_cancel_intensity": intensity("booking_cancel_count"),
        "hotel_search_recency": recency("last_search_at"),
        "hotel_detail_recency": recency("last_detail_at"),
        "booking_start_recency": recency("last_booking_start_at"),
        "search_to_click_rate": smooth("hotel_click_count", "hotel_search_count"),
        "click_to_detail_rate": smooth(
            "hotel_detail_view_count",
            "hotel_click_count",
        ),
        "detail_to_booking_start_rate": smooth(
            "booking_start_count",
            "hotel_detail_view_count",
        ),
        "booking_completion_rate": smooth(
            "booking_complete_count",
            "booking_start_count",
        ),
        "booking_start_without_complete": (
            "if(booking_start_count > booking_complete_count "
            "AND booking_start_count > 0, 1.0, 0.0)"
        ),
        "hotel_consideration_intensity": intensity(
            "hotel_detail_view_count + hotel_click_count"
        ),
        **{
            f"destination_signed_hash_{index:02d}": (
                "if(event_count = 0, 0.0, "
                f"destination_bucket_{index} / toFloat64(event_count))"
            )
            for index in range(16)
        },
        **{
            dimension: (
                "if(checkin_date_count = 0, 0.0, "
                f"toFloat64({count_name}) / toFloat64(checkin_date_count))"
            )
            for dimension, count_name in (
                ("spring_checkin_share", "spring_count"),
                ("summer_checkin_share", "summer_count"),
                ("fall_checkin_share", "fall_count"),
                ("winter_checkin_share", "winter_count"),
                ("lead_time_0_7_share", "lead_0_7_count"),
                ("lead_time_8_30_share", "lead_8_30_count"),
                ("lead_time_gt_30_share", "lead_gt_30_count"),
                ("weekend_checkin_share", "weekend_count"),
            )
        },
        "deal_interest_intensity": intensity("deal_count"),
        "deal_interest_recency": recency("last_deal_at"),
        "price_interest_intensity": intensity("price_count"),
        "budget_price_share": (
            "if(price_count = 0, 0.0, "
            "toFloat64(budget_price_count) / toFloat64(price_count))"
        ),
        "premium_price_share": (
            "if(price_count = 0, 0.0, "
            "toFloat64(premium_price_count) / toFloat64(price_count))"
        ),
        "free_cancellation_interest": intensity("free_cancellation_count"),
        "breakfast_interest": intensity("breakfast_count"),
        "benefit_breadth": (
            "least(1.0, toFloat64((deal_count > 0) + "
            "(free_cancellation_count > 0) + (breakfast_count > 0)) / 3.0)"
        ),
        "promotion_impression_intensity": intensity(
            "promotion_impression_count"
        ),
        "promotion_click_intensity": intensity("promotion_click_count"),
        "campaign_redirect_intensity": intensity("campaign_redirect_count"),
        "campaign_landing_intensity": intensity("campaign_landing_count"),
        "promotion_click_rate": smooth(
            "promotion_click_count",
            "promotion_impression_count",
        ),
        "campaign_landing_rate": smooth(
            "campaign_landing_count",
            "promotion_click_count",
        ),
        "promotion_response_recency": recency("last_response_at"),
        "promotion_response_intensity": intensity(
            "promotion_click_count + campaign_landing_count"
        ),
        "has_hotel_interest": (
            "if(hotel_search_count + hotel_click_count + "
            "hotel_detail_view_count > 0, 1.0, 0.0)"
        ),
        "single_destination_concentration": (
            "if(destination_count = 1, 1.0, 0.0)"
        ),
        "destination_breadth": (
            "least(1.0, toFloat64(destination_count) / 3.0)"
        ),
        "has_booking_abandonment": (
            "if(booking_start_count > booking_complete_count "
            "AND booking_start_count > 0, 1.0, 0.0)"
        ),
        "has_promotion_response": (
            "if(promotion_click_count + campaign_landing_count > 0, 1.0, 0.0)"
        ),
        "has_benefit_interest": (
            "if(deal_count + free_cancellation_count + "
            "breakfast_count > 0, 1.0, 0.0)"
        ),
        "has_recent_activity": (
            "if(dateDiff('day', greatest(last_search_at, last_detail_at, "
            "last_booking_start_at, last_response_at), window_end_value) "
            "<= 30, 1.0, 0.0)"
        ),
        "engagement_composite": (
            "least(1.0, if(hotel_detail_view_count + hotel_click_count >= 2, "
            "0.34, 0.0) + if(booking_start_count > booking_complete_count "
            "AND booking_start_count > 0, 0.33, 0.0) + "
            "if(promotion_click_count + campaign_landing_count > 0, "
            "0.33, 0.0))"
        ),
    }
    vector_terms = order_vector_terms_by_manifest(vector_terms_by_name)
    aggregate_sql = ",\n                ".join(aggregates)
    vector_sql = ",\n                ".join(vector_terms)
    return f"""
        INSERT INTO user_behavior_vectors (
            project_id, user_id, vector_dim, vector_values, vector_version,
            source, window_start, window_end, updated_at
        )
        SELECT
            {{project_id:String}},
            user_id,
            toUInt16({{vector_dim:UInt16}}),
            arrayMap(
                value -> toFloat32(
                    if(vector_norm = 0.0, 0.0, value / vector_norm)
                ),
                raw_vector_values
            ),
            {{vector_version:String}},
            {{source:String}},
            toDateTime64(parseDateTimeBestEffort({{window_start:String}}), 3, 'UTC'),
            window_end_value,
            now64(3)
        FROM (
            SELECT
                user_id,
                window_end_value,
                raw_vector_values,
                sqrt(arraySum(value -> value * value, raw_vector_values))
                    AS vector_norm
            FROM (
                SELECT
                    user_id,
                    window_end_value,
                    [{vector_sql}] AS raw_vector_values
                FROM (
                    SELECT
                        user_id,
                        count() AS event_count,
                        toDateTime64(parseDateTimeBestEffort({{window_end:String}}), 3, 'UTC') AS window_end_value,
                        {aggregate_sql}
                    FROM raw_events
                    WHERE project_id = {{project_id:String}}
                      AND validation_status = 'valid'
                      AND event_time >= toDateTime64(parseDateTimeBestEffort({{window_start:String}}), 3, 'UTC')
                      AND event_time < toDateTime64(parseDateTimeBestEffort({{window_end:String}}), 3, 'UTC')
                    GROUP BY user_id
                )
                WHERE event_count > 0
            )
        )
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
