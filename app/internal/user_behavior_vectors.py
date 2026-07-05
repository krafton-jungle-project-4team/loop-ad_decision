from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Protocol

from app.internal.schemas import (
    UserBehaviorVectorBuildRequest,
    UserBehaviorVectorSource,
)


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
    source: UserBehaviorVectorSource
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

    def build(
        self,
        request: UserBehaviorVectorBuildRequest,
    ) -> UserBehaviorVectorBuildResult:
        window_end = self._resolve_now()
        window_start = window_end - timedelta(days=request.window_days)
        processed_user_count = self._repository.count_expedia_users(
            window_start=window_start,
            window_end=window_end,
        )

        if processed_user_count > 0:
            self._repository.insert_expedia_user_vectors(
                project_id=request.project_id,
                vector_version=request.vector_version,
                source=request.source.value,
                window_start=window_start,
                window_end=window_end,
            )

        return UserBehaviorVectorBuildResult(
            project_id=request.project_id,
            vector_version=request.vector_version,
            source=request.source,
            vector_dim=self.VECTOR_DIM,
            processed_user_count=processed_user_count,
            window_start=window_start,
            window_end=window_end,
            status="completed",
        )

    def _resolve_now(self) -> datetime:
        value = self._now or datetime.now(UTC)
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class UserBehaviorVectorBuildRepository:
    VECTOR_DIM = 64

    def __init__(self, client: ClickHouseBatchClient) -> None:
        self._client = client

    def count_expedia_users(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> int:
        result = self._client.query(
            """
            SELECT countDistinct(toString(user_id)) AS processed_user_count
            FROM expedia_hotel_events
            WHERE date_time >= parseDateTimeBestEffort({window_start:String})
              AND date_time < parseDateTimeBestEffort({window_end:String})
            """,
            parameters={
                "window_start": _clickhouse_datetime(window_start),
                "window_end": _clickhouse_datetime(window_end),
            },
        )
        return int(_first_scalar(result, "processed_user_count"))

    def insert_expedia_user_vectors(
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
        command = getattr(self._client, "command", None)
        if callable(command):
            command(_build_expedia_insert_sql(), parameters=parameters)
            return
        self._client.query(_build_expedia_insert_sql(), parameters=parameters)


def _build_expedia_insert_sql() -> str:
    hotel_cluster_counts = [
        f"countIf(modulo(toUInt32(hotel_cluster), 32) = {index}) "
        f"AS hotel_cluster_bucket_{index}_count"
        for index in range(32)
    ]
    destination_counts = [
        f"countIf(modulo(toUInt32(srch_destination_id), 16) = {index}) "
        f"AS destination_bucket_{index}_count"
        for index in range(16)
    ]
    channel_counts = [
        f"countIf(modulo(toUInt32(channel), 10) = {index}) "
        f"AS channel_bucket_{index}_count"
        for index in range(10)
    ]
    aggregate_terms = ",\n                    ".join(
        [
            *hotel_cluster_counts,
            *destination_counts,
            *channel_counts,
        ]
    )
    vector_terms = [
        "mobile_ratio",
        "package_ratio",
        "booking_rate",
        "family_trip_ratio",
        "stay_nights_score",
        "near_checkin_ratio",
        *[
            _ratio_term(f"hotel_cluster_bucket_{index}_count")
            for index in range(32)
        ],
        *[
            _ratio_term(f"destination_bucket_{index}_count")
            for index in range(16)
        ],
        *[_ratio_term(f"channel_bucket_{index}_count") for index in range(10)],
    ]
    if len(vector_terms) != 64:
        raise RuntimeError("expedia user behavior vector must contain 64 features")
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
                toString(user_id) AS user_id,
                count() AS event_count,
                avg(toFloat64(is_mobile)) AS mobile_ratio,
                avg(toFloat64(is_package)) AS package_ratio,
                avg(toFloat64(is_booking)) AS booking_rate,
                avg(if(srch_children_cnt > 0, 1.0, 0.0)) AS family_trip_ratio,
                least(
                    greatest(
                        avg(
                            if(
                                isNull(srch_ci) OR isNull(srch_co),
                                0.0,
                                toFloat64(dateDiff('day', srch_ci, srch_co))
                            )
                        ) / 14.0,
                        0.0
                    ),
                    1.0
                ) AS stay_nights_score,
                avg(
                    if(
                        NOT isNull(srch_ci)
                        AND dateDiff('day', toDate(date_time), srch_ci) BETWEEN 0 AND 7,
                        1.0,
                        0.0
                    )
                ) AS near_checkin_ratio,
                    {aggregate_terms}
            FROM expedia_hotel_events
            WHERE date_time >= parseDateTimeBestEffort({{window_start:String}})
              AND date_time < parseDateTimeBestEffort({{window_end:String}})
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

