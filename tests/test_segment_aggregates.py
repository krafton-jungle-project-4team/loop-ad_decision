from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.analysis.clickhouse_repository import (
    ClickHouseAnalysisRepository,
    SEGMENT_AGGREGATE_QUERY,
    USER_PRIMARY_SEGMENT_QUERY,
)
from app.analysis.segments import build_segment_key, normalize_dimensions, normalize_dimension_value
from app.analysis.time_window import build_analysis_window


class FakeResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.queries: list[str] = []
        self.parameters: list[dict[str, object] | None] = []

    def query(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        self.queries.append(query)
        self.parameters.append(parameters)
        return FakeResult(self.rows)


def aggregate_row(
    *,
    product_view_count: int = 100,
    user_count: int = 30,
    add_to_cart_count: int = 20,
    checkout_start_count: int = 10,
    purchase_count: int = 5,
    ad_impression_count: int = 0,
    ad_click_count: int = 0,
) -> tuple[object, ...]:
    return (
        "30s",
        " Male ",
        "Mobile Web",
        "Kakao",
        "Fresh Food",
        user_count,
        50,
        200,
        product_view_count,
        add_to_cart_count,
        checkout_start_count,
        purchase_count,
        ad_impression_count,
        ad_click_count,
        "12345.67",
    )


def test_normalize_dimensions_and_segment_key_are_stable() -> None:
    dimensions = normalize_dimensions(
        {
            "age_group": " 30s ",
            "gender": "Male",
            "device_type": "Mobile Web",
            "acquisition_channel": "",
            "primary_category": None,
        }
    )

    assert dimensions == {
        "age_group": "30s",
        "gender": "male",
        "device_type": "mobile_web",
        "acquisition_channel": "unknown",
        "primary_category": "unknown",
    }
    assert (
        build_segment_key(dimensions)
        == "age_30s__gender_male__device_mobile_web__channel_unknown__category_unknown"
    )
    assert normalize_dimension_value("(not set)") == "unknown"


def test_fetch_segment_aggregates_calculates_metrics_and_null_rates() -> None:
    client = FakeClickHouseClient(
        [
            aggregate_row(
                add_to_cart_count=0,
                checkout_start_count=0,
                purchase_count=0,
                ad_impression_count=0,
                ad_click_count=0,
            )
        ]
    )
    repository = ClickHouseAnalysisRepository(client)
    window = build_analysis_window(date(2021, 1, 4), "Asia/Seoul")

    aggregates = repository.fetch_segment_aggregates(project_id=1, window=window)

    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.segment_key == "age_30s__gender_male__device_mobile_web__channel_kakao__category_fresh_food"
    assert aggregate.revenue == Decimal("12345.67")
    assert aggregate.view_to_cart_rate == Decimal("0")
    assert aggregate.cart_to_checkout_rate is None
    assert aggregate.checkout_to_purchase_rate is None
    assert aggregate.ctr is None
    assert client.parameters[0]["window_start_utc"] == "2021-01-03T15:00:00+00:00"
    assert client.parameters[0]["window_end_utc"] == "2021-01-04T15:00:00+00:00"


def test_fetch_segment_aggregates_filters_invalid_samples() -> None:
    client = FakeClickHouseClient([aggregate_row(product_view_count=10, user_count=3)])
    repository = ClickHouseAnalysisRepository(client)
    window = build_analysis_window(date(2021, 1, 4), "Asia/Seoul")

    assert repository.fetch_segment_aggregates(project_id=1, window=window) == []


def test_user_primary_segment_query_is_separate_and_uses_same_segment_key_rules() -> None:
    client = FakeClickHouseClient(
        [
            (
                "user-1",
                "30s",
                " Male ",
                "Mobile Web",
                "Kakao",
                "Fresh Food",
            )
        ]
    )
    repository = ClickHouseAnalysisRepository(client)
    window = build_analysis_window(date(2021, 1, 4), "Asia/Seoul")

    candidates = repository.fetch_user_primary_segment_candidates(project_id=1, window=window)

    assert SEGMENT_AGGREGATE_QUERY != USER_PRIMARY_SEGMENT_QUERY
    assert len(candidates) == 1
    assert candidates[0].external_user_id == "user-1"
    assert candidates[0].segment_key == "age_30s__gender_male__device_mobile_web__channel_kakao__category_fresh_food"
    assert client.queries[0] == USER_PRIMARY_SEGMENT_QUERY
