from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.metrics.schemas import FunnelMetricFilters
from app.root_causes.repository import (
    RootCauseRepository,
    build_grouped_funnel_counts_query,
)


class FakeResult:
    result_rows: list[tuple[object, ...]] = []


class FakeClickHouseClient:
    def __init__(self) -> None:
        self.query_text: str | None = None
        self.parameters: dict[str, object] | None = None

    def query(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        self.query_text = query
        self.parameters = parameters
        return FakeResult()


def test_build_grouped_funnel_counts_query_uses_allowlisted_dimension_column() -> None:
    query, _ = build_grouped_funnel_counts_query(None, "product_id")

    assert "product_id AS dimension_value" in query

    with pytest.raises(ValueError):
        build_grouped_funnel_counts_query(None, "unsafe_column")


def test_build_grouped_funnel_counts_query_passes_filters_as_parameters() -> None:
    query, parameters = build_grouped_funnel_counts_query(
        FunnelMetricFilters(channel="kakao"),
        "inventory_status",
    )

    assert "channel = {filter_channel:String}" in query
    assert parameters == {"filter_channel": "kakao"}


def test_fetch_grouped_funnel_counts_formats_windows_as_kst_strings() -> None:
    fake_client = FakeClickHouseClient()
    repository = RootCauseRepository(fake_client)

    repository.fetch_grouped_funnel_counts(
        project_id="loopad-demo-shop",
        window_start=datetime(2026, 6, 24, 8, 0, tzinfo=ZoneInfo("UTC")),
        window_end=datetime(2026, 6, 24, 9, 0, tzinfo=ZoneInfo("UTC")),
        baseline_start=datetime(2026, 6, 24, 7, 0, tzinfo=ZoneInfo("UTC")),
        baseline_end=datetime(2026, 6, 24, 8, 0, tzinfo=ZoneInfo("UTC")),
        filters=FunnelMetricFilters(campaign_id="campaign-1"),
        dimension="product_id",
        limit=25,
    )

    assert fake_client.parameters is not None
    assert fake_client.parameters["current_window_start"] == "2026-06-24T17:00:00.000+09:00"
    assert fake_client.parameters["current_window_end"] == "2026-06-24T18:00:00.000+09:00"
    assert fake_client.parameters["baseline_start"] == "2026-06-24T16:00:00.000+09:00"
    assert fake_client.parameters["baseline_end"] == "2026-06-24T17:00:00.000+09:00"
    assert fake_client.parameters["filter_campaign_id"] == "campaign-1"
    assert fake_client.parameters["candidate_limit"] == 25


def test_build_grouped_funnel_counts_query_excludes_null_and_empty_dimension_values() -> None:
    query, _ = build_grouped_funnel_counts_query(None, "inventory_status")

    assert "inventory_status IS NOT NULL" in query
    assert "inventory_status != ''" in query


def test_build_grouped_funnel_counts_query_returns_current_and_baseline_counts() -> None:
    query, _ = build_grouped_funnel_counts_query(None, "channel")

    assert "current_counts AS" in query
    assert "baseline_counts AS" in query
    assert "FULL OUTER JOIN baseline_counts" in query
    assert "coalesce(current_counts.dimension_value, baseline_counts.dimension_value)" in query
    assert "current_product_view_sessions" in query
    assert "baseline_product_view_sessions" in query


def test_fetch_grouped_funnel_counts_maps_current_and_baseline_rows() -> None:
    class Result:
        result_rows = [
            ("paid", 100, 20, 10, 5, 200, 80, 30, 10),
        ]

    class Client:
        def query(self, query: str, parameters: dict[str, object] | None = None) -> Result:
            return Result()

    rows = RootCauseRepository(Client()).fetch_grouped_funnel_counts(
        project_id="loopad-demo-shop",
        window_start=datetime(2026, 6, 24, 17, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        window_end=datetime(2026, 6, 24, 18, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        baseline_start=datetime(2026, 6, 24, 16, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        baseline_end=datetime(2026, 6, 24, 17, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        filters=None,
        dimension="channel",
        limit=10,
    )

    assert len(rows) == 1
    assert rows[0].dimension_value == "paid"
    assert rows[0].current_product_view_sessions == 100
    assert rows[0].baseline_add_to_cart_sessions == 80
