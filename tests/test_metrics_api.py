from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.db.clickhouse import ClickHouseClient, get_clickhouse_client_factory
from app.main import app


class FakeResult:
    result_rows = [(1000, 90, 50, 25)]


class FakeClickHouseClient:
    def __init__(self) -> None:
        self.query_text: str | None = None
        self.parameters: dict[str, object] | None = None

    def query(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        self.query_text = query
        self.parameters = parameters
        return FakeResult()


def test_post_metrics_funnel_returns_metrics_and_segment() -> None:
    fake_client = FakeClickHouseClient()

    @contextmanager
    def override_client() -> Iterator[ClickHouseClient]:
        yield fake_client

    app.dependency_overrides[get_clickhouse_client_factory] = lambda: override_client
    try:
        response = TestClient(app).post(
            "/metrics/funnel",
            json={
                "project_id": "loopad-demo-shop",
                "window_start": "2026-06-24T17:00:00+09:00",
                "window_end": "2026-06-24T18:00:00+09:00",
                "filters": {
                    "channel": "kakao",
                    "campaign_id": "campaign_fresh_food_01",
                    "category": "fresh_food",
                },
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["segment"]["channel"] == "kakao"
    assert body["segment"]["campaign_id"] == "campaign_fresh_food_01"
    assert body["segment"]["device"] is None
    assert body["metrics"]["product_view_sessions"] == 1000
    assert body["metrics"]["view_to_cart_rate"] == 0.09
    assert body["metrics"]["view_to_cart_dropoff_rate"] == 0.91
    assert fake_client.parameters is not None
    assert fake_client.parameters["project_id"] == "loopad-demo-shop"
    assert fake_client.parameters["filter_channel"] == "kakao"


def test_post_metrics_funnel_rejects_window_start_after_window_end() -> None:
    response = TestClient(app).post(
        "/metrics/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T18:00:00+09:00",
            "window_end": "2026-06-24T17:00:00+09:00",
        },
    )

    assert response.status_code == 400


def test_post_metrics_funnel_rejects_naive_datetime() -> None:
    response = TestClient(app).post(
        "/metrics/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00",
            "window_end": "2026-06-24T18:00:00+09:00",
        },
    )

    assert response.status_code == 400


def test_post_metrics_funnel_rejects_unknown_filter_key() -> None:
    response = TestClient(app).post(
        "/metrics/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "filters": {
                "channel": "kakao",
                "unsafe_column": "should-not-pass",
            },
        },
    )

    assert response.status_code == 400


def test_post_metrics_funnel_normalizes_timezone_before_query() -> None:
    fake_client = FakeClickHouseClient()

    @contextmanager
    def override_client() -> Iterator[ClickHouseClient]:
        yield fake_client

    app.dependency_overrides[get_clickhouse_client_factory] = lambda: override_client
    try:
        response = TestClient(app).post(
            "/metrics/funnel",
            json={
                "project_id": "loopad-demo-shop",
                "window_start": datetime(2026, 6, 24, 8, 0, tzinfo=ZoneInfo("UTC")).isoformat(),
                "window_end": datetime(2026, 6, 24, 9, 0, tzinfo=ZoneInfo("UTC")).isoformat(),
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_client.parameters is not None
    assert fake_client.parameters["window_start"] == "2026-06-24T17:00:00.000+09:00"
    assert fake_client.parameters["window_end"] == "2026-06-24T18:00:00.000+09:00"
