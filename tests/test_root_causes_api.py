from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.db.clickhouse import ClickHouseClient, get_clickhouse_client_factory
from app.main import app


class FakeResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(self, results_by_call: list[list[tuple[object, ...]]]) -> None:
        self.results_by_call = results_by_call
        self.queries: list[str] = []
        self.parameters: list[dict[str, object]] = []

    def query(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        if len(self.parameters) >= len(self.results_by_call):
            raise AssertionError("FakeClickHouseClient received more queries than expected")
        self.queries.append(query)
        self.parameters.append(parameters or {})
        return FakeResult(self.results_by_call[len(self.parameters) - 1])


def test_post_root_causes_funnel_returns_target_candidates_and_summary() -> None:
    fake_client = FakeClickHouseClient(
        results_by_call=[
            [(1000, 90, 50, 25)],
            [(1000, 200, 50, 25)],
            [("out_of_stock", 300, 15, 5, 2, 300, 150, 20, 10)],
        ]
    )

    @contextmanager
    def override_client() -> Iterator[ClickHouseClient]:
        yield fake_client

    app.dependency_overrides[get_clickhouse_client_factory] = lambda: override_client
    try:
        response = TestClient(app).post(
            "/root-causes/funnel",
            json={
                "project_id": "loopad-demo-shop",
                "window_start": "2026-06-24T17:00:00+09:00",
                "window_end": "2026-06-24T18:00:00+09:00",
                "baseline_start": "2026-06-24T16:00:00+09:00",
                "baseline_end": "2026-06-24T17:00:00+09:00",
                "candidate_dimensions": ["inventory_status"],
                "include_volume_anomalies": False,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "critical"
    assert body["target_anomaly"]["metric"] == "view_to_cart_rate"
    assert body["candidates"][0]["dimension"] == "inventory_status"
    assert body["candidates"][0]["value"] == "out_of_stock"
    assert body["summary_message"]
    assert len(fake_client.parameters) == 3
    assert fake_client.parameters[2]["current_window_start"] == "2026-06-24T17:00:00.000+09:00"
    assert fake_client.parameters[2]["baseline_start"] == "2026-06-24T16:00:00.000+09:00"


def test_post_root_causes_funnel_rejects_naive_datetime() -> None:
    response = TestClient(app).post(
        "/root-causes/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00",
            "window_end": "2026-06-24T18:00:00+09:00",
        },
    )

    assert response.status_code == 400


def test_post_root_causes_funnel_rejects_unknown_candidate_dimension() -> None:
    response = TestClient(app).post(
        "/root-causes/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "candidate_dimensions": ["unsafe_column"],
        },
    )

    assert response.status_code == 400


def test_post_root_causes_funnel_rejects_filtered_candidate_dimension() -> None:
    response = TestClient(app).post(
        "/root-causes/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "filters": {"channel": "kakao"},
            "candidate_dimensions": ["channel"],
        },
    )

    assert response.status_code == 400


def test_post_root_causes_funnel_rejects_warning_threshold_above_critical() -> None:
    response = TestClient(app).post(
        "/root-causes/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "warning_abs_drop": 0.20,
            "critical_abs_drop": 0.10,
        },
    )

    assert response.status_code == 400


def test_post_root_causes_funnel_can_use_default_baseline_window() -> None:
    fake_client = FakeClickHouseClient(
        results_by_call=[
            [(1000, 90, 50, 25)],
            [(1000, 200, 50, 25)],
            [("sku-1", 200, 20, 8, 4, 200, 80, 20, 10)],
        ]
    )

    @contextmanager
    def override_client() -> Iterator[ClickHouseClient]:
        yield fake_client

    app.dependency_overrides[get_clickhouse_client_factory] = lambda: override_client
    try:
        response = TestClient(app).post(
            "/root-causes/funnel",
            json={
                "project_id": "loopad-demo-shop",
                "window_start": "2026-06-24T17:00:00+09:00",
                "window_end": "2026-06-24T18:00:00+09:00",
                "candidate_dimensions": ["product_id"],
                "include_volume_anomalies": False,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["baseline_start"] == "2026-06-24T16:00:00+09:00"
    assert body["baseline_end"] == "2026-06-24T17:00:00+09:00"
    assert fake_client.parameters[1]["window_start"] == "2026-06-24T16:00:00.000+09:00"
    assert fake_client.parameters[2]["baseline_start"] == "2026-06-24T16:00:00.000+09:00"
