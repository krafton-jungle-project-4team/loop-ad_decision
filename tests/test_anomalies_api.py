from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.db.clickhouse import ClickHouseClient, get_clickhouse_client_factory
from app.main import app


class FakeResult:
    def __init__(self, row: tuple[int, int, int, int]) -> None:
        self.result_rows = [row]


class FakeClickHouseClient:
    def __init__(self, rows: list[tuple[int, int, int, int]]) -> None:
        self.rows = rows
        self.parameters: list[dict[str, object]] = []

    def query(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        if len(self.parameters) >= len(self.rows):
            raise AssertionError("FakeClickHouseClient received more queries than expected")
        self.parameters.append(parameters or {})
        return FakeResult(self.rows[len(self.parameters) - 1])


def test_post_anomalies_funnel_returns_warning_anomaly() -> None:
    fake_client = FakeClickHouseClient(
        rows=[
            (1000, 90, 50, 25),
            (1000, 150, 50, 25),
        ]
    )

    @contextmanager
    def override_client() -> Iterator[ClickHouseClient]:
        yield fake_client

    app.dependency_overrides[get_clickhouse_client_factory] = lambda: override_client
    try:
        response = TestClient(app).post(
            "/anomalies/funnel",
            json={
                "project_id": "loopad-demo-shop",
                "window_start": "2026-06-24T17:00:00+09:00",
                "window_end": "2026-06-24T18:00:00+09:00",
                "baseline_start": "2026-06-24T16:00:00+09:00",
                "baseline_end": "2026-06-24T17:00:00+09:00",
                "filters": {
                    "channel": "kakao",
                    "age_group": "30s",
                    "gender": "male",
                    "category": "fresh_food",
                },
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "warning"
    assert body["segment"]["channel"] == "kakao"
    assert body["segment"]["age_group"] == "30s"
    assert len(body["evaluations"]) == 4
    assert len(body["anomalies"]) == 1

    anomaly = body["anomalies"][0]
    assert anomaly["metric"] == "view_to_cart_rate"
    assert anomaly["funnel_step"] == "product_view_to_add_to_cart"
    assert anomaly["severity"] == "warning"
    assert anomaly["current_value"] == pytest.approx(0.09)
    assert anomaly["baseline_value"] == pytest.approx(0.15)
    assert anomaly["drop_point"] == pytest.approx(0.06)
    assert anomaly["relative_drop"] == pytest.approx(0.4)
    assert anomaly["message"] == (
        "product_view_to_add_to_cart conversion rate dropped from 0.1500 to 0.0900."
    )

    assert len(fake_client.parameters) == 2
    assert fake_client.parameters[0]["window_start"] == "2026-06-24T17:00:00.000+09:00"
    assert fake_client.parameters[0]["window_end"] == "2026-06-24T18:00:00.000+09:00"
    assert fake_client.parameters[1]["window_start"] == "2026-06-24T16:00:00.000+09:00"
    assert fake_client.parameters[1]["window_end"] == "2026-06-24T17:00:00.000+09:00"


def test_post_anomalies_funnel_uses_previous_equal_window_as_default_baseline() -> None:
    fake_client = FakeClickHouseClient(
        rows=[
            (1000, 100, 50, 25),
            (1000, 100, 50, 25),
        ]
    )

    @contextmanager
    def override_client() -> Iterator[ClickHouseClient]:
        yield fake_client

    app.dependency_overrides[get_clickhouse_client_factory] = lambda: override_client
    try:
        response = TestClient(app).post(
            "/anomalies/funnel",
            json={
                "project_id": "loopad-demo-shop",
                "window_start": "2026-06-24T17:00:00+09:00",
                "window_end": "2026-06-24T18:00:00+09:00",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["baseline_start"] == "2026-06-24T16:00:00+09:00"
    assert body["baseline_end"] == "2026-06-24T17:00:00+09:00"
    assert fake_client.parameters[1]["window_start"] == "2026-06-24T16:00:00.000+09:00"
    assert fake_client.parameters[1]["window_end"] == "2026-06-24T17:00:00.000+09:00"


def test_post_anomalies_funnel_rejects_naive_datetime() -> None:
    response = TestClient(app).post(
        "/anomalies/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00",
            "window_end": "2026-06-24T18:00:00+09:00",
        },
    )

    assert response.status_code == 400


def test_post_anomalies_funnel_rejects_window_start_after_window_end() -> None:
    response = TestClient(app).post(
        "/anomalies/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T18:00:00+09:00",
            "window_end": "2026-06-24T17:00:00+09:00",
        },
    )

    assert response.status_code == 400


def test_post_anomalies_funnel_rejects_partial_baseline_window() -> None:
    response = TestClient(app).post(
        "/anomalies/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "baseline_start": "2026-06-24T16:00:00+09:00",
        },
    )

    assert response.status_code == 400


def test_post_anomalies_funnel_rejects_warning_threshold_above_critical() -> None:
    response = TestClient(app).post(
        "/anomalies/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "warning_abs_drop": 0.20,
            "critical_abs_drop": 0.10,
        },
    )

    assert response.status_code == 400
