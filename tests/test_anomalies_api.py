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
    assert body["summary_message"]
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


def test_post_anomalies_funnel_route_is_registered() -> None:
    fake_client = FakeClickHouseClient(
        rows=[
            (1000, 200, 100, 50),
            (1000, 200, 100, 50),
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


def test_post_anomalies_funnel_returns_volume_warning_without_mixing_evaluations() -> None:
    fake_client = FakeClickHouseClient(
        rows=[
            (1000, 100, 50, 20),
            (2000, 200, 100, 40),
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
                "critical_volume_relative_drop": 0.60,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "warning"
    assert len(body["evaluations"]) == 4
    assert body["anomalies"] == []
    assert len(body["volume_evaluations"]) == 4
    assert len(body["volume_anomalies"]) == 4
    assert body["volume_anomalies"][0]["metric"] == "product_view_sessions"
    assert body["volume_anomalies"][0]["severity"] == "warning"
    assert body["volume_anomalies"][0]["relative_drop"] == pytest.approx(0.5)
    assert body["primary_anomaly"]["metric"] == "product_view_sessions"
    assert body["summary_message"] == (
        "product_view_sessions dropped by 50.0% compared with baseline."
    )


def test_post_anomalies_funnel_can_disable_volume_anomalies() -> None:
    fake_client = FakeClickHouseClient(
        rows=[
            (1000, 100, 50, 20),
            (2000, 200, 100, 40),
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
                "include_volume_anomalies": False,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "normal"
    assert body["volume_evaluations"] == []
    assert body["volume_anomalies"] == []
    assert body["primary_anomaly"] is None
    assert body["summary_message"] == "No funnel anomaly detected."


def test_post_anomalies_funnel_primary_anomaly_prefers_most_severe_drop() -> None:
    fake_client = FakeClickHouseClient(
        rows=[
            (1000, 90, 50, 25),
            (2000, 300, 100, 50),
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
    assert body["status"] == "critical"
    assert body["primary_anomaly"]["metric"] == "add_to_cart_sessions"
    assert body["primary_anomaly"]["severity"] == "critical"
    assert body["primary_anomaly"]["relative_drop"] == pytest.approx(0.7)


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


def test_post_anomalies_funnel_rejects_overlapping_baseline_window() -> None:
    response = TestClient(app).post(
        "/anomalies/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "baseline_start": "2026-06-24T16:30:00+09:00",
            "baseline_end": "2026-06-24T17:30:00+09:00",
        },
    )

    assert response.status_code == 400


def test_post_anomalies_funnel_rejects_volume_warning_threshold_above_critical() -> None:
    response = TestClient(app).post(
        "/anomalies/funnel",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "warning_volume_relative_drop": 0.70,
            "critical_volume_relative_drop": 0.50,
        },
    )

    assert response.status_code == 400
