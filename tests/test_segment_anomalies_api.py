from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.db.clickhouse import ClickHouseClient, get_clickhouse_client_factory
from app.main import app


class FakeResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(
        self,
        segment_rows: list[tuple[Any, ...]],
        count_rows: list[tuple[int, int, int, int]],
    ) -> None:
        self.segment_rows = segment_rows
        self.count_rows = count_rows
        self.segment_parameters: list[dict[str, object]] = []
        self.count_parameters: list[dict[str, object]] = []

    def query(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        if "SELECT DISTINCT" in query:
            self.segment_parameters.append(parameters or {})
            return FakeResult(self.segment_rows)

        if len(self.count_parameters) >= len(self.count_rows):
            raise AssertionError("FakeClickHouseClient received more funnel queries than expected")
        self.count_parameters.append(parameters or {})
        return FakeResult([self.count_rows[len(self.count_parameters) - 1]])


def post_segments(
    fake_client: FakeClickHouseClient,
    payload: dict[str, object],
):
    @contextmanager
    def override_client() -> Iterator[ClickHouseClient]:
        yield fake_client

    app.dependency_overrides[get_clickhouse_client_factory] = lambda: override_client
    try:
        return TestClient(app).post("/anomalies/funnel/segments", json=payload)
    finally:
        app.dependency_overrides.clear()


def segment_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "project_id": "loopad-demo-shop",
        "window_start": "2026-06-24T17:00:00+09:00",
        "window_end": "2026-06-24T18:00:00+09:00",
        "segment_by": ["channel"],
    }
    payload.update(overrides)
    return payload


def test_post_segment_anomalies_route_is_registered() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("kakao",)],
        count_rows=[(1000, 200, 100, 50), (1000, 200, 100, 50)],
    )

    response = post_segments(fake_client, segment_payload())

    assert response.status_code == 200


def test_post_segment_anomalies_returns_only_warning_or_critical_channel_segments() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("kakao",), ("naver",), ("meta",)],
        count_rows=[
            (1000, 90, 50, 25),
            (1000, 150, 50, 25),
            (1000, 200, 100, 50),
            (1000, 200, 100, 50),
            (10, 0, 0, 0),
            (10, 0, 0, 0),
        ],
    )

    response = post_segments(fake_client, segment_payload(include_volume_anomalies=False))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "warning"
    assert body["total_segments_discovered"] == 3
    assert body["total_segments_evaluated"] == 3
    assert [segment["segment"]["channel"] for segment in body["segments"]] == ["kakao"]


def test_post_segment_anomalies_handles_two_column_segments() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("kakao", "fresh_food")],
        count_rows=[(1000, 90, 50, 25), (1000, 150, 50, 25)],
    )

    response = post_segments(
        fake_client,
        segment_payload(segment_by=["channel", "category"], include_volume_anomalies=False),
    )

    assert response.status_code == 200
    segment = response.json()["segments"][0]["segment"]
    assert segment["channel"] == "kakao"
    assert segment["category"] == "fresh_food"


def test_post_segment_anomalies_result_includes_pr5_fields() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("kakao",)],
        count_rows=[(1000, 90, 50, 25), (1000, 150, 50, 25)],
    )

    response = post_segments(fake_client, segment_payload())

    assert response.status_code == 200
    segment = response.json()["segments"][0]
    assert "volume_evaluations" in segment
    assert "volume_anomalies" in segment
    assert "primary_anomaly" in segment
    assert "summary_message" in segment


def test_post_segment_anomalies_includes_volume_only_anomaly() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("kakao",)],
        count_rows=[(1000, 200, 100, 40), (2000, 400, 200, 80)],
    )

    response = post_segments(fake_client, segment_payload())

    assert response.status_code == 200
    segment = response.json()["segments"][0]
    assert segment["anomalies"] == []
    assert segment["volume_anomalies"]
    assert segment["status"] == "critical"


def test_post_segment_anomalies_includes_rate_only_anomaly() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("kakao",)],
        count_rows=[(1000, 90, 50, 25), (1000, 150, 50, 25)],
    )

    response = post_segments(fake_client, segment_payload(include_volume_anomalies=False))

    assert response.status_code == 200
    segment = response.json()["segments"][0]
    assert segment["anomalies"]
    assert segment["volume_anomalies"] == []
    assert segment["status"] == "warning"


def test_post_segment_anomalies_excludes_normal_and_insufficient_segments() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("normal",), ("insufficient",)],
        count_rows=[
            (1000, 200, 100, 50),
            (1000, 200, 100, 50),
            (10, 0, 0, 0),
            (10, 0, 0, 0),
        ],
    )

    response = post_segments(fake_client, segment_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "normal"
    assert body["segments"] == []


def test_post_segment_anomalies_returns_insufficient_data_when_all_segments_insufficient() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("kakao",)],
        count_rows=[(10, 0, 0, 0), (10, 0, 0, 0)],
    )

    response = post_segments(fake_client, segment_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["segments"] == []


def test_post_segment_anomalies_sorts_critical_before_warning() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("warning",), ("critical",)],
        count_rows=[
            (1000, 90, 50, 25),
            (1000, 150, 50, 25),
            (1000, 200, 100, 40),
            (2000, 400, 200, 80),
        ],
    )

    response = post_segments(fake_client, segment_payload())

    assert response.status_code == 200
    assert [segment["segment"]["channel"] for segment in response.json()["segments"]] == [
        "critical",
        "warning",
    ]


def test_post_segment_anomalies_sorts_same_severity_by_score() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("smaller",), ("larger",)],
        count_rows=[
            (1000, 70, 50, 25),
            (1000, 100, 50, 25),
            (1000, 60, 50, 25),
            (1000, 100, 50, 25),
        ],
    )

    response = post_segments(
        fake_client,
        segment_payload(
            include_volume_anomalies=False,
            critical_abs_drop=0.90,
            critical_relative_drop=0.90,
        ),
    )

    assert response.status_code == 200
    assert [segment["segment"]["channel"] for segment in response.json()["segments"]] == [
        "larger",
        "smaller",
    ]


def test_post_segment_anomalies_applies_limit_and_candidate_limit() -> None:
    fake_client = FakeClickHouseClient(
        segment_rows=[("first",), ("second",), ("third",)],
        count_rows=[
            (1000, 60, 50, 25),
            (1000, 100, 50, 25),
            (1000, 70, 50, 25),
            (1000, 100, 50, 25),
            (1000, 80, 50, 25),
            (1000, 100, 50, 25),
        ],
    )

    response = post_segments(
        fake_client,
        segment_payload(
            limit=2,
            candidate_limit=7,
            include_volume_anomalies=False,
            critical_abs_drop=0.90,
            critical_relative_drop=0.90,
        ),
    )

    assert response.status_code == 200
    assert len(response.json()["segments"]) == 2
    assert fake_client.segment_parameters[0]["segment_limit"] == 7


@pytest.mark.parametrize(
    "overrides",
    [
        {"segment_by": []},
        {"segment_by": ["channel", "category", "device"]},
        {"segment_by": ["channel", "channel"]},
        {"segment_by": ["unsupported"]},
        {"base_filters": {"channel": "kakao"}, "segment_by": ["channel"]},
        {"window_start": "2026-06-24T17:00:00", "window_end": "2026-06-24T18:00:00+09:00"},
        {"baseline_start": "2026-06-24T16:00:00+09:00"},
        {
            "baseline_start": "2026-06-24T16:30:00+09:00",
            "baseline_end": "2026-06-24T17:30:00+09:00",
        },
        {"warning_abs_drop": 0.20, "critical_abs_drop": 0.10},
        {"warning_volume_relative_drop": 0.70, "critical_volume_relative_drop": 0.50},
    ],
)
def test_post_segment_anomalies_rejects_invalid_requests(
    overrides: dict[str, object],
) -> None:
    response = TestClient(app).post(
        "/anomalies/funnel/segments",
        json=segment_payload(**overrides),
    )

    assert response.status_code == 400
