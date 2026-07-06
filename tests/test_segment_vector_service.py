from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pytest

from app.analysis.repositories import SegmentVectorRecord, UserBehaviorVectorRecord
from app.analysis.vector_service import (
    SegmentVectorBuildRequest,
    SegmentVectorService,
)


class FakeSegmentVectorRepository:
    def __init__(
        self,
        *,
        snapshot: SegmentVectorRecord | None = None,
        latest: SegmentVectorRecord | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.latest = latest
        self.snapshot_calls: list[Mapping[str, str]] = []
        self.latest_calls: list[Mapping[str, str]] = []
        self.saved: list[SegmentVectorRecord] = []

    def get_by_segment_snapshot(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_id: str,
        vector_version: str,
    ) -> SegmentVectorRecord | None:
        self.snapshot_calls.append(
            {
                "project_id": project_id,
                "promotion_id": promotion_id,
                "analysis_id": analysis_id,
                "segment_id": segment_id,
                "vector_version": vector_version,
            }
        )
        return self.snapshot

    def get_latest_by_segment(
        self,
        *,
        project_id: str,
        promotion_id: str,
        segment_id: str,
        vector_version: str,
    ) -> SegmentVectorRecord | None:
        self.latest_calls.append(
            {
                "project_id": project_id,
                "promotion_id": promotion_id,
                "segment_id": segment_id,
                "vector_version": vector_version,
            }
        )
        return self.latest

    def save(self, vector: SegmentVectorRecord) -> None:
        self.saved.append(vector)


class FakeUserBehaviorVectorRepository:
    def __init__(self, vectors: list[UserBehaviorVectorRecord]) -> None:
        self.vectors = vectors
        self.calls: list[Mapping[str, Any]] = []

    def list_by_user_ids(
        self,
        *,
        project_id: str,
        user_ids: Sequence[str],
        vector_version: str = "v1",
    ) -> list[UserBehaviorVectorRecord]:
        self.calls.append(
            {
                "project_id": project_id,
                "user_ids": list(user_ids),
                "vector_version": vector_version,
            }
        )
        return self.vectors


def vector_record(
    *,
    user_id: str = "user_001",
    values: list[float] | None = None,
    vector_dim: int = 64,
) -> UserBehaviorVectorRecord:
    return UserBehaviorVectorRecord(
        project_id="hotel-client-a",
        user_id=user_id,
        vector_dim=vector_dim,
        vector_values=values or [0.1] * 64,
        vector_version="v1",
        source="batch_profile",
    )


def build_request(
    *,
    analysis_id: str = "analysis_banner_001",
    segment_id: str = "seg_repeat_hotel_no_booking",
    candidate_user_ids: Sequence[str] = ("user_001", "user_002"),
) -> SegmentVectorBuildRequest:
    return SegmentVectorBuildRequest(
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        analysis_id=analysis_id,
        segment_id=segment_id,
        candidate_user_ids=candidate_user_ids,
    )


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def test_segment_vector_service_averages_candidate_user_vectors() -> None:
    first = [1.0, 0.0, *([0.0] * 62)]
    second = [0.0, 1.0, *([0.0] * 62)]
    store = FakeSegmentVectorRepository()
    reader = FakeUserBehaviorVectorRepository(
        [
            vector_record(user_id="user_001", values=first),
            vector_record(user_id="user_002", values=second),
        ]
    )
    service = SegmentVectorService(
        segment_vector_repository=store,
        user_behavior_vector_repository=reader,
    )

    result = service.prepare_segment_vector(
        build_request(candidate_user_ids=("user_001", "user_001", "user_002"))
    )

    assert reader.calls == [
        {
            "project_id": "hotel-client-a",
            "user_ids": ["user_001", "user_002"],
            "vector_version": "v1",
        }
    ]
    assert len(store.saved) == 1
    saved = store.saved[0]
    assert saved.segment_vector_id == result.segment_vector_id
    assert saved.segment_vector_id.startswith(
        "segvec_seg_repeat_hotel_no_booking_v1_"
    )
    assert len(saved.segment_vector_id) <= 100
    assert saved.vector_dim == 64
    assert saved.source == "decision_analysis"
    assert saved.vector_values == result.vector_values
    assert vector_norm(saved.vector_values) == pytest.approx(1.0)
    assert saved.vector_values[0] == pytest.approx(1 / math.sqrt(2))
    assert saved.vector_values[1] == pytest.approx(1 / math.sqrt(2))


def test_segment_vector_service_uses_deterministic_fixture_fallback() -> None:
    first_store = FakeSegmentVectorRepository()
    second_store = FakeSegmentVectorRepository()
    first_service = SegmentVectorService(
        segment_vector_repository=first_store,
        user_behavior_vector_repository=FakeUserBehaviorVectorRepository([]),
    )
    second_service = SegmentVectorService(
        segment_vector_repository=second_store,
        user_behavior_vector_repository=FakeUserBehaviorVectorRepository([]),
    )

    first_result = first_service.prepare_segment_vector(
        build_request(candidate_user_ids=())
    )
    second_result = second_service.prepare_segment_vector(
        build_request(candidate_user_ids=())
    )

    assert first_store.saved[0].source == "fixture"
    assert first_result.source == "fixture"
    assert first_result.segment_vector_id == second_result.segment_vector_id
    assert first_result.vector_values == second_result.vector_values
    assert vector_norm(first_result.vector_values) == pytest.approx(1.0)


def test_segment_vector_service_reuses_existing_snapshot_vector() -> None:
    existing = SegmentVectorRecord(
        segment_vector_id="segvec_existing_v1",
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        promotion_run_id=None,
        analysis_id="analysis_banner_001",
        segment_id="seg_repeat_hotel_no_booking",
        vector_dim=64,
        vector_values=[1.0, *([0.0] * 63)],
        vector_version="v1",
        source="decision_analysis",
    )
    store = FakeSegmentVectorRepository(snapshot=existing)
    reader = FakeUserBehaviorVectorRepository([vector_record()])
    service = SegmentVectorService(
        segment_vector_repository=store,
        user_behavior_vector_repository=reader,
    )

    result = service.prepare_segment_vector(build_request())

    assert result.segment_vector_id == "segvec_existing_v1"
    assert result.vector_values == existing.vector_values
    assert reader.calls == []
    assert store.saved == []
    assert store.latest_calls == []


def test_segment_vector_service_copies_latest_vector_into_new_analysis_snapshot() -> None:
    latest = SegmentVectorRecord(
        segment_vector_id="segvec_existing_v1",
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        promotion_run_id=None,
        analysis_id="analysis_banner_001",
        segment_id="seg_repeat_hotel_no_booking",
        vector_dim=64,
        vector_values=[1.0, *([0.0] * 63)],
        vector_version="v1",
        source="decision_analysis",
    )
    store = FakeSegmentVectorRepository(latest=latest)
    reader = FakeUserBehaviorVectorRepository([vector_record()])
    service = SegmentVectorService(
        segment_vector_repository=store,
        user_behavior_vector_repository=reader,
    )

    result = service.prepare_segment_vector(
        build_request(analysis_id="analysis_banner_002")
    )

    assert result.segment_vector_id != latest.segment_vector_id
    assert result.segment_vector_id.startswith(
        "segvec_seg_repeat_hotel_no_booking_v1_"
    )
    assert result.vector_values == latest.vector_values
    assert result.source == latest.source
    assert reader.calls == []
    assert len(store.saved) == 1
    saved = store.saved[0]
    assert saved.segment_vector_id == result.segment_vector_id
    assert saved.analysis_id == "analysis_banner_002"
    assert saved.segment_id == latest.segment_id
    assert saved.vector_values == latest.vector_values
    assert saved.source == latest.source


def test_segment_vector_service_rejects_non_64_dimensional_user_vector() -> None:
    store = FakeSegmentVectorRepository()
    reader = FakeUserBehaviorVectorRepository(
        [vector_record(values=[0.1] * 63, vector_dim=64)]
    )
    service = SegmentVectorService(
        segment_vector_repository=store,
        user_behavior_vector_repository=reader,
    )

    with pytest.raises(ValueError, match="64 values"):
        service.prepare_segment_vector(build_request())

    assert store.saved == []
