from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.decision.models import (
    ExistingSegment,
    NearestSegmentCentroid,
    SegmentMatchingConfig,
    UserSegmentCandidate,
)
from app.decision.services import UserSegmentMatchingService


ANALYSIS_DATE = date(2021, 1, 4)


class FakeMatchingRepository:
    def __init__(self) -> None:
        self.memberships: list[dict[str, object]] = []
        self.replace_calls: list[dict[str, object]] = []
        self.nearest_calls: list[dict[str, object]] = []
        self.timezone = "Asia/Seoul"
        self.project_key = "demo-shop"
        self.config = SegmentMatchingConfig(
            embedding_version="segment_match_v1",
            similarity_threshold=Decimal("0.7000"),
        )
        self.default_segment = ExistingSegment(id=1, segment_key="default")
        self.nearest: NearestSegmentCentroid | None = NearestSegmentCentroid(
            segment_id=10,
            segment_key="age_30s__gender_male__device_mobile_web__channel_kakao__category_fresh_food",
            cosine_distance=Decimal("0.20"),
        )

    def get_project_timezone(self, *, project_id: int) -> str:
        return self.timezone

    def get_project_key(self, *, project_id: int) -> str:
        return self.project_key

    def get_segment_matching_config(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> SegmentMatchingConfig:
        return self.config

    def get_default_segment(self, *, project_id: int) -> ExistingSegment | None:
        return self.default_segment

    def find_nearest_segment_centroid(
        self,
        *,
        project_id: int,
        analysis_date: date,
        embedding_version: str,
        user_vector: list[float],
    ) -> NearestSegmentCentroid | None:
        self.nearest_calls.append(
            {
                "project_id": project_id,
                "analysis_date": analysis_date,
                "embedding_version": embedding_version,
                "user_vector": user_vector,
            }
        )
        return self.nearest

    def replace_primary_membership(
        self,
        *,
        project_id: int,
        external_user_id: str,
        segment_id: int,
        analysis_date: date,
        confidence: Decimal,
        reason_json: dict[str, object],
        run_id: int | None,
    ) -> None:
        self.replace_calls.append(
            {
                "project_id": project_id,
                "external_user_id": external_user_id,
                "segment_id": segment_id,
                "analysis_date": analysis_date,
                "confidence": confidence,
                "reason_json": reason_json,
                "run_id": run_id,
            }
        )
        self.memberships = [
            membership
            for membership in self.memberships
            if not (
                membership["project_id"] == project_id
                and membership["external_user_id"] == external_user_id
                and membership["analysis_date"] == analysis_date
                and membership["is_primary"] is True
                and membership["segment_id"] != segment_id
            )
        ]
        self.memberships.append(
            {
                "project_id": project_id,
                "external_user_id": external_user_id,
                "segment_id": segment_id,
                "analysis_date": analysis_date,
                "is_primary": True,
                "confidence": confidence,
                "reason_json": reason_json,
                "run_id": run_id,
            }
        )


class FakeCandidateRepository:
    def __init__(self, candidates: list[UserSegmentCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, object]] = []

    def fetch_user_segment_candidates(self, *, project_id, window):
        self.calls.append((project_id, window))
        return self.candidates


def candidate(
    *,
    external_user_id: str = "user-1",
    age_group: str = "30s",
    gender: str = " Male ",
    device: str = "Mobile Web",
    channel: str = "Kakao",
    category: str = "Fresh Food",
) -> UserSegmentCandidate:
    return UserSegmentCandidate(
        external_user_id=external_user_id,
        dimensions={
            "age_group": age_group,
            "gender": gender,
            "device_type": device,
            "acquisition_channel": channel,
            "primary_category": category,
        },
    )


def test_nearest_centroid_above_threshold_creates_segment_membership() -> None:
    repository = FakeMatchingRepository()
    candidate_repository = FakeCandidateRepository([candidate()])

    result = UserSegmentMatchingService(repository, candidate_repository).run(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=77,
    )

    assert result.matched_count == 1
    assert result.skipped_count == 0
    assert repository.memberships[0]["segment_id"] == 10
    assert repository.replace_calls[0]["confidence"] == Decimal("0.80")
    assert repository.replace_calls[0]["reason_json"] == {
        "matching_source": "embedding_ann",
        "embedding_version": "segment_match_v1",
        "similarity": 0.8,
        "threshold": 0.7,
        "nearest_segment_key": repository.nearest.segment_key,
        "fallback_reason": None,
    }
    assert candidate_repository.calls[0][0] == "demo-shop"


def test_distance_is_converted_to_similarity_before_threshold_comparison() -> None:
    repository = FakeMatchingRepository()
    repository.config = SegmentMatchingConfig(
        embedding_version="segment_match_v1",
        similarity_threshold=Decimal("0.7500"),
    )
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="segment",
        cosine_distance=Decimal("0.20"),
    )

    result = UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 10
    assert repository.memberships[0]["confidence"] == Decimal("0.80")


def test_distance_0_1_becomes_similarity_0_9_and_matches_nearest_segment() -> None:
    repository = FakeMatchingRepository()
    repository.config = SegmentMatchingConfig(
        embedding_version="segment_match_v1",
        similarity_threshold=Decimal("0.7000"),
    )
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="distance-0-1-segment",
        cosine_distance=Decimal("0.10"),
    )

    result = UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    assert result.matched_count == 1
    assert result.skipped_count == 0
    assert repository.memberships[0]["segment_id"] == 10
    assert repository.replace_calls[0]["confidence"] == Decimal("0.90")
    assert repository.replace_calls[0]["reason_json"]["similarity"] == 0.9
    assert repository.replace_calls[0]["reason_json"]["fallback_reason"] is None


def test_distance_0_5_becomes_similarity_0_5_and_falls_back_to_default() -> None:
    repository = FakeMatchingRepository()
    repository.config = SegmentMatchingConfig(
        embedding_version="segment_match_v1",
        similarity_threshold=Decimal("0.7000"),
    )
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="distance-0-5-segment",
        cosine_distance=Decimal("0.50"),
    )

    result = UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    assert result.matched_count == 0
    assert result.skipped_count == 1
    assert repository.memberships[0]["segment_id"] == 1
    assert repository.replace_calls[0]["confidence"] == Decimal("0.50")
    assert repository.replace_calls[0]["reason_json"]["similarity"] == 0.5
    assert repository.replace_calls[0]["reason_json"]["fallback_reason"] == "below_threshold"


def test_distance_0_3_similarity_0_7_matches_threshold_inclusive() -> None:
    repository = FakeMatchingRepository()
    repository.config = SegmentMatchingConfig(
        embedding_version="segment_match_v1",
        similarity_threshold=Decimal("0.7000"),
    )
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="distance-0-3-segment",
        cosine_distance=Decimal("0.30"),
    )

    result = UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    assert result.matched_count == 1
    assert result.skipped_count == 0
    assert repository.memberships[0]["segment_id"] == 10
    assert repository.replace_calls[0]["confidence"] == Decimal("0.70")
    assert repository.replace_calls[0]["reason_json"]["similarity"] == 0.7
    assert repository.replace_calls[0]["reason_json"]["fallback_reason"] is None


def test_below_threshold_falls_back_to_default_primary_membership() -> None:
    repository = FakeMatchingRepository()
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="near-but-not-enough",
        cosine_distance=Decimal("0.35"),
    )

    result = UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    assert result.matched_count == 0
    assert result.skipped_count == 1
    assert repository.memberships[0]["segment_id"] == 1
    assert repository.replace_calls[0]["confidence"] == Decimal("0.65")
    assert repository.replace_calls[0]["reason_json"]["similarity"] == 0.65
    assert repository.replace_calls[0]["reason_json"]["fallback_reason"] == "below_threshold"
    assert repository.replace_calls[0]["reason_json"]["nearest_segment_key"] == "near-but-not-enough"


def test_confidence_is_clamped_when_raw_similarity_is_negative() -> None:
    repository = FakeMatchingRepository()
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="opposite-segment",
        cosine_distance=Decimal("1.50"),
    )

    UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    assert repository.replace_calls[0]["confidence"] == Decimal("0")
    assert repository.replace_calls[0]["reason_json"]["similarity"] == -0.5
    assert repository.replace_calls[0]["reason_json"]["fallback_reason"] == "below_threshold"


def test_negative_0_2_raw_similarity_clamps_confidence_but_remains_in_reason_json() -> None:
    repository = FakeMatchingRepository()
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="negative-similarity-segment",
        cosine_distance=Decimal("1.20"),
    )

    UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    assert repository.replace_calls[0]["confidence"] == Decimal("0")
    assert repository.replace_calls[0]["reason_json"]["similarity"] == -0.2
    assert repository.replace_calls[0]["reason_json"]["fallback_reason"] == "below_threshold"


def test_missing_centroid_candidate_writes_default_membership_with_zero_confidence() -> None:
    repository = FakeMatchingRepository()
    repository.nearest = None

    result = UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=None)

    assert result.matched_count == 0
    assert result.skipped_count == 1
    assert repository.memberships[0]["segment_id"] == 1
    assert repository.replace_calls[0]["confidence"] == Decimal("0")
    assert repository.replace_calls[0]["reason_json"]["similarity"] is None
    assert repository.replace_calls[0]["reason_json"]["fallback_reason"] == "no_candidate_segment"


def test_fallback_replaces_existing_primary_membership_for_same_user_day() -> None:
    repository = FakeMatchingRepository()
    repository.nearest = NearestSegmentCentroid(
        segment_id=10,
        segment_key="near-but-not-enough",
        cosine_distance=Decimal("0.35"),
    )
    repository.memberships.append(
        {
            "project_id": 1,
            "external_user_id": "user-1",
            "segment_id": 99,
            "analysis_date": ANALYSIS_DATE,
            "is_primary": True,
            "confidence": Decimal("1.0"),
            "reason_json": {},
            "run_id": 1,
        }
    )

    UserSegmentMatchingService(
        repository,
        FakeCandidateRepository([candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)

    primary_memberships = [
        membership
        for membership in repository.memberships
        if membership["external_user_id"] == "user-1"
        and membership["analysis_date"] == ANALYSIS_DATE
        and membership["is_primary"] is True
    ]
    assert len(primary_memberships) == 1
    assert primary_memberships[0]["segment_id"] == 1
