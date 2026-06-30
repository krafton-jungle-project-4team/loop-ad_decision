from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.decision.models import ExistingSegment, UserSegmentCandidate
from app.decision.services import UserSegmentMatchingService


ANALYSIS_DATE = date(2021, 1, 4)
SEGMENT_KEY = "age_30s__gender_male__device_mobile_web__channel_kakao__category_fresh_food"


class FakeMatchingRepository:
    def __init__(self, existing_segments: dict[str, ExistingSegment]) -> None:
        self.existing_segments = existing_segments
        self.memberships: list[dict[str, object]] = []
        self.list_existing_project_ids: list[int] = []
        self.replace_calls: list[dict[str, object]] = []
        self.timezone = "Asia/Seoul"
        self.project_key = "demo-shop"

    def get_project_timezone(self, *, project_id: int) -> str:
        return self.timezone

    def get_project_key(self, *, project_id: int) -> str:
        return self.project_key

    def list_existing_segments(self, *, project_id: int) -> dict[str, ExistingSegment]:
        self.list_existing_project_ids.append(project_id)
        return dict(self.existing_segments)

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
        existing = next(
            (
                membership
                for membership in self.memberships
                if membership["project_id"] == project_id
                and membership["external_user_id"] == external_user_id
                and membership["segment_id"] == segment_id
                and membership["analysis_date"] == analysis_date
            ),
            None,
        )
        if existing is not None:
            existing["is_primary"] = True
            existing["confidence"] = confidence
            existing["reason_json"] = reason_json
            existing["run_id"] = run_id
            return
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


def test_existing_segment_key_creates_user_segment_membership() -> None:
    repository = FakeMatchingRepository(
        {SEGMENT_KEY: ExistingSegment(id=10, segment_key=SEGMENT_KEY)}
    )
    candidate_repository = FakeCandidateRepository([candidate()])

    result = UserSegmentMatchingService(repository, candidate_repository).run(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=77,
    )

    assert result.matched_count == 1
    assert result.skipped_count == 0
    assert repository.memberships[0]["segment_id"] == 10
    assert repository.memberships[0]["is_primary"] is True
    assert repository.replace_calls[0]["reason_json"]["segment_key"] == SEGMENT_KEY
    assert candidate_repository.calls[0][0] == "demo-shop"


def test_missing_segment_key_skips_membership_without_creating_segment() -> None:
    repository = FakeMatchingRepository({})
    candidate_repository = FakeCandidateRepository([candidate()])

    result = UserSegmentMatchingService(repository, candidate_repository).run(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=None,
    )

    assert result.matched_count == 0
    assert result.skipped_count == 1
    assert repository.memberships == []
    assert repository.existing_segments == {}
    assert repository.replace_calls == []


def test_replacing_primary_membership_leaves_one_primary_per_user_day() -> None:
    repository = FakeMatchingRepository(
        {SEGMENT_KEY: ExistingSegment(id=10, segment_key=SEGMENT_KEY)}
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
    assert primary_memberships[0]["segment_id"] == 10


def test_unmatched_user_remains_without_membership_for_default_fallback() -> None:
    repository = FakeMatchingRepository(
        {SEGMENT_KEY: ExistingSegment(id=10, segment_key=SEGMENT_KEY)}
    )
    candidate_repository = FakeCandidateRepository(
        [candidate(external_user_id="user-2", age_group="40s")]
    )

    result = UserSegmentMatchingService(repository, candidate_repository).run(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=77,
    )

    assert result.matched_count == 0
    assert result.skipped_count == 1
    assert repository.memberships == []
