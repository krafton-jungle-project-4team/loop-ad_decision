from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from app.analysis.segments import SEGMENT_DIMENSIONS, normalize_dimension_value
from app.decision.models import ExistingSegment, UserSegmentCandidate
from app.decision.services import UserSegmentMatchingService


ANALYSIS_DATE = date(2021, 1, 4)
FULL_DIMENSIONS = {
    "age_group": "30s",
    "gender": "Male",
    "device_type": "Mobile Web",
    "acquisition_channel": "Kakao",
    "primary_category": "Fresh Food",
}
DEFAULT_MATCHING_CONFIG = {
    "dimension_weights": {
        "primary_category": 3,
        "acquisition_channel": 2,
        "device_type": 1,
        "age_group": 1,
        "gender": 1,
    },
    "min_score": 3,
}


class FakeMatchingRepository:
    def __init__(self, existing_segments: list[ExistingSegment]) -> None:
        self.existing_segments = existing_segments
        self.memberships: list[dict[str, object]] = []
        self.list_existing_calls: list[tuple[int, date]] = []
        self.replace_calls: list[dict[str, object]] = []
        self.timezone = "Asia/Seoul"
        self.project_key = "demo-shop"

    def get_project_timezone(self, *, project_id: int) -> str:
        return self.timezone

    def get_project_key(self, *, project_id: int) -> str:
        return self.project_key

    def list_existing_segments(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> list[ExistingSegment]:
        self.list_existing_calls.append((project_id, analysis_date))
        return list(self.existing_segments)

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
    age_group: object = "30s",
    gender: object = " Male ",
    device: object = "Mobile Web",
    channel: object = "Kakao",
    category: object = "Fresh Food",
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


def segment(
    *,
    segment_id: int,
    segment_key: str | None = None,
    dimensions: dict[str, object] | None = None,
    matching_config: Any | None = None,
) -> ExistingSegment:
    raw_dimensions = FULL_DIMENSIONS if dimensions is None else dimensions
    return ExistingSegment(
        id=segment_id,
        segment_key=segment_key or f"segment-{segment_id}",
        dimensions={
            dimension: normalize_dimension_value(value)
            for dimension, value in raw_dimensions.items()
        },
        matching_config=matching_config,
    )


def run_service(
    repository: FakeMatchingRepository,
    candidates: list[UserSegmentCandidate] | None = None,
):
    return UserSegmentMatchingService(
        repository,
        FakeCandidateRepository(candidates or [candidate()]),
    ).run(project_id=1, analysis_date=ANALYSIS_DATE, run_id=77)


def selected_reason(repository: FakeMatchingRepository) -> dict[str, object]:
    return repository.replace_calls[0]["reason_json"]  # type: ignore[return-value]


def test_exact_match_still_wins_and_writes_weighted_reason_json() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=20,
                segment_key="category-only",
                dimensions={"primary_category": "Fresh Food"},
                matching_config=DEFAULT_MATCHING_CONFIG,
            ),
            segment(
                segment_id=10,
                segment_key="full-match",
                matching_config=DEFAULT_MATCHING_CONFIG,
            ),
        ]
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
    assert repository.memberships[0]["confidence"] > Decimal("0.3750")
    assert repository.list_existing_calls == [(1, ANALYSIS_DATE)]
    assert candidate_repository.calls[0][0] == "demo-shop"
    reason = selected_reason(repository)
    assert reason["matching_source"] == "weighted_user_segment_matching_v1"
    assert reason["selected_segment_key"] == "full-match"
    assert reason["score"] == 8
    assert reason["dimension_weights"] == DEFAULT_MATCHING_CONFIG["dimension_weights"]
    assert set(reason["matched_dimensions"]) == set(SEGMENT_DIMENSIONS)


def test_category_only_score_three_creates_membership() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 10
    assert repository.memberships[0]["confidence"] == Decimal("0.3750")
    reason = selected_reason(repository)
    assert reason["score"] == 3
    assert reason["threshold"] == 3
    assert reason["confidence"] == 0.375
    assert reason["matched_dimensions"] == ["primary_category"]


def test_score_below_default_threshold_creates_no_membership() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"device_type": "Mobile Web"},
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 0
    assert result.skipped_count == 1
    assert repository.memberships == []
    assert repository.replace_calls == []


def test_multiple_partial_matches_choose_higher_weighted_score() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                segment_key="channel",
                dimensions={"acquisition_channel": "Kakao"},
            ),
            segment(
                segment_id=20,
                segment_key="category",
                dimensions={"primary_category": "Fresh Food"},
            ),
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 20
    assert selected_reason(repository)["selected_segment_key"] == "category"


def test_equal_scores_use_highest_single_matched_weight_tie_break() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                segment_key="channel-device",
                dimensions={
                    "acquisition_channel": "Kakao",
                    "device_type": "Mobile Web",
                },
            ),
            segment(
                segment_id=20,
                segment_key="category",
                dimensions={"primary_category": "Fresh Food"},
            ),
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 20
    assert selected_reason(repository)["selected_segment_key"] == "category"


def test_segment_specific_weights_drive_highest_single_weight_tie_break() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                segment_key="lower-id-channel-device",
                dimensions={
                    "acquisition_channel": "Kakao",
                    "device_type": "Mobile Web",
                },
                matching_config={
                    "dimension_weights": {
                        "acquisition_channel": 2,
                        "device_type": 2,
                    },
                    "min_score": 1,
                },
            ),
            segment(
                segment_id=20,
                segment_key="higher-id-category-age",
                dimensions={
                    "primary_category": "Fresh Food",
                    "age_group": "30s",
                },
                matching_config={
                    "dimension_weights": {
                        "primary_category": 3,
                        "age_group": 1,
                    },
                    "min_score": 1,
                },
            ),
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 20
    reason = selected_reason(repository)
    assert reason["selected_segment_key"] == "higher-id-category-age"
    assert reason["score"] == 4
    assert reason["dimension_weights"]["primary_category"] == 3
    assert reason["dimension_weights"]["age_group"] == 1


def test_equal_score_and_highest_weight_use_more_matched_dimensions_tie_break() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                segment_key="two-dimensions",
                dimensions={
                    "primary_category": "Fresh Food",
                    "acquisition_channel": "Kakao",
                },
                matching_config={
                    "dimension_weights": {
                        "primary_category": 1,
                        "acquisition_channel": 1,
                    },
                    "min_score": 1,
                },
            ),
            segment(
                segment_id=20,
                segment_key="three-dimensions",
                dimensions={
                    "primary_category": "Fresh Food",
                    "device_type": "Mobile Web",
                    "age_group": "30s",
                },
                matching_config={
                    "dimension_weights": {
                        "primary_category": 1,
                        "device_type": 0.5,
                        "age_group": 0.5,
                    },
                    "min_score": 1,
                },
            ),
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 20
    assert selected_reason(repository)["selected_segment_key"] == "three-dimensions"


def test_fully_equal_tie_break_uses_lower_segment_id() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=20,
                segment_key="segment-20",
                dimensions={"primary_category": "Fresh Food"},
            ),
            segment(
                segment_id=10,
                segment_key="segment-10",
                dimensions={"primary_category": "Fresh Food"},
            ),
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 10
    assert selected_reason(repository)["selected_segment_key"] == "segment-10"


@pytest.mark.parametrize("category", [None, "", "unknown", "N/A"])
def test_null_empty_and_unknown_values_add_no_score(category: object) -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
            )
        ]
    )

    result = run_service(repository, [candidate(category=category)])

    assert result.matched_count == 0
    assert result.skipped_count == 1
    assert repository.memberships == []


def test_absent_segment_dimension_is_skipped_not_a_mismatch() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
            )
        ]
    )

    result = run_service(
        repository,
        [
            candidate(
                age_group="60s",
                gender="Female",
                device="Desktop",
                channel="Email",
                category="Fresh Food",
            )
        ],
    )

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 10
    assert selected_reason(repository)["matched_dimensions"] == ["primary_category"]


def test_adding_device_type_to_rule_json_starts_awarding_device_weight() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                segment_key="category-only",
                dimensions={"primary_category": "Fresh Food"},
            ),
            segment(
                segment_id=20,
                segment_key="category-device",
                dimensions={
                    "primary_category": "Fresh Food",
                    "device_type": "Mobile Web",
                },
            ),
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 20
    assert selected_reason(repository)["score"] == 4
    assert selected_reason(repository)["matched_dimensions"] == [
        "device_type",
        "primary_category",
    ]


def test_analysis_provided_weights_can_change_selected_segment() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                segment_key="category-default",
                dimensions={"primary_category": "Fresh Food"},
            ),
            segment(
                segment_id=20,
                segment_key="channel-boosted",
                dimensions={"acquisition_channel": "Kakao"},
                matching_config={"dimension_weights": {"acquisition_channel": 5}},
            ),
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    assert repository.memberships[0]["segment_id"] == 20
    assert selected_reason(repository)["selected_segment_key"] == "channel-boosted"
    assert selected_reason(repository)["dimension_weights"]["acquisition_channel"] == 5


def test_partial_weight_override_keeps_unlisted_defaults() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={
                    "primary_category": "Fresh Food",
                    "acquisition_channel": "Kakao",
                },
                matching_config={"dimension_weights": {"primary_category": 5}},
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    reason = selected_reason(repository)
    assert reason["score"] == 7
    assert reason["dimension_weights"]["primary_category"] == 5
    assert reason["dimension_weights"]["acquisition_channel"] == 2


def test_malformed_and_non_positive_overrides_fall_back_to_defaults() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={
                    "primary_category": "Fresh Food",
                    "acquisition_channel": "Kakao",
                },
                matching_config={
                    "dimension_weights": {
                        "primary_category": 0,
                        "acquisition_channel": "not-a-number",
                    },
                    "min_score": -1,
                },
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    reason = selected_reason(repository)
    assert reason["score"] == 5
    assert reason["threshold"] == 3
    assert reason["dimension_weights"]["primary_category"] == 3
    assert reason["dimension_weights"]["acquisition_channel"] == 2


def test_malformed_dimension_weights_object_uses_default_weights_only() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
                matching_config={
                    "dimension_weights": ["bad"],
                    "min_score": 2,
                },
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    reason = selected_reason(repository)
    assert reason["score"] == 3
    assert reason["threshold"] == 2
    assert reason["dimension_weights"] == DEFAULT_MATCHING_CONFIG["dimension_weights"]


def test_malformed_min_score_keeps_valid_weight_overrides() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
                matching_config={
                    "dimension_weights": {"primary_category": 5},
                    "min_score": "bad",
                },
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    reason = selected_reason(repository)
    assert reason["score"] == 5
    assert reason["threshold"] == 3
    assert reason["dimension_weights"]["primary_category"] == 5


def test_only_malformed_dimension_weight_keys_fall_back_to_defaults() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={
                    "primary_category": "Fresh Food",
                    "acquisition_channel": "Kakao",
                    "device_type": "Mobile Web",
                },
                matching_config={
                    "dimension_weights": {
                        "primary_category": 0,
                        "acquisition_channel": 4,
                        "device_type": "bad",
                    },
                },
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    reason = selected_reason(repository)
    assert reason["score"] == 8
    assert reason["dimension_weights"]["primary_category"] == 3
    assert reason["dimension_weights"]["acquisition_channel"] == 4
    assert reason["dimension_weights"]["device_type"] == 1


def test_malformed_matching_config_uses_defaults() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
                matching_config=["malformed"],
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    reason = selected_reason(repository)
    assert reason["score"] == 3
    assert reason["threshold"] == 3
    assert reason["dimension_weights"] == DEFAULT_MATCHING_CONFIG["dimension_weights"]


def test_missing_metric_config_uses_defaults() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
                matching_config=None,
            )
        ]
    )

    result = run_service(repository)

    assert result.matched_count == 1
    reason = selected_reason(repository)
    assert reason["score"] == 3
    assert reason["threshold"] == 3


def test_replacing_primary_membership_leaves_one_primary_per_user_day() -> None:
    repository = FakeMatchingRepository(
        [
            segment(
                segment_id=10,
                dimensions={"primary_category": "Fresh Food"},
            )
        ]
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

    result = run_service(repository)

    primary_memberships = [
        membership
        for membership in repository.memberships
        if membership["external_user_id"] == "user-1"
        and membership["analysis_date"] == ANALYSIS_DATE
        and membership["is_primary"] is True
    ]
    assert result.matched_count == 1
    assert len(primary_memberships) == 1
    assert primary_memberships[0]["segment_id"] == 10
