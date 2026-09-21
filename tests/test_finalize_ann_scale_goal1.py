from __future__ import annotations

from scripts.finalize_ann_scale_goal1 import (
    actual_scale_points,
    candidates_from_results,
)


def test_candidates_from_results_preserves_scale_and_policy_boundaries() -> None:
    result = {
        "screening_candidates": [
            _candidate_row(ann_p95_ms=119.9, filter_first_p95_ms=80.0),
        ]
    }

    (candidate,) = candidates_from_results((result,))

    assert candidate.scale_survivor is True
    assert candidate.policy_survivor is False


def test_actual_scale_points_use_best_survivor_and_hide_failed_ann() -> None:
    result = {
        "cohort_size": 50_000,
        "summaries": [
            _summary("exact_all", p95_ms=100.0),
            _summary("filter_first_exact", p95_ms=80.0),
            _summary(
                "ann_first",
                p95_ms=119.0,
                requested_k=5_000,
                recall=0.95,
            ),
            _summary(
                "ann_first",
                p95_ms=70.0,
                requested_k=10_000,
                recall=0.80,
            ),
        ],
    }

    points = actual_scale_points((result,))
    ann = next(item for item in points if item.plan == "ann_first")

    assert ann.p95_ms == 119.0
    assert ann.quality_passed is True


def test_actual_scale_points_mark_no_survivor_without_inventing_a_point() -> None:
    result = {
        "cohort_size": 50_000,
        "summaries": [
            _summary("exact_all", p95_ms=100.0),
            _summary("filter_first_exact", p95_ms=80.0),
            _summary(
                "ann_first",
                p95_ms=70.0,
                requested_k=5_000,
                recall=0.80,
            ),
        ],
    }

    points = actual_scale_points((result,))
    ann = next(item for item in points if item.plan == "ann_first")

    assert ann.p95_ms == 70.0
    assert ann.quality_passed is False


def _candidate_row(
    *, ann_p95_ms: float, filter_first_p95_ms: float
) -> dict[str, object]:
    return {
        "scenario_id": "scenario-t",
        "candidate_type": "intent_matched",
        "scenario_set": "tuning",
        "corpus_user_count": 50_000,
        "requested_k": 5_000,
        "exact_positive_count": 100,
        "recall": 0.90,
        "ann_p95_ms": ann_p95_ms,
        "exact_all_p95_ms": 100.0,
        "filter_first_p95_ms": filter_first_p95_ms,
        "filter_first_results_equal": True,
        "hnsw_index_used": True,
        "temp_spill": False,
        "oom": False,
    }


def _summary(
    plan: str,
    *,
    p95_ms: float,
    requested_k: int | None = None,
    recall: float = 1.0,
) -> dict[str, object]:
    return {
        "scenario_id": "scenario-t",
        "scenario_set": "tuning",
        "candidate_type": "intent_matched",
        "plan": plan,
        "requested_k": requested_k,
        "p95_ms": p95_ms,
        "run_count": 10,
        "hard_match_ratio": 0.10,
        "expected_member_ratio": 0.03,
        "exact_positive_count": 100,
        "final_user_count": 100,
        "intersection_count": 100,
        "recall": recall,
        "index_used": plan == "ann_first",
        "temp_spill": False,
        "oom": False,
    }
