from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from typing import Any


CRITERION_EVIDENCE = "evidence"
CRITERION_QUALITY = "quality"
VERDICT_PASSED = "passed"
VERDICT_FAILED = "failed"
VERDICT_INCONCLUSIVE = "inconclusive"

_RANK_NAMES = {1: "one", 2: "two", 3: "three"}


@dataclass(frozen=True, slots=True)
class RankedOutcome:
    rank: int
    actual_rate: float
    baseline_rate: float


def summarize_rank_quality(
    scenarios: Sequence[Sequence[RankedOutcome]],
) -> dict[str, Any]:
    observed_scenarios: list[list[RankedOutcome]] = []
    by_rank: dict[int, list[RankedOutcome]] = {1: [], 2: [], 3: []}
    multi_candidate_count = 0
    three_rank_count = 0
    comparable_scenario_count = 0
    rank_one_strict_best_count = 0
    rank_one_tied_best_count = 0
    concordant_pair_count = 0
    discordant_pair_count = 0
    tied_pair_count = 0
    rank_one_minus_two_differences: list[float] = []

    for scenario in scenarios:
        ordered = sorted(
            (outcome for outcome in scenario if 1 <= outcome.rank <= 3),
            key=lambda outcome: outcome.rank,
        )
        if not ordered or ordered[0].baseline_rate <= 0:
            continue
        observed_scenarios.append(ordered)
        outcomes_by_rank = {outcome.rank: outcome for outcome in ordered}
        for rank, outcome in outcomes_by_rank.items():
            by_rank[rank].append(outcome)
        if len(ordered) < 2:
            continue

        multi_candidate_count += 1
        if all(rank in outcomes_by_rank for rank in (1, 2, 3)):
            three_rank_count += 1
        rank_one = outcomes_by_rank.get(1)
        rank_two = outcomes_by_rank.get(2)
        if rank_one is not None and rank_two is not None:
            rank_one_minus_two_differences.append(
                (rank_one.actual_rate - rank_two.actual_rate) * 100.0
            )

        scenario_has_directional_pair = False
        for index, earlier in enumerate(ordered):
            for later in ordered[index + 1 :]:
                if _rates_tied(earlier.actual_rate, later.actual_rate):
                    tied_pair_count += 1
                    continue
                scenario_has_directional_pair = True
                if earlier.actual_rate > later.actual_rate:
                    concordant_pair_count += 1
                else:
                    discordant_pair_count += 1
        if not scenario_has_directional_pair:
            if rank_one is not None and _is_tied_for_best(rank_one, ordered):
                rank_one_tied_best_count += 1
            continue

        comparable_scenario_count += 1
        if rank_one is None:
            continue
        if _is_strictly_best(rank_one, ordered):
            rank_one_strict_best_count += 1
        elif _is_tied_for_best(rank_one, ordered):
            rank_one_tied_best_count += 1

    observed_count = len(observed_scenarios)
    directional_pair_count = concordant_pair_count + discordant_pair_count
    total_pair_count = directional_pair_count + tied_pair_count
    metrics: dict[str, Any] = {
        "observed_outcome_scenario_count": observed_count,
        "multi_candidate_scenario_count": multi_candidate_count,
        "three_rank_scenario_count": three_rank_count,
        "rank_comparable_scenario_count": comparable_scenario_count,
        "rank_comparable_scenario_rate": _optional_rate(
            comparable_scenario_count,
            observed_count,
        ),
        "rank_one_is_best_rate": _optional_rate(
            rank_one_strict_best_count,
            comparable_scenario_count,
        ),
        "rank_one_tied_best_rate": _optional_rate(
            rank_one_tied_best_count,
            multi_candidate_count,
        ),
        "rank_one_strict_best_count": rank_one_strict_best_count,
        "rank_one_tied_best_count": rank_one_tied_best_count,
        "pairwise_rank_accuracy": _optional_rate(
            concordant_pair_count,
            directional_pair_count,
        ),
        "pairwise_rank_comparison_count": directional_pair_count,
        "pairwise_rank_total_pair_count": total_pair_count,
        "pairwise_rank_concordant_count": concordant_pair_count,
        "pairwise_rank_discordant_count": discordant_pair_count,
        "pairwise_rank_tie_count": tied_pair_count,
        "pairwise_rank_tie_rate": _optional_rate(
            tied_pair_count,
            total_pair_count,
        ),
        "mean_rank_one_minus_rank_two_outcome_rate_percentage_points": _mean(
            rank_one_minus_two_differences
        ),
    }
    for rank, rank_name in _RANK_NAMES.items():
        rank_results = by_rank[rank]
        metrics[f"rank_{rank_name}_result_count"] = len(rank_results)
        metrics[f"rank_{rank_name}_scenario_coverage_rate"] = _optional_rate(
            len(rank_results),
            observed_count,
        )
        metrics[f"rank_{rank_name}_beats_baseline_rate"] = _optional_rate(
            sum(
                outcome.actual_rate > outcome.baseline_rate
                for outcome in rank_results
            ),
            len(rank_results),
        )
        metrics[f"mean_rank_{rank_name}_lift_percentage_points"] = (
            _mean(
                (outcome.actual_rate - outcome.baseline_rate) * 100.0
                for outcome in rank_results
            )
            if rank_results
            else None
        )
    return metrics


def criterion_result(
    actual: int | float | None,
    operator: str,
    threshold: int | float,
    *,
    category: str,
) -> dict[str, Any]:
    if category not in {CRITERION_EVIDENCE, CRITERION_QUALITY}:
        raise ValueError(f"unsupported criterion category: {category}")
    if operator not in {">=", "<=", ">"}:
        raise ValueError(f"unsupported criterion operator: {operator}")
    passed = False
    if actual is not None:
        if operator == ">=":
            passed = actual >= threshold
        elif operator == "<=":
            passed = actual <= threshold
        else:
            passed = actual > threshold
    return {
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "category": category,
        "passed": passed,
    }


def determine_final_verdict(
    criteria_results: Mapping[str, Mapping[str, Any]],
) -> str:
    evidence_results = [
        result
        for result in criteria_results.values()
        if result.get("category") == CRITERION_EVIDENCE
    ]
    if not evidence_results or any(
        not bool(result.get("passed")) for result in evidence_results
    ):
        return VERDICT_INCONCLUSIVE
    if all(bool(result.get("passed")) for result in criteria_results.values()):
        return VERDICT_PASSED
    return VERDICT_FAILED


def _is_strictly_best(
    rank_one: RankedOutcome,
    ordered: Sequence[RankedOutcome],
) -> bool:
    return all(
        rank_one.rank == other.rank
        or (
            rank_one.actual_rate > other.actual_rate
            and not _rates_tied(rank_one.actual_rate, other.actual_rate)
        )
        for other in ordered
    )


def _is_tied_for_best(
    rank_one: RankedOutcome,
    ordered: Sequence[RankedOutcome],
) -> bool:
    best_rate = max(outcome.actual_rate for outcome in ordered)
    return _rates_tied(rank_one.actual_rate, best_rate) and any(
        outcome.rank != rank_one.rank
        and _rates_tied(outcome.actual_rate, rank_one.actual_rate)
        for outcome in ordered
    )


def _rates_tied(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)
