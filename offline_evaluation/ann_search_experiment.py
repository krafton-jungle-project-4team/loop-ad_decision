"""Offline ANN search benchmark records, policy synthesis, and artifacts.

This module is deliberately kept outside ``app`` so benchmark-only policy
search cannot become part of the Decision request path by accident.  The
runtime policy implementation is a separate, explicitly approved change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from statistics import NormalDist, fmean
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_VERSION = "audience_search.benchmark.v1"
SCALE_EXPERIMENT_VERSION = "audience_search.benchmark.v2"
SUPPORTED_EXPERIMENT_VERSIONS = frozenset(
    {EXPERIMENT_VERSION, SCALE_EXPERIMENT_VERSION}
)
TARGET_RECALL = 0.95
HNSW_INDEX_NAME = "idx_user_behavior_vector_search_embedding_hnsw"

CORPUS_SIZES = (50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000)
MIN_ANN_CANDIDATE_GRID = (5_000, 10_000, 20_000, 50_000)
K_SAFETY_FACTOR_GRID = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
MAX_CORPUS_FRACTION_GRID = (0.05, 0.10, 0.15, 0.25)
SCORE_PASS_SAMPLE_SIZE_GRID = (2_000, 5_000, 10_000, 20_000, 50_000)
EF_SEARCH_GRID = (50, 100, 200, 400)
ITERATIVE_SCAN_GRID = ("strict_order", "relaxed_order")
MAX_SCAN_TUPLES_GRID = (20_000, 50_000, 100_000)
CANDIDATE_RATIO_ANCHORS = (0.01, 0.05, 0.10, 0.25)

DEFAULT_HARD_MATCH_RATIO_BANDS = ((0.0, 0.05), (0.05, 0.20), (0.20, 1.0000001))
DEFAULT_EXPECTED_MEMBER_RATIO_BANDS = (
    (0.0, 0.01),
    (0.01, 0.05),
    (0.05, 0.10),
    (0.10, 0.25),
    (0.25, 1.0000001),
)
DEFAULT_CANDIDATE_TYPES = (
    "intent_matched",
    "target_destination_affinity",
    "funnel_recovery",
    "benefit_value_seeker",
    "general_destination_explorer",
)


class SearchPlan(StrEnum):
    CURRENT_RUNTIME = "current_runtime"
    EXACT_ALL = "exact_all"
    FILTER_FIRST_EXACT = "filter_first_exact"
    ANN_FIRST = "ann_first"


class CacheMode(StrEnum):
    WARM = "warm"
    DB_COLD = "db_cold"


class BenchmarkPhase(StrEnum):
    SCREENING = "screening"
    TUNING = "tuning"
    CONFIRMATION = "confirmation"


class ScenarioSet(StrEnum):
    TUNING = "tuning"
    CONFIRMATION = "confirmation"


@dataclass(frozen=True, slots=True, order=True)
class HnswSettings:
    ef_search: int
    iterative_scan: str
    max_scan_tuples: int

    def __post_init__(self) -> None:
        if self.ef_search <= 0 or self.max_scan_tuples <= 0:
            raise ValueError("HNSW numeric settings must be positive")
        if self.iterative_scan not in ITERATIVE_SCAN_GRID:
            raise ValueError("unsupported hnsw.iterative_scan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, order=True)
class KPolicyParameters:
    min_candidates: int
    k_safety_factor: float
    max_corpus_fraction: float

    def __post_init__(self) -> None:
        if self.min_candidates <= 0 or self.k_safety_factor <= 0:
            raise ValueError("ANN K parameters must be positive")
        if not 0 < self.max_corpus_fraction <= 1:
            raise ValueError("max_corpus_fraction must be in (0, 1]")

    def proposed_k(
        self,
        *,
        corpus_user_count: int,
        expected_member_count: float,
    ) -> int | None:
        if corpus_user_count <= 0 or expected_member_count < 0:
            raise ValueError("corpus and expected member counts are invalid")
        candidate_count = min(
            corpus_user_count,
            max(
                self.min_candidates,
                math.ceil(self.k_safety_factor * expected_member_count),
            ),
        )
        if candidate_count / corpus_user_count > self.max_corpus_fraction:
            return None
        return candidate_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    experiment_version: str
    phase: BenchmarkPhase
    measured: bool
    scenario_id: str
    candidate_type: str
    corpus_user_count: int
    hard_match_user_count: int
    score_pass_sample_size: int
    estimated_score_pass_rate: float
    actual_score_pass_rate: float
    plan: SearchPlan
    cache_mode: CacheMode
    exclusion_ratio: float
    duration_ms: float
    final_user_count: int
    exact_positive_count: int
    intersection_count: int
    requested_k: int | None = None
    hnsw: HnswSettings | None = None
    index_name: str | None = None
    temp_spill: bool = False
    oom: bool = False
    peak_rss_bytes: int | None = None
    theoretical_min_k: int | None = None
    scenario_set: ScenarioSet = ScenarioSet.CONFIRMATION
    ann_attempt_count: int = 0
    requested_k_history: tuple[int, ...] = ()
    audited_user_count: int = 0
    used_exact_fallback: bool = False
    stage_durations_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.experiment_version not in SUPPORTED_EXPERIMENT_VERSIONS:
            raise ValueError("unsupported ANN benchmark experiment version")
        if not self.scenario_id or not self.candidate_type:
            raise ValueError("scenario and candidate type are required")
        counts = (
            self.corpus_user_count,
            self.hard_match_user_count,
            self.score_pass_sample_size,
            self.final_user_count,
            self.exact_positive_count,
            self.intersection_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("benchmark counts must not be negative")
        if self.corpus_user_count <= 0 or self.score_pass_sample_size <= 0:
            raise ValueError("corpus and sample size must be positive")
        if self.hard_match_user_count > self.corpus_user_count:
            raise ValueError("hard match count exceeds corpus")
        if self.intersection_count > min(
            self.final_user_count, self.exact_positive_count
        ):
            raise ValueError("intersection count exceeds result counts")
        for value in (
            self.estimated_score_pass_rate,
            self.actual_score_pass_rate,
            self.exclusion_ratio,
        ):
            if not 0 <= value <= 1:
                raise ValueError("benchmark ratios must be in [0, 1]")
        if self.duration_ms < 0:
            raise ValueError("duration must not be negative")
        if self.plan == SearchPlan.ANN_FIRST:
            if self.requested_k is None or self.hnsw is None:
                raise ValueError("ANN observations require K and HNSW settings")
            if not 0 < self.requested_k <= self.corpus_user_count:
                raise ValueError("ANN requested K is outside the corpus")
        elif self.requested_k is not None or self.hnsw is not None:
            raise ValueError("Exact observations must not contain ANN settings")
        if self.peak_rss_bytes is not None and self.peak_rss_bytes < 0:
            raise ValueError("peak RSS must not be negative")
        if self.ann_attempt_count < 0 or self.audited_user_count < 0:
            raise ValueError("runtime work-amplification counts must not be negative")
        if any(value <= 0 for value in self.requested_k_history):
            raise ValueError("runtime K history must contain positive values")
        if any(value < 0 for value in self.stage_durations_ms.values()):
            raise ValueError("runtime stage durations must not be negative")
        if self.plan != SearchPlan.CURRENT_RUNTIME and (
            self.ann_attempt_count
            or self.requested_k_history
            or self.audited_user_count
            or self.used_exact_fallback
            or self.stage_durations_ms
        ):
            raise ValueError(
                "runtime work-amplification fields require current_runtime plan"
            )

    @property
    def hard_match_ratio(self) -> float:
        return self.hard_match_user_count / self.corpus_user_count

    @property
    def expected_member_count(self) -> float:
        return self.hard_match_user_count * self.estimated_score_pass_rate

    @property
    def expected_member_ratio(self) -> float:
        return self.expected_member_count / self.corpus_user_count

    @property
    def precision(self) -> float:
        if self.final_user_count == 0:
            return 1.0
        return self.intersection_count / self.final_user_count

    @property
    def recall(self) -> float:
        if self.exact_positive_count == 0:
            return 1.0
        return self.intersection_count / self.exact_positive_count

    @property
    def recall_lower_bound(self) -> float:
        if self.exact_positive_count == 0:
            return 0.0
        return wilson_lower_bound(
            successes=self.intersection_count,
            trials=self.exact_positive_count,
            confidence=0.95,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkObservation":
        hnsw_payload = payload.get("hnsw")
        return cls(
            experiment_version=str(payload["experiment_version"]),
            phase=BenchmarkPhase(str(payload["phase"])),
            measured=bool(payload["measured"]),
            scenario_id=str(payload["scenario_id"]),
            candidate_type=str(payload["candidate_type"]),
            corpus_user_count=int(payload["corpus_user_count"]),
            hard_match_user_count=int(payload["hard_match_user_count"]),
            score_pass_sample_size=int(payload["score_pass_sample_size"]),
            estimated_score_pass_rate=float(payload["estimated_score_pass_rate"]),
            actual_score_pass_rate=float(payload["actual_score_pass_rate"]),
            plan=SearchPlan(str(payload["plan"])),
            cache_mode=CacheMode(str(payload["cache_mode"])),
            exclusion_ratio=float(payload.get("exclusion_ratio", 0.0)),
            duration_ms=float(payload["duration_ms"]),
            final_user_count=int(payload["final_user_count"]),
            exact_positive_count=int(payload["exact_positive_count"]),
            intersection_count=int(payload["intersection_count"]),
            requested_k=(
                int(payload["requested_k"])
                if payload.get("requested_k") is not None
                else None
            ),
            hnsw=(
                HnswSettings(
                    ef_search=int(hnsw_payload["ef_search"]),
                    iterative_scan=str(hnsw_payload["iterative_scan"]),
                    max_scan_tuples=int(hnsw_payload["max_scan_tuples"]),
                )
                if isinstance(hnsw_payload, Mapping)
                else None
            ),
            index_name=(
                str(payload["index_name"])
                if payload.get("index_name") is not None
                else None
            ),
            temp_spill=bool(payload.get("temp_spill", False)),
            oom=bool(payload.get("oom", False)),
            peak_rss_bytes=(
                int(payload["peak_rss_bytes"])
                if payload.get("peak_rss_bytes") is not None
                else None
            ),
            theoretical_min_k=(
                int(payload["theoretical_min_k"])
                if payload.get("theoretical_min_k") is not None
                else None
            ),
            scenario_set=ScenarioSet(
                str(payload.get("scenario_set", ScenarioSet.CONFIRMATION.value))
            ),
            ann_attempt_count=int(payload.get("ann_attempt_count", 0)),
            requested_k_history=tuple(
                int(value) for value in payload.get("requested_k_history", ())
            ),
            audited_user_count=int(payload.get("audited_user_count", 0)),
            used_exact_fallback=bool(payload.get("used_exact_fallback", False)),
            stage_durations_ms={
                str(key): float(value)
                for key, value in payload.get("stage_durations_ms", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["plan"] = self.plan.value
        payload["cache_mode"] = self.cache_mode.value
        payload["scenario_set"] = self.scenario_set.value
        payload["hnsw"] = self.hnsw.to_dict() if self.hnsw else None
        payload["precision"] = self.precision
        payload["recall"] = self.recall
        payload["recall_lower_bound"] = self.recall_lower_bound
        payload["hard_match_ratio"] = self.hard_match_ratio
        payload["expected_member_ratio"] = self.expected_member_ratio
        return payload


@dataclass(frozen=True, slots=True)
class RunSummary:
    scenario_id: str
    candidate_type: str
    corpus_user_count: int
    hard_match_user_count: int
    score_pass_sample_size: int
    estimated_score_pass_rate: float
    actual_score_pass_rate: float
    plan: SearchPlan
    cache_mode: CacheMode
    exclusion_ratio: float
    requested_k: int | None
    hnsw: HnswSettings | None
    run_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    final_user_count: int
    exact_positive_count: int
    intersection_count: int
    precision: float
    recall: float
    recall_lower_bound: float
    index_used: bool
    temp_spill: bool
    oom: bool
    peak_rss_bytes: int | None
    theoretical_min_k: int | None
    scenario_set: ScenarioSet
    duration_samples_ms: tuple[float, ...]
    max_ann_attempt_count: int
    requested_k_histories: tuple[tuple[int, ...], ...]
    max_audited_user_count: int
    exact_fallback_rate: float
    stage_p95_ms: Mapping[str, float]

    @property
    def hard_match_ratio(self) -> float:
        return self.hard_match_user_count / self.corpus_user_count

    @property
    def expected_member_count(self) -> float:
        return self.hard_match_user_count * self.estimated_score_pass_rate

    @property
    def expected_member_ratio(self) -> float:
        return self.expected_member_count / self.corpus_user_count

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan"] = self.plan.value
        payload["cache_mode"] = self.cache_mode.value
        payload["scenario_set"] = self.scenario_set.value
        payload["hnsw"] = self.hnsw.to_dict() if self.hnsw else None
        payload.pop("duration_samples_ms", None)
        payload["hard_match_ratio"] = self.hard_match_ratio
        payload["expected_member_ratio"] = self.expected_member_ratio
        return payload


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    target_recall: float = TARGET_RECALL
    minimum_exact_positives: int = 100
    ann_p95_ratio_max: float = 0.70
    ann_p99_ratio_max: float = 1.0
    ann_peak_rss_ratio_max: float = 1.25
    filter_first_p95_ratio_max: float = 0.85
    global_hnsw_local_p95_ratio_max: float = 1.10
    score_pass_relative_error_p95_max: float = 0.10
    score_pass_p95_ratio_max: float = 1.05
    minimum_policy_run_count: int = 300
    minimum_cold_run_count: int = 30
    bootstrap_iterations: int = 1_000
    bootstrap_confidence: float = 0.95
    require_db_cold_confirmation: bool = True
    required_exclusion_ratio: float | None = 0.05
    required_candidate_types: tuple[str, ...] = DEFAULT_CANDIDATE_TYPES
    require_disjoint_scenario_holdout: bool = True
    hard_match_ratio_bands: tuple[tuple[float, float], ...] = (
        DEFAULT_HARD_MATCH_RATIO_BANDS
    )
    expected_member_ratio_bands: tuple[tuple[float, float], ...] = (
        DEFAULT_EXPECTED_MEMBER_RATIO_BANDS
    )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    plan: SearchPlan
    requested_k: int | None
    hnsw: HnswSettings | None
    rule_index: int | None
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.value,
            "requested_k": self.requested_k,
            "hnsw": self.hnsw.to_dict() if self.hnsw else None,
            "rule_index": self.rule_index,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class PolicySynthesisResult:
    policy: Mapping[str, Any]
    summaries: tuple[RunSummary, ...]
    fixtures: tuple[Mapping[str, Any], ...]
    comparison: Mapping[str, Any]
    fallbacks: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _Point:
    scenario_id: str
    candidate_type: str
    corpus_user_count: int
    hard_match_user_count: int
    reference_score_pass_rate: float
    hard_band: tuple[float, float]
    expected_band: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _RuleCandidate:
    min_users: int
    max_users: int
    hard_band: tuple[float, float]
    expected_band: tuple[float, float]
    plan: SearchPlan
    sample_size: int
    k_policy: KPolicyParameters | None
    hnsw: HnswSettings | None
    points: tuple[_Point, ...]
    worst_p95_ratio: float
    validated_recall: float
    validated_recall_lower_bound: float


def load_observations(path: Path) -> list[BenchmarkObservation]:
    observations: list[BenchmarkObservation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                payload = json.loads(value)
                observations.append(BenchmarkObservation.from_dict(payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid benchmark observation at line {line_number}: {exc}"
                ) from exc
    if not observations:
        raise ValueError("benchmark observation file is empty")
    return observations


def write_observations(
    observations: Iterable[BenchmarkObservation],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.to_dict(), sort_keys=True))
            handle.write("\n")


def summarize_observations(
    observations: Sequence[BenchmarkObservation],
    *,
    phase: BenchmarkPhase | None = None,
) -> list[RunSummary]:
    selected = [item for item in observations if item.measured]
    if phase is not None:
        selected = [item for item in selected if item.phase == phase]
    if not selected:
        raise ValueError("no measured benchmark observations were selected")

    grouped: dict[tuple[Any, ...], list[BenchmarkObservation]] = defaultdict(list)
    for item in selected:
        grouped[_summary_key(item)].append(item)

    summaries: list[RunSummary] = []
    for runs in grouped.values():
        first = runs[0]
        durations = [item.duration_ms for item in runs]
        final_counts = {item.final_user_count for item in runs}
        exact_counts = {item.exact_positive_count for item in runs}
        intersections = {item.intersection_count for item in runs}
        if len(exact_counts) != 1:
            raise ValueError(
                "the same benchmark cell used non-deterministic ground truth counts"
            )
        if first.plan in {
            SearchPlan.EXACT_ALL,
            SearchPlan.FILTER_FIRST_EXACT,
        } and (len(final_counts) != 1 or len(intersections) != 1):
            raise ValueError(
                "the same exact benchmark cell produced non-deterministic results"
            )
        memory_values = [
            item.peak_rss_bytes for item in runs if item.peak_rss_bytes is not None
        ]
        stage_names = sorted(
            {
                name
                for item in runs
                for name in item.stage_durations_ms
            }
        )
        requested_k_histories = tuple(
            sorted({item.requested_k_history for item in runs})
        )
        summaries.append(
            RunSummary(
                scenario_id=first.scenario_id,
                candidate_type=first.candidate_type,
                corpus_user_count=first.corpus_user_count,
                hard_match_user_count=first.hard_match_user_count,
                score_pass_sample_size=first.score_pass_sample_size,
                estimated_score_pass_rate=first.estimated_score_pass_rate,
                actual_score_pass_rate=first.actual_score_pass_rate,
                plan=first.plan,
                cache_mode=first.cache_mode,
                exclusion_ratio=first.exclusion_ratio,
                requested_k=first.requested_k,
                hnsw=first.hnsw,
                run_count=len(runs),
                mean_ms=fmean(durations),
                p50_ms=percentile(durations, 0.50),
                p95_ms=percentile(durations, 0.95),
                p99_ms=percentile(durations, 0.99),
                # HNSW may visit a different graph frontier after max_scan_tuples
                # is reached. Preserve the worst observed counts while precision,
                # recall, and Wilson bounds below are also minimized per run.
                final_user_count=min(final_counts),
                exact_positive_count=first.exact_positive_count,
                intersection_count=min(intersections),
                precision=min(item.precision for item in runs),
                recall=min(item.recall for item in runs),
                recall_lower_bound=min(item.recall_lower_bound for item in runs),
                index_used=(
                    first.plan != SearchPlan.ANN_FIRST
                    or all(item.index_name == HNSW_INDEX_NAME for item in runs)
                ),
                temp_spill=any(item.temp_spill for item in runs),
                oom=any(item.oom for item in runs),
                peak_rss_bytes=max(memory_values) if memory_values else None,
                theoretical_min_k=first.theoretical_min_k,
                scenario_set=first.scenario_set,
                duration_samples_ms=tuple(sorted(durations)),
                max_ann_attempt_count=max(
                    item.ann_attempt_count for item in runs
                ),
                requested_k_histories=requested_k_histories,
                max_audited_user_count=max(
                    item.audited_user_count for item in runs
                ),
                exact_fallback_rate=(
                    sum(item.used_exact_fallback for item in runs) / len(runs)
                ),
                stage_p95_ms={
                    name: percentile(
                        [
                            item.stage_durations_ms.get(name, 0.0)
                            for item in runs
                        ],
                        0.95,
                    )
                    for name in stage_names
                },
            )
        )
    return sorted(summaries, key=_summary_sort_key)


def synthesize_policy(
    observations: Sequence[BenchmarkObservation],
    *,
    vector_version: str,
    manifest_hash: str,
    dataset_hash: str,
    scenario_set_hash: str,
    environment: Mapping[str, Any],
    config: SynthesisConfig | None = None,
    status: str = "candidate",
) -> PolicySynthesisResult:
    if status not in {"preliminary", "candidate"}:
        raise ValueError("policy status must be preliminary or candidate")
    cfg = config or SynthesisConfig()
    if cfg.require_disjoint_scenario_holdout:
        _validate_disjoint_scenario_holdout(
            observations,
            config=cfg,
        )
    phase = _highest_available_phase(observations)
    summaries = summarize_observations(observations, phase=phase)
    warm_primary = [
        item
        for item in summaries
        if item.cache_mode == CacheMode.WARM and item.exclusion_ratio == 0.0
        and item.scenario_set == ScenarioSet.CONFIRMATION
    ]
    if not warm_primary:
        raise ValueError("warm-cache primary measurements are required")

    sample_size = _select_score_pass_sample_size(warm_primary, cfg)
    reference_sample_size = max(item.score_pass_sample_size for item in warm_primary)
    points = _build_points(
        warm_primary,
        reference_sample_size=reference_sample_size,
        config=cfg,
    )
    corpus_sizes = sorted({point.corpus_user_count for point in points})
    if not corpus_sizes:
        raise ValueError("no benchmark scenario points are available")

    lookup = _summary_lookup(warm_primary)
    support_lookup = _support_summary_lookup(summaries)
    rule_candidates: list[_RuleCandidate] = []
    fallbacks: list[Mapping[str, Any]] = []
    for index, max_users in enumerate(corpus_sizes):
        previous_users = corpus_sizes[index - 1] if index else None
        min_users = (
            previous_users + 1 if previous_users is not None else max_users
        )
        for hard_band in cfg.hard_match_ratio_bands:
            for expected_band in cfg.expected_member_ratio_bands:
                current_points = _points_in_bucket(
                    points,
                    corpus_user_count=max_users,
                    hard_band=hard_band,
                    expected_band=expected_band,
                )
                if not current_points:
                    fallbacks.append(
                        _fallback_record(
                            min_users,
                            max_users,
                            hard_band,
                            expected_band,
                            "unobserved_bucket",
                        )
                    )
                    continue
                if not _has_candidate_type_coverage(
                    current_points, cfg.required_candidate_types
                ):
                    fallbacks.append(
                        _fallback_record(
                            min_users,
                            max_users,
                            hard_band,
                            expected_band,
                            "candidate_type_coverage_missing",
                        )
                    )
                    continue

                boundary_points = list(current_points)
                if previous_users is not None:
                    previous_points = _points_in_bucket(
                        points,
                        corpus_user_count=previous_users,
                        hard_band=hard_band,
                        expected_band=expected_band,
                    )
                    if _has_candidate_type_coverage(
                        previous_points, cfg.required_candidate_types
                    ):
                        boundary_points.extend(previous_points)
                    else:
                        fallbacks.append(
                            _fallback_record(
                                min_users,
                                max_users,
                                hard_band,
                                expected_band,
                                "adjacent_boundary_unvalidated",
                            )
                        )
                        continue

                ann_candidate = _best_ann_candidate(
                    points=boundary_points,
                    min_users=min_users,
                    max_users=max_users,
                    hard_band=hard_band,
                    expected_band=expected_band,
                    sample_size=sample_size,
                    lookup=lookup,
                    support_lookup=support_lookup,
                    config=cfg,
                )
                if ann_candidate is not None:
                    rule_candidates.append(ann_candidate)
                    continue

                exact_plan = _exact_plan_for_points(
                    boundary_points,
                    sample_size=sample_size,
                    lookup=lookup,
                    config=cfg,
                )
                if exact_plan == SearchPlan.FILTER_FIRST_EXACT:
                    rule_candidates.append(
                        _RuleCandidate(
                            min_users=min_users,
                            max_users=max_users,
                            hard_band=hard_band,
                            expected_band=expected_band,
                            plan=exact_plan,
                            sample_size=sample_size,
                            k_policy=None,
                            hnsw=None,
                            points=tuple(boundary_points),
                            worst_p95_ratio=0.0,
                            validated_recall=1.0,
                            validated_recall_lower_bound=1.0,
                        )
                    )
                else:
                    fallbacks.append(
                        _fallback_record(
                            min_users,
                            max_users,
                            hard_band,
                            expected_band,
                            "exact_all_is_safest",
                        )
                    )

    rule_candidates = _consolidate_global_hnsw(
        rule_candidates,
        lookup=lookup,
        support_lookup=support_lookup,
        config=cfg,
    )
    validated_max_users = max(corpus_sizes)
    rules = [_rule_payload(candidate) for candidate in rule_candidates]
    policy: dict[str, Any] = {
        "version": EXPERIMENT_VERSION,
        "status": status,
        "vector_version": vector_version,
        "manifest_hash": manifest_hash,
        "dataset_hash": dataset_hash,
        "scenario_set_hash": scenario_set_hash,
        "environment": dict(environment),
        "validated_max_users": validated_max_users,
        "target_recall": cfg.target_recall,
        "score_pass_sample_size": sample_size,
        "default_plan": SearchPlan.EXACT_ALL.value,
        "ratio_bounds": "min_inclusive_max_exclusive",
        "rules": rules,
    }
    validate_policy_structure(policy)
    fixtures = tuple(build_implementation_fixtures(policy))
    comparison = build_implementation_comparison(policy)
    if corpus_sizes[0] > 1:
        fallbacks.append(
            {
                "min_users": 1,
                "max_users": corpus_sizes[0] - 1,
                "reason": "below_smallest_validated_user_range",
                "plan": SearchPlan.EXACT_ALL.value,
            }
        )
    fallbacks.append(
        {
            "min_users": validated_max_users + 1,
            "max_users": None,
            "reason": "outside_validated_user_range",
            "plan": SearchPlan.EXACT_ALL.value,
        }
    )
    return PolicySynthesisResult(
        policy=policy,
        summaries=tuple(summaries),
        fixtures=fixtures,
        comparison=comparison,
        fallbacks=tuple(fallbacks),
    )


def build_phase_report(
    observations: Sequence[BenchmarkObservation],
    *,
    phase: BenchmarkPhase,
) -> Mapping[str, Any]:
    """Turn screening/tuning output into the exact cells for the next phase."""

    if phase not in (BenchmarkPhase.SCREENING, BenchmarkPhase.TUNING):
        raise ValueError("phase report supports screening or tuning only")
    summaries = [
        item
        for item in summarize_observations(observations, phase=phase)
        if item.cache_mode == CacheMode.WARM and item.exclusion_ratio == 0.0
    ]
    exact_by_cell: dict[tuple[str, int, int], RunSummary] = {}
    for item in summaries:
        if item.plan not in {
            SearchPlan.EXACT_ALL,
            SearchPlan.FILTER_FIRST_EXACT,
        }:
            continue
        key = (
            item.scenario_id,
            item.corpus_user_count,
            item.score_pass_sample_size,
        )
        current = exact_by_cell.get(key)
        if current is None or (item.p95_ms, item.p99_ms) < (
            current.p95_ms,
            current.p99_ms,
        ):
            exact_by_cell[key] = item

    retained: list[Mapping[str, Any]] = []
    tuning_groups: dict[tuple[str, int, int, int], list[RunSummary]] = defaultdict(list)
    for ann in summaries:
        if ann.plan != SearchPlan.ANN_FIRST:
            continue
        exact = exact_by_cell.get(
            (
                ann.scenario_id,
                ann.corpus_user_count,
                ann.score_pass_sample_size,
            )
        )
        if exact is None:
            continue
        if phase == BenchmarkPhase.SCREENING:
            if (
                min(ann.run_count, exact.run_count) >= 10
                and ann.recall >= 0.90
                and ann.p95_ms <= exact.p95_ms * 1.20
                and ann.index_used
                and not ann.temp_spill
                and not ann.oom
            ):
                retained.append(_phase_cell_payload(ann, exact))
            continue
        if (
            min(ann.run_count, exact.run_count) >= 10
            and ann.precision == 1.0
            and ann.recall >= TARGET_RECALL
            and ann.recall_lower_bound >= TARGET_RECALL
            and ann.index_used
            and not ann.temp_spill
            and not ann.oom
        ):
            assert ann.requested_k is not None
            tuning_groups[
                (
                    ann.scenario_id,
                    ann.corpus_user_count,
                    ann.score_pass_sample_size,
                    ann.requested_k,
                )
            ].append(ann)
    if phase == BenchmarkPhase.TUNING:
        for group in tuning_groups.values():
            winner = min(
                group,
                key=lambda item: (
                    item.p95_ms,
                    item.p99_ms,
                    item.hnsw or HnswSettings(1, "strict_order", 1),
                ),
            )
            exact = exact_by_cell[
                (
                    winner.scenario_id,
                    winner.corpus_user_count,
                    winner.score_pass_sample_size,
                )
            ]
            retained.append(_phase_cell_payload(winner, exact))
    retained.sort(
        key=lambda item: (
            item["corpus_user_count"],
            item["scenario_id"],
            item["score_pass_sample_size"],
            item["requested_k"],
        )
    )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "phase": phase.value,
        "next_phase": (
            BenchmarkPhase.TUNING.value
            if phase == BenchmarkPhase.SCREENING
            else BenchmarkPhase.CONFIRMATION.value
        ),
        "retained_cell_count": len(retained),
        "cells": retained,
    }


def select_policy(
    policy: Mapping[str, Any],
    *,
    corpus_user_count: int,
    hard_match_user_count: int,
    estimated_score_pass_rate: float,
    vector_version: str,
    manifest_hash: str,
) -> PolicyDecision:
    if corpus_user_count <= 0:
        raise ValueError("corpus_user_count must be positive")
    if not 0 <= hard_match_user_count <= corpus_user_count:
        raise ValueError("hard_match_user_count is invalid")
    if not 0 <= estimated_score_pass_rate <= 1:
        raise ValueError("estimated_score_pass_rate must be in [0, 1]")
    if str(policy.get("version")) != EXPERIMENT_VERSION:
        return _exact_fallback("policy_version_mismatch")
    if str(policy.get("vector_version")) != vector_version:
        return _exact_fallback("vector_version_mismatch")
    if str(policy.get("manifest_hash")) != manifest_hash:
        return _exact_fallback("manifest_hash_mismatch")
    try:
        validate_policy_structure(policy)
        validated_max_users = int(policy["validated_max_users"])
    except (KeyError, TypeError, ValueError):
        return _exact_fallback("policy_structure_invalid")
    if corpus_user_count > validated_max_users:
        return _exact_fallback("outside_validated_user_range")

    hard_ratio = hard_match_user_count / corpus_user_count
    expected_count = hard_match_user_count * estimated_score_pass_rate
    expected_ratio = expected_count / corpus_user_count
    rules = policy.get("rules")
    if not isinstance(rules, list):
        return _exact_fallback("policy_rules_invalid")
    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            return _exact_fallback("policy_rule_invalid")
        if not _rule_matches(
            rule,
            corpus_user_count=corpus_user_count,
            hard_match_ratio=hard_ratio,
            expected_member_ratio=expected_ratio,
        ):
            continue
        try:
            plan = SearchPlan(str(rule["plan"]))
        except (KeyError, ValueError):
            return _exact_fallback("policy_plan_invalid")
        if plan != SearchPlan.ANN_FIRST:
            return PolicyDecision(
                plan=plan,
                requested_k=None,
                hnsw=None,
                rule_index=rule_index,
                fallback_reason=None,
            )
        ann = rule.get("ann")
        if not isinstance(ann, Mapping):
            return _exact_fallback("ann_rule_invalid")
        try:
            k_policy = KPolicyParameters(
                min_candidates=int(ann["min_candidates"]),
                k_safety_factor=float(ann["k_safety_factor"]),
                max_corpus_fraction=float(ann["max_corpus_fraction"]),
            )
            hnsw = HnswSettings(
                ef_search=int(ann["ef_search"]),
                iterative_scan=str(ann["iterative_scan"]),
                max_scan_tuples=int(ann["max_scan_tuples"]),
            )
            requested_k = k_policy.proposed_k(
                corpus_user_count=corpus_user_count,
                expected_member_count=expected_count,
            )
        except (KeyError, TypeError, ValueError):
            return _exact_fallback("ann_rule_invalid")
        if requested_k is None:
            return _exact_fallback("ann_candidate_fraction_exceeded")
        return PolicyDecision(
            plan=plan,
            requested_k=requested_k,
            hnsw=hnsw,
            rule_index=rule_index,
            fallback_reason=None,
        )
    return _exact_fallback("no_validated_rule")


def build_implementation_fixtures(
    policy: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    fixtures: list[Mapping[str, Any]] = []

    def add_fixture(fixture_id: str, inputs: Mapping[str, Any]) -> None:
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "inputs": dict(inputs),
                "expected": select_policy(policy, **inputs).to_dict(),
            }
        )

    rules = policy.get("rules", [])
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                continue
            evidence = rule.get("evidence")
            representative = (
                evidence.get("representative")
                if isinstance(evidence, Mapping)
                else None
            )
            if not isinstance(representative, Mapping):
                continue
            representative_users = int(representative["corpus_user_count"])
            representative_hard = int(representative["hard_match_user_count"])
            representative_pass = float(
                representative["estimated_score_pass_rate"]
            )
            representative_hard_ratio = (
                representative_hard / representative_users
            )
            representative_expected_ratio = (
                representative_hard_ratio * representative_pass
            )
            boundary_cases = (
                ("representative", representative_users),
                ("min_users", int(rule["min_users"])),
                ("max_users", int(rule["max_users"])),
            )
            for label, corpus_users in boundary_cases:
                add_fixture(
                    f"rule_{index}_{rule['plan']}_{label}",
                    _fixture_inputs_for_ratios(
                        policy,
                        corpus_user_count=corpus_users,
                        hard_match_ratio=representative_hard_ratio,
                        expected_member_ratio=representative_expected_ratio,
                    ),
                )
            epsilon = 1 / representative_users
            hard_bounds = (
                float(rule["min_hard_match_ratio"]),
                max(
                    float(rule["min_hard_match_ratio"]),
                    float(rule["max_hard_match_ratio"]) - epsilon,
                ),
            )
            for label, hard_ratio in zip(
                ("hard_min", "hard_max_inside"), hard_bounds, strict=True
            ):
                expected_ratio = min(representative_expected_ratio, hard_ratio)
                if expected_ratio < float(rule["min_expected_member_ratio"]):
                    continue
                add_fixture(
                    f"rule_{index}_{rule['plan']}_{label}",
                    _fixture_inputs_for_ratios(
                        policy,
                        corpus_user_count=representative_users,
                        hard_match_ratio=hard_ratio,
                        expected_member_ratio=expected_ratio,
                    ),
                )
            expected_bounds = (
                float(rule["min_expected_member_ratio"]),
                max(
                    float(rule["min_expected_member_ratio"]),
                    float(rule["max_expected_member_ratio"]) - epsilon,
                ),
            )
            for label, expected_ratio in zip(
                ("expected_min", "expected_max_inside"),
                expected_bounds,
                strict=True,
            ):
                if expected_ratio > representative_hard_ratio:
                    continue
                add_fixture(
                    f"rule_{index}_{rule['plan']}_{label}",
                    _fixture_inputs_for_ratios(
                        policy,
                        corpus_user_count=representative_users,
                        hard_match_ratio=representative_hard_ratio,
                        expected_member_ratio=expected_ratio,
                    ),
                )
    over_max_inputs = {
        "corpus_user_count": int(policy["validated_max_users"]) + 1,
        "hard_match_user_count": 0,
        "estimated_score_pass_rate": 0.0,
        "vector_version": str(policy["vector_version"]),
        "manifest_hash": str(policy["manifest_hash"]),
    }
    add_fixture("fallback_outside_validated_range", over_max_inputs)
    mismatch_inputs = {
        "corpus_user_count": min(50_000, int(policy["validated_max_users"])),
        "hard_match_user_count": 1,
        "estimated_score_pass_rate": 1.0,
        "vector_version": str(policy["vector_version"]),
        "manifest_hash": "mismatch",
    }
    add_fixture("fallback_manifest_mismatch", mismatch_inputs)
    return fixtures


def _fixture_inputs_for_ratios(
    policy: Mapping[str, Any],
    *,
    corpus_user_count: int,
    hard_match_ratio: float,
    expected_member_ratio: float,
) -> Mapping[str, Any]:
    hard_match_user_count = min(
        corpus_user_count,
        max(0, round(corpus_user_count * hard_match_ratio)),
    )
    estimated_score_pass_rate = (
        min(
            1.0,
            max(
                0.0,
                expected_member_ratio
                * corpus_user_count
                / hard_match_user_count,
            ),
        )
        if hard_match_user_count
        else 0.0
    )
    return {
        "corpus_user_count": corpus_user_count,
        "hard_match_user_count": hard_match_user_count,
        "estimated_score_pass_rate": estimated_score_pass_rate,
        "vector_version": str(policy["vector_version"]),
        "manifest_hash": str(policy["manifest_hash"]),
    }


def validate_implementation_fixtures(
    policy: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
) -> None:
    validate_policy_structure(policy)
    for fixture in fixtures:
        inputs = fixture.get("inputs")
        expected = fixture.get("expected")
        if not isinstance(inputs, Mapping) or not isinstance(expected, Mapping):
            raise ValueError("implementation fixture is malformed")
        actual = select_policy(policy, **inputs).to_dict()
        if actual != dict(expected):
            raise ValueError(
                f"fixture {fixture.get('fixture_id')} drifted: "
                f"expected {expected}, got {actual}"
            )


def validate_policy_structure(policy: Mapping[str, Any]) -> None:
    """Reject malformed or overlapping runtime buckets conservatively."""

    validated_max_users = int(policy["validated_max_users"])
    if validated_max_users <= 0:
        raise ValueError("validated_max_users must be positive")
    rules = policy.get("rules")
    if not isinstance(rules, list):
        raise ValueError("policy rules must be a list")
    rectangles: list[tuple[int, int, float, float, float, float]] = []
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise ValueError("policy rule must be an object")
        rectangle = (
            int(rule["min_users"]),
            int(rule["max_users"]),
            float(rule["min_hard_match_ratio"]),
            float(rule["max_hard_match_ratio"]),
            float(rule["min_expected_member_ratio"]),
            float(rule["max_expected_member_ratio"]),
        )
        min_users, max_users, min_hard, max_hard, min_expected, max_expected = (
            rectangle
        )
        if not 1 <= min_users <= max_users <= validated_max_users:
            raise ValueError("policy user bucket is invalid")
        if not 0 <= min_hard < max_hard <= 1.0000001:
            raise ValueError("policy hard-match bucket is invalid")
        if not 0 <= min_expected < max_expected <= 1.0000001:
            raise ValueError("policy expected-member bucket is invalid")
        try:
            plan = SearchPlan(str(rule["plan"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("policy plan is invalid") from exc
        if plan == SearchPlan.CURRENT_RUNTIME:
            raise ValueError("current_runtime is a benchmark-only plan")
        if plan == SearchPlan.ANN_FIRST:
            ann = rule.get("ann")
            if not isinstance(ann, Mapping):
                raise ValueError("ANN policy rule is missing settings")
            KPolicyParameters(
                min_candidates=int(ann["min_candidates"]),
                k_safety_factor=float(ann["k_safety_factor"]),
                max_corpus_fraction=float(ann["max_corpus_fraction"]),
            )
            HnswSettings(
                ef_search=int(ann["ef_search"]),
                iterative_scan=str(ann["iterative_scan"]),
                max_scan_tuples=int(ann["max_scan_tuples"]),
            )
            validated_recall = float(ann["validated_recall"])
            validated_lower = float(ann["validated_recall_lower_bound"])
            if not TARGET_RECALL <= validated_lower <= validated_recall <= 1.0:
                raise ValueError("ANN validated recall evidence is invalid")
        rectangles.append(rectangle)
    for left_index, left in enumerate(rectangles):
        for right in rectangles[left_index + 1 :]:
            users_overlap = max(left[0], right[0]) <= min(left[1], right[1])
            hard_overlap = max(left[2], right[2]) < min(left[3], right[3])
            expected_overlap = max(left[4], right[4]) < min(left[5], right[5])
            if users_overlap and hard_overlap and expected_overlap:
                raise ValueError("policy buckets overlap")


def build_implementation_comparison(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    ann_rules = [
        rule
        for rule in policy.get("rules", [])
        if isinstance(rule, Mapping) and rule.get("plan") == SearchPlan.ANN_FIRST.value
    ]
    ann_min_users = (
        min(int(rule["min_users"]) for rule in ann_rules) if ann_rules else None
    )
    candidate_values = {
        key: sorted({rule["ann"][key] for rule in ann_rules})
        for key in (
            "min_candidates",
            "k_safety_factor",
            "max_corpus_fraction",
            "ef_search",
            "iterative_scan",
            "max_scan_tuples",
            "validated_recall",
            "validated_recall_lower_bound",
        )
    }
    return {
        "current": {
            "exact_user_limit": 50_000,
            "transition_user_limit": 500_000,
            "min_ann_candidates": 10_000,
            "ann_k_safety_factor": 1.5,
            "max_ann_corpus_fraction": 0.25,
            "target_threshold_recall": TARGET_RECALL,
            "hnsw": {
                "ef_search": 100,
                "iterative_scan": "strict_order",
                "max_scan_tuples": 20_000,
            },
        },
        "candidate": {
            "ann_min_users": ann_min_users,
            "score_pass_sample_size": policy["score_pass_sample_size"],
            "validated_max_users": policy["validated_max_users"],
            "target_threshold_recall": policy["target_recall"],
            "transition_runtime_audit": "remove_after_separate_approval",
            **candidate_values,
        },
    }


def write_synthesis_artifacts(
    result: PolicySynthesisResult,
    *,
    output_dir: Path,
) -> Mapping[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "policy": output_dir / "candidate-policy.json",
        "fixtures": output_dir / "implementation-fixtures.json",
        "comparison": output_dir / "current-vs-candidate.json",
        "fallbacks": output_dir / "unvalidated-fallbacks.json",
        "summary_json": output_dir / "benchmark-summary.json",
        "summary_csv": output_dir / "benchmark-summary.csv",
        "current_runtime": output_dir / "current-runtime-work-amplification.json",
        "report": output_dir / "report.md",
    }
    _write_json(paths["policy"], result.policy)
    _write_json(paths["fixtures"], {"fixtures": list(result.fixtures)})
    _write_json(paths["comparison"], result.comparison)
    _write_json(paths["fallbacks"], {"fallbacks": list(result.fallbacks)})
    _write_json(
        paths["summary_json"],
        {"summaries": [item.to_dict() for item in result.summaries]},
    )
    _write_summary_csv(paths["summary_csv"], result.summaries)
    _write_json(
        paths["current_runtime"],
        build_current_runtime_report(result.summaries),
    )
    paths["report"].write_text(_render_report(result), encoding="utf-8")
    return paths


def experiment_manifest_template() -> Mapping[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "source_cutoff": "2015-01-01T00:00:00+00:00",
        "window_days": 730,
        "vector_version": "hotel_behavior.v2",
        "corpus_sizes": list(CORPUS_SIZES),
        "candidate_types": list(DEFAULT_CANDIDATE_TYPES),
        "scenario_sets": [item.value for item in ScenarioSet],
        "candidate_ratio_anchors": list(CANDIDATE_RATIO_ANCHORS),
        "grids": {
            "min_ann_candidates": list(MIN_ANN_CANDIDATE_GRID),
            "k_safety_factor": list(K_SAFETY_FACTOR_GRID),
            "max_corpus_fraction": list(MAX_CORPUS_FRACTION_GRID),
            "score_pass_sample_size": list(SCORE_PASS_SAMPLE_SIZE_GRID),
            "ef_search": list(EF_SEARCH_GRID),
            "iterative_scan": list(ITERATIVE_SCAN_GRID),
            "max_scan_tuples": list(MAX_SCAN_TUPLES_GRID),
        },
        "phases": {
            "screening": {"warmup": 5, "measured": 10},
            "tuning": {"warmup": 3, "measured": 10},
            "confirmation": {
                "warmup": 10,
                "warm_measured": 300,
                "db_cold_measured": 30,
            },
            "macro": {
                "segments_per_request": 3,
                "concurrency": [1, 4],
                "warmup_seconds": 120,
                "measured_seconds": 900,
            },
        },
        "quality_gates": {
            "precision": 1.0,
            "recall_lower_bound": TARGET_RECALL,
            "ann_p95_ratio_max": 0.70,
            "ann_p99_ratio_max": 1.0,
            "filter_first_p95_ratio_max": 0.85,
            "ann_peak_rss_ratio_max": 1.25,
            "required_index": HNSW_INDEX_NAME,
            "bootstrap_iterations": 1_000,
            "bootstrap_confidence": 0.95,
        },
    }


def enumerate_unique_candidate_counts(
    *,
    corpus_user_count: int,
    expected_member_count: float,
) -> tuple[int, ...]:
    values = {
        max(1, min(corpus_user_count, math.ceil(corpus_user_count * ratio)))
        for ratio in CANDIDATE_RATIO_ANCHORS
    }
    for minimum in MIN_ANN_CANDIDATE_GRID:
        for factor in K_SAFETY_FACTOR_GRID:
            raw = min(
                corpus_user_count,
                max(minimum, math.ceil(factor * expected_member_count)),
            )
            values.add(raw)
    return tuple(sorted(values))


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def wilson_lower_bound(
    *,
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval counts are invalid")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = proportion + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (center - margin) / denominator)


def bootstrap_percentile_ratio_upper_bound(
    numerator_ms: Sequence[float],
    denominator_ms: Sequence[float],
    *,
    percentile_quantile: float = 0.95,
    confidence: float = 0.95,
    iterations: int = 1_000,
    seed: int = 20260718,
) -> float:
    """Return a deterministic bootstrap upper bound for a latency ratio."""

    if not numerator_ms or not denominator_ms:
        raise ValueError("bootstrap latency samples must not be empty")
    if iterations <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap configuration is invalid")
    if any(value < 0 for value in (*numerator_ms, *denominator_ms)):
        raise ValueError("bootstrap latency samples must not be negative")
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(iterations):
        numerator = [rng.choice(numerator_ms) for _ in numerator_ms]
        denominator = [rng.choice(denominator_ms) for _ in denominator_ms]
        denominator_value = percentile(denominator, percentile_quantile)
        if denominator_value == 0:
            ratios.append(math.inf)
            continue
        ratios.append(
            percentile(numerator, percentile_quantile) / denominator_value
        )
    return percentile(ratios, confidence)


def build_current_runtime_report(
    summaries: Sequence[RunSummary],
) -> Mapping[str, Any]:
    cells = []
    for item in summaries:
        if item.plan != SearchPlan.CURRENT_RUNTIME:
            continue
        cells.append(
            {
                "scenario_id": item.scenario_id,
                "scenario_set": item.scenario_set.value,
                "candidate_type": item.candidate_type,
                "corpus_user_count": item.corpus_user_count,
                "cache_mode": item.cache_mode.value,
                "exclusion_ratio": item.exclusion_ratio,
                "run_count": item.run_count,
                "p50_ms": item.p50_ms,
                "p95_ms": item.p95_ms,
                "p99_ms": item.p99_ms,
                "max_ann_attempt_count": item.max_ann_attempt_count,
                "requested_k_histories": [
                    list(history) for history in item.requested_k_histories
                ],
                "max_audited_user_count": item.max_audited_user_count,
                "exact_fallback_rate": item.exact_fallback_rate,
                "stage_p95_ms": dict(item.stage_p95_ms),
            }
        )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "cell_count": len(cells),
        "cells": cells,
    }


def _summary_key(item: BenchmarkObservation) -> tuple[Any, ...]:
    return (
        item.scenario_id,
        item.scenario_set,
        item.candidate_type,
        item.corpus_user_count,
        item.hard_match_user_count,
        item.score_pass_sample_size,
        item.estimated_score_pass_rate,
        item.actual_score_pass_rate,
        item.plan,
        item.cache_mode,
        item.exclusion_ratio,
        item.requested_k,
        item.hnsw,
    )


def _summary_sort_key(item: RunSummary) -> tuple[Any, ...]:
    return (
        item.corpus_user_count,
        item.scenario_set.value,
        item.scenario_id,
        item.score_pass_sample_size,
        item.plan.value,
        item.requested_k or 0,
        item.hnsw or HnswSettings(1, "strict_order", 1),
    )


def _highest_available_phase(
    observations: Sequence[BenchmarkObservation],
) -> BenchmarkPhase:
    measured = {item.phase for item in observations if item.measured}
    for phase in (
        BenchmarkPhase.CONFIRMATION,
        BenchmarkPhase.TUNING,
        BenchmarkPhase.SCREENING,
    ):
        if phase in measured:
            return phase
    raise ValueError("no measured benchmark phase is available")


def _validate_disjoint_scenario_holdout(
    observations: Sequence[BenchmarkObservation],
    *,
    config: SynthesisConfig,
) -> None:
    selected = [
        item
        for item in observations
        if item.measured and item.plan == SearchPlan.EXACT_ALL
    ]
    tuning = {
        (item.candidate_type, item.scenario_id)
        for item in selected
        if item.scenario_set == ScenarioSet.TUNING
    }
    confirmation = {
        (item.candidate_type, item.scenario_id)
        for item in selected
        if item.scenario_set == ScenarioSet.CONFIRMATION
    }
    if {scenario_id for _, scenario_id in tuning} & {
        scenario_id for _, scenario_id in confirmation
    }:
        raise ValueError("tuning and confirmation scenario IDs must be disjoint")
    for candidate_type in config.required_candidate_types:
        if not any(value[0] == candidate_type for value in tuning):
            raise ValueError(
                f"tuning scenario coverage missing for {candidate_type}"
            )
        if not any(value[0] == candidate_type for value in confirmation):
            raise ValueError(
                f"confirmation scenario coverage missing for {candidate_type}"
            )
    tuning_buckets = {
        _scenario_holdout_bucket(item, config)
        for item in selected
        if item.scenario_set == ScenarioSet.TUNING
    }
    for item in selected:
        if item.scenario_set != ScenarioSet.CONFIRMATION:
            continue
        if _scenario_holdout_bucket(item, config) not in tuning_buckets:
            raise ValueError(
                "tuning scenario bucket coverage missing for confirmation cell: "
                f"{item.scenario_id}@{item.corpus_user_count}"
            )


def _scenario_holdout_bucket(
    item: BenchmarkObservation,
    config: SynthesisConfig,
) -> tuple[Any, ...]:
    return (
        item.candidate_type,
        item.corpus_user_count,
        _ratio_band(item.hard_match_ratio, config.hard_match_ratio_bands),
        _ratio_band(
            item.expected_member_ratio,
            config.expected_member_ratio_bands,
        ),
    )


def _select_score_pass_sample_size(
    summaries: Sequence[RunSummary],
    config: SynthesisConfig,
) -> int:
    sizes = sorted({item.score_pass_sample_size for item in summaries})
    reference_size = max(sizes)
    exact_lookup = {
        (
            item.scenario_id,
            item.corpus_user_count,
            item.score_pass_sample_size,
            item.plan,
        ): item
        for item in summaries
        if item.plan in (SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT)
    }
    reference = {
        (item.scenario_id, item.corpus_user_count): item
        for item in summaries
        if item.score_pass_sample_size == reference_size
        and item.plan == SearchPlan.EXACT_ALL
    }
    for size in sizes:
        candidates = [
            item
            for item in summaries
            if item.score_pass_sample_size == size
            and item.plan == SearchPlan.EXACT_ALL
        ]
        if not candidates:
            continue
        errors: list[float] = []
        stable = True
        for item in candidates:
            baseline = reference.get((item.scenario_id, item.corpus_user_count))
            if baseline is None:
                stable = False
                break
            if item.p95_ms > baseline.p95_ms * config.score_pass_p95_ratio_max:
                stable = False
                break
            if _selected_exact_plan_for_sample(
                item,
                lookup=exact_lookup,
                config=config,
            ) != _selected_exact_plan_for_sample(
                baseline,
                lookup=exact_lookup,
                config=config,
            ):
                stable = False
                break
            if _k_bucket_signature(item) != _k_bucket_signature(baseline):
                stable = False
                break
            denominator = max(item.actual_score_pass_rate, 1 / item.corpus_user_count)
            errors.append(
                abs(item.estimated_score_pass_rate - item.actual_score_pass_rate)
                / denominator
            )
            if _ratio_band(
                item.expected_member_ratio,
                config.expected_member_ratio_bands,
            ) != _ratio_band(
                baseline.expected_member_ratio,
                config.expected_member_ratio_bands,
            ):
                stable = False
                break
        if stable and errors and percentile(errors, 0.95) <= (
            config.score_pass_relative_error_p95_max
        ):
            return size
    return reference_size


def _selected_exact_plan_for_sample(
    exact_all: RunSummary,
    *,
    lookup: Mapping[tuple[str, int, int, SearchPlan], RunSummary],
    config: SynthesisConfig,
) -> SearchPlan:
    filtered = lookup.get(
        (
            exact_all.scenario_id,
            exact_all.corpus_user_count,
            exact_all.score_pass_sample_size,
            SearchPlan.FILTER_FIRST_EXACT,
        )
    )
    if filtered is not None and _filter_first_passes(filtered, exact_all, config):
        return SearchPlan.FILTER_FIRST_EXACT
    return SearchPlan.EXACT_ALL


def _k_bucket_signature(summary: RunSummary) -> tuple[int | None, ...]:
    signature: list[int | None] = []
    for minimum in MIN_ANN_CANDIDATE_GRID:
        for factor in K_SAFETY_FACTOR_GRID:
            for cap in MAX_CORPUS_FRACTION_GRID:
                proposed = KPolicyParameters(minimum, factor, cap).proposed_k(
                    corpus_user_count=summary.corpus_user_count,
                    expected_member_count=summary.expected_member_count,
                )
                if proposed is None:
                    signature.append(None)
                    continue
                ratio = proposed / summary.corpus_user_count
                signature.append(
                    next(
                        (
                            index
                            for index, upper in enumerate(
                                (*CANDIDATE_RATIO_ANCHORS, 1.0)
                            )
                            if ratio <= upper
                        ),
                        len(CANDIDATE_RATIO_ANCHORS),
                    )
                )
    return tuple(signature)


def _p95_ratio_upper_bound(
    numerator: RunSummary,
    denominator: RunSummary,
    config: SynthesisConfig,
) -> float:
    identity = "|".join(
        (
            numerator.scenario_id,
            str(numerator.corpus_user_count),
            numerator.plan.value,
            denominator.plan.value,
            str(numerator.requested_k or 0),
            str(numerator.hnsw or ""),
        )
    )
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    return bootstrap_percentile_ratio_upper_bound(
        numerator.duration_samples_ms,
        denominator.duration_samples_ms,
        iterations=config.bootstrap_iterations,
        confidence=config.bootstrap_confidence,
        seed=seed,
    )


def _filter_first_passes(
    filtered: RunSummary,
    exact: RunSummary,
    config: SynthesisConfig,
) -> bool:
    if min(filtered.run_count, exact.run_count) < config.minimum_policy_run_count:
        return False
    if (
        filtered.final_user_count != exact.final_user_count
        or filtered.intersection_count != exact.exact_positive_count
        or filtered.precision != 1.0
        or filtered.recall != 1.0
    ):
        return False
    if filtered.p95_ms > exact.p95_ms * config.filter_first_p95_ratio_max:
        return False
    if (
        _p95_ratio_upper_bound(filtered, exact, config)
        > config.filter_first_p95_ratio_max
    ):
        return False
    return filtered.p99_ms <= exact.p99_ms


def _build_points(
    summaries: Sequence[RunSummary],
    *,
    reference_sample_size: int,
    config: SynthesisConfig,
) -> list[_Point]:
    seen: set[tuple[str, int]] = set()
    points: list[_Point] = []
    for item in summaries:
        if (
            item.plan != SearchPlan.EXACT_ALL
            or item.score_pass_sample_size != reference_sample_size
        ):
            continue
        identity = (item.scenario_id, item.corpus_user_count)
        if identity in seen:
            continue
        seen.add(identity)
        hard_band = _ratio_band(item.hard_match_ratio, config.hard_match_ratio_bands)
        expected_band = _ratio_band(
            item.expected_member_ratio,
            config.expected_member_ratio_bands,
        )
        if hard_band is None or expected_band is None:
            continue
        points.append(
            _Point(
                scenario_id=item.scenario_id,
                candidate_type=item.candidate_type,
                corpus_user_count=item.corpus_user_count,
                hard_match_user_count=item.hard_match_user_count,
                reference_score_pass_rate=item.estimated_score_pass_rate,
                hard_band=hard_band,
                expected_band=expected_band,
            )
        )
    return points


def _summary_lookup(
    summaries: Sequence[RunSummary],
) -> Mapping[tuple[Any, ...], RunSummary]:
    lookup: dict[tuple[Any, ...], RunSummary] = {}
    for item in summaries:
        key = (
            item.scenario_id,
            item.corpus_user_count,
            item.score_pass_sample_size,
            item.plan,
            item.requested_k,
            item.hnsw,
        )
        if key in lookup:
            raise ValueError("duplicate summarized benchmark cell")
        lookup[key] = item
    return lookup


def _support_summary_lookup(
    summaries: Sequence[RunSummary],
) -> Mapping[tuple[Any, ...], RunSummary]:
    lookup: dict[tuple[Any, ...], RunSummary] = {}
    for item in summaries:
        key = (
            item.scenario_id,
            item.corpus_user_count,
            item.score_pass_sample_size,
            item.plan,
            item.requested_k,
            item.hnsw,
            item.cache_mode,
            item.exclusion_ratio,
        )
        if key in lookup:
            raise ValueError("duplicate supporting benchmark cell")
        lookup[key] = item
    return lookup


def _points_in_bucket(
    points: Sequence[_Point],
    *,
    corpus_user_count: int,
    hard_band: tuple[float, float],
    expected_band: tuple[float, float],
) -> list[_Point]:
    return [
        point
        for point in points
        if point.corpus_user_count == corpus_user_count
        and point.hard_band == hard_band
        and point.expected_band == expected_band
    ]


def _has_candidate_type_coverage(
    points: Sequence[_Point],
    required: Sequence[str],
) -> bool:
    return set(required).issubset({point.candidate_type for point in points})


def _best_ann_candidate(
    *,
    points: Sequence[_Point],
    min_users: int,
    max_users: int,
    hard_band: tuple[float, float],
    expected_band: tuple[float, float],
    sample_size: int,
    lookup: Mapping[tuple[Any, ...], RunSummary],
    support_lookup: Mapping[tuple[Any, ...], RunSummary],
    config: SynthesisConfig,
) -> _RuleCandidate | None:
    hnsw_settings = sorted(
        {
            summary.hnsw
            for summary in lookup.values()
            if summary.plan == SearchPlan.ANN_FIRST and summary.hnsw is not None
        }
    )
    candidates: list[_RuleCandidate] = []
    for minimum in MIN_ANN_CANDIDATE_GRID:
        for factor in K_SAFETY_FACTOR_GRID:
            for cap in MAX_CORPUS_FRACTION_GRID:
                k_policy = KPolicyParameters(minimum, factor, cap)
                for hnsw in hnsw_settings:
                    ratios: list[float] = []
                    recalls: list[float] = []
                    recall_lower_bounds: list[float] = []
                    passed = True
                    for point in points:
                        exact = _fastest_exact_summary(
                            point,
                            sample_size=sample_size,
                            lookup=lookup,
                            config=config,
                        )
                        sample_exact = lookup.get(
                            (
                                point.scenario_id,
                                point.corpus_user_count,
                                sample_size,
                                SearchPlan.EXACT_ALL,
                                None,
                                None,
                            )
                        )
                        if exact is None or sample_exact is None:
                            passed = False
                            break
                        requested_k = k_policy.proposed_k(
                            corpus_user_count=point.corpus_user_count,
                            expected_member_count=sample_exact.expected_member_count,
                        )
                        if requested_k is None:
                            passed = False
                            break
                        ann = lookup.get(
                            (
                                point.scenario_id,
                                point.corpus_user_count,
                                sample_size,
                                SearchPlan.ANN_FIRST,
                                requested_k,
                                hnsw,
                            )
                        )
                        if ann is None or not _ann_passes(ann, exact, config):
                            passed = False
                            break
                        current = lookup.get(
                            (
                                point.scenario_id,
                                point.corpus_user_count,
                                sample_size,
                                SearchPlan.CURRENT_RUNTIME,
                                None,
                                None,
                            )
                        )
                        if current is None or ann.p99_ms > current.p99_ms:
                            passed = False
                            break
                        if not _supporting_ann_passes(
                            point,
                            sample_size=sample_size,
                            requested_k=requested_k,
                            hnsw=hnsw,
                            lookup=support_lookup,
                            config=config,
                        ):
                            passed = False
                            break
                        ratios.append(ann.p95_ms / exact.p95_ms)
                        recalls.append(ann.recall)
                        recall_lower_bounds.append(ann.recall_lower_bound)
                    if passed and ratios:
                        candidates.append(
                            _RuleCandidate(
                                min_users=min_users,
                                max_users=max_users,
                                hard_band=hard_band,
                                expected_band=expected_band,
                                plan=SearchPlan.ANN_FIRST,
                                sample_size=sample_size,
                                k_policy=k_policy,
                                hnsw=hnsw,
                                points=tuple(points),
                                worst_p95_ratio=max(ratios),
                                validated_recall=min(recalls),
                                validated_recall_lower_bound=min(
                                    recall_lower_bounds
                                ),
                            )
                        )
    if not candidates:
        return None
    return min(candidates, key=_rule_candidate_sort_key)


def _ann_passes(
    ann: RunSummary,
    exact: RunSummary,
    config: SynthesisConfig,
    *,
    require_plan_diagnostic: bool = True,
    minimum_run_count: int | None = None,
    require_p99: bool = True,
) -> bool:
    required_runs = minimum_run_count or config.minimum_policy_run_count
    if min(ann.run_count, exact.run_count) < required_runs:
        return False
    if ann.exact_positive_count < config.minimum_exact_positives:
        return False
    if ann.precision != 1.0:
        return False
    if ann.recall < config.target_recall:
        return False
    if ann.recall_lower_bound < config.target_recall:
        return False
    if ann.p95_ms > exact.p95_ms * config.ann_p95_ratio_max:
        return False
    if _p95_ratio_upper_bound(ann, exact, config) > config.ann_p95_ratio_max:
        return False
    if require_p99 and ann.p99_ms > exact.p99_ms * config.ann_p99_ratio_max:
        return False
    if ann.oom:
        return False
    if require_plan_diagnostic and (not ann.index_used or ann.temp_spill):
        return False
    if ann.peak_rss_bytes is None or exact.peak_rss_bytes is None:
        return False
    return ann.peak_rss_bytes <= (
        exact.peak_rss_bytes * config.ann_peak_rss_ratio_max
    )


def _supporting_ann_passes(
    point: _Point,
    *,
    sample_size: int,
    requested_k: int,
    hnsw: HnswSettings,
    lookup: Mapping[tuple[Any, ...], RunSummary],
    config: SynthesisConfig,
) -> bool:
    modes: list[tuple[CacheMode, float]] = []
    if config.require_db_cold_confirmation:
        modes.append((CacheMode.DB_COLD, 0.0))
    if config.required_exclusion_ratio is not None:
        modes.append((CacheMode.WARM, config.required_exclusion_ratio))
    for cache_mode, exclusion_ratio in modes:
        exact = _fastest_supporting_exact_summary(
            point,
            sample_size=sample_size,
            cache_mode=cache_mode,
            exclusion_ratio=exclusion_ratio,
            lookup=lookup,
            config=config,
        )
        ann = lookup.get(
            (
                point.scenario_id,
                point.corpus_user_count,
                sample_size,
                SearchPlan.ANN_FIRST,
                requested_k,
                hnsw,
                cache_mode,
                exclusion_ratio,
            )
        )
        if exact is None or ann is None or not _ann_passes(
            ann,
            exact,
            config,
            require_plan_diagnostic=cache_mode != CacheMode.DB_COLD,
            minimum_run_count=(
                config.minimum_cold_run_count
                if cache_mode == CacheMode.DB_COLD
                else config.minimum_policy_run_count
            ),
            require_p99=cache_mode != CacheMode.DB_COLD,
        ):
            return False
    return True


def _fastest_supporting_exact_summary(
    point: _Point,
    *,
    sample_size: int,
    cache_mode: CacheMode,
    exclusion_ratio: float,
    lookup: Mapping[tuple[Any, ...], RunSummary],
    config: SynthesisConfig,
) -> RunSummary | None:
    base = (
        point.scenario_id,
        point.corpus_user_count,
        sample_size,
    )
    exact = lookup.get(
        (*base, SearchPlan.EXACT_ALL, None, None, cache_mode, exclusion_ratio)
    )
    filtered = lookup.get(
        (
            *base,
            SearchPlan.FILTER_FIRST_EXACT,
            None,
            None,
            cache_mode,
            exclusion_ratio,
        )
    )
    if exact is None:
        return None
    if filtered is not None and _filter_first_passes(filtered, exact, config):
        return filtered
    return exact


def _fastest_exact_summary(
    point: _Point,
    *,
    sample_size: int,
    lookup: Mapping[tuple[Any, ...], RunSummary],
    config: SynthesisConfig,
) -> RunSummary | None:
    exact = lookup.get(
        (
            point.scenario_id,
            point.corpus_user_count,
            sample_size,
            SearchPlan.EXACT_ALL,
            None,
            None,
        )
    )
    filtered = lookup.get(
        (
            point.scenario_id,
            point.corpus_user_count,
            sample_size,
            SearchPlan.FILTER_FIRST_EXACT,
            None,
            None,
        )
    )
    if exact is None:
        return None
    if filtered is None:
        return exact
    if _filter_first_passes(filtered, exact, config):
        return filtered
    return exact


def _exact_plan_for_points(
    points: Sequence[_Point],
    *,
    sample_size: int,
    lookup: Mapping[tuple[Any, ...], RunSummary],
    config: SynthesisConfig,
) -> SearchPlan:
    for point in points:
        exact = lookup.get(
            (
                point.scenario_id,
                point.corpus_user_count,
                sample_size,
                SearchPlan.EXACT_ALL,
                None,
                None,
            )
        )
        filtered = lookup.get(
            (
                point.scenario_id,
                point.corpus_user_count,
                sample_size,
                SearchPlan.FILTER_FIRST_EXACT,
                None,
                None,
            )
        )
        if (
            exact is None
            or filtered is None
            or not _filter_first_passes(filtered, exact, config)
        ):
            return SearchPlan.EXACT_ALL
    return SearchPlan.FILTER_FIRST_EXACT


def _consolidate_global_hnsw(
    candidates: Sequence[_RuleCandidate],
    *,
    lookup: Mapping[tuple[Any, ...], RunSummary],
    support_lookup: Mapping[tuple[Any, ...], RunSummary],
    config: SynthesisConfig,
) -> list[_RuleCandidate]:
    ann_candidates = [
        candidate for candidate in candidates if candidate.plan == SearchPlan.ANN_FIRST
    ]
    if len(ann_candidates) < 2:
        return list(candidates)
    settings = sorted(
        {
            summary.hnsw
            for summary in lookup.values()
            if summary.plan == SearchPlan.ANN_FIRST and summary.hnsw is not None
        }
    )
    eligible: list[tuple[float, HnswSettings]] = []
    for hnsw in settings:
        worst_ratio = 0.0
        valid = True
        for candidate in ann_candidates:
            assert candidate.k_policy is not None
            for point in candidate.points:
                sample_exact = lookup.get(
                    (
                        point.scenario_id,
                        point.corpus_user_count,
                        candidate.sample_size,
                        SearchPlan.EXACT_ALL,
                        None,
                        None,
                    )
                )
                fastest_exact = _fastest_exact_summary(
                    point,
                    sample_size=candidate.sample_size,
                    lookup=lookup,
                    config=config,
                )
                if sample_exact is None or fastest_exact is None:
                    valid = False
                    break
                requested_k = candidate.k_policy.proposed_k(
                    corpus_user_count=point.corpus_user_count,
                    expected_member_count=sample_exact.expected_member_count,
                )
                ann = lookup.get(
                    (
                        point.scenario_id,
                        point.corpus_user_count,
                        candidate.sample_size,
                        SearchPlan.ANN_FIRST,
                        requested_k,
                        hnsw,
                    )
                )
                if ann is None or not _ann_passes(ann, fastest_exact, config):
                    valid = False
                    break
                if not _supporting_ann_passes(
                    point,
                    sample_size=candidate.sample_size,
                    requested_k=requested_k,
                    hnsw=hnsw,
                    lookup=support_lookup,
                    config=config,
                ):
                    valid = False
                    break
                ratio = ann.p95_ms / fastest_exact.p95_ms
                if ratio > (
                    candidate.worst_p95_ratio
                    * config.global_hnsw_local_p95_ratio_max
                ):
                    valid = False
                    break
                worst_ratio = max(worst_ratio, ratio)
            if not valid:
                break
        if valid:
            eligible.append((worst_ratio, hnsw))
    if not eligible:
        return list(candidates)
    _, selected = min(
        eligible,
        key=lambda item: (
            item[0],
            item[1].ef_search,
            item[1].max_scan_tuples,
            item[1].iterative_scan,
        ),
    )
    return [
        replace(candidate, hnsw=selected)
        if candidate.plan == SearchPlan.ANN_FIRST
        else candidate
        for candidate in candidates
    ]


def _rule_candidate_sort_key(candidate: _RuleCandidate) -> tuple[Any, ...]:
    assert candidate.k_policy is not None and candidate.hnsw is not None
    return (
        candidate.k_policy.min_candidates,
        candidate.k_policy.k_safety_factor,
        candidate.k_policy.max_corpus_fraction,
        candidate.worst_p95_ratio,
        candidate.hnsw.ef_search,
        candidate.hnsw.max_scan_tuples,
        candidate.hnsw.iterative_scan,
    )


def _phase_cell_payload(
    ann: RunSummary,
    exact: RunSummary,
) -> Mapping[str, Any]:
    assert ann.requested_k is not None and ann.hnsw is not None
    return {
        "scenario_id": ann.scenario_id,
        "candidate_type": ann.candidate_type,
        "corpus_user_count": ann.corpus_user_count,
        "score_pass_sample_size": ann.score_pass_sample_size,
        "requested_k": ann.requested_k,
        "hnsw": ann.hnsw.to_dict(),
        "recall": ann.recall,
        "recall_lower_bound": ann.recall_lower_bound,
        "ann_p95_ms": ann.p95_ms,
        "exact_plan": exact.plan.value,
        "exact_p95_ms": exact.p95_ms,
        "p95_ratio": ann.p95_ms / exact.p95_ms,
    }


def _rule_payload(candidate: _RuleCandidate) -> Mapping[str, Any]:
    representative = candidate.points[0]
    payload: dict[str, Any] = {
        "min_users": candidate.min_users,
        "max_users": candidate.max_users,
        "min_hard_match_ratio": candidate.hard_band[0],
        "max_hard_match_ratio": candidate.hard_band[1],
        "min_expected_member_ratio": candidate.expected_band[0],
        "max_expected_member_ratio": candidate.expected_band[1],
        "plan": candidate.plan.value,
        "evidence": {
            "scenario_count": len(candidate.points),
            "candidate_types": sorted(
                {point.candidate_type for point in candidate.points}
            ),
            "corpus_anchors": sorted(
                {point.corpus_user_count for point in candidate.points}
            ),
            "worst_p95_ratio": candidate.worst_p95_ratio,
            "representative": {
                "scenario_id": representative.scenario_id,
                "corpus_user_count": representative.corpus_user_count,
                "hard_match_user_count": representative.hard_match_user_count,
                "estimated_score_pass_rate": (
                    representative.reference_score_pass_rate
                ),
            },
        },
    }
    if candidate.plan == SearchPlan.ANN_FIRST:
        assert candidate.k_policy is not None and candidate.hnsw is not None
        payload["ann"] = {
            **candidate.k_policy.to_dict(),
            **candidate.hnsw.to_dict(),
            "validated_recall": candidate.validated_recall,
            "validated_recall_lower_bound": (
                candidate.validated_recall_lower_bound
            ),
        }
    return payload


def _rule_matches(
    rule: Mapping[str, Any],
    *,
    corpus_user_count: int,
    hard_match_ratio: float,
    expected_member_ratio: float,
) -> bool:
    try:
        return (
            int(rule["min_users"]) <= corpus_user_count <= int(rule["max_users"])
            and float(rule["min_hard_match_ratio"]) <= hard_match_ratio
            < float(rule["max_hard_match_ratio"])
            and float(rule["min_expected_member_ratio"]) <= expected_member_ratio
            < float(rule["max_expected_member_ratio"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ratio_band(
    value: float,
    bands: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    return next((band for band in bands if band[0] <= value < band[1]), None)


def _fallback_record(
    min_users: int,
    max_users: int,
    hard_band: tuple[float, float],
    expected_band: tuple[float, float],
    reason: str,
) -> Mapping[str, Any]:
    return {
        "min_users": min_users,
        "max_users": max_users,
        "min_hard_match_ratio": hard_band[0],
        "max_hard_match_ratio": hard_band[1],
        "min_expected_member_ratio": expected_band[0],
        "max_expected_member_ratio": expected_band[1],
        "reason": reason,
        "plan": SearchPlan.EXACT_ALL.value,
    }


def _exact_fallback(reason: str) -> PolicyDecision:
    return PolicyDecision(
        plan=SearchPlan.EXACT_ALL,
        requested_k=None,
        hnsw=None,
        rule_index=None,
        fallback_reason=reason,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, summaries: Sequence[RunSummary]) -> None:
    rows = [item.to_dict() for item in summaries]
    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
            if key != "hnsw"
        }
        | {"ef_search", "iterative_scan", "max_scan_tuples"}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            hnsw = row.pop("hnsw", None)
            if isinstance(hnsw, Mapping):
                row.update(hnsw)
            writer.writerow(row)


def _render_report(result: PolicySynthesisResult) -> str:
    rules = result.policy.get("rules", [])
    ann_count = sum(
        1
        for rule in rules
        if isinstance(rule, Mapping) and rule.get("plan") == SearchPlan.ANN_FIRST.value
    )
    filter_count = sum(
        1
        for rule in rules
        if isinstance(rule, Mapping)
        and rule.get("plan") == SearchPlan.FILTER_FIRST_EXACT.value
    )
    return "\n".join(
        (
            "# Audience ANN search benchmark",
            "",
            f"- Policy status: `{result.policy['status']}`",
            f"- Validated max users: `{result.policy['validated_max_users']}`",
            f"- Score-pass sample size: `{result.policy['score_pass_sample_size']}`",
            f"- ANN rules: `{ann_count}`",
            f"- Filter-first Exact rules: `{filter_count}`",
            f"- Exact fallback buckets: `{len(result.fallbacks)}`",
            "",
            "Runtime remains unchanged until this candidate policy is separately approved.",
            "",
        )
    )
