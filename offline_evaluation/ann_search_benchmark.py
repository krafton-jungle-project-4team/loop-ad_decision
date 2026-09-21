"""Live database adapter for the offline ANN search experiment.

The adapter operates on one already prepared, disposable PostgreSQL cohort at
a time.  It never creates or alters Data Contract schema objects.  Corpus
loading and container restarts remain explicit environment preparation steps.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.analysis.audience_search import (
    AudienceSearchMethod,
    CandidateAudienceSearchService,
    SearchCandidate,
)
from app.analysis.behavior_vector_schema import CandidateBehaviorSpec
from app.analysis.audience_search_repository import (
    AudienceSearchContext,
    HardMatchAggregateRequest,
    PgClickHouseAudienceVectorSearchRepository,
    _hard_predicate_query,
    _vector_literal,
    hard_predicates_support_batch_aggregate,
)
from app.analysis.repositories import PsycopgPostgresExecutor
from app.audience_exclusions import POSTGRES_EXCLUSION_RELATION
from offline_evaluation.ann_search_experiment import (
    BenchmarkObservation,
    BenchmarkPhase,
    CacheMode,
    DEFAULT_CANDIDATE_TYPES,
    EF_SEARCH_GRID,
    EXPERIMENT_VERSION,
    HNSW_INDEX_NAME,
    HnswSettings,
    ITERATIVE_SCAN_GRID,
    MAX_SCAN_TUPLES_GRID,
    SCORE_PASS_SAMPLE_SIZE_GRID,
    SUPPORTED_EXPERIMENT_VERSIONS,
    ScenarioSet,
    SearchPlan,
    enumerate_unique_candidate_counts,
    select_policy,
    wilson_lower_bound,
)
from offline_evaluation.ann_search_macro import MacroMode, MacroObservation
from offline_evaluation.ann_search_scale_series import (
    COHORT_SIGNAL_RELATION,
    SCALE_EXPERIMENT_VERSION,
    ScaleCohortScope,
    cohort_bound_hard_match_query,
    cohort_bound_hard_match_users_query,
    cohort_bound_reference_sample_query,
    validate_cohort_bound_users,
)


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    scenario_id: str
    candidate_type: str
    query_vector: tuple[float, ...]
    score_threshold: float
    hard_predicate_keys: tuple[str, ...]
    predicate_parameters: Mapping[str, tuple[str, ...] | tuple[int, ...]]
    compiler_provenance: Mapping[str, str]
    scenario_set: ScenarioSet = ScenarioSet.CONFIRMATION

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.candidate_type:
            raise ValueError("scenario id and candidate type are required")
        if self.candidate_type not in DEFAULT_CANDIDATE_TYPES:
            raise ValueError("scenario candidate type is not production-supported")
        if len(self.query_vector) != 64:
            raise ValueError("ANN benchmark query vector must contain 64 values")
        if any(not math.isfinite(value) for value in self.query_vector):
            raise ValueError("ANN benchmark query vector must contain finite values")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("scenario score threshold must be in [0, 1]")
        required_provenance = {
            "manifest_hash",
            "calibration_version",
            "calibration_hash",
            "query_compiler_version",
            "query_compiler_hash",
            "template_id",
            "template_semantic_hash",
        }
        if required_provenance != set(self.compiler_provenance):
            raise ValueError("scenario compiler provenance is incomplete")
        if any(not value for value in self.compiler_provenance.values()):
            raise ValueError("scenario compiler provenance values are required")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkScenario":
        parameters = payload.get("predicate_parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("predicate_parameters must be an object")
        provenance = payload.get("compiler_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("compiler_provenance must be an object")
        return cls(
            scenario_id=str(payload["scenario_id"]),
            candidate_type=str(payload["candidate_type"]),
            query_vector=tuple(float(value) for value in payload["query_vector"]),
            score_threshold=float(payload["score_threshold"]),
            hard_predicate_keys=tuple(
                str(value) for value in payload["hard_predicate_keys"]
            ),
            predicate_parameters={
                str(key): tuple(value)
                for key, value in parameters.items()
            },
            compiler_provenance={
                str(key): str(value) for key, value in provenance.items()
            },
            scenario_set=ScenarioSet(
                str(payload.get("scenario_set", ScenarioSet.CONFIRMATION.value))
            ),
        )

    def candidate_behavior_spec(self, *, vector_version: str) -> CandidateBehaviorSpec:
        return CandidateBehaviorSpec(
            candidate_type=self.candidate_type,
            schema_version=vector_version,
            vector_version=vector_version,
            hard_predicate_keys=self.hard_predicate_keys,
            predicate_parameters=self.predicate_parameters,
            query_vector=self.query_vector,
            active_blocks=(),
            block_weights={},
            score_threshold=self.score_threshold,
            calibration_version=self.compiler_provenance["calibration_version"],
            manifest_hash=self.compiler_provenance["manifest_hash"],
            calibration_hash=self.compiler_provenance["calibration_hash"],
            query_compiler_version=self.compiler_provenance[
                "query_compiler_version"
            ],
            query_compiler_hash=self.compiler_provenance["query_compiler_hash"],
            template_id=self.compiler_provenance["template_id"],
            template_semantic_hash=self.compiler_provenance[
                "template_semantic_hash"
            ],
        )


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    project_id: str
    vector_version: str
    manifest_hash: str
    scenarios: tuple[BenchmarkScenario, ...]
    campaign_id: str | None = None
    promotion_id: str | None = None
    experiment_version: str = EXPERIMENT_VERSION

    @classmethod
    def load(cls, path: Path) -> "BenchmarkManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment_version = str(payload.get("experiment_version"))
        if experiment_version not in SUPPORTED_EXPERIMENT_VERSIONS:
            raise ValueError("benchmark manifest version is invalid")
        scenarios = tuple(
            BenchmarkScenario.from_dict(item) for item in payload["scenarios"]
        )
        if len({item.scenario_id for item in scenarios}) != len(scenarios):
            raise ValueError("benchmark scenario IDs must be unique")
        if not scenarios:
            raise ValueError("benchmark manifest requires at least one scenario")
        manifest_hash = str(payload["manifest_hash"])
        if any(
            scenario.compiler_provenance["manifest_hash"] != manifest_hash
            for scenario in scenarios
        ):
            raise ValueError("scenario compiler manifest hash mismatches")
        campaign_id = (
            str(payload["campaign_id"]) if payload.get("campaign_id") else None
        )
        promotion_id = (
            str(payload["promotion_id"]) if payload.get("promotion_id") else None
        )
        if (campaign_id is None) != (promotion_id is None):
            raise ValueError("campaign_id and promotion_id must be provided together")
        return cls(
            project_id=str(payload["project_id"]),
            vector_version=str(payload["vector_version"]),
            manifest_hash=manifest_hash,
            scenarios=scenarios,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            experiment_version=experiment_version,
        )

    def require_scenario(self, scenario_id: str) -> BenchmarkScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise ValueError(f"unknown benchmark scenario: {scenario_id}")


@dataclass(frozen=True, slots=True)
class CellRunConfig:
    phase: BenchmarkPhase
    cache_mode: CacheMode
    warmups: int
    repetitions: int
    sample_sizes: tuple[int, ...] = SCORE_PASS_SAMPLE_SIZE_GRID
    hnsw_settings: tuple[HnswSettings, ...] = (
        HnswSettings(100, "strict_order", 20_000),
    )
    requested_k_values: tuple[int, ...] = ()
    plans: tuple[SearchPlan, ...] = tuple(SearchPlan)
    exclusion_ratio: float = 0.0
    random_seed: int = 20260718

    def __post_init__(self) -> None:
        if self.warmups < 0 or self.repetitions <= 0:
            raise ValueError("warmups/repetitions are invalid")
        if self.cache_mode == CacheMode.DB_COLD and (
            self.warmups != 0 or self.repetitions != 1
        ):
            raise ValueError(
                "DB-cold mode requires one run after an external DB restart"
            )
        if not self.sample_sizes or any(value <= 0 for value in self.sample_sizes):
            raise ValueError("score-pass sample sizes must be positive")
        if not self.plans or len(set(self.plans)) != len(self.plans):
            raise ValueError("benchmark plans must be non-empty and unique")
        if not 0 <= self.exclusion_ratio <= 1:
            raise ValueError("exclusion ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class _ExecutionCell:
    sample_size: int
    plan: SearchPlan
    requested_k: int | None = None
    hnsw: HnswSettings | None = None


@dataclass(frozen=True, slots=True)
class _PlanDiagnostic:
    index_name: str | None
    temp_spill: bool
    explain: Any


@dataclass(frozen=True, slots=True)
class _RuntimeWork:
    ann_attempt_count: int = 0
    requested_k_history: tuple[int, ...] = ()
    audited_user_count: int = 0
    used_exact_fallback: bool = False
    stage_durations_ms: Mapping[str, float] | None = None


class _RuntimeTraceRepository:
    """Transparent repository proxy that records current-runtime work."""

    def __init__(self, repository: Any) -> None:
        self._repository = repository
        self.requested_k_history: list[int] = []
        self.audited_user_count = 0
        self.stage_durations_ms: dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._repository, name)
        if not callable(value):
            return value

        def traced(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter_ns()
            result = value(*args, **kwargs)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            self.stage_durations_ms[name] = (
                self.stage_durations_ms.get(name, 0.0) + elapsed
            )
            if name in {"materialize_ann_candidates", "ann_search"}:
                self.requested_k_history.append(int(kwargs["limit"]))
            if name in {"audit_materialized_nonretrieved", "audit_nonretrieved"}:
                self.audited_user_count += int(result[0])
            return result

        return traced

    def to_work(self, method: AudienceSearchMethod) -> _RuntimeWork:
        return _RuntimeWork(
            ann_attempt_count=len(self.requested_k_history),
            requested_k_history=tuple(self.requested_k_history),
            audited_user_count=self.audited_user_count,
            used_exact_fallback=method == AudienceSearchMethod.EXACT_FALLBACK,
            stage_durations_ms=dict(self.stage_durations_ms),
        )


class LiveAnnSearchBenchmark:
    def __init__(
        self,
        *,
        postgres_connection: Any,
        clickhouse: Any,
        scale_cohort_scope: ScaleCohortScope | None = None,
    ) -> None:
        self._connection = postgres_connection
        self._postgres = PsycopgPostgresExecutor(postgres_connection)
        self._clickhouse = clickhouse
        self._repository = PgClickHouseAudienceVectorSearchRepository(
            postgres=self._postgres,
            clickhouse=clickhouse,
            predicate_chunk_size=1_000,
        )
        self._scale_cohort_scope = scale_cohort_scope
        self._cohort_user_ids_by_generation: dict[str, tuple[str, ...]] = {}
        self._cohort_hard_counts: dict[tuple[str, str], int] = {}

    def _require_manifest_mode(self, manifest: BenchmarkManifest) -> None:
        if self._scale_cohort_scope is None:
            if manifest.experiment_version == SCALE_EXPERIMENT_VERSION:
                raise RuntimeError(
                    "benchmark v2 requires an explicit scale cohort scope"
                )
            return
        if manifest.experiment_version != SCALE_EXPERIMENT_VERSION:
            raise RuntimeError("scale cohort scope requires benchmark v2 manifest")
        if manifest.campaign_id is not None or manifest.promotion_id is not None:
            raise RuntimeError(
                "Goal 1 scale-series does not run exclusion confirmation"
            )

    @property
    def _experiment_version(self) -> str:
        return (
            SCALE_EXPERIMENT_VERSION
            if self._scale_cohort_scope is not None
            else EXPERIMENT_VERSION
        )

    def preflight(self, manifest: BenchmarkManifest) -> Mapping[str, Any]:
        self._require_manifest_mode(manifest)
        with self._connection.transaction():
            context = self._repository.get_context(
                project_id=manifest.project_id,
                vector_version=manifest.vector_version,
                campaign_id=manifest.campaign_id,
                promotion_id=manifest.promotion_id,
            )
            postgres = self._postgres.fetchone(
                """
                SELECT
                    current_setting('server_version') AS postgres_version,
                    extversion AS pgvector_version
                FROM pg_extension
                WHERE extname = 'vector'
                """
            )
            index = self._postgres.fetchone(
                """
                SELECT indexrelid::regclass::text AS index_name,
                       indisvalid, indisready,
                       pg_relation_size(indexrelid) AS index_size_bytes
                FROM pg_index
                WHERE indexrelid = %s::regclass
                """,
                (HNSW_INDEX_NAME,),
            )
        if context.manifest_hash != manifest.manifest_hash:
            raise RuntimeError("active vector generation manifest hash mismatches")
        if index is None or not bool(index["indisvalid"]) or not bool(index["indisready"]):
            raise RuntimeError("HNSW index must be valid and ready")
        clickhouse_version = self._clickhouse.query("SELECT version() AS version")
        clickhouse_rows = _named_rows(clickhouse_version)
        cohort_membership_count: int | None = None
        cohort_signal_count: int | None = None
        if self._scale_cohort_scope is not None:
            membership_result = self._clickhouse.query(
                """
                SELECT uniqExact(user_id) AS user_count
                FROM ann_benchmark_scale_membership
                WHERE scale_series_id = {scale_series_id:String}
                  AND cohort_rank <= {cohort_size:UInt64}
                """,
                parameters=self._scale_cohort_scope.clickhouse_parameters(),
            )
            membership_rows = _named_rows(membership_result)
            cohort_membership_count = (
                int(membership_rows[0]["user_count"]) if membership_rows else 0
            )
            if cohort_membership_count != self._scale_cohort_scope.cohort_size:
                raise RuntimeError("ClickHouse scale membership is incomplete")
            if context.corpus_user_count != self._scale_cohort_scope.cohort_size:
                raise RuntimeError(
                    "PostgreSQL cohort differs from ClickHouse scale membership"
                )
            signal_result = self._clickhouse.query(
                f"""
                SELECT uniqExact(user_id) AS user_count
                FROM {COHORT_SIGNAL_RELATION}
                WHERE scale_series_id = {{scale_series_id:String}}
                  AND cohort_rank <= {{cohort_size:UInt64}}
                """,
                parameters=self._scale_cohort_scope.clickhouse_parameters(),
            )
            signal_rows = _named_rows(signal_result)
            cohort_signal_count = (
                int(signal_rows[0]["user_count"]) if signal_rows else 0
            )
            if cohort_signal_count != self._scale_cohort_scope.cohort_size:
                raise RuntimeError("ClickHouse frozen signal relation is incomplete")
        return {
            "project_id": manifest.project_id,
            "vector_version": manifest.vector_version,
            "manifest_hash": context.manifest_hash,
            "vector_generation_id": context.vector_generation_id,
            "corpus_user_count": context.corpus_user_count,
            "window_start": context.window_start.isoformat(),
            "source_cutoff": context.source_cutoff.isoformat(),
            "source_revision_cutoff": context.source_revision_cutoff.isoformat(),
            "postgres_version": postgres["postgres_version"] if postgres else None,
            "pgvector_version": postgres["pgvector_version"] if postgres else None,
            "clickhouse_version": (
                str(clickhouse_rows[0]["version"]) if clickhouse_rows else None
            ),
            "hnsw_index": dict(index),
            "excluded_user_count": (
                context.exclusion_context.excluded_user_count
                if context.exclusion_context is not None
                else 0
            ),
            "scale_series_id": (
                self._scale_cohort_scope.scale_series_id
                if self._scale_cohort_scope is not None
                else None
            ),
            "cohort_membership_count": cohort_membership_count,
            "cohort_signal_count": cohort_signal_count,
        }

    def run_macro_request(
        self,
        *,
        manifest: BenchmarkManifest,
        scenarios: Sequence[BenchmarkScenario],
        mode: MacroMode,
        sample_size: int,
        concurrency: int,
        measured: bool,
        policy: Mapping[str, Any] | None = None,
    ) -> MacroObservation:
        if self._scale_cohort_scope is not None:
            raise RuntimeError(
                "macro benchmark is outside Goal 1 scale-series scope"
            )
        if len(scenarios) != 3 or len({item.scenario_id for item in scenarios}) != 3:
            raise ValueError("macro benchmark requires three distinct scenarios")
        if mode == MacroMode.CANDIDATE_POLICY and policy is None:
            raise ValueError("candidate macro benchmark requires a policy")
        context = self._load_context(manifest)
        sampler = _BackendRssSampler.from_connection(self._connection)
        selected_plans: list[str] = []
        requested_values: list[int | None] = []
        final_counts: list[int] = []
        temp_bytes_before = self._postgres_temp_bytes()
        with sampler:
            started = time.perf_counter_ns()
            hard_counts = self._macro_hard_counts(
                manifest=manifest,
                scenarios=scenarios,
                context=context,
            )
            for scenario in scenarios:
                with self._connection.transaction():
                    pass_rate = self._repository.estimate_score_pass_rate(
                        project_id=manifest.project_id,
                        vector_generation_id=context.vector_generation_id,
                        vector_version=manifest.vector_version,
                        source_revision_cutoff=context.source_revision_cutoff,
                        source_cutoff=context.source_cutoff,
                        window_start=context.window_start,
                        query_vector=scenario.query_vector,
                        score_threshold=scenario.score_threshold,
                        hard_predicate_keys=scenario.hard_predicate_keys,
                        predicate_parameters=scenario.predicate_parameters,
                        sample_size=sample_size,
                        exclusion_context=context.exclusion_context,
                    )
                    hard_count = hard_counts[scenario.scenario_id]
                    if mode == MacroMode.CURRENT_RUNTIME:
                        result = CandidateAudienceSearchService(
                            self._repository
                        ).search(
                            project_id=manifest.project_id,
                            vector_generation_id=context.vector_generation_id,
                            source_cutoff=context.source_cutoff,
                            spec=scenario.candidate_behavior_spec(
                                vector_version=manifest.vector_version
                            ),
                            corpus_user_count=context.corpus_user_count,
                            hard_match_user_count=hard_count,
                            estimated_score_pass_rate=pass_rate,
                        )
                        members = list(result.members)
                        if not members and result.members_relation:
                            members = self._materialized_members(
                                result.members_relation
                            )
                        selected_plans.append(result.method.value)
                        requested_values.append(result.requested_k or None)
                    else:
                        assert policy is not None
                        decision = select_policy(
                            policy,
                            corpus_user_count=context.corpus_user_count,
                            hard_match_user_count=hard_count,
                            estimated_score_pass_rate=pass_rate,
                            vector_version=manifest.vector_version,
                            manifest_hash=manifest.manifest_hash,
                        )
                        selected_plans.append(decision.plan.value)
                        requested_values.append(decision.requested_k)
                        if decision.plan == SearchPlan.FILTER_FIRST_EXACT:
                            members = self._filter_first_exact(
                                manifest=manifest,
                                scenario=scenario,
                                context=context,
                            )
                        elif decision.plan == SearchPlan.ANN_FIRST:
                            assert decision.requested_k is not None
                            assert decision.hnsw is not None
                            members = self._ann_first(
                                manifest=manifest,
                                scenario=scenario,
                                context=context,
                                requested_k=decision.requested_k,
                                hnsw=decision.hnsw,
                            )
                        else:
                            members = self._repository.exact_search(
                                project_id=manifest.project_id,
                                vector_generation_id=context.vector_generation_id,
                                vector_version=manifest.vector_version,
                                source_cutoff=context.source_cutoff,
                                query_vector=scenario.query_vector,
                                score_threshold=scenario.score_threshold,
                                hard_predicate_keys=scenario.hard_predicate_keys,
                                predicate_parameters=scenario.predicate_parameters,
                            )
                    final_counts.append(len({item.user_id for item in members}))
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
        return MacroObservation(
            experiment_version=EXPERIMENT_VERSION,
            mode=mode,
            concurrency=concurrency,
            measured=measured,
            scenario_ids=tuple(item.scenario_id for item in scenarios),  # type: ignore[arg-type]
            duration_ms=duration_ms,
            selected_plans=tuple(selected_plans),  # type: ignore[arg-type]
            requested_k=tuple(requested_values),  # type: ignore[arg-type]
            final_user_counts=tuple(final_counts),  # type: ignore[arg-type]
            peak_rss_bytes=sampler.peak_rss_bytes,
            temp_spill=self._postgres_temp_bytes() > temp_bytes_before,
        )

    def _postgres_temp_bytes(self) -> int:
        with self._connection.transaction():
            row = self._postgres.fetchone(
                """
                SELECT temp_bytes
                FROM pg_stat_database
                WHERE datname = current_database()
                """
            )
        return int(row["temp_bytes"] or 0) if row is not None else 0

    def _macro_hard_counts(
        self,
        *,
        manifest: BenchmarkManifest,
        scenarios: Sequence[BenchmarkScenario],
        context: AudienceSearchContext,
    ) -> Mapping[str, int]:
        batchable = [
            item
            for item in scenarios
            if hard_predicates_support_batch_aggregate(item.hard_predicate_keys)
        ]
        counts: dict[str, int] = {}
        if batchable:
            counts.update(
                self._repository.count_hard_matches_batch(
                    project_id=manifest.project_id,
                    vector_version=manifest.vector_version,
                    source_revision_cutoff=context.source_revision_cutoff,
                    window_start=context.window_start,
                    window_end=context.source_cutoff,
                    requests=tuple(
                        HardMatchAggregateRequest(
                            segment_id=item.scenario_id,
                            hard_predicate_keys=item.hard_predicate_keys,
                            predicate_parameters=item.predicate_parameters,
                        )
                        for item in batchable
                    ),
                    exclusion_context=context.exclusion_context,
                )
            )
        for scenario in scenarios:
            if scenario in batchable:
                continue
            counts[scenario.scenario_id] = self._repository.count_hard_matches(
                project_id=manifest.project_id,
                vector_version=manifest.vector_version,
                source_revision_cutoff=context.source_revision_cutoff,
                window_start=context.window_start,
                window_end=context.source_cutoff,
                hard_predicate_keys=scenario.hard_predicate_keys,
                predicate_parameters=scenario.predicate_parameters,
                exclusion_context=context.exclusion_context,
            )
        return counts

    def run_scenario(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        config: CellRunConfig,
        diagnostics_dir: Path | None = None,
    ) -> list[BenchmarkObservation]:
        expected_set = (
            ScenarioSet.CONFIRMATION
            if config.phase == BenchmarkPhase.CONFIRMATION
            else ScenarioSet.TUNING
        )
        if scenario.scenario_set != expected_set:
            raise ValueError(
                f"{config.phase.value} requires {expected_set.value} scenarios"
            )
        if config.cache_mode == CacheMode.DB_COLD:
            return self._run_db_cold_scenario(
                manifest=manifest,
                scenario=scenario,
                config=config,
            )
        diagnostic_started_at = datetime.now(UTC)
        preflight = self.preflight(manifest)
        corpus_user_count = int(preflight["corpus_user_count"])
        context = self._load_context(manifest)
        excluded_user_count = (
            context.exclusion_context.excluded_user_count
            if context.exclusion_context is not None
            else 0
        )
        observed_exclusion_ratio = excluded_user_count / (
            context.corpus_user_count + excluded_user_count
        ) if excluded_user_count else 0.0
        if abs(observed_exclusion_ratio - config.exclusion_ratio) > 0.005:
            raise RuntimeError(
                "prepared Data Contract exclusion ratio does not match run config: "
                f"expected {config.exclusion_ratio}, got {observed_exclusion_ratio}"
            )
        exact_members = self._exact_members(
            manifest=manifest,
            scenario=scenario,
            context=context,
        )
        exact_user_ids = {member.user_id for member in exact_members}
        exact_positive_count = len(exact_members)
        theoretical_min_k = _minimum_rank_for_recall_gate(
            exact_members,
            target_recall=0.95,
        )

        estimates = {
            sample_size: self._estimate_inputs(
                manifest=manifest,
                scenario=scenario,
                context=context,
                sample_size=sample_size,
            )
            for sample_size in config.sample_sizes
        }
        requested_k_values = set(config.requested_k_values)
        if not requested_k_values:
            for hard_count, pass_rate in estimates.values():
                requested_k_values.update(
                    enumerate_unique_candidate_counts(
                        corpus_user_count=corpus_user_count,
                        expected_member_count=hard_count * pass_rate,
                    )
                )
        requested_k_values = {
            value for value in requested_k_values if 0 < value <= corpus_user_count
        }
        cells = self._execution_cells(
            sample_sizes=config.sample_sizes,
            requested_k_values=tuple(sorted(requested_k_values)),
            hnsw_settings=config.hnsw_settings,
            plans=config.plans,
        )
        diagnostics = (
            self._ann_diagnostics(
                manifest=manifest,
                scenario=scenario,
                context=context,
                cells=cells,
                diagnostics_dir=diagnostics_dir,
            )
            if diagnostics_dir is not None
            else {}
        )

        observations: list[BenchmarkObservation] = []
        rng = random.Random(config.random_seed)
        for measured, count in ((False, config.warmups), (True, config.repetitions)):
            for iteration in range(count):
                ordered_cells = list(cells)
                rng.shuffle(ordered_cells)
                for cell in ordered_cells:
                    (
                        hard_count,
                        pass_rate,
                        duration_ms,
                        members,
                        peak_rss_bytes,
                        runtime_work,
                    ) = self._execute_cell(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                        cell=cell,
                    )
                    final_ids = {member.user_id for member in members}
                    if (
                        cell.plan == SearchPlan.FILTER_FIRST_EXACT
                        and final_ids != exact_user_ids
                    ):
                        raise RuntimeError(
                            "filter_first_exact diverged from exact_all ground truth"
                        )
                    if (
                        cell.plan == SearchPlan.ANN_FIRST
                        and not final_ids.issubset(exact_user_ids)
                    ):
                        raise RuntimeError(
                            "ANN final membership contains a non-exact-positive user"
                        )
                    diagnostic = diagnostics.get((cell.requested_k, cell.hnsw))
                    actual_pass_rate = (
                        exact_positive_count / hard_count if hard_count else 0.0
                    )
                    observations.append(
                        BenchmarkObservation(
                            experiment_version=self._experiment_version,
                            phase=config.phase,
                            measured=measured,
                            scenario_id=scenario.scenario_id,
                            candidate_type=scenario.candidate_type,
                            corpus_user_count=corpus_user_count,
                            hard_match_user_count=hard_count,
                            score_pass_sample_size=cell.sample_size,
                            estimated_score_pass_rate=pass_rate,
                            actual_score_pass_rate=actual_pass_rate,
                            plan=cell.plan,
                            cache_mode=config.cache_mode,
                            exclusion_ratio=config.exclusion_ratio,
                            duration_ms=duration_ms,
                            final_user_count=len(final_ids),
                            exact_positive_count=exact_positive_count,
                            intersection_count=len(final_ids & exact_user_ids),
                            requested_k=cell.requested_k,
                            hnsw=cell.hnsw,
                            index_name=diagnostic.index_name if diagnostic else None,
                            temp_spill=diagnostic.temp_spill if diagnostic else False,
                            oom=False,
                            peak_rss_bytes=peak_rss_bytes,
                            theoretical_min_k=theoretical_min_k,
                            scenario_set=scenario.scenario_set,
                            ann_attempt_count=runtime_work.ann_attempt_count,
                            requested_k_history=(
                                runtime_work.requested_k_history
                            ),
                            audited_user_count=runtime_work.audited_user_count,
                            used_exact_fallback=runtime_work.used_exact_fallback,
                            stage_durations_ms=(
                                runtime_work.stage_durations_ms or {}
                            ),
                        )
                    )
        if diagnostics_dir is not None:
            self._write_clickhouse_query_log(
                manifest=manifest,
                scenario=scenario,
                started_at=diagnostic_started_at,
                ended_at=datetime.now(UTC),
                diagnostics_dir=diagnostics_dir,
            )
        return observations

    def export_ground_truth(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        output_path: Path,
        reference_sample_size: int = 50_000,
    ) -> Mapping[str, Any]:
        """Persist the exact full-corpus rank and final-positive membership."""

        if output_path.suffix != ".gz":
            raise ValueError("ground truth output must use a .gz suffix")
        preflight = self.preflight(manifest)
        context = self._load_context(manifest)
        exact_members = self._exact_members(
            manifest=manifest,
            scenario=scenario,
            context=context,
        )
        final_user_ids = {member.user_id for member in exact_members}
        hard_count, estimated_pass_rate = self._estimate_inputs(
            manifest=manifest,
            scenario=scenario,
            context=context,
            sample_size=reference_sample_size,
        )
        exclusion_sql = ""
        exclusion_params: tuple[Any, ...] = ()
        if context.exclusion_context is not None:
            exclusion_sql = f"""
              AND NOT EXISTS (
                  SELECT 1
                  FROM {POSTGRES_EXCLUSION_RELATION} AS excluded
                  WHERE excluded.project_id = search.project_id
                    AND excluded.promotion_id = %s
                    AND excluded.user_id = search.user_id
                    AND excluded.state IN ('reserved', 'consumed')
              )
            """
            exclusion_params = (context.exclusion_context.promotion_id,)
        query = f"""
            SELECT search.user_id,
                   1 - (search.embedding <=> %s::vector) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              {exclusion_sql}
            ORDER BY behavior_fit_score DESC, search.user_id ASC
        """
        params = (
            _vector_literal(scenario.query_vector),
            manifest.project_id,
            manifest.vector_version,
            context.source_cutoff,
            context.vector_generation_id,
            *exclusion_params,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_hash = hashlib.sha256()
        row_count = 0
        score_pass_count = 0
        with self._connection.transaction():
            self._postgres.execute("SET LOCAL enable_indexscan = off")
            self._postgres.execute("SET LOCAL enable_bitmapscan = off")
            with self._connection.cursor(
                name=f"ann_ground_truth_{scenario.scenario_id[:24]}"
            ) as cursor:
                cursor.itersize = 10_000
                cursor.execute(query, params)
                with gzip.open(output_path, "wt", encoding="utf-8", newline="") as raw:
                    writer = csv.writer(raw)
                    writer.writerow(
                        (
                            "cosine_rank",
                            "user_id",
                            "behavior_fit_score",
                            "score_pass",
                            "final_positive",
                        )
                    )
                    for row_count, row in enumerate(cursor, start=1):
                        user_id = str(row[0])
                        score = float(row[1])
                        score_pass = score >= scenario.score_threshold
                        score_pass_count += int(score_pass)
                        corpus_hash.update(user_id.encode("utf-8"))
                        corpus_hash.update(b"\n")
                        writer.writerow(
                            (
                                row_count,
                                user_id,
                                format(score, ".17g"),
                                int(score_pass),
                                int(user_id in final_user_ids),
                            )
                        )
        if row_count != int(preflight["corpus_user_count"]):
            raise RuntimeError("ground truth row count differs from active corpus")
        actual_pass_rate = len(final_user_ids) / hard_count if hard_count else 0.0
        summary = {
            "experiment_version": self._experiment_version,
            "scenario_id": scenario.scenario_id,
            "candidate_type": scenario.candidate_type,
            "vector_generation_id": context.vector_generation_id,
            "manifest_hash": context.manifest_hash,
            "corpus_user_count": row_count,
            "corpus_rank_sha256": corpus_hash.hexdigest(),
            "hard_match_user_count": hard_count,
            "score_pass_user_count": score_pass_count,
            "exact_positive_count": len(final_user_ids),
            "reference_sample_size": reference_sample_size,
            "estimated_score_pass_rate": estimated_pass_rate,
            "actual_score_pass_rate": actual_pass_rate,
            "estimated_member_count": hard_count * estimated_pass_rate,
            "theoretical_min_k": _minimum_rank_for_recall_gate(
                exact_members,
                target_recall=0.95,
            ),
            "ground_truth_csv_gz": str(output_path),
        }
        summary_path = output_path.with_suffix("").with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _run_db_cold_scenario(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        config: CellRunConfig,
    ) -> list[BenchmarkObservation]:
        if config.exclusion_ratio != 0.0:
            raise ValueError("DB-cold confirmation does not support exclusion mode")
        if len(config.sample_sizes) != 1 or len(config.plans) != 1:
            raise ValueError("DB-cold mode requires one sample size and one plan")
        plan = config.plans[0]
        if plan == SearchPlan.ANN_FIRST:
            if len(config.requested_k_values) != 1 or len(config.hnsw_settings) != 1:
                raise ValueError("DB-cold ANN requires one K and one HNSW setting")
            requested_k = config.requested_k_values[0]
            hnsw = config.hnsw_settings[0]
        else:
            if config.requested_k_values:
                raise ValueError("DB-cold Exact plan must not specify requested K")
            requested_k = None
            hnsw = None
        context = self._load_cold_context(manifest)
        cell = _ExecutionCell(
            sample_size=config.sample_sizes[0],
            plan=plan,
            requested_k=requested_k,
            hnsw=hnsw,
        )
        (
            hard_count,
            pass_rate,
            duration_ms,
            members,
            peak_rss,
            runtime_work,
        ) = self._execute_cell(
            manifest=manifest,
            scenario=scenario,
            context=context,
            cell=cell,
        )
        exact_members = (
            members
            if plan == SearchPlan.EXACT_ALL
            else self._exact_members(
                manifest=manifest,
                scenario=scenario,
                context=context,
            )
        )
        exact_user_ids = {member.user_id for member in exact_members}
        final_user_ids = {member.user_id for member in members}
        if (
            plan == SearchPlan.FILTER_FIRST_EXACT
            and final_user_ids != exact_user_ids
        ):
            raise RuntimeError(
                "filter_first_exact diverged from exact_all ground truth"
            )
        if (
            plan == SearchPlan.ANN_FIRST
            and not final_user_ids.issubset(exact_user_ids)
        ):
            raise RuntimeError(
                "ANN final membership contains a non-exact-positive user"
            )
        exact_positive_count = len(exact_user_ids)
        return [
            BenchmarkObservation(
                experiment_version=self._experiment_version,
                phase=config.phase,
                measured=True,
                scenario_id=scenario.scenario_id,
                candidate_type=scenario.candidate_type,
                corpus_user_count=context.corpus_user_count,
                hard_match_user_count=hard_count,
                score_pass_sample_size=cell.sample_size,
                estimated_score_pass_rate=pass_rate,
                actual_score_pass_rate=(
                    exact_positive_count / hard_count if hard_count else 0.0
                ),
                plan=plan,
                cache_mode=CacheMode.DB_COLD,
                exclusion_ratio=0.0,
                duration_ms=duration_ms,
                final_user_count=len(final_user_ids),
                exact_positive_count=exact_positive_count,
                intersection_count=len(final_user_ids & exact_user_ids),
                requested_k=requested_k,
                hnsw=hnsw,
                index_name=None,
                temp_spill=False,
                oom=False,
                peak_rss_bytes=peak_rss,
                theoretical_min_k=_minimum_rank_for_recall_gate(
                    exact_members,
                    target_recall=0.95,
                ),
                scenario_set=scenario.scenario_set,
                ann_attempt_count=runtime_work.ann_attempt_count,
                requested_k_history=runtime_work.requested_k_history,
                audited_user_count=runtime_work.audited_user_count,
                used_exact_fallback=runtime_work.used_exact_fallback,
                stage_durations_ms=runtime_work.stage_durations_ms or {},
            )
        ]

    def _load_cold_context(self, manifest: BenchmarkManifest) -> AudienceSearchContext:
        with self._connection.transaction():
            row = self._postgres.fetchone(
                """
                SELECT vector_generation_id, manifest_hash,
                       window_end AS source_cutoff, source_revision_cutoff,
                       window_start, expected_user_count, synced_user_count
                FROM user_behavior_vector_search_generations
                WHERE project_id = %s
                  AND vector_version = %s
                  AND status = 'activated'
                  AND is_active = true
                """,
                (manifest.project_id, manifest.vector_version),
            )
        if row is None:
            raise RuntimeError("active benchmark vector generation is required")
        if str(row["manifest_hash"]) != manifest.manifest_hash:
            raise RuntimeError("active vector generation manifest hash mismatches")
        expected = int(row["expected_user_count"])
        if expected <= 0 or int(row["synced_user_count"]) != expected:
            raise RuntimeError("cold benchmark generation counts are incomplete")
        return AudienceSearchContext(
            vector_generation_id=str(row["vector_generation_id"]),
            manifest_hash=str(row["manifest_hash"]),
            source_cutoff=row["source_cutoff"],
            source_revision_cutoff=row["source_revision_cutoff"],
            window_start=row["window_start"],
            corpus_user_count=expected,
            exclusion_context=None,
        )

    def _load_context(self, manifest: BenchmarkManifest) -> Any:
        with self._connection.transaction():
            context = self._repository.get_context(
                project_id=manifest.project_id,
                vector_version=manifest.vector_version,
                campaign_id=manifest.campaign_id,
                promotion_id=manifest.promotion_id,
            )
        return context

    def _estimate_inputs(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        sample_size: int,
    ) -> tuple[int, float]:
        with self._connection.transaction():
            hard_count = self._cohort_hard_match_count(
                manifest=manifest,
                scenario=scenario,
                context=context,
            )
            if self._scale_cohort_scope is not None:
                if sample_size != (
                    self._scale_cohort_scope.effective_reference_sample_size
                ):
                    raise ValueError(
                        "scale benchmark P must use the fixed cohort-bounded "
                        "reference sample size"
                    )
                pass_rate = self._cohort_score_pass_rate(
                    manifest=manifest,
                    scenario=scenario,
                    context=context,
                )
            else:
                pass_rate = self._repository.estimate_score_pass_rate(
                    project_id=manifest.project_id,
                    vector_generation_id=context.vector_generation_id,
                    vector_version=manifest.vector_version,
                    source_revision_cutoff=context.source_revision_cutoff,
                    source_cutoff=context.source_cutoff,
                    window_start=context.window_start,
                    query_vector=scenario.query_vector,
                    score_threshold=scenario.score_threshold,
                    hard_predicate_keys=scenario.hard_predicate_keys,
                    predicate_parameters=scenario.predicate_parameters,
                    sample_size=sample_size,
                    exclusion_context=context.exclusion_context,
                )
        return hard_count, pass_rate

    def _cohort_hard_match_count(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> int:
        key = (context.vector_generation_id, scenario.scenario_id)
        cached = self._cohort_hard_counts.get(key)
        if cached is not None:
            return cached
        if self._scale_cohort_scope is not None:
            count = self._cohort_hard_match_count_clickhouse(
                manifest=manifest,
                scenario=scenario,
                context=context,
            )
        else:
            count = self._repository.count_hard_matches_for_user_ids(
                project_id=manifest.project_id,
                vector_generation_id=context.vector_generation_id,
                vector_version=manifest.vector_version,
                source_cutoff=context.source_cutoff,
                hard_predicate_keys=scenario.hard_predicate_keys,
                predicate_parameters=scenario.predicate_parameters,
                user_ids=self._cohort_user_ids(context.vector_generation_id),
            )
        self._cohort_hard_counts[key] = count
        return count

    def _scale_clickhouse_parameters(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> dict[str, Any]:
        scope = self._scale_cohort_scope
        if scope is None:
            raise RuntimeError("scale cohort parameters require a scale scope")
        if context.exclusion_context is not None:
            raise RuntimeError(
                "Goal 1 scale-series does not run exclusion confirmation"
            )
        return {
            "project_id": manifest.project_id,
            "vector_version": manifest.vector_version,
            "source_revision_cutoff": context.source_revision_cutoff.isoformat(),
            "raw_event_received_cutoff": (
                context.source_revision_cutoff.isoformat()
            ),
            "window_start": context.window_start.isoformat(),
            "window_end": context.source_cutoff.isoformat(),
            "destinations": list(
                scenario.predicate_parameters.get("destinations", ())
            ),
            "season_months": list(
                scenario.predicate_parameters.get("season_months", ())
            ),
            "benefit_keys": list(
                scenario.predicate_parameters.get("benefit_keys", ())
            ),
            **scope.clickhouse_parameters(),
        }

    def _cohort_hard_match_count_clickhouse(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> int:
        result = self._clickhouse.query(
            cohort_bound_hard_match_query(scenario.hard_predicate_keys),
            parameters=self._scale_clickhouse_parameters(
                manifest=manifest,
                scenario=scenario,
                context=context,
            ),
        )
        rows = _named_rows(result)
        return int(rows[0]["matching_user_count"]) if rows else 0

    def _cohort_score_pass_rate(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> float:
        scope = self._scale_cohort_scope
        if scope is None:
            raise RuntimeError("cohort score estimate requires a scale scope")
        result = self._clickhouse.query(
            cohort_bound_reference_sample_query(scenario.hard_predicate_keys),
            parameters=self._scale_clickhouse_parameters(
                manifest=manifest,
                scenario=scenario,
                context=context,
            ),
        )
        sample_user_ids = tuple(
            str(row["user_id"]) for row in _named_rows(result)
        )
        if not sample_user_ids:
            return 0.0
        cohort_user_ids = self._cohort_user_ids(context.vector_generation_id)
        validate_cohort_bound_users(
            cohort_user_ids=cohort_user_ids,
            reference_sample_user_ids=sample_user_ids,
        )
        if len(sample_user_ids) > scope.effective_reference_sample_size:
            raise RuntimeError("reference sample exceeds the fixed cohort bound")
        self._repository._replace_temp_user_ids(
            table_name="audience_hard_match_sample",
            user_ids=sample_user_ids,
        )
        row = self._postgres.fetchone(
            """
            SELECT
                count(*) AS sampled_count,
                count(*) FILTER (
                    WHERE 1 - (search.embedding <=> %s::vector) >= %s
                ) AS passed_count
            FROM user_behavior_vector_search AS search
            JOIN audience_hard_match_sample AS sample USING (user_id)
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.window_end = %s
              AND search.vector_generation_id = %s
            """,
            (
                _vector_literal(scenario.query_vector),
                scenario.score_threshold,
                manifest.project_id,
                manifest.vector_version,
                context.source_cutoff,
                context.vector_generation_id,
            ),
        )
        sampled_count = int(row["sampled_count"]) if row is not None else 0
        if sampled_count != len(set(sample_user_ids)):
            raise RuntimeError(
                "cohort reference sample is not fully present in PostgreSQL"
            )
        return int(row["passed_count"]) / sampled_count if sampled_count else 0.0

    def _cohort_user_ids(self, vector_generation_id: str) -> tuple[str, ...]:
        cached = self._cohort_user_ids_by_generation.get(vector_generation_id)
        if cached is not None:
            return cached
        rows = self._postgres.fetchall(
            """
            SELECT user_id
            FROM user_behavior_vector_search
            WHERE vector_generation_id = %s
            ORDER BY user_id
            """,
            (vector_generation_id,),
        )
        user_ids = tuple(str(row["user_id"]) for row in rows)
        self._cohort_user_ids_by_generation[vector_generation_id] = user_ids
        return user_ids

    def _exact_members(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> list[SearchCandidate]:
        with self._connection.transaction():
            if self._scale_cohort_scope is not None:
                return self._scale_exact_search(
                    manifest=manifest,
                    scenario=scenario,
                    context=context,
                )
            return self._repository.exact_search(
                project_id=manifest.project_id,
                vector_generation_id=context.vector_generation_id,
                vector_version=manifest.vector_version,
                source_cutoff=context.source_cutoff,
                query_vector=scenario.query_vector,
                score_threshold=scenario.score_threshold,
                hard_predicate_keys=scenario.hard_predicate_keys,
                predicate_parameters=scenario.predicate_parameters,
            )

    def _execution_cells(
        self,
        *,
        sample_sizes: Sequence[int],
        requested_k_values: Sequence[int],
        hnsw_settings: Sequence[HnswSettings],
        plans: Sequence[SearchPlan],
    ) -> list[_ExecutionCell]:
        cells: list[_ExecutionCell] = []
        for sample_size in sample_sizes:
            if SearchPlan.CURRENT_RUNTIME in plans:
                cells.append(_ExecutionCell(sample_size, SearchPlan.CURRENT_RUNTIME))
            if SearchPlan.EXACT_ALL in plans:
                cells.append(_ExecutionCell(sample_size, SearchPlan.EXACT_ALL))
            if SearchPlan.FILTER_FIRST_EXACT in plans:
                cells.append(
                    _ExecutionCell(sample_size, SearchPlan.FILTER_FIRST_EXACT)
                )
            if SearchPlan.ANN_FIRST in plans:
                cells.extend(
                    _ExecutionCell(
                        sample_size,
                        SearchPlan.ANN_FIRST,
                        requested_k=requested_k,
                        hnsw=hnsw,
                    )
                    for requested_k in requested_k_values
                    for hnsw in hnsw_settings
                )
        return cells

    def _execute_cell(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        cell: _ExecutionCell,
    ) -> tuple[
        int,
        float,
        float,
        list[SearchCandidate],
        int | None,
        _RuntimeWork,
    ]:
        sampler = _BackendRssSampler.from_connection(self._connection)
        runtime_work = _RuntimeWork()
        with sampler:
            started = time.perf_counter_ns()
            with self._connection.transaction():
                hard_started = time.perf_counter_ns()
                if self._scale_cohort_scope is not None:
                    hard_count = self._cohort_hard_match_count_clickhouse(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                    )
                else:
                    self._repository.count_hard_matches(
                        project_id=manifest.project_id,
                        vector_version=manifest.vector_version,
                        source_revision_cutoff=context.source_revision_cutoff,
                        window_start=context.window_start,
                        window_end=context.source_cutoff,
                        hard_predicate_keys=scenario.hard_predicate_keys,
                        predicate_parameters=scenario.predicate_parameters,
                        exclusion_context=context.exclusion_context,
                    )
                    hard_count = self._cohort_hard_match_count(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                    )
                hard_duration = (time.perf_counter_ns() - hard_started) / 1_000_000
                estimate_started = time.perf_counter_ns()
                if self._scale_cohort_scope is not None:
                    if cell.sample_size != (
                        self._scale_cohort_scope.effective_reference_sample_size
                    ):
                        raise ValueError(
                            "scale benchmark P must use the fixed cohort-bounded "
                            "reference sample size"
                        )
                    pass_rate = self._cohort_score_pass_rate(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                    )
                else:
                    pass_rate = self._repository.estimate_score_pass_rate(
                        project_id=manifest.project_id,
                        vector_generation_id=context.vector_generation_id,
                        vector_version=manifest.vector_version,
                        source_revision_cutoff=context.source_revision_cutoff,
                        source_cutoff=context.source_cutoff,
                        window_start=context.window_start,
                        query_vector=scenario.query_vector,
                        score_threshold=scenario.score_threshold,
                        hard_predicate_keys=scenario.hard_predicate_keys,
                        predicate_parameters=scenario.predicate_parameters,
                        sample_size=cell.sample_size,
                        exclusion_context=context.exclusion_context,
                    )
                estimate_duration = (
                    time.perf_counter_ns() - estimate_started
                ) / 1_000_000
                search_started = time.perf_counter_ns()
                if cell.plan == SearchPlan.CURRENT_RUNTIME:
                    runtime_repository = (
                        _ScaleRuntimeRepository(
                            benchmark=self,
                            manifest=manifest,
                            scenario=scenario,
                            context=context,
                        )
                        if self._scale_cohort_scope is not None
                        else self._repository
                    )
                    traced_repository = _RuntimeTraceRepository(runtime_repository)
                    result = CandidateAudienceSearchService(
                        traced_repository
                    ).search(
                        project_id=manifest.project_id,
                        vector_generation_id=context.vector_generation_id,
                        source_cutoff=context.source_cutoff,
                        spec=scenario.candidate_behavior_spec(
                            vector_version=manifest.vector_version
                        ),
                        corpus_user_count=context.corpus_user_count,
                        hard_match_user_count=hard_count,
                        estimated_score_pass_rate=pass_rate,
                    )
                    members = list(result.members)
                    if not members and result.members_relation:
                        members = self._materialized_members(
                            result.members_relation
                        )
                    traced_work = traced_repository.to_work(result.method)
                    runtime_work = _RuntimeWork(
                        ann_attempt_count=traced_work.ann_attempt_count,
                        requested_k_history=traced_work.requested_k_history,
                        audited_user_count=traced_work.audited_user_count,
                        used_exact_fallback=traced_work.used_exact_fallback,
                        stage_durations_ms={
                            "hard_count": hard_duration,
                            "score_pass_estimate": estimate_duration,
                            **(traced_work.stage_durations_ms or {}),
                        },
                    )
                elif cell.plan == SearchPlan.EXACT_ALL:
                    members = (
                        self._scale_exact_search(
                            manifest=manifest,
                            scenario=scenario,
                            context=context,
                        )
                        if self._scale_cohort_scope is not None
                        else self._repository.exact_search(
                            project_id=manifest.project_id,
                            vector_generation_id=context.vector_generation_id,
                            vector_version=manifest.vector_version,
                            source_cutoff=context.source_cutoff,
                            query_vector=scenario.query_vector,
                            score_threshold=scenario.score_threshold,
                            hard_predicate_keys=scenario.hard_predicate_keys,
                            predicate_parameters=scenario.predicate_parameters,
                        )
                    )
                elif cell.plan == SearchPlan.FILTER_FIRST_EXACT:
                    members = self._filter_first_exact(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                    )
                else:
                    assert cell.requested_k is not None and cell.hnsw is not None
                    members = self._ann_first(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                        requested_k=cell.requested_k,
                        hnsw=cell.hnsw,
                    )
                search_duration = (
                    time.perf_counter_ns() - search_started
                ) / 1_000_000
                if cell.plan == SearchPlan.CURRENT_RUNTIME:
                    runtime_work = _RuntimeWork(
                        ann_attempt_count=runtime_work.ann_attempt_count,
                        requested_k_history=runtime_work.requested_k_history,
                        audited_user_count=runtime_work.audited_user_count,
                        used_exact_fallback=runtime_work.used_exact_fallback,
                        stage_durations_ms={
                            **(runtime_work.stage_durations_ms or {}),
                            "search_total": search_duration,
                        },
                    )
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
        return (
            hard_count,
            pass_rate,
            duration_ms,
            members,
            sampler.peak_rss_bytes,
            runtime_work,
        )

    def _materialized_members(self, relation: str) -> list[SearchCandidate]:
        if relation not in {"audience_exact_members", "audience_ann_members"}:
            raise ValueError("unexpected current-runtime member relation")
        rows = self._postgres.fetchall(
            f"SELECT user_id, behavior_fit_score, retrieval_rank "
            f"FROM {relation} ORDER BY retrieval_rank, user_id"
        )
        return _candidates(rows)

    def _filter_first_exact(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> list[SearchCandidate]:
        user_ids = self._hard_match_user_ids(
            manifest=manifest,
            scenario=scenario,
            context=context,
        )
        self._load_hard_match_temp(user_ids)
        rows = self._postgres.fetchall(
            _exact_plan_sql(
                plan=SearchPlan.FILTER_FIRST_EXACT,
                explain=False,
                exclude_promotion_users=context.exclusion_context is not None,
            ),
            _exact_plan_params(
                manifest=manifest,
                scenario=scenario,
                context=context,
                exclusion_promotion_id=(
                    context.exclusion_context.promotion_id
                    if context.exclusion_context is not None
                    else None
                ),
            ),
        )
        return _candidates(rows)

    def _hard_match_user_ids(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> list[str]:
        if self._scale_cohort_scope is not None:
            result = self._clickhouse.query(
                cohort_bound_hard_match_users_query(
                    scenario.hard_predicate_keys
                ),
                parameters=self._scale_clickhouse_parameters(
                    manifest=manifest,
                    scenario=scenario,
                    context=context,
                ),
            )
            user_ids = [str(row["user_id"]) for row in _named_rows(result)]
            validate_cohort_bound_users(
                cohort_user_ids=self._cohort_user_ids(
                    context.vector_generation_id
                ),
                hard_match_user_ids=user_ids,
                filter_first_user_ids=user_ids,
            )
            return user_ids
        result = self._clickhouse.query(
            _hard_predicate_query(
                scenario.hard_predicate_keys,
                filter_user_ids=False,
                restrict_to_vector_population=True,
                exclude_promotion_users=context.exclusion_context is not None,
            ),
            parameters={
                "project_id": manifest.project_id,
                "vector_version": manifest.vector_version,
                "source_revision_cutoff": context.source_revision_cutoff.isoformat(),
                "raw_event_received_cutoff": (
                    context.source_revision_cutoff.isoformat()
                ),
                "window_start": context.window_start.isoformat(),
                "window_end": context.source_cutoff.isoformat(),
                "destinations": list(
                    scenario.predicate_parameters.get("destinations", ())
                ),
                "season_months": list(
                    scenario.predicate_parameters.get("season_months", ())
                ),
                "benefit_keys": list(
                    scenario.predicate_parameters.get("benefit_keys", ())
                ),
                **(
                    {
                        "exclusion_promotion_id": (
                            context.exclusion_context.promotion_id
                        ),
                        "exclusion_revision": context.exclusion_context.revision,
                    }
                    if context.exclusion_context is not None
                    else {}
                ),
            },
        )
        return [str(row["user_id"]) for row in _named_rows(result)]

    def _load_hard_match_temp(self, user_ids: Sequence[str]) -> None:
        self._postgres.execute(
            "CREATE TEMP TABLE benchmark_hard_matches "
            "(user_id text PRIMARY KEY) ON COMMIT DROP"
        )
        for offset in range(0, len(user_ids), 10_000):
            self._postgres.execute(
                "INSERT INTO benchmark_hard_matches (user_id) "
                "SELECT DISTINCT user_id FROM unnest(%s::text[]) rows(user_id) "
                "ON CONFLICT DO NOTHING",
                (list(user_ids[offset : offset + 10_000]),),
            )

    def _ann_first(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        requested_k: int,
        hnsw: HnswSettings,
    ) -> list[SearchCandidate]:
        self._set_hnsw(hnsw)
        rows = self._postgres.fetchall(
            _ann_sql(
                explain=False,
                exclude_promotion_users=context.exclusion_context is not None,
            ),
            _ann_params(
                manifest=manifest,
                scenario=scenario,
                context=context,
                requested_k=requested_k,
                exclusion_promotion_id=(
                    context.exclusion_context.promotion_id
                    if context.exclusion_context is not None
                    else None
                ),
            ),
        )
        if self._scale_cohort_scope is not None:
            return self._scale_exact_filter_user_ids(
                manifest=manifest,
                scenario=scenario,
                context=context,
                user_ids=[str(row["user_id"]) for row in rows],
            )
        return self._repository.exact_filter_candidates(
            project_id=manifest.project_id,
            vector_generation_id=context.vector_generation_id,
            vector_version=manifest.vector_version,
            source_cutoff=context.source_cutoff,
            query_vector=scenario.query_vector,
            score_threshold=scenario.score_threshold,
            hard_predicate_keys=scenario.hard_predicate_keys,
            predicate_parameters=scenario.predicate_parameters,
            user_ids=[str(row["user_id"]) for row in rows],
        )

    def _scale_exact_search(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> list[SearchCandidate]:
        rows = self._postgres.fetchall(
            _exact_plan_sql(
                plan=SearchPlan.EXACT_ALL,
                explain=False,
                exclude_promotion_users=False,
            ),
            _exact_plan_params(
                manifest=manifest,
                scenario=scenario,
                context=context,
                exclusion_promotion_id=None,
            ),
        )
        return self._scale_filter_candidates(
            manifest=manifest,
            scenario=scenario,
            context=context,
            candidates=_candidates(rows),
        )

    def _scale_filter_candidates(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        candidates: Sequence[SearchCandidate],
    ) -> list[SearchCandidate]:
        if not candidates:
            return []
        hard_ids = set(
            self._hard_match_user_ids(
                manifest=manifest,
                scenario=scenario,
                context=context,
            )
        )
        return [item for item in candidates if item.user_id in hard_ids]

    def _scale_exact_filter_user_ids(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        user_ids: Sequence[str],
    ) -> list[SearchCandidate]:
        if not user_ids:
            return []
        self._repository._replace_temp_user_ids(
            table_name="audience_ann_candidates",
            user_ids=user_ids,
        )
        rows = self._postgres.fetchall(
            """
            SELECT search.user_id,
                   1 - (search.embedding <=> %s::vector) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            JOIN audience_ann_candidates AS candidate USING (user_id)
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              AND 1 - (search.embedding <=> %s::vector) >= %s
            ORDER BY behavior_fit_score DESC, search.user_id ASC
            """,
            (
                _vector_literal(scenario.query_vector),
                manifest.project_id,
                manifest.vector_version,
                context.source_cutoff,
                context.vector_generation_id,
                _vector_literal(scenario.query_vector),
                scenario.score_threshold,
            ),
        )
        return self._scale_filter_candidates(
            manifest=manifest,
            scenario=scenario,
            context=context,
            candidates=_candidates(rows),
        )

    def _scale_audit_nonretrieved(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        excluded_user_ids: Sequence[str],
        sample_size: int,
    ) -> tuple[int, int]:
        self._repository._replace_temp_user_ids(
            table_name="audience_ann_retrieved",
            user_ids=excluded_user_ids,
        )
        rows = self._postgres.fetchall(
            """
            SELECT search.user_id,
                   1 - (search.embedding <=> %s::vector) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              AND NOT EXISTS (
                  SELECT 1 FROM audience_ann_retrieved AS retrieved
                  WHERE retrieved.user_id = search.user_id
              )
            ORDER BY md5(search.user_id || %s) ASC, search.user_id ASC
            LIMIT %s
            """,
            (
                _vector_literal(scenario.query_vector),
                manifest.project_id,
                manifest.vector_version,
                context.source_cutoff,
                context.vector_generation_id,
                f"{manifest.project_id}:{context.vector_generation_id}:"
                f"{context.source_cutoff}",
                sample_size,
            ),
        )
        sampled = _candidates(rows)
        score_passed = [
            item for item in sampled
            if item.behavior_fit_score >= scenario.score_threshold
        ]
        missed = self._scale_filter_candidates(
            manifest=manifest,
            scenario=scenario,
            context=context,
            candidates=score_passed,
        )
        return len(sampled), len(missed)

    def _ann_diagnostics(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        cells: Sequence[_ExecutionCell],
        diagnostics_dir: Path | None,
    ) -> Mapping[tuple[int | None, HnswSettings | None], _PlanDiagnostic]:
        diagnostics: dict[
            tuple[int | None, HnswSettings | None], _PlanDiagnostic
        ] = {}
        ann_cells = {
            (cell.requested_k, cell.hnsw)
            for cell in cells
            if cell.plan == SearchPlan.ANN_FIRST
        }
        if diagnostics_dir is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            self._write_exact_diagnostics(
                manifest=manifest,
                scenario=scenario,
                context=context,
                diagnostics_dir=diagnostics_dir,
            )
        for requested_k, hnsw in sorted(
            ann_cells,
            key=lambda item: (item[0] or 0, item[1] or HnswSettings(1, "strict_order", 1)),
        ):
            assert requested_k is not None and hnsw is not None
            with self._connection.transaction():
                self._set_hnsw(hnsw)
                row = self._postgres.fetchone(
                    _ann_sql(
                        explain=True,
                        exclude_promotion_users=(
                            context.exclusion_context is not None
                        ),
                    ),
                    _ann_params(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                        requested_k=requested_k,
                        exclusion_promotion_id=(
                            context.exclusion_context.promotion_id
                            if context.exclusion_context is not None
                            else None
                        ),
                    ),
                )
            explain = next(iter(row.values())) if row else None
            index_names = _find_plan_values(explain, "Index Name")
            temp_blocks = sum(
                int(value or 0)
                for key in ("Temp Read Blocks", "Temp Written Blocks")
                for value in _find_plan_values(explain, key)
            )
            diagnostic = _PlanDiagnostic(
                index_name=(
                    HNSW_INDEX_NAME if HNSW_INDEX_NAME in index_names else None
                ),
                temp_spill=temp_blocks > 0,
                explain=explain,
            )
            diagnostics[(requested_k, hnsw)] = diagnostic
            if diagnostics_dir is not None:
                filename = (
                    f"{scenario.scenario_id}-k{requested_k}-ef{hnsw.ef_search}-"
                    f"{hnsw.iterative_scan}-scan{hnsw.max_scan_tuples}.json"
                )
                (diagnostics_dir / filename).write_text(
                    json.dumps(explain, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return diagnostics

    def _write_exact_diagnostics(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
        diagnostics_dir: Path,
    ) -> None:
        exclusion_promotion_id = (
            context.exclusion_context.promotion_id
            if context.exclusion_context is not None
            else None
        )
        for plan in (SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT):
            hard_user_ids: list[str] | None = None
            if plan == SearchPlan.FILTER_FIRST_EXACT:
                hard_user_ids = self._hard_match_user_ids(
                    manifest=manifest,
                    scenario=scenario,
                    context=context,
                )
            with self._connection.transaction():
                if hard_user_ids is not None:
                    self._load_hard_match_temp(hard_user_ids)
                row = self._postgres.fetchone(
                    _exact_plan_sql(
                        plan=plan,
                        explain=True,
                        exclude_promotion_users=(
                            context.exclusion_context is not None
                        ),
                    ),
                    _exact_plan_params(
                        manifest=manifest,
                        scenario=scenario,
                        context=context,
                        exclusion_promotion_id=exclusion_promotion_id,
                    ),
                )
            explain = next(iter(row.values())) if row else None
            (diagnostics_dir / f"{scenario.scenario_id}-{plan.value}.json").write_text(
                json.dumps(explain, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _set_hnsw(self, hnsw: HnswSettings) -> None:
        for name, value in (
            ("hnsw.ef_search", str(hnsw.ef_search)),
            ("hnsw.iterative_scan", hnsw.iterative_scan),
            ("hnsw.max_scan_tuples", str(hnsw.max_scan_tuples)),
        ):
            self._postgres.execute(
                "SELECT set_config(%s, %s, true)",
                (name, value),
            )

    def _write_clickhouse_query_log(
        self,
        *,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        started_at: datetime,
        ended_at: datetime,
        diagnostics_dir: Path,
    ) -> None:
        output_path = diagnostics_dir / (
            f"{scenario.scenario_id}-clickhouse-query-log.json"
        )
        try:
            command = getattr(self._clickhouse, "command", None)
            if callable(command):
                command("SYSTEM FLUSH LOGS")
            result = self._clickhouse.query(
                """
                SELECT
                    event_time_microseconds,
                    query_id,
                    type,
                    query_duration_ms,
                    read_rows,
                    read_bytes,
                    result_rows,
                    memory_usage,
                    ProfileEvents,
                    Settings,
                    query
                FROM system.query_log
                WHERE event_time_microseconds >= {started_at:DateTime64(6, 'UTC')}
                  AND event_time_microseconds <= {ended_at:DateTime64(6, 'UTC')}
                  AND type IN ('QueryFinish', 'ExceptionWhileProcessing')
                  AND query NOT ILIKE '%system.query_log%'
                  AND (
                      query ILIKE '%user_behavior%'
                      OR query ILIKE '%promotion_audience_exclusions%'
                  )
                ORDER BY event_time_microseconds, query_id
                """,
                parameters={"started_at": started_at, "ended_at": ended_at},
            )
            payload: Mapping[str, Any] = {
                "project_id": manifest.project_id,
                "scenario_id": scenario.scenario_id,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "queries": _named_rows(result),
            }
        except Exception as exc:  # diagnostic absence must be explicit in artifacts
            payload = {
                "project_id": manifest.project_id,
                "scenario_id": scenario.scenario_id,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "queries": [],
                "capture_error": f"{type(exc).__name__}: {exc}",
            }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


class _ScaleRuntimeRepository:
    """Run the production policy with the frozen scale-safe hard backend.

    Selection, K growth, audit, and fallback behavior are the production
    implementation. Only candidate hard filtering uses the exact frozen
    one-row-per-user signal relation instead of rescanning raw events once per
    1,000-candidate repository chunk.
    """

    def __init__(
        self,
        *,
        benchmark: LiveAnnSearchBenchmark,
        manifest: BenchmarkManifest,
        scenario: BenchmarkScenario,
        context: Any,
    ) -> None:
        self._benchmark = benchmark
        self._manifest = manifest
        self._scenario = scenario
        self._context = context

    def exact_search(self, **_: Any) -> list[SearchCandidate]:
        return self._benchmark._scale_exact_search(
            manifest=self._manifest,
            scenario=self._scenario,
            context=self._context,
        )

    def ann_search(self, *, limit: int, **_: Any) -> list[SearchCandidate]:
        self._benchmark._set_hnsw(HnswSettings(100, "strict_order", 20_000))
        rows = self._benchmark._postgres.fetchall(
            _ann_sql(explain=False, exclude_promotion_users=False),
            _ann_params(
                manifest=self._manifest,
                scenario=self._scenario,
                context=self._context,
                requested_k=limit,
                exclusion_promotion_id=None,
            ),
        )
        return _candidates(rows)

    def exact_filter_candidates(
        self, *, user_ids: Sequence[str], **_: Any
    ) -> list[SearchCandidate]:
        return self._benchmark._scale_exact_filter_user_ids(
            manifest=self._manifest,
            scenario=self._scenario,
            context=self._context,
            user_ids=user_ids,
        )

    def audit_nonretrieved(
        self,
        *,
        excluded_user_ids: Sequence[str],
        sample_size: int,
        **_: Any,
    ) -> tuple[int, int]:
        return self._benchmark._scale_audit_nonretrieved(
            manifest=self._manifest,
            scenario=self._scenario,
            context=self._context,
            excluded_user_ids=excluded_user_ids,
            sample_size=sample_size,
        )


def full_hnsw_grid() -> tuple[HnswSettings, ...]:
    return tuple(
        HnswSettings(ef_search, iterative_scan, max_scan_tuples)
        for ef_search in EF_SEARCH_GRID
        for iterative_scan in ITERATIVE_SCAN_GRID
        for max_scan_tuples in MAX_SCAN_TUPLES_GRID
    )


def _exact_plan_sql(
    *,
    plan: SearchPlan,
    explain: bool,
    exclude_promotion_users: bool,
) -> str:
    if plan not in (SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT):
        raise ValueError("exact diagnostic plan is invalid")
    prefix = (
        "EXPLAIN (ANALYZE, BUFFERS, WAL, SETTINGS, FORMAT JSON) "
        if explain
        else ""
    )
    hard_join = (
        "JOIN benchmark_hard_matches AS hard USING (user_id)"
        if plan == SearchPlan.FILTER_FIRST_EXACT
        else ""
    )
    exclusion_sql = ""
    if exclude_promotion_users:
        exclusion_sql = f"""
          AND NOT EXISTS (
              SELECT 1
              FROM {POSTGRES_EXCLUSION_RELATION} AS excluded
              WHERE excluded.project_id = search.project_id
                AND excluded.promotion_id = %s
                AND excluded.user_id = search.user_id
                AND excluded.state IN ('reserved', 'consumed')
          )
        """
    return prefix + f"""
        SELECT search.user_id,
               1 - (search.embedding <=> %s::vector) AS behavior_fit_score
        FROM user_behavior_vector_search AS search
        {hard_join}
        WHERE search.project_id = %s
          AND search.vector_version = %s
          AND search.vector_dim = 64
          AND search.window_end = %s
          AND search.vector_generation_id = %s
          {exclusion_sql}
          AND 1 - (search.embedding <=> %s::vector) >= %s
        ORDER BY behavior_fit_score DESC, search.user_id ASC
    """


def _exact_plan_params(
    *,
    manifest: BenchmarkManifest,
    scenario: BenchmarkScenario,
    context: Any,
    exclusion_promotion_id: str | None,
) -> tuple[Any, ...]:
    vector = _vector_literal(scenario.query_vector)
    exclusion_params: tuple[Any, ...] = (
        (exclusion_promotion_id,) if exclusion_promotion_id is not None else ()
    )
    return (
        vector,
        manifest.project_id,
        manifest.vector_version,
        context.source_cutoff,
        context.vector_generation_id,
        *exclusion_params,
        vector,
        scenario.score_threshold,
    )


def _ann_sql(*, explain: bool, exclude_promotion_users: bool) -> str:
    prefix = (
        "EXPLAIN (ANALYZE, BUFFERS, WAL, SETTINGS, FORMAT JSON) "
        if explain
        else ""
    )
    exclusion_sql = ""
    if exclude_promotion_users:
        exclusion_sql = f"""
          AND NOT EXISTS (
              SELECT 1
              FROM {POSTGRES_EXCLUSION_RELATION} AS excluded
              WHERE excluded.project_id = search.project_id
                AND excluded.promotion_id = %s
                AND excluded.user_id = search.user_id
                AND excluded.state IN ('reserved', 'consumed')
          )
        """
    return prefix + f"""
        SELECT search.user_id,
               1 - (search.embedding <=> %s::vector) AS behavior_fit_score
        FROM user_behavior_vector_search AS search
        WHERE search.project_id = %s
          AND search.vector_version = %s
          AND search.vector_dim = 64
          AND search.window_end = %s
          AND search.vector_generation_id = %s
          {exclusion_sql}
        ORDER BY search.embedding <=> %s::vector
        LIMIT %s
    """


def _ann_params(
    *,
    manifest: BenchmarkManifest,
    scenario: BenchmarkScenario,
    context: Any,
    requested_k: int,
    exclusion_promotion_id: str | None,
) -> tuple[Any, ...]:
    vector = _vector_literal(scenario.query_vector)
    exclusion_params: tuple[Any, ...] = (
        (exclusion_promotion_id,) if exclusion_promotion_id is not None else ()
    )
    return (
        vector,
        manifest.project_id,
        manifest.vector_version,
        context.source_cutoff,
        context.vector_generation_id,
        *exclusion_params,
        vector,
        requested_k,
    )


def _minimum_rank_for_recall_gate(
    exact_members: Sequence[SearchCandidate],
    *,
    target_recall: float,
) -> int | None:
    if not exact_members:
        return None
    ranks = sorted(member.retrieval_rank for member in exact_members)
    for retrieved_count, rank in enumerate(ranks, start=1):
        recall = retrieved_count / len(ranks)
        lower = wilson_lower_bound(
            successes=retrieved_count,
            trials=len(ranks),
            confidence=0.95,
        )
        if recall >= target_recall and lower >= target_recall:
            return rank
    return None


def _named_rows(result: Any) -> list[Mapping[str, Any]]:
    if hasattr(result, "named_results"):
        return list(result.named_results())
    column_names = list(getattr(result, "column_names", ()))
    return [
        dict(zip(column_names, row, strict=True))
        for row in getattr(result, "result_rows", ())
    ]


def _candidates(rows: Sequence[Mapping[str, Any]]) -> list[SearchCandidate]:
    return [
        SearchCandidate(
            user_id=str(row["user_id"]),
            behavior_fit_score=float(row["behavior_fit_score"]),
            retrieval_rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]


def _find_plan_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            if child_key == key:
                found.append(child_value)
            found.extend(_find_plan_values(child_value, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_plan_values(child, key))
    return found


class _BackendRssSampler:
    def __init__(
        self,
        process_id: int | None,
        *,
        interval_seconds: float = 0.01,
        postgres_container: str | None = None,
    ) -> None:
        self._process_id = process_id
        self._interval_seconds = interval_seconds
        self._postgres_container = postgres_container
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._docker_sampler: subprocess.Popen[str] | None = None
        self._sample_lock = threading.Lock()
        self.peak_rss_bytes: int | None = None

    @classmethod
    def from_connection(cls, connection: Any) -> "_BackendRssSampler":
        info = getattr(connection, "info", None)
        process_id = getattr(info, "backend_pid", None)
        return cls(
            int(process_id) if process_id is not None else None,
            postgres_container=os.environ.get("ANN_POSTGRES_CONTAINER") or None,
        )

    def __enter__(self) -> "_BackendRssSampler":
        if self._process_id is None:
            return self
        initial = _postgres_rss_bytes(self._process_id)
        if initial is None and self._postgres_container is not None:
            self._docker_sampler = _start_docker_rss_sampler(
                self._postgres_container,
                self._process_id,
            )
            initial = self._sample_docker()
        if initial is None:
            self._close_docker_sampler()
            return self
        self.peak_rss_bytes = initial
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()
        self._close_docker_sampler()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        if self._process_id is None:
            return
        with self._sample_lock:
            value = (
                self._sample_docker()
                if self._docker_sampler is not None
                else _postgres_rss_bytes(self._process_id)
            )
        if value is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, value)

    def _sample_docker(self) -> int | None:
        sampler = self._docker_sampler
        if sampler is None or sampler.stdin is None or sampler.stdout is None:
            return None
        if sampler.poll() is not None:
            return None
        try:
            sampler.stdin.write("sample\n")
            sampler.stdin.flush()
            value = sampler.stdout.readline().strip()
            return int(value) if value else None
        except (BrokenPipeError, OSError, ValueError):
            return None

    def _close_docker_sampler(self) -> None:
        sampler = self._docker_sampler
        self._docker_sampler = None
        if sampler is None:
            return
        if sampler.stdin is not None:
            try:
                sampler.stdin.close()
            except OSError:
                pass
        try:
            sampler.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            sampler.terminate()
            try:
                sampler.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait(timeout=1.0)


def _start_docker_rss_sampler(
    postgres_container: str,
    process_id: int,
) -> subprocess.Popen[str] | None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", postgres_container):
        return None
    script = """
pid="$1"
while IFS= read -r _; do
  if [ ! -r "/proc/$pid/comm" ] || [ ! -r "/proc/$pid/status" ]; then
    printf '\\n'
    continue
  fi
  command=$(cat "/proc/$pid/comm" 2>/dev/null || true)
  case "$command" in
    *postgres*) awk '/^VmRSS:/ { printf "%.0f\\n", $2 * 1024; found=1 }
                       END { if (!found) printf "\\n" }' "/proc/$pid/status" ;;
    *) printf '\\n' ;;
  esac
done
""".strip()
    try:
        return subprocess.Popen(
            (
                "docker",
                "exec",
                "-i",
                postgres_container,
                "sh",
                "-c",
                script,
                "ann-rss-sampler",
                str(process_id),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except OSError:
        return None


def _postgres_rss_bytes(process_id: int) -> int | None:
    proc_dir = Path("/proc") / str(process_id)
    if proc_dir.exists():
        try:
            command = (proc_dir / "comm").read_text(encoding="utf-8").strip().lower()
            if "postgres" not in command:
                return None
            for line in (proc_dir / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (IndexError, OSError, ValueError):
            return None
        return None
    try:
        command = subprocess.run(
            ("ps", "-o", "comm=", "-p", str(process_id)),
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        ).stdout.strip().lower()
        if "postgres" not in command:
            return None
        rss = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(process_id)),
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        ).stdout.strip()
        return int(rss) * 1024 if rss else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
