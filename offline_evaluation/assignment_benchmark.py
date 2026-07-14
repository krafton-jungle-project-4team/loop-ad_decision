from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing
import os
import platform
import resource
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from offline_evaluation.assignment_corpus import (
    FrozenAssignmentCorpus,
    controlled_thread_environment,
    cpu_model,
    thread_settings,
)
from offline_evaluation.assignment_matchers import (
    AdaptiveHNSWMatcher,
    ExactOracle,
    FixedHNSWMatcher,
    MatcherConfig,
    MatcherOutput,
    concatenate_outputs,
    cross_validate_exact_with_faiss,
)


BENCHMARK_FORMAT_VERSION = "loopad.assignment-benchmark.v5"
DISAGREEMENT_BOUND_METHOD = "wilson_one_sided_95"
BENCHMARK_SOURCE_PATHS = (
    "offline_evaluation/assignment_corpus.py",
    "offline_evaluation/assignment_matchers.py",
    "offline_evaluation/assignment_benchmark.py",
    "offline_evaluation/assignment_shadow.py",
    "scripts/export_segment_assignment_corpus.py",
    "scripts/benchmark_segment_assignments.py",
    "pyproject.toml",
)
_SPAWN_ENV_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class AgreementMetrics:
    user_count: int
    exact_top1_recall: float
    pre_rescue_segment_agreement: float
    post_rescue_segment_agreement: float
    fallback_agreement: float
    assignment_agreement: float
    oracle_fallback_count: int
    oracle_fallback_rate: float
    candidate_fallback_count: int
    candidate_fallback_rate: float
    false_fallback_count: int
    false_nonfallback_count: int
    rescue_count: int
    rescue_rate: float
    rescue_reason_counts: Mapping[str, int]
    observed_disagreement_count: int
    observed_disagreement_rate: float
    disagreement_rate_upper_95: float
    disagreement_bound_method: str = DISAGREEMENT_BOUND_METHOD


@dataclass(frozen=True, slots=True)
class TimingMetrics:
    index_build_seconds: float
    match_seconds: float
    output_assembly_seconds: float
    end_to_end_seconds: float
    end_to_end_definition: str
    end_to_end_p50_ms: float
    end_to_end_p95_ms: float
    end_to_end_sample_count: int
    end_to_end_sample_definition: str
    percentile_method: str
    trial_samples: tuple[Mapping[str, float | int], ...]
    match_users_per_second: float
    match_batch_latency_p50_ms: float
    match_batch_latency_p95_ms: float
    latency_sample_count: int
    latency_sample_definition: str
    peak_rss_bytes: int
    peak_rss_definition: str


@dataclass(frozen=True, slots=True)
class _TrialTiming:
    index_build_seconds: float
    match_seconds: float
    output_assembly_seconds: float
    end_to_end_seconds: float
    match_users_per_second: float
    batch_latency_ms: tuple[float, ...]
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class MatcherBenchmark:
    matcher: str
    agreement: AgreementMetrics
    timing: TimingMetrics
    execution_environment: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    format_version: str
    created_at: str
    corpus_manifest: Mapping[str, Any]
    matcher_config: Mapping[str, Any]
    requested_warmup_user_count: int
    warmup_user_count: int
    timing_trial_count: int
    timing_trial_schedule: tuple[tuple[str, ...], ...]
    benchmark_code_identity: Mapping[str, Any]
    execution_environment: Mapping[str, Any]
    faiss_flat_crosscheck: str
    matchers: tuple[MatcherBenchmark, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(
    corpus: FrozenAssignmentCorpus,
    *,
    config: MatcherConfig,
    warmup_user_count: int = 16,
    timing_trial_count: int = 3,
) -> BenchmarkReport:
    if warmup_user_count < 0:
        raise ValueError("warmup_user_count must be non-negative")
    if timing_trial_count < 2:
        raise ValueError("timing_trial_count must be at least 2")
    effective_warmup_user_count = min(
        warmup_user_count,
        corpus.manifest.user_count,
    )
    code_identity = benchmark_code_identity()
    trial_results, trial_schedule = _run_interleaved_matcher_trials(
        corpus=corpus,
        config=config,
        warmup_user_count=effective_warmup_user_count,
        timing_trial_count=timing_trial_count,
    )
    exact_output, exact_timing, crosscheck, exact_environment = trial_results.pop(
        "exact"
    )
    fixed_output, fixed_timing, _, fixed_environment = trial_results.pop(
        "fixed"
    )
    fixed_agreement = compare_to_oracle(exact_output, fixed_output)
    del fixed_output
    gc.collect()
    adaptive_output, adaptive_timing, _, adaptive_environment = trial_results.pop(
        "adaptive"
    )
    adaptive_agreement = compare_to_oracle(exact_output, adaptive_output)
    if (
        benchmark_code_identity()["benchmark_source_sha256"]
        != code_identity["benchmark_source_sha256"]
    ):
        raise RuntimeError("benchmark source changed while the benchmark was running")

    return BenchmarkReport(
        format_version=BENCHMARK_FORMAT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        corpus_manifest=corpus.manifest.to_dict(),
        matcher_config=config.to_dict(),
        requested_warmup_user_count=warmup_user_count,
        warmup_user_count=effective_warmup_user_count,
        timing_trial_count=timing_trial_count,
        timing_trial_schedule=trial_schedule,
        benchmark_code_identity=code_identity,
        execution_environment=benchmark_execution_environment(
            process_role="benchmark_orchestrator_parent",
        ),
        faiss_flat_crosscheck=crosscheck,
        matchers=(
            MatcherBenchmark(
                matcher="exact_numpy_batched_matmul",
                agreement=compare_to_oracle(exact_output, exact_output),
                timing=exact_timing,
                execution_environment=exact_environment,
            ),
            MatcherBenchmark(
                matcher="fixed_faiss_hnsw_exact_candidate_rerank",
                agreement=fixed_agreement,
                timing=fixed_timing,
                execution_environment=fixed_environment,
            ),
            MatcherBenchmark(
                matcher="adaptive_faiss_hnsw_exact_rescue",
                agreement=adaptive_agreement,
                timing=adaptive_timing,
                execution_environment=adaptive_environment,
            ),
        ),
    )


def _run_interleaved_matcher_trials(
    *,
    corpus: FrozenAssignmentCorpus,
    config: MatcherConfig,
    warmup_user_count: int,
    timing_trial_count: int,
) -> tuple[
    dict[
        str,
        tuple[MatcherOutput, TimingMetrics, str, Mapping[str, Any]],
    ],
    tuple[tuple[str, ...], ...],
]:
    matcher_kinds = ("exact", "fixed", "adaptive")
    outputs: dict[str, MatcherOutput] = {}
    timings: dict[str, list[_TrialTiming]] = {
        matcher_kind: [] for matcher_kind in matcher_kinds
    }
    crosschecks: dict[str, str] = {}
    environments: dict[str, list[Mapping[str, Any]]] = {
        matcher_kind: [] for matcher_kind in matcher_kinds
    }
    schedule: list[tuple[str, ...]] = []
    for trial_index in range(timing_trial_count):
        rotation = trial_index % len(matcher_kinds)
        trial_order = matcher_kinds[rotation:] + matcher_kinds[:rotation]
        schedule.append(trial_order)
        for matcher_kind in trial_order:
            trial_output, timing, trial_crosscheck, environment = _run_isolated_matcher(
                matcher_kind,
                corpus=corpus,
                config=config,
                warmup_user_count=warmup_user_count,
            )
            if matcher_kind in outputs:
                if not _matcher_outputs_identical(
                    outputs[matcher_kind],
                    trial_output,
                ):
                    raise RuntimeError(
                        f"isolated {matcher_kind} benchmark output changed "
                        "across timing trials"
                    )
                if trial_crosscheck != crosschecks[matcher_kind]:
                    raise RuntimeError(
                        f"isolated {matcher_kind} cross-check changed "
                        "across timing trials"
                    )
                del trial_output
            else:
                outputs[matcher_kind] = trial_output
                crosschecks[matcher_kind] = trial_crosscheck
            timings[matcher_kind].append(timing)
            environments[matcher_kind].append(environment)
        gc.collect()
    return (
        {
            matcher_kind: (
                outputs[matcher_kind],
                _aggregate_trial_timings(
                    timings[matcher_kind],
                    batch_size=config.exact_batch_size,
                ),
                crosschecks[matcher_kind],
                _aggregate_trial_environments(environments[matcher_kind]),
            )
            for matcher_kind in matcher_kinds
        },
        tuple(schedule),
    )


def _run_isolated_matcher(
    matcher_kind: str,
    *,
    corpus: FrozenAssignmentCorpus,
    config: MatcherConfig,
    warmup_user_count: int,
) -> tuple[MatcherOutput, _TrialTiming, str, Mapping[str, Any]]:
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_matcher_worker_entrypoint,
        args=(
            send_connection,
            matcher_kind,
            corpus,
            config,
            warmup_user_count,
        ),
    )
    with _temporary_thread_environment(config.blas_threads):
        process.start()
    send_connection.close()
    try:
        outcome, payload = receive_connection.recv()
    except EOFError as exc:
        process.join()
        raise RuntimeError(
            f"isolated {matcher_kind} benchmark exited without a result "
            f"(exit code {process.exitcode})"
        ) from exc
    finally:
        receive_connection.close()
    process.join()
    if outcome == "error":
        raise RuntimeError(
            f"isolated {matcher_kind} benchmark failed:\n{payload}"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            f"isolated {matcher_kind} benchmark exited with code {process.exitcode}"
        )
    return payload


def _run_matcher_worker_entrypoint(
    connection: Any,
    matcher_kind: str,
    corpus: FrozenAssignmentCorpus,
    config: MatcherConfig,
    warmup_user_count: int,
) -> None:
    try:
        result = _run_matcher_worker(
            matcher_kind,
            corpus,
            config,
            warmup_user_count,
        )
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    else:
        connection.send(("ok", result))
    finally:
        connection.close()


def _run_matcher_worker(
    matcher_kind: str,
    corpus: FrozenAssignmentCorpus,
    config: MatcherConfig,
    warmup_user_count: int,
) -> tuple[MatcherOutput, _TrialTiming, str, Mapping[str, Any]]:
    factories: dict[str, Callable[[], Any]] = {
        "exact": lambda: ExactOracle(
            segment_ids=corpus.segment_ids,
            segment_vectors=corpus.segment_vectors,
            config=config,
        ),
        "fixed": lambda: FixedHNSWMatcher(
            segment_ids=corpus.segment_ids,
            segment_vectors=corpus.segment_vectors,
            config=config,
        ),
        "adaptive": lambda: AdaptiveHNSWMatcher(
            segment_ids=corpus.segment_ids,
            segment_vectors=corpus.segment_vectors,
            config=config,
        ),
    }
    try:
        factory = factories[matcher_kind]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark matcher: {matcher_kind}") from exc
    _numpy()
    with limit_blas_threads(config.blas_threads):
        if matcher_kind in {"fixed", "adaptive"}:
            __import__("faiss").omp_set_num_threads(config.faiss_threads)
        matcher, build_seconds, build_peak_rss = _build_matcher(factory)
        output, timing = _time_matcher(
            matcher,
            corpus=corpus,
            build_seconds=build_seconds,
            build_peak_rss=build_peak_rss,
            warmup_user_count=warmup_user_count,
            batch_size=config.exact_batch_size,
        )
        crosscheck = "not_applicable"
        if matcher_kind == "exact":
            pairs = corpus.manifest.user_count * corpus.manifest.segment_count
            crosscheck = "not_run_pair_limit"
            if pairs <= config.flat_crosscheck_max_pairs:
                __import__("faiss").omp_set_num_threads(config.faiss_threads)
                cross_validate_exact_with_faiss(
                    oracle=matcher,
                    user_ids=corpus.user_ids,
                    user_vectors=corpus.user_vectors,
                )
                crosscheck = "passed"
        return output, timing, crosscheck, benchmark_execution_environment(
            process_role="matcher_spawn_child",
        )


@contextmanager
def limit_blas_threads(thread_count: int):
    if thread_count <= 0:
        raise ValueError("thread_count must be positive")
    threadpoolctl = __import__("threadpoolctl")
    with threadpoolctl.threadpool_limits(
        limits=thread_count,
        user_api="blas",
    ):
        yield


@contextmanager
def _temporary_thread_environment(thread_count: int):
    with _SPAWN_ENV_LOCK:
        controlled = controlled_thread_environment(thread_count)
        previous = {name: os.environ.get(name) for name in controlled}
        os.environ.update(controlled)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def compare_to_oracle(
    oracle: MatcherOutput,
    candidate: MatcherOutput,
) -> AgreementMetrics:
    np = _numpy()
    if oracle.user_ids != candidate.user_ids or oracle.segment_ids != candidate.segment_ids:
        raise ValueError("oracle and candidate outputs must use identical IDs and order")
    user_count = oracle.user_count
    if user_count == 0:
        raise ValueError("benchmark outputs must not be empty")
    pre_indices = (
        candidate.pre_rescue_top_segment_indices
        if candidate.pre_rescue_top_segment_indices is not None
        else candidate.top_segment_indices
    )
    pre_agreement = pre_indices == oracle.top_segment_indices
    post_agreement = candidate.top_segment_indices == oracle.top_segment_indices
    fallback_agreement = candidate.fallback_mask == oracle.fallback_mask
    assignment_agreement = np.logical_and(
        fallback_agreement,
        np.logical_or(oracle.fallback_mask, post_agreement),
    )
    false_fallback = np.logical_and(candidate.fallback_mask, ~oracle.fallback_mask)
    false_nonfallback = np.logical_and(~candidate.fallback_mask, oracle.fallback_mask)
    disagreement = ~assignment_agreement
    if candidate.candidate_indices is None:
        recall = np.ones(user_count, dtype=np.bool_)
    else:
        recall = np.any(
            candidate.candidate_indices == oracle.top_segment_indices[:, None],
            axis=1,
        )
    reason_counts: dict[str, int] = {}
    for reasons in candidate.rescue_reasons:
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    disagreement_count = int(np.count_nonzero(disagreement))
    oracle_fallback_count = int(np.count_nonzero(oracle.fallback_mask))
    candidate_fallback_count = int(np.count_nonzero(candidate.fallback_mask))
    return AgreementMetrics(
        user_count=user_count,
        exact_top1_recall=float(np.mean(recall)),
        pre_rescue_segment_agreement=float(np.mean(pre_agreement)),
        post_rescue_segment_agreement=float(np.mean(post_agreement)),
        fallback_agreement=float(np.mean(fallback_agreement)),
        assignment_agreement=float(np.mean(assignment_agreement)),
        oracle_fallback_count=oracle_fallback_count,
        oracle_fallback_rate=oracle_fallback_count / user_count,
        candidate_fallback_count=candidate_fallback_count,
        candidate_fallback_rate=candidate_fallback_count / user_count,
        false_fallback_count=int(np.count_nonzero(false_fallback)),
        false_nonfallback_count=int(np.count_nonzero(false_nonfallback)),
        rescue_count=int(np.count_nonzero(candidate.rescued_mask)),
        rescue_rate=float(np.mean(candidate.rescued_mask)),
        rescue_reason_counts=reason_counts,
        observed_disagreement_count=disagreement_count,
        observed_disagreement_rate=disagreement_count / user_count,
        disagreement_rate_upper_95=one_sided_wilson_upper(
            disagreement_count,
            user_count,
        ),
    )


def one_sided_wilson_upper(errors: int, sample_size: int) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if errors < 0 or errors > sample_size:
        raise ValueError("errors must be in [0, sample_size]")
    z = 1.6448536269514722
    observed = errors / sample_size
    z_squared = z * z
    denominator = 1.0 + z_squared / sample_size
    center = observed + z_squared / (2.0 * sample_size)
    radius = z * math.sqrt(
        observed * (1.0 - observed) / sample_size
        + z_squared / (4.0 * sample_size * sample_size)
    )
    return min(1.0, (center + radius) / denominator)


def write_benchmark_report(
    report: BenchmarkReport,
    destination: Path,
    *,
    overwrite: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8") as output:
        output.write(json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n")


def _build_matcher(factory: Callable[[], Any]) -> tuple[Any, float, int]:
    monitor = _PeakRSSMonitor()
    start = time.perf_counter()
    with monitor:
        matcher = factory()
    return matcher, time.perf_counter() - start, monitor.peak_rss_bytes


def _time_matcher(
    matcher: Any,
    *,
    corpus: FrozenAssignmentCorpus,
    build_seconds: float,
    build_peak_rss: int,
    warmup_user_count: int,
    batch_size: int,
) -> tuple[MatcherOutput, _TrialTiming]:
    if warmup_user_count > 0:
        stop = min(warmup_user_count, corpus.manifest.user_count)
        matcher.match(
            user_ids=corpus.user_ids[:stop],
            user_vectors=corpus.user_vectors[:stop],
        )
    outputs: list[MatcherOutput] = []
    batch_latency_ms: list[float] = []
    total_seconds = 0.0
    monitor = _PeakRSSMonitor()
    with monitor:
        for start in range(0, corpus.manifest.user_count, batch_size):
            stop = min(start + batch_size, corpus.manifest.user_count)
            batch_started = time.perf_counter()
            output = matcher.match(
                user_ids=corpus.user_ids[start:stop],
                user_vectors=corpus.user_vectors[start:stop],
            )
            elapsed = time.perf_counter() - batch_started
            outputs.append(output)
            total_seconds += elapsed
            batch_latency_ms.append(elapsed * 1000.0)
        assembly_started = time.perf_counter()
        combined = concatenate_outputs(outputs)
        output_assembly_seconds = time.perf_counter() - assembly_started
    match_users_per_second = (
        corpus.manifest.user_count / total_seconds if total_seconds > 0.0 else math.inf
    )
    return combined, _TrialTiming(
        index_build_seconds=build_seconds,
        match_seconds=total_seconds,
        output_assembly_seconds=output_assembly_seconds,
        end_to_end_seconds=(
            build_seconds + total_seconds + output_assembly_seconds
        ),
        match_users_per_second=match_users_per_second,
        batch_latency_ms=tuple(batch_latency_ms),
        peak_rss_bytes=max(build_peak_rss, monitor.peak_rss_bytes),
    )


def _aggregate_trial_timings(
    trials: Sequence[_TrialTiming],
    *,
    batch_size: int,
) -> TimingMetrics:
    if not trials:
        raise ValueError("at least one timing trial is required")
    np = _numpy()

    def percentile(values: Sequence[float], quantile: float) -> float:
        return float(
            np.percentile(
                np.asarray(values, dtype=np.float64),
                quantile,
                method="linear",
            )
        )

    end_to_end_ms = [trial.end_to_end_seconds * 1000.0 for trial in trials]
    pooled_batch_latency_ms = [
        latency
        for trial in trials
        for latency in trial.batch_latency_ms
    ]
    return TimingMetrics(
        index_build_seconds=percentile(
            [trial.index_build_seconds for trial in trials], 50
        ),
        match_seconds=percentile([trial.match_seconds for trial in trials], 50),
        output_assembly_seconds=percentile(
            [trial.output_assembly_seconds for trial in trials], 50
        ),
        end_to_end_seconds=percentile(
            [trial.end_to_end_seconds for trial in trials], 50
        ),
        end_to_end_definition=(
            "median of independent fresh-child trials: warmup-excluded index "
            "construction + all measured match batches + output assembly inside "
            "the already-started child; excludes spawn startup, corpus IPC/"
            "deserialization, oracle cross-check, and result IPC"
        ),
        end_to_end_p50_ms=percentile(end_to_end_ms, 50),
        end_to_end_p95_ms=percentile(end_to_end_ms, 95),
        end_to_end_sample_count=len(trials),
        end_to_end_sample_definition=(
            "one sample per independent fresh spawn child, from index build start "
            "through measured match batches and output assembly; exclusions match "
            "end_to_end_definition"
        ),
        percentile_method="numpy_linear",
        trial_samples=tuple(
            {
                "trial_index": index,
                "index_build_seconds": trial.index_build_seconds,
                "match_seconds": trial.match_seconds,
                "output_assembly_seconds": trial.output_assembly_seconds,
                "end_to_end_seconds": trial.end_to_end_seconds,
                "match_users_per_second": trial.match_users_per_second,
                "peak_rss_bytes": trial.peak_rss_bytes,
            }
            for index, trial in enumerate(trials, start=1)
        ),
        match_users_per_second=percentile(
            [trial.match_users_per_second for trial in trials], 50
        ),
        match_batch_latency_p50_ms=percentile(pooled_batch_latency_ms, 50),
        match_batch_latency_p95_ms=percentile(pooled_batch_latency_ms, 95),
        latency_sample_count=len(pooled_batch_latency_ms),
        latency_sample_definition=(
            "pooled warmup-excluded matcher.match wall time per configured user "
            f"batch (up to {batch_size} users) across all timing trials; excludes "
            "one-time index build and output assembly"
        ),
        peak_rss_bytes=max(trial.peak_rss_bytes for trial in trials),
        peak_rss_definition=(
            "maximum absolute ru_maxrss across fresh matcher trial children after "
            "corpus deserialization; includes normalized vectors, index, per-batch "
            "outputs, and combined output; excludes parent and other matcher processes"
        ),
    )


def _aggregate_trial_environments(
    environments: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not environments:
        raise ValueError("at least one trial environment is required")
    stable = {
        key: value
        for key, value in environments[0].items()
        if key != "process_id"
    }
    for environment in environments[1:]:
        candidate = {
            key: value
            for key, value in environment.items()
            if key != "process_id"
        }
        if candidate != stable:
            raise RuntimeError("matcher execution environment changed across trials")
    return {
        **environments[0],
        "timing_trial_process_count": len(environments),
        "trial_process_ids": [
            int(environment["process_id"]) for environment in environments
        ],
    }


def _matcher_outputs_identical(left: MatcherOutput, right: MatcherOutput) -> bool:
    if (
        left.user_ids != right.user_ids
        or left.segment_ids != right.segment_ids
        or left.fallback_reasons != right.fallback_reasons
        or left.rescue_reasons != right.rescue_reasons
    ):
        return False
    np = _numpy()
    for field_name in (
        "top_segment_indices",
        "raw_scores",
        "fallback_mask",
        "candidate_indices",
        "candidate_counts",
        "rescued_mask",
        "pre_rescue_top_segment_indices",
        "pre_rescue_raw_scores",
        "pre_rescue_fallback_mask",
    ):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        if left_value is None or right_value is None:
            if left_value is not right_value:
                return False
            continue
        if not bool(np.array_equal(left_value, right_value, equal_nan=True)):
            return False
    return True


class _PeakRSSMonitor:
    def __init__(self) -> None:
        self.peak_rss_bytes = _resource_peak_rss_bytes()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            psutil = __import__("psutil")
            self._process = psutil.Process(os.getpid())
        except ImportError:
            self._process = None

    def __enter__(self) -> _PeakRSSMonitor:
        if self._process is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_bytes = max(self.peak_rss_bytes, _resource_peak_rss_bytes())

    def _sample(self) -> None:
        while not self._stop.wait(0.005):
            try:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    int(self._process.memory_info().rss),
                )
            except Exception:
                return


def _resource_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def benchmark_execution_environment(
    *,
    process_role: str,
    matcher_process_isolation: str = "fresh_spawn_child_per_matcher_trial",
) -> dict[str, Any]:
    np = _numpy()
    try:
        faiss_version = str(__import__("faiss").__version__)
    except ImportError:
        faiss_version = None
    try:
        psutil_version = str(__import__("psutil").__version__)
    except ImportError:
        psutil_version = None
    return {
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "faiss_version": faiss_version,
        "psutil_version": psutil_version,
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "thread_settings": thread_settings(),
        "process_role": process_role,
        "process_id": os.getpid(),
        "matcher_process_isolation": matcher_process_isolation,
        "rss_shared_page_note": (
            "RSS is OS-reported resident memory in a spawn child; shared library "
            "pages may be counted in each child"
        ),
    }


def benchmark_code_identity(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    root = repository_root or Path(__file__).resolve().parents[1]
    file_hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for relative_path in BENCHMARK_SOURCE_PATHS:
        payload = (root / relative_path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        file_hashes[relative_path] = digest
        encoded_path = relative_path.encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(4, "big"))
        aggregate.update(encoded_path)
        aggregate.update(bytes.fromhex(digest))
    status = _git_output(
        root,
        "status",
        "--porcelain",
        "--",
        *BENCHMARK_SOURCE_PATHS,
    )
    return {
        "git_commit": _git_output(root, "rev-parse", "HEAD"),
        "git_tree": _git_output(root, "rev-parse", "HEAD^{tree}"),
        "git_branch": _git_output(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "benchmark_sources_dirty": None if status is None else bool(status),
        "benchmark_source_sha256": aggregate.hexdigest(),
        "benchmark_source_files_sha256": file_hashes,
    }


def _git_output(repository_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _numpy() -> Any:
    try:
        return __import__("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "assignment benchmark requires the 'assignment-benchmark' optional extra"
        ) from exc
