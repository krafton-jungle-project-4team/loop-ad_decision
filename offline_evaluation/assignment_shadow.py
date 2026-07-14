from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from offline_evaluation.assignment_benchmark import (
    benchmark_code_identity,
    benchmark_execution_environment,
    compare_to_oracle,
    limit_blas_threads,
)
from offline_evaluation.assignment_corpus import FrozenAssignmentCorpus
from offline_evaluation.assignment_matchers import (
    AdaptiveHNSWMatcher,
    ExactOracle,
    FixedHNSWMatcher,
    MatcherConfig,
    MatcherOutput,
)


SHADOW_FORMAT_VERSION = "loopad.assignment-shadow.v3"
SHADOW_WRITE_POLICY = "read_only_report_no_assignment_or_publication_writes"


@dataclass(frozen=True, slots=True)
class AssignmentShadowReport:
    format_version: str
    created_at: str
    corpus_sha256: str
    corpus_manifest: Mapping[str, Any]
    matcher_config: Mapping[str, Any]
    benchmark_code_identity: Mapping[str, Any]
    execution_environment: Mapping[str, Any]
    write_policy: str
    fixed_hnsw_metrics: Mapping[str, Any]
    adaptive_hnsw_metrics: Mapping[str, Any]
    mismatch_example_limit: int
    mismatch_user_count: int
    mismatch_examples_truncated: bool
    mismatch_examples: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_assignment_shadow(
    corpus: FrozenAssignmentCorpus,
    *,
    config: MatcherConfig,
    max_mismatch_examples: int = 100,
) -> AssignmentShadowReport:
    if max_mismatch_examples < 0:
        raise ValueError("max_mismatch_examples must be non-negative")
    with limit_blas_threads(config.blas_threads):
        return _run_assignment_shadow_controlled(
            corpus,
            config=config,
            max_mismatch_examples=max_mismatch_examples,
        )


def _run_assignment_shadow_controlled(
    corpus: FrozenAssignmentCorpus,
    *,
    config: MatcherConfig,
    max_mismatch_examples: int,
) -> AssignmentShadowReport:
    code_identity = benchmark_code_identity()
    exact = ExactOracle(
        segment_ids=corpus.segment_ids,
        segment_vectors=corpus.segment_vectors,
        config=config,
    ).match(user_ids=corpus.user_ids, user_vectors=corpus.user_vectors)
    fixed = FixedHNSWMatcher(
        segment_ids=corpus.segment_ids,
        segment_vectors=corpus.segment_vectors,
        config=config,
    ).match(user_ids=corpus.user_ids, user_vectors=corpus.user_vectors)
    adaptive = AdaptiveHNSWMatcher(
        segment_ids=corpus.segment_ids,
        segment_vectors=corpus.segment_vectors,
        config=config,
    ).match(user_ids=corpus.user_ids, user_vectors=corpus.user_vectors)
    fixed_metrics = compare_to_oracle(exact, fixed)
    adaptive_metrics = compare_to_oracle(exact, adaptive)
    np = __import__("numpy")
    fixed_equal_mask = _assignment_equal_mask(exact, fixed)
    adaptive_equal_mask = _assignment_equal_mask(exact, adaptive)
    mismatch_indices = np.flatnonzero(
        np.logical_not(np.logical_and(fixed_equal_mask, adaptive_equal_mask))
    )
    mismatch_user_count = int(mismatch_indices.size)
    examples: list[Mapping[str, Any]] = []
    for raw_index in mismatch_indices[:max_mismatch_examples]:
        index = int(raw_index)
        user_id = corpus.user_ids[index]
        fixed_equal = _assignment_equal(exact, fixed, index)
        adaptive_equal = _assignment_equal(exact, adaptive, index)
        examples.append(
            {
                "user_id": user_id,
                "exact": _result_payload(exact, index),
                "fixed_hnsw": _result_payload(fixed, index),
                "adaptive_hnsw": _result_payload(adaptive, index),
                "rescue_reasons": list(adaptive.rescue_reasons[index]),
                "fixed_assignment_agrees": fixed_equal,
                "adaptive_assignment_agrees": adaptive_equal,
            }
        )
    if (
        benchmark_code_identity()["benchmark_source_sha256"]
        != code_identity["benchmark_source_sha256"]
    ):
        raise RuntimeError("benchmark source changed while shadow was running")
    return AssignmentShadowReport(
        format_version=SHADOW_FORMAT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        corpus_sha256=corpus.manifest.corpus_sha256,
        corpus_manifest=corpus.manifest.to_dict(),
        matcher_config=config.to_dict(),
        benchmark_code_identity=code_identity,
        execution_environment=benchmark_execution_environment(
            process_role="shadow_single_process",
            matcher_process_isolation=(
                "single_process_exact_fixed_adaptive_sequential"
            ),
        ),
        write_policy=SHADOW_WRITE_POLICY,
        fixed_hnsw_metrics=asdict(fixed_metrics),
        adaptive_hnsw_metrics=asdict(adaptive_metrics),
        mismatch_example_limit=max_mismatch_examples,
        mismatch_user_count=mismatch_user_count,
        mismatch_examples_truncated=(
            mismatch_user_count > max_mismatch_examples
        ),
        mismatch_examples=tuple(examples),
    )


def write_shadow_report(
    report: AssignmentShadowReport,
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


def _assignment_equal(
    left: MatcherOutput,
    right: MatcherOutput,
    index: int,
) -> bool:
    if bool(left.fallback_mask[index]) != bool(right.fallback_mask[index]):
        return False
    if bool(left.fallback_mask[index]):
        return True
    return int(left.top_segment_indices[index]) == int(right.top_segment_indices[index])


def _assignment_equal_mask(left: MatcherOutput, right: MatcherOutput) -> Any:
    np = __import__("numpy")
    fallback_equal = left.fallback_mask == right.fallback_mask
    segment_equal = left.top_segment_indices == right.top_segment_indices
    return np.logical_and(
        fallback_equal,
        np.logical_or(left.fallback_mask, segment_equal),
    )


def _result_payload(output: MatcherOutput, index: int) -> dict[str, Any]:
    segment_index = int(output.top_segment_indices[index])
    return {
        "top_segment_id": (
            output.segment_ids[segment_index] if segment_index >= 0 else None
        ),
        "raw_cosine_score": (
            float(output.raw_scores[index])
            if segment_index >= 0
            else None
        ),
        "fallback": bool(output.fallback_mask[index]),
        "fallback_reason": output.fallback_reasons[index],
        "rescued": bool(output.rescued_mask[index]),
    }
