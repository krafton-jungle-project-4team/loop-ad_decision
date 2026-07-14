from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from offline_evaluation.assignment_corpus import VECTOR_DIM, normalize_float32_matrix


SIMILARITY_THRESHOLD = 0.65
FALLBACK_REASON_BELOW_THRESHOLD = "below_threshold"
FALLBACK_REASON_NO_CANDIDATE = "no_candidate"
RESCUE_UNDERFILL = "candidate_underfill"
RESCUE_THRESHOLD_BAND = "threshold_band"
RESCUE_LOW_MARGIN = "low_margin"
RESCUE_INVALID_ANN = "invalid_ann_result"


class InvalidANNResultError(ValueError):
    """Raised when FAISS returns a structurally unusable result batch."""


@dataclass(frozen=True, slots=True)
class MatcherConfig:
    dimension: int = VECTOR_DIM
    threshold: float = SIMILARITY_THRESHOLD
    exact_batch_size: int = 1024
    candidate_k: int = 50
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    rescue_threshold_band: float = 0.02
    rescue_margin: float = 0.02
    faiss_threads: int = 1
    blas_threads: int = 1
    flat_crosscheck_max_pairs: int = 1_000_000

    def __post_init__(self) -> None:
        if self.dimension != VECTOR_DIM:
            raise ValueError("assignment matcher dimension must be 64")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.exact_batch_size <= 0:
            raise ValueError("exact_batch_size must be positive")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if self.hnsw_m <= 0 or self.hnsw_ef_construction <= 0 or self.hnsw_ef_search <= 0:
            raise ValueError("HNSW M and ef values must be positive")
        if self.rescue_threshold_band < 0.0 or self.rescue_margin < 0.0:
            raise ValueError("rescue band and margin must be non-negative")
        if self.rescue_margin > 0.0 and self.candidate_k < 2:
            raise ValueError("candidate_k must be at least 2 when margin rescue is enabled")
        if self.faiss_threads <= 0:
            raise ValueError("faiss_threads must be positive")
        if self.blas_threads <= 0:
            raise ValueError("blas_threads must be positive")
        if self.flat_crosscheck_max_pairs < 0:
            raise ValueError("flat_crosscheck_max_pairs must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatcherOutput:
    user_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    top_segment_indices: Any
    raw_scores: Any
    fallback_mask: Any
    fallback_reasons: tuple[str | None, ...]
    candidate_indices: Any | None
    candidate_counts: Any
    rescued_mask: Any
    rescue_reasons: tuple[tuple[str, ...], ...]
    pre_rescue_top_segment_indices: Any | None = None
    pre_rescue_raw_scores: Any | None = None
    pre_rescue_fallback_mask: Any | None = None

    @property
    def user_count(self) -> int:
        return len(self.user_ids)


class ExactOracle:
    def __init__(
        self,
        *,
        segment_ids: Sequence[str],
        segment_vectors: Any,
        config: MatcherConfig,
    ) -> None:
        np = _numpy()
        if not segment_ids:
            raise ValueError("exact oracle requires at least one segment")
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("segment IDs must be unique")
        segments = normalize_float32_matrix(segment_vectors, name="segment_vectors")
        if len(segment_ids) != segments.shape[0]:
            raise ValueError("segment ID count must match segment vectors")
        order = np.argsort(np.asarray(tuple(str(value) for value in segment_ids)), kind="stable")
        self.segment_ids = tuple(str(segment_ids[int(index)]) for index in order)
        self.segment_vectors = np.ascontiguousarray(segments[order], dtype=np.float32)
        self.config = config

    def match(self, *, user_ids: Sequence[str], user_vectors: Any) -> MatcherOutput:
        np = _numpy()
        users = normalize_float32_matrix(user_vectors, name="user_vectors")
        if len(user_ids) != users.shape[0]:
            raise ValueError("user ID count must match user vectors")
        top_indices = np.empty(users.shape[0], dtype=np.int32)
        top_scores = np.empty(users.shape[0], dtype=np.float32)
        for start in range(0, users.shape[0], self.config.exact_batch_size):
            stop = min(start + self.config.exact_batch_size, users.shape[0])
            scores = users[start:stop] @ self.segment_vectors.T
            indices = np.argmax(scores, axis=1)
            top_indices[start:stop] = indices.astype(np.int32, copy=False)
            top_scores[start:stop] = scores[np.arange(stop - start), indices]
        fallback = top_scores < self.config.threshold
        reasons = tuple(
            FALLBACK_REASON_BELOW_THRESHOLD if bool(value) else None
            for value in fallback
        )
        return MatcherOutput(
            user_ids=tuple(str(value) for value in user_ids),
            segment_ids=self.segment_ids,
            top_segment_indices=top_indices,
            raw_scores=top_scores,
            fallback_mask=fallback,
            fallback_reasons=reasons,
            candidate_indices=None,
            candidate_counts=np.full(users.shape[0], len(self.segment_ids), dtype=np.int32),
            rescued_mask=np.zeros(users.shape[0], dtype=np.bool_),
            rescue_reasons=tuple(() for _ in user_ids),
        )


class FixedHNSWMatcher:
    def __init__(
        self,
        *,
        segment_ids: Sequence[str],
        segment_vectors: Any,
        config: MatcherConfig,
    ) -> None:
        np = _numpy()
        faiss = _faiss()
        exact = ExactOracle(
            segment_ids=segment_ids,
            segment_vectors=segment_vectors,
            config=config,
        )
        self.segment_ids = exact.segment_ids
        self.segment_vectors = exact.segment_vectors
        self.config = config
        faiss.omp_set_num_threads(config.faiss_threads)
        self.index = faiss.IndexHNSWFlat(
            config.dimension,
            config.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.index.hnsw.efConstruction = config.hnsw_ef_construction
        self.index.hnsw.efSearch = config.hnsw_ef_search
        self.index.add(np.ascontiguousarray(self.segment_vectors, dtype=np.float32))

    def match(self, *, user_ids: Sequence[str], user_vectors: Any) -> MatcherOutput:
        np = _numpy()
        users = normalize_float32_matrix(user_vectors, name="user_vectors")
        if len(user_ids) != users.shape[0]:
            raise ValueError("user ID count must match user vectors")
        expected_candidates = min(self.config.candidate_k, len(self.segment_ids))
        distances, labels = self.index.search(users, expected_candidates)
        return _rerank_ann_candidates(
            user_ids=user_ids,
            users=users,
            segment_ids=self.segment_ids,
            segment_vectors=self.segment_vectors,
            distances=distances,
            labels=labels,
            config=self.config,
        )


class AdaptiveHNSWMatcher(FixedHNSWMatcher):
    def match(self, *, user_ids: Sequence[str], user_vectors: Any) -> MatcherOutput:
        np = _numpy()
        users = normalize_float32_matrix(user_vectors, name="user_vectors")
        try:
            fixed = super().match(user_ids=user_ids, user_vectors=users)
        except InvalidANNResultError:
            return _rescue_invalid_ann_batch(
                user_ids=user_ids,
                users=users,
                segment_ids=self.segment_ids,
                segment_vectors=self.segment_vectors,
                config=self.config,
            )
        expected = min(self.config.candidate_k, len(self.segment_ids))
        reasons: list[tuple[str, ...]] = []
        for index in range(len(user_ids)):
            invalid = (
                fixed.fallback_reasons[index] == FALLBACK_REASON_NO_CANDIDATE
                or RESCUE_INVALID_ANN in fixed.rescue_reasons[index]
            )
            score = float(fixed.raw_scores[index])
            candidate_row = fixed.candidate_indices[index]
            valid_labels = [int(value) for value in candidate_row if int(value) >= 0]
            if len(valid_labels) != len(set(valid_labels)):
                invalid = True
            margin = _candidate_margin(
                user=users[index],
                candidate_indices=valid_labels,
                segment_vectors=self.segment_vectors,
            )
            reasons.append(
                adaptive_rescue_reasons(
                    candidate_count=int(fixed.candidate_counts[index]),
                    expected_candidate_count=expected,
                    segment_count=len(self.segment_ids),
                    top_score=score,
                    top_two_margin=margin,
                    invalid_ann=invalid,
                    config=self.config,
                )
            )
        rescued = np.asarray([bool(value) for value in reasons], dtype=np.bool_)
        if not bool(np.any(rescued)):
            return MatcherOutput(
                **{
                    **_output_payload(fixed),
                    "rescued_mask": rescued,
                    "rescue_reasons": tuple(reasons),
                    "pre_rescue_top_segment_indices": fixed.top_segment_indices.copy(),
                    "pre_rescue_raw_scores": fixed.raw_scores.copy(),
                    "pre_rescue_fallback_mask": fixed.fallback_mask.copy(),
                }
            )

        rescue_positions = np.flatnonzero(rescued)
        exact = ExactOracle(
            segment_ids=self.segment_ids,
            segment_vectors=self.segment_vectors,
            config=self.config,
        ).match(
            user_ids=[str(user_ids[int(index)]) for index in rescue_positions],
            user_vectors=users[rescue_positions],
        )
        top_indices = fixed.top_segment_indices.copy()
        scores = fixed.raw_scores.copy()
        fallback = fixed.fallback_mask.copy()
        fallback_reasons = list(fixed.fallback_reasons)
        for exact_index, output_index in enumerate(rescue_positions):
            position = int(output_index)
            top_indices[position] = exact.top_segment_indices[exact_index]
            scores[position] = exact.raw_scores[exact_index]
            fallback[position] = exact.fallback_mask[exact_index]
            fallback_reasons[position] = exact.fallback_reasons[exact_index]
        return MatcherOutput(
            user_ids=fixed.user_ids,
            segment_ids=fixed.segment_ids,
            top_segment_indices=top_indices,
            raw_scores=scores,
            fallback_mask=fallback,
            fallback_reasons=tuple(fallback_reasons),
            candidate_indices=fixed.candidate_indices,
            candidate_counts=fixed.candidate_counts,
            rescued_mask=rescued,
            rescue_reasons=tuple(reasons),
            pre_rescue_top_segment_indices=fixed.top_segment_indices.copy(),
            pre_rescue_raw_scores=fixed.raw_scores.copy(),
            pre_rescue_fallback_mask=fixed.fallback_mask.copy(),
        )


def adaptive_rescue_reasons(
    *,
    candidate_count: int,
    expected_candidate_count: int,
    segment_count: int,
    top_score: float,
    top_two_margin: float | None,
    invalid_ann: bool,
    config: MatcherConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate_count < expected_candidate_count:
        reasons.append(RESCUE_UNDERFILL)
    if invalid_ann or not math.isfinite(top_score):
        reasons.append(RESCUE_INVALID_ANN)
    full_exact_candidates = expected_candidate_count >= segment_count
    if not full_exact_candidates and math.isfinite(top_score):
        if abs(top_score - config.threshold) <= config.rescue_threshold_band:
            reasons.append(RESCUE_THRESHOLD_BAND)
        if top_two_margin is not None and top_two_margin <= config.rescue_margin:
            reasons.append(RESCUE_LOW_MARGIN)
    return tuple(reasons)


def cross_validate_exact_with_faiss(
    *,
    oracle: ExactOracle,
    user_ids: Sequence[str],
    user_vectors: Any,
    atol: float = 1e-5,
) -> None:
    np = _numpy()
    faiss = _faiss()
    users = normalize_float32_matrix(user_vectors, name="user_vectors")
    oracle_output = oracle.match(user_ids=user_ids, user_vectors=users)
    index = faiss.IndexFlatIP(oracle.config.dimension)
    index.add(np.ascontiguousarray(oracle.segment_vectors, dtype=np.float32))
    distances, labels = index.search(users, len(oracle.segment_ids))
    distances = np.asarray(distances)
    labels = np.asarray(labels)
    expected_shape = (users.shape[0], len(oracle.segment_ids))
    if distances.shape != expected_shape or labels.shape != expected_shape:
        raise AssertionError("FAISS IndexFlatIP returned an invalid full-search shape")
    expected_labels = np.arange(len(oracle.segment_ids), dtype=np.int64)
    scores_by_label = np.empty(expected_shape, dtype=np.float32)
    for row_index in range(users.shape[0]):
        row_labels = labels[row_index].astype(np.int64, copy=False)
        if not bool(np.array_equal(np.sort(row_labels), expected_labels)):
            raise AssertionError(
                "FAISS IndexFlatIP full search did not return every segment once"
            )
        if not bool(np.isfinite(distances[row_index]).all()):
            raise AssertionError("FAISS IndexFlatIP returned a non-finite score")
        scores_by_label[row_index, row_labels] = distances[row_index]
    numpy_scores = users @ oracle.segment_vectors.T
    if not bool(np.allclose(scores_by_label, numpy_scores, atol=atol, rtol=0.0)):
        raise AssertionError("NumPy exact scores disagree with FAISS IndexFlatIP scores")
    flat_top_indices = np.argmax(scores_by_label, axis=1)
    flat_top_scores = scores_by_label[
        np.arange(users.shape[0]),
        flat_top_indices,
    ]
    if not bool(np.array_equal(flat_top_indices, oracle_output.top_segment_indices)):
        raise AssertionError("NumPy exact oracle disagrees with FAISS IndexFlatIP top-1")
    if not bool(np.allclose(flat_top_scores, oracle_output.raw_scores, atol=atol, rtol=0.0)):
        raise AssertionError("NumPy exact oracle disagrees with FAISS IndexFlatIP scores")


def concatenate_outputs(outputs: Sequence[MatcherOutput]) -> MatcherOutput:
    np = _numpy()
    if not outputs:
        raise ValueError("at least one matcher output is required")
    segment_ids = outputs[0].segment_ids
    if any(output.segment_ids != segment_ids for output in outputs):
        raise ValueError("matcher outputs use different segment order")
    candidates = None
    if all(output.candidate_indices is not None for output in outputs):
        candidates = np.concatenate([output.candidate_indices for output in outputs], axis=0)

    def optional_array(name: str) -> Any | None:
        values = [getattr(output, name) for output in outputs]
        if all(value is not None for value in values):
            return np.concatenate(values, axis=0)
        return None

    return MatcherOutput(
        user_ids=tuple(value for output in outputs for value in output.user_ids),
        segment_ids=segment_ids,
        top_segment_indices=np.concatenate([output.top_segment_indices for output in outputs]),
        raw_scores=np.concatenate([output.raw_scores for output in outputs]),
        fallback_mask=np.concatenate([output.fallback_mask for output in outputs]),
        fallback_reasons=tuple(
            value for output in outputs for value in output.fallback_reasons
        ),
        candidate_indices=candidates,
        candidate_counts=np.concatenate([output.candidate_counts for output in outputs]),
        rescued_mask=np.concatenate([output.rescued_mask for output in outputs]),
        rescue_reasons=tuple(
            value for output in outputs for value in output.rescue_reasons
        ),
        pre_rescue_top_segment_indices=optional_array("pre_rescue_top_segment_indices"),
        pre_rescue_raw_scores=optional_array("pre_rescue_raw_scores"),
        pre_rescue_fallback_mask=optional_array("pre_rescue_fallback_mask"),
    )


def _rerank_ann_candidates(
    *,
    user_ids: Sequence[str],
    users: Any,
    segment_ids: tuple[str, ...],
    segment_vectors: Any,
    distances: Any,
    labels: Any,
    config: MatcherConfig,
) -> MatcherOutput:
    np = _numpy()
    distances = np.asarray(distances)
    labels = np.asarray(labels)
    if labels.ndim != 2 or distances.shape != labels.shape or labels.shape[0] != users.shape[0]:
        raise InvalidANNResultError("FAISS search returned an invalid result shape")
    candidate_indices = np.full(labels.shape, -1, dtype=np.int32)
    top_indices = np.full(users.shape[0], -1, dtype=np.int32)
    top_scores = np.full(users.shape[0], np.nan, dtype=np.float32)
    candidate_counts = np.zeros(users.shape[0], dtype=np.int32)
    invalid_rows = np.zeros(users.shape[0], dtype=np.bool_)
    for row_index in range(users.shape[0]):
        valid: list[int] = []
        seen: set[int] = set()
        for distance, raw_label in zip(distances[row_index], labels[row_index], strict=True):
            try:
                if isinstance(raw_label, (bool, np.bool_)):
                    raise TypeError("boolean ANN label")
                label = operator.index(raw_label)
                if isinstance(distance, (bool, np.bool_, str, bytes)):
                    raise TypeError("non-numeric ANN distance")
                finite_distance = math.isfinite(float(distance))
            except (TypeError, ValueError, OverflowError):
                invalid_rows[row_index] = True
                continue
            if label == -1:
                continue
            if label < 0 or label >= len(segment_ids) or not finite_distance:
                invalid_rows[row_index] = True
                continue
            if label in seen:
                invalid_rows[row_index] = True
                continue
            seen.add(label)
            valid.append(label)
        if valid:
            ordered = sorted(valid)
            candidate_indices[row_index, : len(ordered)] = ordered
            candidate_counts[row_index] = len(ordered)
            candidate_scores = users[row_index] @ segment_vectors[ordered].T
            best_local = int(np.argmax(candidate_scores))
            top_indices[row_index] = ordered[best_local]
            top_scores[row_index] = candidate_scores[best_local]
    fallback = np.logical_or(
        top_indices < 0,
        np.logical_and(np.isfinite(top_scores), top_scores < config.threshold),
    )
    reasons: list[str | None] = []
    for index in range(users.shape[0]):
        if top_indices[index] < 0 or invalid_rows[index] and not math.isfinite(float(top_scores[index])):
            reasons.append(FALLBACK_REASON_NO_CANDIDATE)
        elif bool(fallback[index]):
            reasons.append(FALLBACK_REASON_BELOW_THRESHOLD)
        else:
            reasons.append(None)
    rescue_reasons = tuple(
        (RESCUE_INVALID_ANN,) if bool(value) else () for value in invalid_rows
    )
    return MatcherOutput(
        user_ids=tuple(str(value) for value in user_ids),
        segment_ids=segment_ids,
        top_segment_indices=top_indices,
        raw_scores=top_scores,
        fallback_mask=fallback,
        fallback_reasons=tuple(reasons),
        candidate_indices=candidate_indices,
        candidate_counts=candidate_counts,
        rescued_mask=np.zeros(users.shape[0], dtype=np.bool_),
        rescue_reasons=rescue_reasons,
    )


def _rescue_invalid_ann_batch(
    *,
    user_ids: Sequence[str],
    users: Any,
    segment_ids: tuple[str, ...],
    segment_vectors: Any,
    config: MatcherConfig,
) -> MatcherOutput:
    np = _numpy()
    exact = ExactOracle(
        segment_ids=segment_ids,
        segment_vectors=segment_vectors,
        config=config,
    ).match(user_ids=user_ids, user_vectors=users)
    user_count = len(user_ids)
    candidate_count = min(config.candidate_k, len(segment_ids))
    return MatcherOutput(
        user_ids=exact.user_ids,
        segment_ids=exact.segment_ids,
        top_segment_indices=exact.top_segment_indices,
        raw_scores=exact.raw_scores,
        fallback_mask=exact.fallback_mask,
        fallback_reasons=exact.fallback_reasons,
        candidate_indices=np.full(
            (user_count, candidate_count),
            -1,
            dtype=np.int32,
        ),
        candidate_counts=np.zeros(user_count, dtype=np.int32),
        rescued_mask=np.ones(user_count, dtype=np.bool_),
        rescue_reasons=tuple((RESCUE_INVALID_ANN,) for _ in user_ids),
        pre_rescue_top_segment_indices=np.full(
            user_count,
            -1,
            dtype=np.int32,
        ),
        pre_rescue_raw_scores=np.zeros(user_count, dtype=np.float32),
        pre_rescue_fallback_mask=np.ones(user_count, dtype=np.bool_),
    )


def _candidate_margin(
    *,
    user: Any,
    candidate_indices: Sequence[int],
    segment_vectors: Any,
) -> float | None:
    np = _numpy()
    if len(candidate_indices) < 2:
        return None
    scores = np.sort(user @ segment_vectors[list(candidate_indices)].T)
    return float(scores[-1] - scores[-2])


def _output_payload(output: MatcherOutput) -> dict[str, Any]:
    return {
        "user_ids": output.user_ids,
        "segment_ids": output.segment_ids,
        "top_segment_indices": output.top_segment_indices,
        "raw_scores": output.raw_scores,
        "fallback_mask": output.fallback_mask,
        "fallback_reasons": output.fallback_reasons,
        "candidate_indices": output.candidate_indices,
        "candidate_counts": output.candidate_counts,
        "rescued_mask": output.rescued_mask,
        "rescue_reasons": output.rescue_reasons,
        "pre_rescue_top_segment_indices": output.pre_rescue_top_segment_indices,
        "pre_rescue_raw_scores": output.pre_rescue_raw_scores,
        "pre_rescue_fallback_mask": output.pre_rescue_fallback_mask,
    }


def _numpy() -> Any:
    try:
        return __import__("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "assignment benchmark requires the 'assignment-benchmark' optional extra"
        ) from exc


def _faiss() -> Any:
    try:
        return __import__("faiss")
    except ImportError as exc:
        raise RuntimeError(
            "FAISS matchers require the 'assignment-benchmark' optional extra"
        ) from exc
