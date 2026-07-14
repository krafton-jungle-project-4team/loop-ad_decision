from __future__ import annotations

import math

import pytest


np = pytest.importorskip("numpy")

from offline_evaluation.assignment_matchers import (  # noqa: E402
    RESCUE_INVALID_ANN,
    RESCUE_LOW_MARGIN,
    RESCUE_THRESHOLD_BAND,
    RESCUE_UNDERFILL,
    AdaptiveHNSWMatcher,
    ExactOracle,
    MatcherConfig,
    adaptive_rescue_reasons,
    cross_validate_exact_with_faiss,
)
from offline_evaluation.assignment_corpus import freeze_corpus  # noqa: E402


def _unit(index: int) -> object:
    values = np.zeros(64, dtype=np.float32)
    values[index] = 1.0
    return values


def test_exact_oracle_uses_lexical_tie_and_threshold_equality() -> None:
    threshold_vector = 0.65 * _unit(0) + math.sqrt(1.0 - 0.65**2) * _unit(1)
    oracle = ExactOracle(
        segment_ids=["seg-z", "seg-a", "seg-threshold"],
        segment_vectors=[_unit(0), _unit(0), threshold_vector],
        config=MatcherConfig(candidate_k=2),
    )
    output = oracle.match(
        user_ids=["tie", "threshold"],
        user_vectors=[_unit(0), _unit(0)],
    )
    assert output.segment_ids[int(output.top_segment_indices[0])] == "seg-a"
    assert bool(output.fallback_mask[0]) is False

    threshold_only = ExactOracle(
        segment_ids=["seg-threshold"],
        segment_vectors=[threshold_vector],
        config=MatcherConfig(candidate_k=2),
    ).match(user_ids=["user"], user_vectors=[_unit(0)])
    assert threshold_only.raw_scores[0] == pytest.approx(0.65, abs=1e-6)
    assert bool(threshold_only.fallback_mask[0]) is False


def test_matcher_config_rejects_non_positive_native_thread_counts() -> None:
    with pytest.raises(ValueError, match="faiss_threads"):
        MatcherConfig(faiss_threads=0)
    with pytest.raises(ValueError, match="blas_threads"):
        MatcherConfig(blas_threads=0)


def test_frozen_corpus_preserves_segment_id_lexical_tie_break() -> None:
    corpus = freeze_corpus(
        user_ids=["raw-user"],
        user_vectors=[_unit(0)],
        segment_ids=["seg-z", "seg-a"],
        segment_vectors=[_unit(0), _unit(0)],
        vector_version="v1",
        source_cutoff_at="2026-01-01T00:00:00Z",
        distribution="tie_fixture",
        random_seed=7,
        git_commit="abc123",
        id_hash_salt="test-salt",
    )

    output = ExactOracle(
        segment_ids=corpus.segment_ids,
        segment_vectors=corpus.segment_vectors,
        config=MatcherConfig(candidate_k=2),
    ).match(user_ids=corpus.user_ids, user_vectors=corpus.user_vectors)

    assert output.segment_ids[int(output.top_segment_indices[0])] == "seg-a"


def test_adaptive_rescue_reason_union_and_full_candidate_shortcut() -> None:
    config = MatcherConfig(
        candidate_k=5,
        rescue_threshold_band=0.02,
        rescue_margin=0.01,
    )
    reasons = adaptive_rescue_reasons(
        candidate_count=3,
        expected_candidate_count=5,
        segment_count=10,
        top_score=0.65,
        top_two_margin=0.005,
        invalid_ann=True,
        config=config,
    )
    assert set(reasons) == {
        RESCUE_UNDERFILL,
        RESCUE_INVALID_ANN,
        RESCUE_THRESHOLD_BAND,
        RESCUE_LOW_MARGIN,
    }
    full = adaptive_rescue_reasons(
        candidate_count=5,
        expected_candidate_count=5,
        segment_count=5,
        top_score=0.65,
        top_two_margin=0.0,
        invalid_ann=False,
        config=config,
    )
    assert full == ()


def test_faiss_flat_crosscheck_and_adaptive_rescue() -> None:
    pytest.importorskip("faiss")
    rng = np.random.default_rng(9)
    segments = rng.normal(size=(32, 64)).astype(np.float32)
    users = rng.normal(size=(16, 64)).astype(np.float32)
    segment_ids = [f"seg-{index:03d}" for index in reversed(range(32))]
    user_ids = [f"user-{index:03d}" for index in range(16)]
    config = MatcherConfig(
        candidate_k=8,
        hnsw_m=8,
        hnsw_ef_construction=40,
        hnsw_ef_search=16,
        rescue_threshold_band=1.0,
        rescue_margin=1.0,
    )
    oracle = ExactOracle(
        segment_ids=segment_ids,
        segment_vectors=segments,
        config=config,
    )
    cross_validate_exact_with_faiss(
        oracle=oracle,
        user_ids=user_ids,
        user_vectors=users,
    )
    exact = oracle.match(user_ids=user_ids, user_vectors=users)
    adaptive = AdaptiveHNSWMatcher(
        segment_ids=segment_ids,
        segment_vectors=segments,
        config=config,
    ).match(user_ids=user_ids, user_vectors=users)
    assert np.all(adaptive.rescued_mask)
    assert np.array_equal(adaptive.top_segment_indices, exact.top_segment_indices)
    assert np.array_equal(adaptive.fallback_mask, exact.fallback_mask)


def test_faiss_flat_crosscheck_rejects_wrong_scores(monkeypatch) -> None:
    oracle = ExactOracle(
        segment_ids=["seg-a", "seg-b"],
        segment_vectors=[_unit(0), _unit(1)],
        config=MatcherConfig(candidate_k=2),
    )

    class WrongFlatIndex:
        def __init__(self, _dimension):
            pass

        def add(self, _vectors):
            pass

        def search(self, users, _candidate_count):
            return (
                np.zeros((users.shape[0], 2), dtype=np.float32),
                np.tile(np.array([1, 0], dtype=np.int64), (users.shape[0], 1)),
            )

    class WrongFaiss:
        IndexFlatIP = WrongFlatIndex

    monkeypatch.setattr(
        "offline_evaluation.assignment_matchers._faiss",
        lambda: WrongFaiss(),
    )

    with pytest.raises(AssertionError, match="scores disagree"):
        cross_validate_exact_with_faiss(
            oracle=oracle,
            user_ids=["user"],
            user_vectors=[_unit(0)],
        )


def test_adaptive_rescues_entire_batch_when_ann_shape_is_invalid() -> None:
    pytest.importorskip("faiss")
    config = MatcherConfig(candidate_k=2, hnsw_m=4)
    matcher = AdaptiveHNSWMatcher(
        segment_ids=["seg-z", "seg-a", "seg-b"],
        segment_vectors=[_unit(0), _unit(0), _unit(1)],
        config=config,
    )

    class InvalidShapeIndex:
        def search(self, users, _candidate_count):
            return (
                np.zeros((users.shape[0], 2), dtype=np.float32),
                np.zeros((users.shape[0], 1), dtype=np.int64),
            )

    matcher.index = InvalidShapeIndex()
    output = matcher.match(
        user_ids=["user-1", "user-2"],
        user_vectors=[_unit(0), _unit(1)],
    )
    exact = ExactOracle(
        segment_ids=matcher.segment_ids,
        segment_vectors=matcher.segment_vectors,
        config=config,
    ).match(
        user_ids=["user-1", "user-2"],
        user_vectors=[_unit(0), _unit(1)],
    )

    assert np.array_equal(output.top_segment_indices, exact.top_segment_indices)
    assert np.array_equal(output.fallback_mask, exact.fallback_mask)
    assert np.all(output.rescued_mask)
    assert np.all(output.candidate_counts == 0)
    assert output.rescue_reasons == (
        (RESCUE_INVALID_ANN,),
        (RESCUE_INVALID_ANN,),
    )
    assert np.all(output.pre_rescue_top_segment_indices == -1)


@pytest.mark.parametrize("malformed_label", (0.5, "0", True))
def test_adaptive_rescues_non_integral_ann_labels(malformed_label) -> None:
    pytest.importorskip("faiss")
    config = MatcherConfig(
        candidate_k=2,
        hnsw_m=4,
        rescue_threshold_band=0.0,
        rescue_margin=0.0,
    )
    matcher = AdaptiveHNSWMatcher(
        segment_ids=["seg-a", "seg-b", "seg-c"],
        segment_vectors=[_unit(0), _unit(1), _unit(2)],
        config=config,
    )

    class MalformedLabelIndex:
        def search(self, users, _candidate_count):
            return (
                np.asarray([[0.9, 0.8]], dtype=np.float32),
                np.asarray([[malformed_label, 1]], dtype=object),
            )

    matcher.index = MalformedLabelIndex()
    output = matcher.match(user_ids=["user"], user_vectors=[_unit(0)])

    assert output.segment_ids[int(output.top_segment_indices[0])] == "seg-a"
    assert bool(output.rescued_mask[0]) is True
    assert set(output.rescue_reasons[0]) == {
        RESCUE_INVALID_ANN,
        RESCUE_UNDERFILL,
    }
