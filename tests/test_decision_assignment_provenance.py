from __future__ import annotations

import json

import pytest

from app.decision.assignment_provenance import (
    CANONICAL_INPUT_VERSION,
    INPUT_MANIFEST_VERSION,
    MATCHER_MANIFEST_VERSION,
    REQUEST_FINGERPRINT_VERSION,
    RESULT_SUMMARY_VERSION,
    build_canonical_request,
    build_input_fingerprint,
    build_input_manifest,
    build_request_fingerprint,
    canonical_json,
    sha256_canonical_json,
    validate_input_manifest,
    validate_result_summary,
)


def test_canonical_json_is_compact_sorted_utf8_and_sha256_is_lowercase() -> None:
    value = {"z": "가", "a": [2, 1]}

    assert canonical_json(value) == '{"a":[2,1],"z":"가"}'
    assert sha256_canonical_json(value) == sha256_canonical_json(
        {"a": (2, 1), "z": "가"}
    )
    assert len(sha256_canonical_json(value)) == 64
    assert sha256_canonical_json(value) == sha256_canonical_json(value).lower()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_non_finite_numbers(invalid: float) -> None:
    with pytest.raises(ValueError, match="NaN or Infinity"):
        canonical_json({"value": invalid})


def test_request_fingerprint_is_order_and_explicit_user_duplicate_invariant() -> None:
    first = _logical_request(
        explicit_user_ids=["user-b", "user-a", "user-b"],
        audience_scope=None,
    )
    second = {
        "expires_in_days": 7,
        "effective_limit": 10_000,
        "explicit_user_ids": ["user-a", "user-b"],
        "vector_source": "booking",
        "vector_version": "v1",
        "generation_id": "generation-1",
        "analysis_id": "analysis-1",
        "promotion_id": "promotion-1",
        "campaign_id": "campaign-1",
        "project_id": "project-1",
        "promotion_run_id": "run-1",
        "audience_scope": None,
    }

    assert build_request_fingerprint(first) == build_request_fingerprint(second)
    assert build_canonical_request(first)["explicit_user_ids"] == [
        "user-a",
        "user-b",
    ]
    assert build_canonical_request(first)["version"] == REQUEST_FINGERPRINT_VERSION


def test_request_fingerprint_includes_expires_and_other_logical_request_values() -> None:
    base = _logical_request()

    changed_expiry = {**base, "expires_in_days": 8}
    changed_limit = {**base, "effective_limit": 9_999}

    assert build_request_fingerprint(base) != build_request_fingerprint(changed_expiry)
    assert build_request_fingerprint(base) != build_request_fingerprint(changed_limit)


def test_explicit_empty_user_set_is_distinct_from_live_audience_request() -> None:
    explicit_empty = _logical_request(
        explicit_user_ids=[],
        audience_scope=None,
    )
    live_audience = _logical_request(
        explicit_user_ids=None,
        audience_scope={"base": "user_behavior_vectors"},
    )

    assert build_request_fingerprint(explicit_empty) != build_request_fingerprint(
        live_audience
    )


def test_request_fingerprint_excludes_execution_and_matcher_metadata() -> None:
    base = _logical_request()
    first = {
        **base,
        "matcher_strategy": "exact_cosine",
        "matcher_version": "matcher-v1",
        "selector_policy_version": "policy-v1",
        "source_cutoff_at": "2026-07-14T00:00:00.000001Z",
        "page_size": 10_000,
        "executed_at": "2026-07-14T00:00:01Z",
        "diagnostics": {"latency_ms": 10},
    }
    second = {
        **base,
        "matcher_strategy": "pgvector_hnsw_rerank",
        "matcher_version": "matcher-v2",
        "selector_policy_version": "policy-v2",
        "source_cutoff_at": "2026-07-15T00:00:00.000001Z",
        "page_size": 100,
        "executed_at": "2026-07-15T00:00:01Z",
        "diagnostics": {"latency_ms": 999},
    }

    assert build_request_fingerprint(first) == build_request_fingerprint(second)


def test_unknown_request_field_is_not_silently_omitted() -> None:
    with pytest.raises(ValueError, match="unsupported logical request field"):
        build_request_fingerprint({**_logical_request(), "new_request_option": True})


def test_input_fingerprint_hashes_only_canonical_input() -> None:
    canonical_input = _canonical_input()
    first = build_input_manifest(
        canonical_input=canonical_input,
        matcher=_matcher(page_size=10_000, strategy="exact_cosine"),
        result_summary=_result_summary(),
    )
    second = build_input_manifest(
        canonical_input=canonical_input,
        matcher=_matcher(
            page_size=100,
            strategy="pgvector_hnsw_rerank",
            candidate_limit=32,
        ),
        result_summary={
            **_result_summary(),
            "page_count": 2,
            "ann_candidate_count": 2,
        },
    )

    assert build_input_fingerprint(first) == build_input_fingerprint(second)
    assert build_input_fingerprint(first) == build_input_fingerprint(canonical_input)


@pytest.mark.parametrize(
    "changed_field",
    [
        "vector_row_id_stream_digest",
        "segment_embedding_identity_digest",
        "experiment_content_mapping_digest",
    ],
)
def test_input_fingerprint_changes_with_canonical_input_digest(
    changed_field: str,
) -> None:
    first = _canonical_input()
    second = {**first, changed_field: "d" * 64}

    assert build_input_fingerprint(first) != build_input_fingerprint(second)


def test_manifest_has_only_versioned_bounded_sections() -> None:
    manifest = build_input_manifest(
        canonical_input=_canonical_input(),
        matcher=_matcher(),
        result_summary=_result_summary(),
    )

    assert set(manifest) == {
        "version",
        "canonical_input",
        "matcher",
        "result_summary",
    }
    assert manifest["version"] == INPUT_MANIFEST_VERSION
    assert manifest["canonical_input"]["version"] == CANONICAL_INPUT_VERSION
    assert (
        manifest["canonical_input"]["source_table"]
        == "user_behavior_vector_revisions"
    )
    assert manifest["matcher"]["version"] == MATCHER_MANIFEST_VERSION
    assert manifest["matcher"]["matcher_version"] == "exact_cosine_v1"
    assert manifest["result_summary"]["version"] == RESULT_SUMMARY_VERSION
    json.dumps(manifest, allow_nan=False)


@pytest.mark.parametrize(
    ("section", "forbidden_key", "forbidden_value"),
    [
        ("canonical_input", "user_ids", ["user-1"]),
        ("canonical_input", "user_vectors", [[0.1] * 64]),
        ("matcher", "benchmark_p95_ms", 12.5),
        ("matcher", "logs", ["query plan"]),
        (
            "matcher",
            "approved_ann_region",
            {"raw_user_ids": ["user-1"], "latency_p95_ms": 12.5},
        ),
        ("result_summary", "http_response", {"status": 200}),
        ("result_summary", "latency_ms", 10),
    ],
)
def test_manifest_rejects_raw_perf_log_and_full_response_payloads(
    section: str,
    forbidden_key: str,
    forbidden_value: object,
) -> None:
    manifest = build_input_manifest(
        canonical_input=_canonical_input(),
        matcher=_matcher(),
        result_summary=_result_summary(),
    )
    manifest[section][forbidden_key] = forbidden_value

    with pytest.raises(ValueError, match="forbidden or unbounded keys"):
        validate_input_manifest(manifest)


def test_result_summary_enforces_one_effective_assignment_set() -> None:
    summary = validate_result_summary(_result_summary())

    assert summary["newly_linked_count"] + summary["reused_existing_count"] == 3
    assert summary["assignment_count"] == 3
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate"] == pytest.approx(1 / 3)
    assert sum(summary["fallback_reason_counts"].values()) == 1
    assert sum(summary["segment_assignment_counts"].values()) == 3
    assert sum(summary["similarity_score_buckets"].values()) == 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"newly_linked_count": 3, "reused_existing_count": 1},
            "newly_linked_count .* reused_existing_count",
        ),
        ({"processed_user_count": 2}, "processed_user_count"),
        ({"skipped_existing_count": 0}, "skipped_existing_count"),
        ({"users_to_match_count": 1}, "users_to_match_count"),
        ({"ann_applied": True}, "ann_applied"),
        (
            {
                "fallback_count": 4,
                "batch_has_fallback": True,
                "fallback_rate": 4 / 3,
                "fallback_reason_counts": {"below_threshold": 4},
            },
            "must not exceed assignment_count",
        ),
        ({"fallback_rate": 0.9}, "must equal fallback_count"),
        (
            {"fallback_reason_counts": {"below_threshold": 2}},
            "must sum to fallback_count",
        ),
        (
            {"segment_assignment_counts": {"segment-a": 1}},
            "must sum to assignment_count",
        ),
        (
            {"similarity_score_buckets": {"gte_0_90": 2}},
            "must sum to assignment_count",
        ),
    ],
)
def test_result_summary_rejects_inconsistent_aggregates(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_result_summary({**_result_summary(), **changes})


def test_zero_assignment_summary_has_null_fallback_rate() -> None:
    summary = {
        **_result_summary(),
        "assignment_count": 0,
        "newly_linked_count": 0,
        "reused_existing_count": 0,
        "skipped_existing_count": 0,
        "processed_user_count": 0,
        "users_to_match_count": 0,
        "fallback_count": 0,
        "batch_has_fallback": False,
        "fallback_rate": None,
        "fallback_reason_counts": {
            "below_threshold": 0,
            "no_candidate": 0,
            "invalid_user_vector": 0,
        },
        "segment_assignment_counts": {},
        "similarity_score_buckets": {},
    }

    assert validate_result_summary(summary)["fallback_rate"] is None


def test_result_summary_rejects_unknown_fallback_reason_key() -> None:
    summary = {
        **_result_summary(),
        "fallback_reason_counts": {
            "unexpected_reason": 1,
            "no_candidate": 0,
            "invalid_user_vector": 0,
        },
    }

    with pytest.raises(ValueError, match="exactly the contract fallback reasons"):
        validate_result_summary(summary)


def _logical_request(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "promotion_run_id": "run-1",
        "project_id": "project-1",
        "campaign_id": "campaign-1",
        "promotion_id": "promotion-1",
        "analysis_id": "analysis-1",
        "generation_id": "generation-1",
        "vector_version": "v1",
        "vector_source": "booking",
        "audience_scope": {"base": "user_behavior_vectors"},
        "explicit_user_ids": None,
        "effective_limit": 10_000,
        "expires_in_days": 7,
    }
    value.update(changes)
    return value


def _canonical_input() -> dict[str, object]:
    return {
        "source_cutoff_at": "2026-07-14T00:00:00.123456Z",
        "source_table": "user_behavior_vector_revisions",
        "selection_version": "user_behavior_vector_revisions_argmax_v1",
        "selection_mode": "live_keyset",
        "vector_version": "v1",
        "vector_source": "booking",
        "user_count": 3,
        "segment_count": 2,
        "dimension": 64,
        "vector_row_id_stream_digest": "a" * 64,
        "segment_embedding_identity_digest": "b" * 64,
        "experiment_content_mapping_digest": "c" * 64,
    }


def _matcher(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "selector_policy_version": "exact_only_v1",
        "backend": "postgresql_pgvector",
        "strategy": "exact_cosine",
        "matcher_version": "exact_cosine_v1",
        "page_size": 10_000,
        "candidate_limit": 32,
        "ann_query_user_batch_size": 256,
        "hnsw_ef_search": 64,
        "exact_rescue_enabled": True,
        "exact_rescue_reasons": [
            "underfill",
            "empty",
            "duplicate",
            "foreign",
            "malformed",
        ],
    }
    value.update(changes)
    return value


def _result_summary() -> dict[str, object]:
    return {
        "assignment_count": 3,
        "newly_linked_count": 2,
        "reused_existing_count": 1,
        "skipped_existing_count": 1,
        "insert_conflict_count": 0,
        "fallback_count": 1,
        "batch_has_fallback": True,
        "fallback_rate": 1 / 3,
        "fallback_reason_counts": {
            "below_threshold": 1,
            "invalid_user_vector": 0,
            "no_candidate": 0,
        },
        "segment_assignment_counts": {
            "segment-a": 2,
            "segment-b": 1,
        },
        "similarity_score_buckets": {
            "0_65_to_0_80": 1,
            "gte_0_90": 2,
        },
        "page_count": 1,
        "processed_user_count": 3,
        "users_to_match_count": 2,
        "ann_candidate_count": 0,
        "exact_reranked_pair_count": 4,
        "ann_underfilled_user_count": 0,
        "exact_rescue_user_count": 0,
        "ann_query_user_count": 0,
        "run_assignment_count": 3,
        "run_fallback_count": 1,
        "assignment_mode": "live_keyset",
        "ann_applied": False,
        "ann_not_applied_reason": "matcher_selected_exact",
        "matching_mode": "exact_cosine",
    }
