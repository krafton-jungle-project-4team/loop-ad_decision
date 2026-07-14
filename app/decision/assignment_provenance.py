from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REQUEST_FINGERPRINT_VERSION = "segment_assignment_request.v1"
INPUT_MANIFEST_VERSION = "segment_assignment_input_manifest.v1"
CANONICAL_INPUT_VERSION = "segment_assignment_canonical_input.v1"
MATCHER_MANIFEST_VERSION = "segment_assignment_matcher.v1"
RESULT_SUMMARY_VERSION = "segment_assignment_result_summary.v1"

SHA256_HEX_LENGTH = 64
MAX_AGGREGATE_BUCKET_COUNT = 10_000
MAX_RESULT_SUMMARY_JSON_BYTES = 512 * 1024
MAX_INPUT_MANIFEST_JSON_BYTES = 768 * 1024
MAX_LABEL_LENGTH = 512

_FALLBACK_REASON_KEYS = frozenset(
    {"below_threshold", "no_candidate", "invalid_user_vector"}
)

_MISSING = object()

_REQUEST_ALIASES = {
    "effective_vector_version": "vector_version",
    "source": "vector_source",
    "effective_source": "vector_source",
    "eligible_user_limit": "effective_limit",
    "user_ids": "explicit_user_ids",
}
_LOGICAL_REQUEST_KEYS = frozenset(
    {
        "promotion_run_id",
        "project_id",
        "campaign_id",
        "promotion_id",
        "analysis_id",
        "generation_id",
        "vector_version",
        "vector_source",
        "audience_scope",
        "explicit_user_ids",
        "effective_limit",
        "expires_in_days",
    }
)
_NON_LOGICAL_REQUEST_KEYS = frozenset(
    {
        "version",
        "matcher",
        "matcher_strategy",
        "matcher_version",
        "matching_mode",
        "selector",
        "selector_version",
        "selector_policy_version",
        "policy",
        "policy_version",
        "cutoff",
        "source_cutoff_at",
        "page_size",
        "configured_page_size",
        "execution_time",
        "executed_at",
        "created_at",
        "completed_at",
        "diagnostics",
        "metrics",
        "timing",
        "latency",
        "duration",
        "input_fingerprint",
        "request_fingerprint",
        "input_manifest_json",
        "result_summary",
    }
)

_CANONICAL_INPUT_KEYS = frozenset(
    {
        "version",
        "source_cutoff_at",
        "source_table",
        "selection_version",
        "selection_mode",
        "vector_version",
        "vector_source",
        "user_count",
        "segment_count",
        "dimension",
        "vector_row_id_stream_digest",
        "vector_row_id_stream_sha256",
        "logical_key_vector_row_id_stream_sha256",
        "segment_embedding_identity_digest",
        "segment_embedding_identity_sha256",
        "experiment_content_mapping_digest",
        "experiment_content_mapping_sha256",
    }
)
_CANONICAL_INPUT_REQUIRED_GROUPS = (
    frozenset({"source_cutoff_at"}),
    frozenset({"source_table"}),
    frozenset({"selection_version"}),
    frozenset({"selection_mode"}),
    frozenset({"vector_version"}),
    frozenset({"vector_source"}),
    frozenset({"user_count"}),
    frozenset({"segment_count"}),
    frozenset({"dimension"}),
    frozenset(
        {
            "vector_row_id_stream_digest",
            "vector_row_id_stream_sha256",
            "logical_key_vector_row_id_stream_sha256",
        }
    ),
    frozenset(
        {
            "segment_embedding_identity_digest",
            "segment_embedding_identity_sha256",
        }
    ),
    frozenset(
        {
            "experiment_content_mapping_digest",
            "experiment_content_mapping_sha256",
        }
    ),
)
_CANONICAL_INPUT_DIGEST_KEYS = frozenset(
    {
        "vector_row_id_stream_digest",
        "vector_row_id_stream_sha256",
        "logical_key_vector_row_id_stream_sha256",
        "segment_embedding_identity_digest",
        "segment_embedding_identity_sha256",
        "experiment_content_mapping_digest",
        "experiment_content_mapping_sha256",
    }
)

_MATCHER_KEYS = frozenset(
    {
        "version",
        "selector_policy_version",
        "backend",
        "strategy",
        "matcher_version",
        "page_size",
        "candidate_limit",
        "query_user_batch_size",
        "ann_query_user_batch_size",
        "hnsw_ef_search",
        "distance_metric",
        "distance_operator",
        "exact_rescue_enabled",
        "exact_rescue_reasons",
    }
)
_MATCHER_REQUIRED_KEYS = frozenset(
    {"selector_policy_version", "backend", "strategy", "matcher_version"}
)

_RESULT_SUMMARY_COUNT_KEYS = frozenset(
    {
        "assignment_count",
        "effective_assignment_count",
        "newly_linked_count",
        "reused_existing_count",
        "skipped_existing_count",
        "insert_conflict_count",
        "fallback_count",
        "page_count",
        "processed_user_count",
        "users_to_match_count",
        "ann_candidate_count",
        "exact_reranked_pair_count",
        "ann_underfilled_user_count",
        "exact_rescue_user_count",
        "ann_query_user_count",
        "run_assignment_count",
        "run_fallback_count",
    }
)
_RESULT_SUMMARY_AGGREGATE_KEYS = frozenset(
    {
        "fallback_reason_counts",
        "segment_assignment_counts",
        "similarity_score_buckets",
    }
)
_RESULT_SUMMARY_STRING_KEYS = frozenset(
    {
        "assignment_mode",
        "ann_not_applied_reason",
        "matching_mode",
    }
)
_RESULT_SUMMARY_KEYS = frozenset(
    {
        "version",
        "batch_has_fallback",
        "fallback_rate",
        "ann_applied",
        *_RESULT_SUMMARY_COUNT_KEYS,
        *_RESULT_SUMMARY_AGGREGATE_KEYS,
        *_RESULT_SUMMARY_STRING_KEYS,
    }
)
_RESULT_SUMMARY_REQUIRED_KEYS = frozenset(
    {
        "assignment_count",
        "newly_linked_count",
        "reused_existing_count",
        "skipped_existing_count",
        "insert_conflict_count",
        "fallback_count",
        "batch_has_fallback",
        "fallback_rate",
        "fallback_reason_counts",
        "segment_assignment_counts",
        "similarity_score_buckets",
        "page_count",
        "processed_user_count",
        "users_to_match_count",
        "ann_candidate_count",
        "exact_reranked_pair_count",
        "ann_underfilled_user_count",
        "exact_rescue_user_count",
        "ann_query_user_count",
        "run_assignment_count",
        "run_fallback_count",
        "assignment_mode",
        "ann_not_applied_reason",
        "ann_applied",
        "matching_mode",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for persisted fingerprints."""

    normalized = _normalize_json_value(value, path="$", permit_tuple=True)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_canonical_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_explicit_user_ids(user_ids: Sequence[str] | None) -> list[str] | None:
    if user_ids is None:
        return None
    if isinstance(user_ids, (str, bytes, bytearray)):
        raise ValueError("explicit_user_ids must be a sequence of user ID strings")
    normalized: set[str] = set()
    for index, user_id in enumerate(user_ids):
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(
                f"explicit_user_ids[{index}] must be a non-empty string"
            )
        normalized.add(user_id)
    return sorted(normalized)


def build_canonical_request(
    logical_request: Mapping[str, Any] | None = None,
    **logical_fields: Any,
) -> dict[str, Any]:
    """Return only fields that define the caller's logical assignment request.

    Execution metadata is deliberately accepted and discarded so callers cannot
    accidentally make retries depend on matcher rollout, cutoff, paging, time, or
    diagnostics. Unknown fields are rejected to make additions to the public
    request contract explicit.
    """

    supplied: dict[str, Any] = {}
    if logical_request is not None:
        if not isinstance(logical_request, Mapping):
            raise ValueError("logical_request must be a mapping")
        supplied.update(logical_request)
    supplied.update(logical_fields)

    canonical_fields: dict[str, Any] = {}
    for input_key, value in supplied.items():
        if not isinstance(input_key, str):
            raise ValueError("logical request keys must be strings")
        if input_key in _NON_LOGICAL_REQUEST_KEYS:
            continue
        canonical_key = _REQUEST_ALIASES.get(input_key, input_key)
        if canonical_key not in _LOGICAL_REQUEST_KEYS:
            raise ValueError(f"unsupported logical request field: {input_key}")
        previous = canonical_fields.get(canonical_key, _MISSING)
        if previous is not _MISSING and previous != value:
            raise ValueError(
                f"conflicting values supplied for logical field {canonical_key}"
            )
        canonical_fields[canonical_key] = value

    if "explicit_user_ids" in canonical_fields:
        canonical_fields["explicit_user_ids"] = normalize_explicit_user_ids(
            canonical_fields["explicit_user_ids"]
        )
    if "audience_scope" in canonical_fields:
        audience_scope = canonical_fields["audience_scope"]
        canonical_fields["audience_scope"] = (
            None
            if audience_scope is None
            else _normalize_audience_scope(audience_scope)
        )

    return _normalize_json_value(
        {"version": REQUEST_FINGERPRINT_VERSION, **canonical_fields},
        path="$",
        permit_tuple=True,
    )


def build_request_fingerprint(
    logical_request: Mapping[str, Any] | None = None,
    **logical_fields: Any,
) -> str:
    return sha256_canonical_json(
        build_canonical_request(logical_request, **logical_fields)
    )


def build_canonical_input(
    canonical_input: Mapping[str, Any] | None = None,
    **input_fields: Any,
) -> dict[str, Any]:
    supplied = _merge_mapping_and_fields(
        canonical_input,
        input_fields,
        field_name="canonical_input",
    )
    normalized = _with_version(
        supplied,
        expected_version=CANONICAL_INPUT_VERSION,
        section_name="canonical_input",
    )
    _reject_unknown_keys(normalized, _CANONICAL_INPUT_KEYS, "canonical_input")
    for required_group in _CANONICAL_INPUT_REQUIRED_GROUPS:
        if not required_group.intersection(normalized):
            names = " or ".join(sorted(required_group))
            raise ValueError(f"canonical_input requires {names}")
    for count_key in ("user_count", "segment_count", "dimension"):
        _validate_non_negative_integer(normalized[count_key], f"canonical_input.{count_key}")
    if normalized["dimension"] == 0:
        raise ValueError("canonical_input.dimension must be greater than zero")
    for key in _CANONICAL_INPUT_DIGEST_KEYS.intersection(normalized):
        _validate_sha256(normalized[key], f"canonical_input.{key}")
    for key in (
        "source_cutoff_at",
        "source_table",
        "selection_version",
        "selection_mode",
        "vector_version",
    ):
        _validate_label(normalized[key], f"canonical_input.{key}")
    if normalized["vector_source"] is not None:
        _validate_label(normalized["vector_source"], "canonical_input.vector_source")
    return normalized


def build_matcher_manifest(
    matcher: Mapping[str, Any] | None = None,
    **matcher_fields: Any,
) -> dict[str, Any]:
    supplied = _merge_mapping_and_fields(
        matcher,
        matcher_fields,
        field_name="matcher",
    )
    normalized = _with_version(
        supplied,
        expected_version=MATCHER_MANIFEST_VERSION,
        section_name="matcher",
    )
    _reject_unknown_keys(normalized, _MATCHER_KEYS, "matcher")
    missing = _MATCHER_REQUIRED_KEYS.difference(normalized)
    if missing:
        raise ValueError(f"matcher is missing required keys: {sorted(missing)}")
    for key in _MATCHER_REQUIRED_KEYS:
        _validate_label(normalized[key], f"matcher.{key}")
    for key in (
        "page_size",
        "candidate_limit",
        "query_user_batch_size",
        "ann_query_user_batch_size",
        "hnsw_ef_search",
    ):
        if key in normalized and normalized[key] is not None:
            _validate_non_negative_integer(normalized[key], f"matcher.{key}")
    if "exact_rescue_enabled" in normalized and not isinstance(
        normalized["exact_rescue_enabled"], bool
    ):
        raise ValueError("matcher.exact_rescue_enabled must be a boolean")
    if "exact_rescue_reasons" in normalized:
        reasons = normalized["exact_rescue_reasons"]
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason for reason in reasons
        ):
            raise ValueError("matcher.exact_rescue_reasons must be a list of strings")
    return normalized


def build_result_summary(
    result_summary: Mapping[str, Any] | None = None,
    **summary_fields: Any,
) -> dict[str, Any]:
    supplied = _merge_mapping_and_fields(
        result_summary,
        summary_fields,
        field_name="result_summary",
    )
    normalized = _with_version(
        supplied,
        expected_version=RESULT_SUMMARY_VERSION,
        section_name="result_summary",
    )
    return validate_result_summary(normalized)


def validate_result_summary(result_summary: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _with_version(
        result_summary,
        expected_version=RESULT_SUMMARY_VERSION,
        section_name="result_summary",
    )
    _reject_unknown_keys(normalized, _RESULT_SUMMARY_KEYS, "result_summary")
    missing = _RESULT_SUMMARY_REQUIRED_KEYS.difference(normalized)
    if missing:
        raise ValueError(
            f"result_summary is missing required keys: {sorted(missing)}"
        )

    for key in _RESULT_SUMMARY_COUNT_KEYS.intersection(normalized):
        _validate_non_negative_integer(normalized[key], f"result_summary.{key}")
    assignment_count = normalized["assignment_count"]
    if "effective_assignment_count" in normalized and (
        normalized["effective_assignment_count"] != assignment_count
    ):
        raise ValueError(
            "result_summary.effective_assignment_count must equal assignment_count"
        )
    if (
        normalized["newly_linked_count"] + normalized["reused_existing_count"]
        != assignment_count
    ):
        raise ValueError(
            "result_summary newly_linked_count + reused_existing_count must equal "
            "assignment_count"
        )
    if normalized["processed_user_count"] != assignment_count:
        raise ValueError(
            "result_summary.processed_user_count must equal assignment_count"
        )
    if (
        normalized["skipped_existing_count"]
        + normalized["insert_conflict_count"]
        != normalized["reused_existing_count"]
    ):
        raise ValueError(
            "result_summary skipped_existing_count + insert_conflict_count must "
            "equal reused_existing_count"
        )
    if (
        normalized["newly_linked_count"]
        + normalized["insert_conflict_count"]
        != normalized["users_to_match_count"]
    ):
        raise ValueError(
            "result_summary newly_linked_count + insert_conflict_count must equal "
            "users_to_match_count"
        )

    fallback_count = normalized["fallback_count"]
    if fallback_count > assignment_count:
        raise ValueError(
            "result_summary.fallback_count must not exceed assignment_count"
        )
    if not isinstance(normalized["batch_has_fallback"], bool):
        raise ValueError("result_summary.batch_has_fallback must be a boolean")
    if normalized["batch_has_fallback"] != (fallback_count > 0):
        raise ValueError(
            "result_summary.batch_has_fallback must match fallback_count > 0"
        )

    fallback_rate = normalized["fallback_rate"]
    expected_rate = None if assignment_count == 0 else fallback_count / assignment_count
    if expected_rate is None:
        if fallback_rate is not None:
            raise ValueError(
                "result_summary.fallback_rate must be null when assignment_count is zero"
            )
    elif (
        isinstance(fallback_rate, bool)
        or not isinstance(fallback_rate, (int, float))
        or not math.isfinite(float(fallback_rate))
        or not math.isclose(
            float(fallback_rate), expected_rate, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(
            "result_summary.fallback_rate must equal fallback_count / assignment_count"
        )
    if fallback_rate is not None and not 0.0 <= float(fallback_rate) <= 1.0:
        raise ValueError("result_summary.fallback_rate must be between zero and one")

    aggregates: dict[str, dict[str, int]] = {}
    for key in _RESULT_SUMMARY_AGGREGATE_KEYS:
        aggregates[key] = _validate_count_aggregate(
            normalized[key],
            field_name=f"result_summary.{key}",
        )
        normalized[key] = aggregates[key]
    if sum(aggregates["fallback_reason_counts"].values()) != fallback_count:
        raise ValueError(
            "result_summary fallback_reason_counts must sum to fallback_count"
        )
    if set(aggregates["fallback_reason_counts"]) != _FALLBACK_REASON_KEYS:
        raise ValueError(
            "result_summary fallback_reason_counts must contain exactly the "
            "contract fallback reasons"
        )
    for key in ("segment_assignment_counts", "similarity_score_buckets"):
        if sum(aggregates[key].values()) != assignment_count:
            raise ValueError(
                f"result_summary {key} must sum to assignment_count"
            )

    if normalized["run_fallback_count"] > normalized["run_assignment_count"]:
        raise ValueError(
            "result_summary.run_fallback_count must not exceed run_assignment_count"
        )
    for key in _RESULT_SUMMARY_STRING_KEYS:
        if key in normalized and normalized[key] is not None:
            _validate_label(normalized[key], f"result_summary.{key}")
    if "ann_applied" in normalized and not isinstance(normalized["ann_applied"], bool):
        raise ValueError("result_summary.ann_applied must be a boolean")
    if "ann_applied" in normalized and normalized["ann_applied"] != (
        normalized["ann_query_user_count"] > 0
    ):
        raise ValueError(
            "result_summary.ann_applied must match ann_query_user_count > 0"
        )

    encoded = canonical_json_bytes(normalized)
    if len(encoded) > MAX_RESULT_SUMMARY_JSON_BYTES:
        raise ValueError(
            "result_summary exceeds the bounded persisted summary size "
            f"of {MAX_RESULT_SUMMARY_JSON_BYTES} bytes"
        )
    return normalized


def build_input_manifest(
    *,
    canonical_input: Mapping[str, Any],
    matcher: Mapping[str, Any],
    result_summary: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {
        "version": INPUT_MANIFEST_VERSION,
        "canonical_input": build_canonical_input(canonical_input),
        "matcher": build_matcher_manifest(matcher),
        "result_summary": build_result_summary(result_summary),
    }
    return validate_input_manifest(manifest)


def validate_input_manifest(input_manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json_value(
        input_manifest,
        path="input_manifest",
        permit_tuple=True,
    )
    expected_keys = {"version", "canonical_input", "matcher", "result_summary"}
    _reject_unknown_keys(normalized, frozenset(expected_keys), "input_manifest")
    if set(normalized) != expected_keys:
        missing = expected_keys.difference(normalized)
        raise ValueError(f"input_manifest is missing required keys: {sorted(missing)}")
    if normalized["version"] != INPUT_MANIFEST_VERSION:
        raise ValueError(
            f"input_manifest.version must be {INPUT_MANIFEST_VERSION!r}"
        )
    manifest = {
        "version": INPUT_MANIFEST_VERSION,
        "canonical_input": build_canonical_input(normalized["canonical_input"]),
        "matcher": build_matcher_manifest(normalized["matcher"]),
        "result_summary": validate_result_summary(normalized["result_summary"]),
    }
    encoded = canonical_json_bytes(manifest)
    if len(encoded) > MAX_INPUT_MANIFEST_JSON_BYTES:
        raise ValueError(
            "input_manifest exceeds the bounded persisted manifest size "
            f"of {MAX_INPUT_MANIFEST_JSON_BYTES} bytes"
        )
    return manifest


def build_input_fingerprint(
    canonical_input_or_manifest: Mapping[str, Any],
) -> str:
    if "canonical_input" in canonical_input_or_manifest:
        canonical_input = validate_input_manifest(canonical_input_or_manifest)[
            "canonical_input"
        ]
    else:
        canonical_input = build_canonical_input(canonical_input_or_manifest)
    return sha256_canonical_json(canonical_input)


def _merge_mapping_and_fields(
    mapping: Mapping[str, Any] | None,
    fields: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    if mapping is None:
        merged: dict[str, Any] = {}
    elif isinstance(mapping, Mapping):
        merged = dict(mapping)
    else:
        raise ValueError(f"{field_name} must be a mapping")
    for key, value in fields.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"conflicting values supplied for {field_name}.{key}")
        merged[key] = value
    return merged


def _with_version(
    section: Mapping[str, Any],
    *,
    expected_version: str,
    section_name: str,
) -> dict[str, Any]:
    if not isinstance(section, Mapping):
        raise ValueError(f"{section_name} must be a mapping")
    normalized = _normalize_json_value(
        section,
        path=section_name,
        permit_tuple=True,
    )
    version = normalized.get("version", expected_version)
    if version != expected_version:
        raise ValueError(f"{section_name}.version must be {expected_version!r}")
    return {"version": expected_version, **normalized}


def _normalize_audience_scope(value: Any) -> Any:
    normalized = _normalize_json_value(
        value,
        path="$.audience_scope",
        permit_tuple=True,
    )
    if not isinstance(normalized, dict):
        raise ValueError("audience_scope must be a JSON object")
    for key in ("user_ids", "explicit_user_ids"):
        if key in normalized:
            normalized[key] = normalize_explicit_user_ids(normalized[key])
    for key in _NON_LOGICAL_REQUEST_KEYS:
        normalized.pop(key, None)
    return normalized


def _normalize_json_value(value: Any, *, path: str, permit_tuple: bool) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or Infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            normalized[key] = _normalize_json_value(
                child,
                path=f"{path}.{key}",
                permit_tuple=permit_tuple,
            )
        return normalized
    if isinstance(value, list) or (permit_tuple and isinstance(value, tuple)):
        return [
            _normalize_json_value(
                child,
                path=f"{path}[{index}]",
                permit_tuple=permit_tuple,
            )
            for index, child in enumerate(value)
        ]
    raise ValueError(f"{path} contains unsupported JSON value {type(value).__name__}")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    section_name: str,
) -> None:
    unknown = set(value).difference(allowed_keys)
    if unknown:
        raise ValueError(
            f"{section_name} contains forbidden or unbounded keys: {sorted(unknown)}"
        )


def _validate_non_negative_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_sha256(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _validate_label(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_LABEL_LENGTH
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string of at most "
            f"{MAX_LABEL_LENGTH} UTF-8 bytes"
        )


def _validate_count_aggregate(value: Any, *, field_name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    if len(value) > MAX_AGGREGATE_BUCKET_COUNT:
        raise ValueError(
            f"{field_name} exceeds the bounded aggregate key count "
            f"of {MAX_AGGREGATE_BUCKET_COUNT}"
        )
    normalized: dict[str, int] = {}
    for key, count in value.items():
        _validate_label(key, f"{field_name} key")
        _validate_non_negative_integer(count, f"{field_name}.{key}")
        normalized[key] = count
    return normalized
