from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

from app.analysis.segments import SEGMENT_DIMENSIONS, UNKNOWN_VALUE, normalize_dimensions


VECTOR_DIMENSION = 64
CATEGORICAL_BUCKETS = 40
BEHAVIOR_RESERVED_START = 40
FUTURE_RESERVED_START = 55
DEFAULT_EMBEDDING_VERSION = "segment_match_v1"
HASH_SEED = "loop-ad-segment-match-v1"

_ALIASES = {
    "device": "device_type",
    "channel": "acquisition_channel",
    "category": "primary_category",
}


def embed_user(attrs: Mapping[str, Any]) -> list[float]:
    """Embed a user into the segment matching vector space."""
    return _embed_attributes(attrs)


def embed_segment(rule_json: Mapping[str, Any]) -> list[float]:
    """Embed a segment rule into the same vector space as users.

    Slots 40-54 are reserved for future behavior features and are excluded from
    MVP matching. Slots 55-63 are reserved for later extensions. Both ranges are
    intentionally kept at zero so v1 matching depends only on categorical rules.
    """
    return _embed_attributes(rule_json)


def _embed_attributes(values: Mapping[str, Any]) -> list[float]:
    normalized = normalize_dimensions(_with_internal_dimension_names(values))
    vector = [0.0] * VECTOR_DIMENSION
    for dimension in SEGMENT_DIMENSIONS:
        value = normalized[dimension]
        if value == UNKNOWN_VALUE:
            continue
        bucket, sign = _hash_feature(dimension, value)
        vector[bucket] += sign
    return _l2_normalize(vector)


def _with_internal_dimension_names(values: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(values)
    for source_name, target_name in _ALIASES.items():
        if target_name not in merged and source_name in merged:
            merged[target_name] = merged[source_name]
    return merged


def _hash_feature(dimension: str, value: str) -> tuple[int, float]:
    payload = f"{HASH_SEED}:{dimension}:{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "big") % CATEGORICAL_BUCKETS
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return bucket, sign


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def is_zero_vector(vector: list[float]) -> bool:
    return all(value == 0 for value in vector)


def to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in vector) + "]"
