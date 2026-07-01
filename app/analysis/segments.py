from __future__ import annotations

import re
from collections.abc import Mapping


SEGMENT_DIMENSIONS = (
    "region",
    "age_group",
    "gender",
    "device_type",
    "acquisition_channel",
    "primary_category",
)

UNKNOWN_VALUE = "unknown"
DEFAULT_SEGMENT_KEY = "default"
UNKNOWN_MARKERS = {"", "unknown", "null", "none", "n/a", "na", "(not set)"}


def normalize_dimension_value(value: object) -> str:
    if value is None:
        return UNKNOWN_VALUE
    normalized = str(value).strip().lower()
    if normalized in UNKNOWN_MARKERS:
        return UNKNOWN_VALUE
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_가-힣-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or UNKNOWN_VALUE


def normalize_dimensions(values: Mapping[str, object]) -> dict[str, str]:
    return {
        dimension: normalize_dimension_value(values.get(dimension))
        for dimension in SEGMENT_DIMENSIONS
    }


def build_segment_key(dimensions: Mapping[str, object]) -> str:
    normalized = normalize_dimensions(dimensions)
    parts = [
        f"region_{normalized['region']}",
        f"age_{normalized['age_group']}",
        f"gender_{normalized['gender']}",
        f"device_{normalized['device_type']}",
        f"channel_{normalized['acquisition_channel']}",
        f"category_{normalized['primary_category']}",
    ]
    return "__".join(parts)


def build_segment_name(dimensions: Mapping[str, object]) -> str:
    normalized = normalize_dimensions(dimensions)
    return " / ".join(normalized[dimension] for dimension in SEGMENT_DIMENSIONS)


def is_default_segment_key(segment_key: str) -> bool:
    return segment_key == DEFAULT_SEGMENT_KEY
