from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "manifests"
    / "hotel_booking_behavior.v2.json"
)


class BehaviorManifestError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_behavior_manifest() -> Mapping[str, Any]:
    try:
        raw = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BehaviorManifestError("behavior vector manifest is unavailable") from exc
    _validate_manifest(raw)
    return MappingProxyType(raw)


def behavior_manifest_hash() -> str:
    payload = json.dumps(
        dict(load_behavior_manifest()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_destination_id(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    return destination_alias_lookup().get(normalized, normalized)


@lru_cache(maxsize=1)
def destination_alias_lookup() -> Mapping[str, str]:
    aliases: dict[str, str] = {}
    manifest = load_behavior_manifest()
    for canonical, values in manifest["destination_aliases"].items():
        canonical_id = _normalize_text(str(canonical))
        if not canonical_id:
            raise BehaviorManifestError("canonical destination id must not be empty")
        for value in (canonical, *values):
            alias = _normalize_text(str(value))
            previous = aliases.setdefault(alias, canonical_id)
            if previous != canonical_id:
                raise BehaviorManifestError(
                    f"destination alias maps to multiple ids: {alias}"
                )
    return MappingProxyType(aliases)


def clickhouse_canonical_destination_sql(source_expression: str) -> str:
    """Build the ClickHouse canonicalization expression from the same manifest.

    `source_expression` must be an internal SQL fragment, never user input.
    Values originating in the manifest are escaped before interpolation.
    """

    normalized = (
        "lowerUTF8(replaceRegexpAll(trimBoth("
        + source_expression
        + "), '\\\\s+', ' '))"
    )
    groups: dict[str, list[str]] = {}
    for alias, canonical in destination_alias_lookup().items():
        groups.setdefault(canonical, []).append(alias)
    arguments: list[str] = []
    for canonical in sorted(groups):
        aliases = ", ".join(
            f"'{_escape_clickhouse_string(value)}'"
            for value in sorted(groups[canonical])
        )
        arguments.extend(
            (
                f"{normalized} IN ({aliases})",
                f"'{_escape_clickhouse_string(canonical)}'",
            )
        )
    arguments.append(normalized)
    return "multiIf(" + ", ".join(arguments) + ")"


def manifest_candidate_block_weights() -> Mapping[str, Mapping[str, float]]:
    raw = load_behavior_manifest()["candidate_block_weights"]
    return MappingProxyType(
        {
            str(candidate): MappingProxyType(
                {str(block): float(weight) for block, weight in weights.items()}
            )
            for candidate, weights in raw.items()
        }
    )


def manifest_candidate_query_indices() -> Mapping[str, tuple[int, ...]]:
    manifest = load_behavior_manifest()
    indices = {
        str(item["name"]): int(item["index"])
        for item in manifest["dimensions"]
    }
    return MappingProxyType(
        {
            str(candidate): tuple(indices[str(name)] for name in names)
            for candidate, names in manifest["candidate_query_dimensions"].items()
        }
    )


def manifest_candidate_hard_predicates() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType(
        {
            str(candidate): tuple(str(value) for value in values)
            for candidate, values in load_behavior_manifest()[
                "candidate_hard_predicates"
            ].items()
        }
    )


def manifest_season_query_indices() -> Mapping[str, int]:
    manifest = load_behavior_manifest()
    indices = {
        str(item["name"]): int(item["index"])
        for item in manifest["dimensions"]
    }
    return MappingProxyType(
        {
            str(season): indices[str(name)]
            for season, name in manifest["season_query_dimensions"].items()
        }
    )


def manifest_intent_benefit_query_indices() -> Mapping[str, tuple[int, ...]]:
    manifest = load_behavior_manifest()
    indices = {
        str(item["name"]): int(item["index"])
        for item in manifest["dimensions"]
    }
    return MappingProxyType(
        {
            str(benefit): tuple(indices[str(name)] for name in names)
            for benefit, names in manifest[
                "intent_benefit_query_dimensions"
            ].items()
        }
    )


def manifest_blocks() -> Mapping[str, range]:
    raw = load_behavior_manifest()["blocks"]
    return MappingProxyType(
        {
            str(name): range(int(bounds[0]), int(bounds[1]) + 1)
            for name, bounds in raw.items()
        }
    )


def order_vector_terms_by_manifest(
    terms_by_name: Mapping[str, str],
) -> tuple[str, ...]:
    names = tuple(
        str(item["name"])
        for item in load_behavior_manifest()["dimensions"]
    )
    missing = sorted(set(names) - set(terms_by_name))
    extra = sorted(set(terms_by_name) - set(names))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise BehaviorManifestError(
            "ClickHouse vector terms do not match behavior manifest: "
            + "; ".join(details)
        )
    return tuple(terms_by_name[name] for name in names)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _escape_clickhouse_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _validate_manifest(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise BehaviorManifestError("behavior vector manifest must be an object")
    required = {
        "schema_version",
        "vector_version",
        "vector_dim",
        "missing_value_policy",
        "normalization",
        "query_block_normalization",
        "window_policy",
        "destination_alias_version",
        "destination_aliases",
        "canonical_destination_id_patterns",
        "blocks",
        "candidate_block_weights",
        "candidate_query_dimensions",
        "candidate_hard_predicates",
        "season_query_dimensions",
        "intent_benefit_query_dimensions",
        "dimensions",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise BehaviorManifestError(
            "behavior vector manifest fields are missing: " + ", ".join(missing)
        )
    if raw["query_block_normalization"] != (
        "l2_per_active_block_then_weight_then_global_l2"
    ):
        raise BehaviorManifestError("query block normalization is unsupported")
    destination_patterns = raw["canonical_destination_id_patterns"]
    if (
        not isinstance(destination_patterns, list)
        or not destination_patterns
        or any(not isinstance(value, str) or not value for value in destination_patterns)
    ):
        raise BehaviorManifestError(
            "canonical destination id patterns must be non-empty strings"
        )
    vector_dim = int(raw["vector_dim"])
    dimensions = raw["dimensions"]
    if vector_dim != 64 or not isinstance(dimensions, list) or len(dimensions) != 64:
        raise BehaviorManifestError("behavior vector manifest must define 64 dimensions")
    indices = [int(item.get("index", -1)) for item in dimensions]
    if indices != list(range(64)):
        raise BehaviorManifestError("behavior vector dimensions must be ordered 0..63")
    names = [str(item.get("name", "")) for item in dimensions]
    if any(not name for name in names) or len(set(names)) != 64:
        raise BehaviorManifestError("behavior vector dimension names must be unique")
    blocks = raw["blocks"]
    covered: list[int] = []
    for bounds in blocks.values():
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise BehaviorManifestError("behavior vector block bounds are invalid")
        covered.extend(range(int(bounds[0]), int(bounds[1]) + 1))
    if sorted(covered) != list(range(64)):
        raise BehaviorManifestError("behavior vector blocks must cover 0..63 once")
    for item in dimensions:
        if item.get("block") not in blocks:
            raise BehaviorManifestError("dimension references an unknown block")
        if not item.get("raw_calculation"):
            raise BehaviorManifestError("dimension raw calculation is required")
        if not isinstance(item.get("query_enabled"), bool):
            raise BehaviorManifestError("dimension query_enabled must be boolean")
    for candidate, weights in raw["candidate_block_weights"].items():
        if not candidate or not isinstance(weights, dict) or not weights:
            raise BehaviorManifestError("candidate block weights are invalid")
        if set(weights) - set(blocks):
            raise BehaviorManifestError("candidate references an unknown block")
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
            raise BehaviorManifestError("candidate block weights must sum to one")
    candidate_types = set(raw["candidate_block_weights"])
    if set(raw["candidate_query_dimensions"]) != candidate_types:
        raise BehaviorManifestError("candidate query dimensions are incomplete")
    if set(raw["candidate_hard_predicates"]) != candidate_types:
        raise BehaviorManifestError("candidate hard predicates are incomplete")
    query_enabled_names = {
        str(item["name"]) for item in dimensions if item["query_enabled"]
    }
    for dimension_names in raw["candidate_query_dimensions"].values():
        if not dimension_names or set(dimension_names) - query_enabled_names:
            raise BehaviorManifestError("candidate query dimension is invalid")
    if set(raw["season_query_dimensions"].values()) - query_enabled_names:
        raise BehaviorManifestError("season query dimension is invalid")
    if any(
        not names or set(names) - query_enabled_names
        for names in raw["intent_benefit_query_dimensions"].values()
    ):
        raise BehaviorManifestError("intent benefit query dimensions are invalid")
    if any(
        not isinstance(predicates, list) or not predicates
        for predicates in raw["candidate_hard_predicates"].values()
    ):
        raise BehaviorManifestError("candidate hard predicates are invalid")
