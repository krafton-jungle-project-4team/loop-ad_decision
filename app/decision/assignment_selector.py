from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


EXACT_MATCHER_STRATEGY = "exact_cosine"
ANN_MATCHER_STRATEGY = "pgvector_hnsw_rerank"
ADAPTIVE_MATCHER_STRATEGY = "adaptive_exact_ann"
MATCHER_SELECTED_EXACT_REASON = "matcher_selected_exact"
MATCHER_SELECTOR_POLICY_VERSION = "assignment_matcher_exact_only_v1"

# Production stays exact-only until a production-backend benchmark approves a
# bounded ANN region. Tests may inject evidence regions through the selector.
PRODUCTION_APPROVED_ANN_REGIONS: tuple[Mapping[str, object], ...] = ()

_REGION_BOUND_NAMES = (
    "user_count",
    "segment_count",
    "dimension",
    "page_size",
    "workload_size",
)


class AssignmentMatcherSelector:
    """Conservatively select exact matching unless an evidence region applies."""

    def __init__(
        self,
        *,
        approved_ann_regions: Sequence[
            Mapping[str, object]
        ] = PRODUCTION_APPROVED_ANN_REGIONS,
        policy_version: str = MATCHER_SELECTOR_POLICY_VERSION,
    ) -> None:
        normalized_policy_version = policy_version.strip()
        if not normalized_policy_version:
            raise ValueError("matcher selector policy_version must not be empty")

        self._policy_version = normalized_policy_version
        self._approved_ann_regions = tuple(
            self._normalize_region(region) for region in approved_ann_regions
        )

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def select(
        self,
        *,
        user_count: int | None,
        segment_count: int | None,
        dimension: int | None,
        page_size: int | None,
        backend: str | None,
    ) -> dict[str, str | int | bool | None]:
        normalized_backend = self._normalize_backend(backend)
        numeric_inputs = (user_count, segment_count, dimension, page_size)
        if (
            normalized_backend is None
            or not all(self._is_positive_int(value) for value in numeric_inputs)
            or user_count is None
            or page_size is None
            or user_count > page_size
        ):
            return self._selection(
                strategy=EXACT_MATCHER_STRATEGY,
                backend=normalized_backend,
                workload_size=None,
            )

        # The positive-integer guard above narrows these values at runtime.
        assert segment_count is not None
        assert dimension is not None
        workload_size = user_count * segment_count * dimension
        workload = {
            "user_count": user_count,
            "segment_count": segment_count,
            "dimension": dimension,
            "page_size": page_size,
            "workload_size": workload_size,
        }
        for region in self._approved_ann_regions:
            if self._region_contains(
                region=region,
                backend=normalized_backend,
                workload=workload,
            ):
                return self._selection(
                    strategy=ANN_MATCHER_STRATEGY,
                    backend=normalized_backend,
                    workload_size=workload_size,
                )

        return self._selection(
            strategy=EXACT_MATCHER_STRATEGY,
            backend=normalized_backend,
            workload_size=workload_size,
        )

    def _selection(
        self,
        *,
        strategy: str,
        backend: str | None,
        workload_size: int | None,
    ) -> dict[str, str | int | bool | None]:
        use_ann = strategy == ANN_MATCHER_STRATEGY
        return {
            "policy_version": self._policy_version,
            "backend": backend,
            "matcher_strategy": strategy,
            "use_ann": use_ann,
            "workload_size": workload_size,
            "ann_not_applied_reason": (
                None if use_ann else MATCHER_SELECTED_EXACT_REASON
            ),
        }

    @classmethod
    def _normalize_region(
        cls,
        region: Mapping[str, object],
    ) -> dict[str, str | int]:
        backend = cls._normalize_backend(region.get("backend"))
        if backend is None:
            raise ValueError("ANN evidence region backend must not be empty")

        normalized: dict[str, str | int] = {"backend": backend}
        for name in _REGION_BOUND_NAMES:
            minimum_key = f"min_{name}"
            maximum_key = f"max_{name}"
            minimum = region.get(minimum_key)
            maximum = region.get(maximum_key)
            if not cls._is_positive_int(minimum) or not cls._is_positive_int(
                maximum
            ):
                raise ValueError(
                    "ANN evidence region bounds must be positive integers: "
                    f"{minimum_key}, {maximum_key}"
                )
            if minimum > maximum:
                raise ValueError(
                    "ANN evidence region minimum exceeded maximum: "
                    f"{minimum_key}, {maximum_key}"
                )
            normalized[minimum_key] = minimum
            normalized[maximum_key] = maximum
        return normalized

    @staticmethod
    def _normalize_backend(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return normalized or None

    @staticmethod
    def _is_positive_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @staticmethod
    def _region_contains(
        *,
        region: Mapping[str, str | int],
        backend: str,
        workload: Mapping[str, int],
    ) -> bool:
        if region["backend"] != backend:
            return False
        return all(
            region[f"min_{name}"]
            <= workload[name]
            <= region[f"max_{name}"]
            for name in _REGION_BOUND_NAMES
        )


def aggregate_matcher_strategy(page_strategies: Iterable[str]) -> str:
    """Return the truthful execution strategy for a sequence of page choices."""

    strategies = set(page_strategies)
    supported = {
        EXACT_MATCHER_STRATEGY,
        ANN_MATCHER_STRATEGY,
        ADAPTIVE_MATCHER_STRATEGY,
    }
    unsupported = strategies - supported
    if unsupported:
        raise ValueError(
            "unsupported page matcher strategy: " + ", ".join(sorted(unsupported))
        )
    if ADAPTIVE_MATCHER_STRATEGY in strategies or (
        EXACT_MATCHER_STRATEGY in strategies
        and ANN_MATCHER_STRATEGY in strategies
    ):
        return ADAPTIVE_MATCHER_STRATEGY
    if ANN_MATCHER_STRATEGY in strategies:
        return ANN_MATCHER_STRATEGY
    return EXACT_MATCHER_STRATEGY
