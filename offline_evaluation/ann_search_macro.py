"""Three-segment end-to-end workload records for the ANN policy benchmark."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from offline_evaluation.ann_search_experiment import EXPERIMENT_VERSION, percentile


class MacroMode(StrEnum):
    CURRENT_RUNTIME = "current_runtime"
    CANDIDATE_POLICY = "candidate_policy"


@dataclass(frozen=True, slots=True)
class MacroObservation:
    experiment_version: str
    mode: MacroMode
    concurrency: int
    measured: bool
    scenario_ids: tuple[str, str, str]
    duration_ms: float
    selected_plans: tuple[str, str, str] = ("", "", "")
    requested_k: tuple[int | None, int | None, int | None] = (None, None, None)
    final_user_counts: tuple[int, int, int] = (0, 0, 0)
    error: str | None = None
    peak_rss_bytes: int | None = None
    temp_spill: bool = False

    def __post_init__(self) -> None:
        if self.experiment_version != EXPERIMENT_VERSION:
            raise ValueError("unsupported macro benchmark version")
        if self.concurrency <= 0 or self.duration_ms < 0:
            raise ValueError("macro benchmark timing is invalid")
        if len(self.scenario_ids) != 3 or len(set(self.scenario_ids)) != 3:
            raise ValueError("macro benchmark requires three distinct scenarios")
        if self.error is None:
            if any(not value for value in self.selected_plans):
                raise ValueError("successful macro requests require selected plans")
            if any(value < 0 for value in self.final_user_counts):
                raise ValueError("macro final counts must not be negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MacroObservation":
        scenario_ids = tuple(str(value) for value in payload["scenario_ids"])
        selected_plans = tuple(str(value) for value in payload["selected_plans"])
        requested_k = tuple(
            int(value) if value is not None else None
            for value in payload.get("requested_k", (None, None, None))
        )
        final_counts = tuple(
            int(value) for value in payload.get("final_user_counts", (0, 0, 0))
        )
        if not (
            len(scenario_ids)
            == len(selected_plans)
            == len(requested_k)
            == len(final_counts)
            == 3
        ):
            raise ValueError("macro benchmark arrays must contain three values")
        return cls(
            experiment_version=str(payload["experiment_version"]),
            mode=MacroMode(str(payload["mode"])),
            concurrency=int(payload["concurrency"]),
            measured=bool(payload["measured"]),
            scenario_ids=scenario_ids,  # type: ignore[arg-type]
            duration_ms=float(payload["duration_ms"]),
            selected_plans=selected_plans,  # type: ignore[arg-type]
            requested_k=requested_k,  # type: ignore[arg-type]
            final_user_counts=final_counts,  # type: ignore[arg-type]
            error=str(payload["error"]) if payload.get("error") else None,
            peak_rss_bytes=(
                int(payload["peak_rss_bytes"])
                if payload.get("peak_rss_bytes") is not None
                else None
            ),
            temp_spill=bool(payload.get("temp_spill", False)),
        )


def load_macro_observations(path: Path) -> list[MacroObservation]:
    observations: list[MacroObservation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                observations.append(MacroObservation.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid macro observation at line {line_number}: {exc}"
                ) from exc
    return observations


def write_macro_observations(
    path: Path,
    observations: Sequence[MacroObservation],
    *,
    append: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.to_dict(), sort_keys=True))
            handle.write("\n")


def build_macro_report(
    observations: Sequence[MacroObservation],
    *,
    required_concurrency: Sequence[int] = (1, 4),
) -> Mapping[str, Any]:
    measured = [item for item in observations if item.measured]
    grouped: dict[tuple[Any, ...], list[MacroObservation]] = defaultdict(list)
    for item in measured:
        grouped[(item.mode, item.concurrency, item.scenario_ids)].append(item)

    summaries: list[dict[str, Any]] = []
    lookup: dict[tuple[int, tuple[str, str, str], MacroMode], dict[str, Any]] = {}
    for (mode, concurrency, scenario_ids), runs in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][2], item[0][0].value),
    ):
        successful = [item for item in runs if item.error is None]
        durations = [item.duration_ms for item in successful]
        summary = {
            "mode": mode.value,
            "concurrency": concurrency,
            "scenario_ids": list(scenario_ids),
            "request_count": len(runs),
            "success_count": len(successful),
            "error_rate": (len(runs) - len(successful)) / len(runs),
            "p50_ms": percentile(durations, 0.50) if durations else None,
            "p95_ms": percentile(durations, 0.95) if durations else None,
            "p99_ms": percentile(durations, 0.99) if durations else None,
            "peak_rss_bytes": max(
                (
                    item.peak_rss_bytes
                    for item in successful
                    if item.peak_rss_bytes is not None
                ),
                default=None,
            ),
            "temp_spill": any(item.temp_spill for item in successful),
        }
        summaries.append(summary)
        lookup[(concurrency, scenario_ids, mode)] = summary

    comparisons: list[dict[str, Any]] = []
    passed = True
    reasons: list[str] = []
    scenario_batches = sorted({item.scenario_ids for item in measured})
    for concurrency in required_concurrency:
        for scenario_ids in scenario_batches:
            baseline = lookup.get(
                (concurrency, scenario_ids, MacroMode.CURRENT_RUNTIME)
            )
            candidate = lookup.get(
                (concurrency, scenario_ids, MacroMode.CANDIDATE_POLICY)
            )
            if baseline is None or candidate is None:
                passed = False
                reasons.append(
                    f"missing mode at concurrency={concurrency}, scenarios={scenario_ids}"
                )
                continue
            baseline_p95 = baseline["p95_ms"]
            candidate_p95 = candidate["p95_ms"]
            baseline_p99 = baseline["p99_ms"]
            candidate_p99 = candidate["p99_ms"]
            cell_passed = (
                baseline["error_rate"] == 0.0
                and candidate["error_rate"] == 0.0
                and not baseline["temp_spill"]
                and not candidate["temp_spill"]
                and baseline_p95 is not None
                and candidate_p95 is not None
                and baseline_p99 is not None
                and candidate_p99 is not None
                and candidate_p95 <= baseline_p95
                and candidate_p99 <= baseline_p99
            )
            passed = passed and cell_passed
            comparisons.append(
                {
                    "concurrency": concurrency,
                    "scenario_ids": list(scenario_ids),
                    "p95_ratio": (
                        candidate_p95 / baseline_p95
                        if baseline_p95 and candidate_p95 is not None
                        else None
                    ),
                    "p99_ratio": (
                        candidate_p99 / baseline_p99
                        if baseline_p99 and candidate_p99 is not None
                        else None
                    ),
                    "passed": cell_passed,
                }
            )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "passed": passed and bool(comparisons),
        "reasons": reasons,
        "summaries": summaries,
        "comparisons": comparisons,
    }
