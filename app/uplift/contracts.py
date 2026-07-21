from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class UpliftTrainingExample:
    experiment_unit_id: str
    project_id: str
    promotion_run_id: str
    ad_experiment_id: str
    segment_id: str
    user_id: str
    audience_snapshot_id: str
    vector_generation_id: str
    features: tuple[float, ...]
    treatment: int
    outcome: int
    treatment_probability: float
    assigned_at: datetime | None = None
    outcome_window_start: datetime | None = None
    outcome_window_end: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_unit_id": self.experiment_unit_id,
            "project_id": self.project_id,
            "promotion_run_id": self.promotion_run_id,
            "ad_experiment_id": self.ad_experiment_id,
            "segment_id": self.segment_id,
            "user_id": self.user_id,
            "audience_snapshot_id": self.audience_snapshot_id,
            "vector_generation_id": self.vector_generation_id,
            "features": list(self.features),
            "treatment": self.treatment,
            "outcome": self.outcome,
            "treatment_probability": self.treatment_probability,
            "assigned_at": _isoformat(self.assigned_at),
            "outcome_window_start": _isoformat(self.outcome_window_start),
            "outcome_window_end": _isoformat(self.outcome_window_end),
        }


@dataclass(frozen=True, slots=True)
class UpliftDatasetBuildResult:
    examples: tuple[UpliftTrainingExample, ...]
    excluded_reason_counts: Mapping[str, int]
    source_unit_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "source_unit_count": self.source_unit_count,
            "training_example_count": len(self.examples),
            "excluded_reason_counts": dict(self.excluded_reason_counts),
        }


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None

