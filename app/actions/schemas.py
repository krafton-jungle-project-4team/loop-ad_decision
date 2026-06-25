from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ActionScalar = str | int | float | bool | None

ROOT_CAUSE_SEVERITY_SCORES = {
    "critical": 1.0,
    "warning": 0.7,
    "normal": 0.3,
    "insufficient_data": 0.1,
}


class CauseEvidence(BaseModel):
    metric_name: str
    current_value: float | int | None = None
    baseline_value: float | int | None = None
    delta: float | int | None = None
    note: str | None = None
    drop_point: float | int | None = None
    relative_drop: float | int | None = None
    message: str | None = None

    model_config = ConfigDict(extra="forbid")


class CauseCandidate(BaseModel):
    cause_id: str
    cause_type: str
    label: str | None = None
    description: str | None = None
    affected_step: str | None = None
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[CauseEvidence] = Field(default_factory=list)
    attributes: dict[str, ActionScalar] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_root_cause_candidate(cls, data: Any) -> Any:
        return normalize_cause_candidate(data)


class ActionRecommendationRequest(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    segment: dict[str, str | None] = Field(default_factory=dict)
    causes: list[CauseCandidate]
    top_n: int = Field(default=5, ge=1, le=20)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_causes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "causes" not in data:
            return data
        normalized = dict(data)
        normalized["causes"] = [
            normalize_cause_candidate(cause)
            for cause in normalized.get("causes", [])
        ]
        return normalized

    @field_validator("window_start", "window_end")
    @classmethod
    def validate_timezone_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("window_start and window_end must be timezone-aware datetimes")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "ActionRecommendationRequest":
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be earlier than window_end")
        return self


class ActionExperiment(BaseModel):
    enabled: bool = True
    primary_metric: str
    guardrail_metrics: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RecommendedAction(BaseModel):
    action_id: str
    action_type: str
    title: str
    description: str
    target_step: str | None
    priority_score: float
    expected_impact: str
    rationale: str
    triggered_by: list[str]
    execution_hint: dict[str, ActionScalar]
    experiment: ActionExperiment | None = None

    model_config = ConfigDict(extra="forbid")


class ActionRecommendationResponse(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    segment: dict[str, str | None]
    recommendations: list[RecommendedAction]

    model_config = ConfigDict(extra="forbid")


def normalize_cause_candidate(data: Any) -> Any:
    raw = model_to_dict(data)
    if not isinstance(raw, dict):
        return data

    if is_root_cause_candidate_payload(raw):
        return normalize_root_cause_payload(raw)

    if isinstance(raw.get("severity"), str):
        raw = dict(raw)
        raw["severity"] = severity_to_score(raw["severity"])
    if "score" in raw and "confidence" not in raw:
        raw = dict(raw)
        raw["confidence"] = confidence_from_score(raw.get("score"))
    return raw


def model_to_dict(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump()
    return data


def is_root_cause_candidate_payload(raw: dict[str, Any]) -> bool:
    root_cause_fields = {"dimension", "value", "metric", "funnel_step", "score", "message"}
    return "cause_id" not in raw and root_cause_fields.issubset(raw)


def normalize_root_cause_payload(raw: dict[str, Any]) -> dict[str, Any]:
    cause_type = str(raw.get("cause_type", "unknown"))
    dimension = raw.get("dimension")
    value = raw.get("value")
    metric = raw.get("metric")
    funnel_step = raw.get("funnel_step")
    message = raw.get("message")
    drop_point = raw.get("drop_point")
    relative_drop = raw.get("relative_drop")

    attributes = {
        "dimension": dimension,
        "value": value,
        "metric": metric,
        "funnel_step": funnel_step,
        "current_value": raw.get("current_value"),
        "baseline_value": raw.get("baseline_value"),
        "drop_point": drop_point,
        "relative_drop": relative_drop,
        "support_share": raw.get("support_share"),
        "excess_lost_sessions": raw.get("excess_lost_sessions"),
    }

    return {
        "cause_id": f"{cause_type}:{dimension}:{value}:{metric}",
        "cause_type": cause_type,
        "label": None,
        "description": message,
        "affected_step": funnel_step,
        "severity": severity_to_score(raw.get("severity")),
        "confidence": confidence_from_score(raw.get("score")),
        "evidence": [
            {
                "metric_name": str(metric),
                "current_value": raw.get("current_value"),
                "baseline_value": raw.get("baseline_value"),
                "delta": drop_point,
                "drop_point": drop_point,
                "relative_drop": relative_drop,
                "note": message,
                "message": message,
            }
        ],
        "attributes": attributes,
    }


def severity_to_score(value: Any) -> float:
    if isinstance(value, str):
        return ROOT_CAUSE_SEVERITY_SCORES.get(value.lower(), 0.5)
    return clamp_float(value, default=0.5)


def confidence_from_score(value: Any) -> float:
    return clamp_float(to_float(value, default=1.5) / 3.0, default=0.5)


def clamp_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(number, 1.0))


def to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
