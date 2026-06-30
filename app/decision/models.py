from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any


JsonObject = dict[str, Any]


@dataclass(slots=True)
class Project:
    id: int
    project_key: str
    timezone: str


@dataclass(frozen=True, slots=True)
class ExistingSegment:
    id: int
    segment_key: str
    dimensions: JsonObject = field(default_factory=dict)
    matching_config: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class UserSegmentCandidate:
    external_user_id: str
    dimensions: JsonObject
    confidence: Decimal = Decimal("1.0")


@dataclass(slots=True)
class SegmentAnomaly:
    id: int
    project_id: int
    segment_id: int
    analysis_date: date
    metric_name: str
    severity: str
    impact_score: Decimal
    status: str
    evidence_json: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class RootCauseCandidate:
    id: int
    anomaly_id: int
    cause_type: str
    cause_key: str
    title: str
    description: str | None
    confidence_score: Decimal
    impact_score: Decimal
    rank_no: int
    evidence_json: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class ActionCatalogItem:
    id: int
    action_key: str
    name: str
    description: str | None
    target_funnel_step: str | None
    default_channel: str
    template_json: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class RecommendationResult:
    id: int
    project_id: int
    segment_id: int
    anomaly_id: int | None
    primary_root_cause_id: int | None
    analysis_date: date
    summary: str
    status: str
    recommendation_json: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class RecommendationAction:
    id: int
    recommendation_result_id: int
    project_id: int
    segment_id: int
    action_catalog_id: int | None
    action_key: str
    title: str
    description: str | None
    priority: int
    expected_effect_metric: str
    expected_effect_direction: str
    expected_effect_value: Decimal | None
    status: str
    metadata: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedContent:
    id: int
    project_id: int
    segment_id: int
    recommendation_action_id: int | None
    variant_key: str
    generation_status: str


@dataclass(slots=True)
class Experiment:
    id: int
    project_id: int
    segment_id: int
    recommendation_action_id: int
    name: str
    objective_metric: str
    target_value: Decimal
    allocation_policy: str
    status: str
    start_date: date
    winner_variant_id: int | None = None


@dataclass(slots=True)
class ExperimentVariant:
    id: int
    experiment_id: int
    project_id: int
    variant_key: str
    name: str
    generated_content_id: int | None
    is_control: bool
    traffic_weight: Decimal
    impression_count: int
    click_count: int
    conversion_count: int
    ctr: Decimal
    conversion_rate: Decimal
    status: str


@dataclass(slots=True)
class SegmentAdMapping:
    id: int
    project_id: int
    segment_id: int
    placement_key: str
    experiment_id: int | None
    experiment_variant_id: int | None
    generated_content_id: int | None
    traffic_weight: Decimal
    is_active: bool
    is_winner: bool


@dataclass(slots=True)
class VariantPerformance:
    experiment_variant_id: int
    ad_impression_count: int
    ad_click_count: int
    attributed_purchase_count: int


@dataclass(slots=True)
class ExperimentResultWindow:
    experiment: Experiment
    variants: list[ExperimentVariant]
    window_start: datetime
    window_end: datetime
