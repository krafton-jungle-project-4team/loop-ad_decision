from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


VARIANT_KEYS = ("control", "treatment_a")

ACTION_STATUS_RECOMMENDED = "recommended"
ACTION_STATUS_FAILED = "failed"
ACTION_STATUS_CONTENT_GENERATED = "content_generated"

GENERATION_STATUS_GENERATED = "generated"
GENERATION_STATUS_FAILED = "failed"

ERROR_TYPE_CONTENT_GENERATION_FAILED = "content_generation_failed"


@dataclass(frozen=True)
class SegmentContext:
    id: int
    segment_key: str
    name: str
    is_default: bool = False
    description: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationActionTarget:
    id: int
    project_id: int | str
    recommendation_result_id: int
    action_key: str
    status: str
    segment: SegmentContext
    analysis_date: date | str
    action_type: str | None = None
    action_title: str | None = None
    action_description: str | None = None
    content_type: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    root_cause: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedContentDraft:
    project_id: int | str
    recommendation_action_id: int
    segment_id: int
    variant_key: str
    content_type: str
    title: str
    body: str
    cta_label: str
    landing_url: str
    image_prompt: str
    generation_model: str
    generation_status: str = GENERATION_STATUS_GENERATED
    created_run_id: int | None = None
    image_url: str | None = None
    media_s3_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_required_fields(self) -> bool:
        return all(
            value.strip()
            for value in (
                self.variant_key,
                self.content_type,
                self.title,
                self.body,
                self.cta_label,
                self.landing_url,
                self.image_prompt,
                self.generation_model,
            )
        )


@dataclass(frozen=True)
class GeneratedContentRecord:
    id: int
    project_id: int | str
    variant_key: str
    generation_status: str
    recommendation_action_id: int | None = None
    segment_id: int | None = None
    created_run_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_default_seed(self) -> bool:
        return self.recommendation_action_id is None


@dataclass(frozen=True)
class ContentGenerationActionResult:
    recommendation_action_id: int
    status: str
    created_variant_keys: tuple[str, ...] = ()
    skipped_variant_keys: tuple[str, ...] = ()
    failed_variant_keys: tuple[str, ...] = ()
    error_message: str | None = None


@dataclass
class ContentGenerationSummary:
    actions_seen: int = 0
    actions_created: int = 0
    actions_skipped: int = 0
    actions_failed: int = 0
    variants_created: int = 0
    variants_skipped: int = 0
    variants_failed: int = 0
    results: list[ContentGenerationActionResult] = field(default_factory=list)

    def add_result(self, result: ContentGenerationActionResult) -> None:
        self.results.append(result)
        if result.status == "failed":
            self.actions_failed += 1
        elif result.status == "skipped":
            self.actions_skipped += 1
        else:
            self.actions_created += 1
        self.variants_created += len(result.created_variant_keys)
        self.variants_skipped += len(result.skipped_variant_keys)
        self.variants_failed += len(result.failed_variant_keys)
