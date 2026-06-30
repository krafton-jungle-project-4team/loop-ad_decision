from __future__ import annotations

from dataclasses import replace

from app.contents.generators import ContentGenerator, PartialContentGenerationError
from app.contents.repository import ContentRepository
from app.contents.types import (
    ACTION_STATUS_CONTENT_GENERATED,
    ACTION_STATUS_FAILED,
    ACTION_STATUS_RECOMMENDED,
    ERROR_TYPE_CONTENT_GENERATION_FAILED,
    GENERATION_STATUS_FAILED,
    VARIANT_KEYS,
    ContentGenerationActionResult,
    ContentGenerationSummary,
    GeneratedContentDraft,
    RecommendationActionTarget,
)


class ContentGenerationService:
    def __init__(
        self,
        *,
        repository: ContentRepository,
        generator: ContentGenerator,
    ) -> None:
        self.repository = repository
        self.generator = generator

    def generate_for_actions(
        self,
        *,
        project_id: int | str,
        analysis_date: str,
        run_id: int | None = None,
        force: bool = False,
    ) -> ContentGenerationSummary:
        summary = ContentGenerationSummary()
        eligible_statuses = self._eligible_statuses(force)
        targets = list(
            self.repository.list_generation_targets(
                project_id=project_id,
                analysis_date=analysis_date,
                eligible_statuses=eligible_statuses,
            )
        )
        summary.actions_seen = len(targets)

        for target in targets:
            result = self._generate_for_target(
                target=target,
                run_id=run_id,
                force=force,
                eligible_statuses=eligible_statuses,
            )
            summary.add_result(result)

        return summary

    def _generate_for_target(
        self,
        *,
        target: RecommendationActionTarget,
        run_id: int | None,
        force: bool,
        eligible_statuses: tuple[str, ...],
    ) -> ContentGenerationActionResult:
        if target.segment.is_default or target.status not in eligible_statuses:
            return ContentGenerationActionResult(
                recommendation_action_id=target.id,
                status="skipped",
                skipped_variant_keys=VARIANT_KEYS,
            )

        created: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []

        try:
            with self.repository.generation_lock(
                project_id=target.project_id,
                recommendation_action_id=target.id,
            ):
                for variant_key in VARIANT_KEYS:
                    existing = self.repository.get_generated_content(
                        project_id=target.project_id,
                        recommendation_action_id=target.id,
                        variant_key=variant_key,
                    )
                    if existing is not None and existing.is_default_seed:
                        skipped.append(variant_key)
                        continue
                    if existing is not None and not force:
                        skipped.append(variant_key)
                        continue

                    draft = self.generator.generate(target=target, variant_key=variant_key)
                    draft = _with_created_run_id(draft, run_id)
                    if not draft.has_required_fields():
                        raise PartialContentGenerationError(
                            f"generated content for {variant_key} is missing required fields",
                            draft,
                        )
                    self.repository.upsert_generated_content(draft=draft, force=force)
                    created.append(variant_key)

                if not failed:
                    self.repository.mark_action_content_generated(recommendation_action_id=target.id)
        except PartialContentGenerationError as exc:
            failed_variant = self._failed_variant_from_partial(exc.draft)
            if exc.draft is not None and exc.draft.has_required_fields():
                self.repository.upsert_generated_content(
                    draft=_as_failed_draft(exc.draft, run_id=run_id),
                    force=force,
                )
            self.repository.mark_action_failed(
                recommendation_action_id=target.id,
                error_type=ERROR_TYPE_CONTENT_GENERATION_FAILED,
                error_message=str(exc),
            )
            return ContentGenerationActionResult(
                recommendation_action_id=target.id,
                status="failed",
                created_variant_keys=tuple(created),
                skipped_variant_keys=tuple(skipped),
                failed_variant_keys=(failed_variant,),
                error_message=str(exc),
            )
        except Exception as exc:
            self.repository.mark_action_failed(
                recommendation_action_id=target.id,
                error_type=ERROR_TYPE_CONTENT_GENERATION_FAILED,
                error_message=str(exc),
            )
            return ContentGenerationActionResult(
                recommendation_action_id=target.id,
                status="failed",
                created_variant_keys=tuple(created),
                skipped_variant_keys=tuple(skipped),
                failed_variant_keys=tuple(
                    key for key in VARIANT_KEYS if key not in created and key not in skipped
                )
                or VARIANT_KEYS,
                error_message=str(exc),
            )

        status = "created" if created else "skipped"
        return ContentGenerationActionResult(
            recommendation_action_id=target.id,
            status=status,
            created_variant_keys=tuple(created),
            skipped_variant_keys=tuple(skipped),
            failed_variant_keys=tuple(failed),
        )

    def _eligible_statuses(self, force: bool) -> tuple[str, ...]:
        if force:
            return (
                ACTION_STATUS_RECOMMENDED,
                ACTION_STATUS_FAILED,
                ACTION_STATUS_CONTENT_GENERATED,
            )
        return (ACTION_STATUS_RECOMMENDED,)

    def _failed_variant_from_partial(self, draft: GeneratedContentDraft | None) -> str:
        if draft is not None and draft.variant_key:
            return draft.variant_key
        return "unknown"


def _with_created_run_id(
    draft: GeneratedContentDraft,
    run_id: int | None,
) -> GeneratedContentDraft:
    if draft.created_run_id == run_id:
        return draft
    return replace(draft, created_run_id=run_id)


def _as_failed_draft(
    draft: GeneratedContentDraft,
    *,
    run_id: int | None,
) -> GeneratedContentDraft:
    metadata = {
        **draft.metadata,
        "error_type": ERROR_TYPE_CONTENT_GENERATION_FAILED,
    }
    return GeneratedContentDraft(
        project_id=draft.project_id,
        recommendation_action_id=draft.recommendation_action_id,
        segment_id=draft.segment_id,
        variant_key=draft.variant_key,
        content_type=draft.content_type,
        title=draft.title,
        body=draft.body,
        cta_label=draft.cta_label,
        landing_url=draft.landing_url,
        image_prompt=draft.image_prompt,
        generation_model=draft.generation_model,
        generation_status=GENERATION_STATUS_FAILED,
        created_run_id=run_id if run_id is not None else draft.created_run_id,
        image_url=draft.image_url,
        media_s3_key=draft.media_s3_key,
        metadata=metadata,
    )
