from __future__ import annotations

import hashlib
import itertools
import json
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence

from psycopg import errors

from app.audience_exclusions import (
    POSTGRES_EXCLUSION_RELATION,
    PromotionAudienceExclusionContext,
    PromotionAudienceExclusionReader,
    SegmentAudienceExclusionError,
)
from app.analysis.segment_suggester import DEFAULT_MAX_SUGGESTED_SEGMENTS


ALLOCATION_POLICY_VERSION = "hotel_segment_allocation.v1"
ALLOCATION_PREVIEW_VERSION = "audience_allocation_preview.v1"
ALLOCATION_POLICY_PRIORITY = {
    "target_destination_affinity": 0,
    "funnel_recovery": 1,
    "benefit_value_seeker": 2,
    "promotion_responsive": 3,
    "general_destination_explorer": 4,
    "intent_matched": 5,
}
ALLOCATION_POLICY_HASH = "sha256:" + hashlib.sha256(
    json.dumps(
        {
            "version": ALLOCATION_POLICY_VERSION,
            "priority": ALLOCATION_POLICY_PRIORITY,
            "normalized_fit": "(behavior_fit_score-threshold)/semantic_margin",
            "tie_break": "segment_id_asc",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class SegmentAudienceAllocationError(SegmentAudienceExclusionError):
    pass


@dataclass(frozen=True, slots=True)
class FinalAudienceAllocation:
    segment_id: str
    source_analysis_id: str
    source_snapshot_id: str
    final_snapshot_id: str
    allocation_plan_id: str
    final_user_count: int
    meets_min_sample_size: bool
    audience_status: str
    exclusion_revision: int


@dataclass(frozen=True, slots=True)
class ConfirmationAllocationResult:
    source_analysis_id: str
    allocation_plan_id: str
    exclusion_revision: int
    allocations: Mapping[str, FinalAudienceAllocation]


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    segment_id: str
    source_analysis_id: str
    snapshot_id: str
    candidate_type: str
    score_threshold: Decimal
    semantic_margin: Decimal
    min_sample_size: int
    segment_vector_id: str


class PostgresExecutor(Protocol):
    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        ...

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        ...

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        ...


class AudienceAllocationRepositoryProtocol(Protocol):
    def confirm_selection(
        self,
        *,
        confirmation_analysis_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        segment_ids: Sequence[str],
        min_sample_size: int,
        source_analysis_id: str | None = None,
    ) -> ConfirmationAllocationResult:
        ...

    def refresh_recommendation_previews(
        self,
        *,
        analysis_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        context: PromotionAudienceExclusionContext | None = None,
        exclude_segment_ids: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        ...

    def release_reserved_target(
        self,
        *,
        target_analysis_id: str,
        segment_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        release_entire_plan: bool = False,
    ) -> PromotionAudienceExclusionContext:
        ...


class AudienceAllocationService:
    def __init__(self, repository: AudienceAllocationRepositoryProtocol) -> None:
        self._repository = repository

    def confirm_selection(self, **kwargs: Any) -> ConfirmationAllocationResult:
        try:
            return self._repository.confirm_selection(**kwargs)
        except (errors.UndefinedTable, errors.UndefinedColumn) as exc:
            raise _allocation_contract_missing(kwargs) from exc

    def refresh_recommendation_previews(self, **kwargs: Any) -> Mapping[str, Any]:
        try:
            return self._repository.refresh_recommendation_previews(**kwargs)
        except (errors.UndefinedTable, errors.UndefinedColumn) as exc:
            raise _allocation_contract_missing(kwargs) from exc

    def release_reserved_target(
        self,
        **kwargs: Any,
    ) -> PromotionAudienceExclusionContext:
        try:
            return self._repository.release_reserved_target(**kwargs)
        except (errors.UndefinedTable, errors.UndefinedColumn) as exc:
            raise _allocation_contract_missing(kwargs) from exc


def _allocation_contract_missing(
    arguments: Mapping[str, Any],
) -> SegmentAudienceAllocationError:
    promotion_id = str(arguments.get("promotion_id") or "")
    segment_id = arguments.get("segment_id")
    return SegmentAudienceAllocationError(
        code="segment_audience_exclusion_contract_missing",
        promotion_id=promotion_id,
        segment_id=str(segment_id) if segment_id is not None else None,
        reason="audience allocation Data Contract is missing or incompatible",
    )


class PostgresAudienceAllocationRepository:
    """Persist confirmation-time allocation without re-running audience search."""

    def __init__(
        self,
        *,
        postgres: PostgresExecutor,
        exclusion_reader: PromotionAudienceExclusionReader,
        configured_candidate_limit: int = DEFAULT_MAX_SUGGESTED_SEGMENTS,
    ) -> None:
        if configured_candidate_limit <= 0:
            raise ValueError("configured_candidate_limit must be positive")
        self._db = postgres
        self._exclusion_reader = exclusion_reader
        self._configured_candidate_limit = configured_candidate_limit

    def confirm_selection(
        self,
        *,
        confirmation_analysis_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        segment_ids: Sequence[str],
        min_sample_size: int,
        source_analysis_id: str | None = None,
    ) -> ConfirmationAllocationResult:
        selected = tuple(sorted(set(segment_ids)))
        if not selected or len(selected) > self._configured_candidate_limit:
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_binding_invalid",
                promotion_id=promotion_id,
                reason=(
                    "confirmation selection exceeds the configured AI candidate "
                    f"limit ({self._configured_candidate_limit})"
                ),
            )
        self._lock_promotion(project_id=project_id, promotion_id=promotion_id)
        sources = self._load_source_snapshots(
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            segment_ids=selected,
            source_analysis_id=source_analysis_id,
        )
        source_analysis_ids = {source.source_analysis_id for source in sources}
        if len(source_analysis_ids) != 1:
            raise SegmentAudienceAllocationError(
                code="segment_audience_source_batch_mismatch",
                promotion_id=promotion_id,
                reason="one confirmation action must use one recommendation analysis",
            )
        source_analysis_id = next(iter(source_analysis_ids))
        retry = self._find_idempotent_result(
            confirmation_analysis_id=confirmation_analysis_id,
            project_id=project_id,
            promotion_id=promotion_id,
            sources=sources,
        )
        if retry is not None:
            return retry
        self._raise_if_sources_already_confirmed(
            confirmation_analysis_id=confirmation_analysis_id,
            project_id=project_id,
            promotion_id=promotion_id,
            sources=sources,
        )
        self._raise_if_segments_already_active(
            project_id=project_id,
            promotion_id=promotion_id,
            sources=sources,
        )

        context = self._exclusion_reader.load_active_exclusion_context(
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
        )
        plan_fingerprint = _plan_fingerprint(sources=sources, context=context)
        plan_id = _allocation_plan_id(
            promotion_id=promotion_id,
            plan_fingerprint=plan_fingerprint,
        )
        self._materialize_winners(
            project_id=project_id,
            promotion_id=promotion_id,
            sources=sources,
        )
        counts = self._winner_counts()
        missing = [
            source.segment_id
            for source in sources
            if counts.get(source.segment_id, 0) <= 0
        ]
        if missing:
            raise SegmentAudienceAllocationError(
                code="segment_audience_allocation_empty",
                promotion_id=promotion_id,
                segment_id=missing[0],
                reason="confirmation allocation produced zero final users",
            )

        self._ensure_target_rows(
            target_analysis_id=confirmation_analysis_id,
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            sources=sources,
        )

        reservation_revision = self._advance_exclusion_revision(
            promotion_id=promotion_id,
            expected_revision=context.revision,
        )

        self._insert_plan(
            plan_id=plan_id,
            confirmation_analysis_id=confirmation_analysis_id,
            source_analysis_id=source_analysis_id,
            promotion_id=promotion_id,
            plan_fingerprint=plan_fingerprint,
            selected_segment_ids=selected,
            exclusion_revision=reservation_revision,
        )
        allocations: dict[str, FinalAudienceAllocation] = {}
        for source in sources:
            final_snapshot_id = _final_snapshot_id(
                plan_id=plan_id,
                segment_id=source.segment_id,
            )
            final_count = counts[source.segment_id]
            audience_status = _audience_status(
                final_user_count=final_count,
                min_sample_size=min_sample_size,
            )
            self._insert_final_snapshot(
                source=source,
                confirmation_analysis_id=confirmation_analysis_id,
                plan_id=plan_id,
                final_snapshot_id=final_snapshot_id,
                final_user_count=final_count,
                min_sample_size=min_sample_size,
                audience_status=audience_status,
                context=context,
            )
            self._insert_final_members(
                source=source,
                plan_id=plan_id,
                final_snapshot_id=final_snapshot_id,
            )
            self._bind_target_audience(
                target_analysis_id=confirmation_analysis_id,
                segment_id=source.segment_id,
                plan_id=plan_id,
                final_snapshot_id=final_snapshot_id,
                final_user_count=final_count,
            )
            allocations[source.segment_id] = FinalAudienceAllocation(
                segment_id=source.segment_id,
                source_analysis_id=source.source_analysis_id,
                source_snapshot_id=source.snapshot_id,
                final_snapshot_id=final_snapshot_id,
                allocation_plan_id=plan_id,
                final_user_count=final_count,
                meets_min_sample_size=final_count >= min_sample_size,
                audience_status=audience_status,
                exclusion_revision=reservation_revision,
            )

        new_context = self._reserve_final_members(
            plan_id=plan_id,
            target_analysis_id=confirmation_analysis_id,
            project_id=project_id,
            promotion_id=promotion_id,
            previous=context,
            reservation_revision=reservation_revision,
        )
        self.refresh_recommendation_previews(
            analysis_id=source_analysis_id,
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            context=new_context,
            exclude_segment_ids=selected,
        )
        return ConfirmationAllocationResult(
            source_analysis_id=source_analysis_id,
            allocation_plan_id=plan_id,
            exclusion_revision=new_context.revision,
            allocations=allocations,
        )

    def refresh_recommendation_previews(
        self,
        *,
        analysis_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        context: PromotionAudienceExclusionContext | None = None,
        exclude_segment_ids: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        context = context or self._exclusion_reader.load_active_exclusion_context(
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
        )
        sources = self._load_unconfirmed_source_snapshots(
            analysis_id=analysis_id,
            project_id=project_id,
            promotion_id=promotion_id,
            exclude_segment_ids=exclude_segment_ids,
        )
        if len(sources) > self._configured_candidate_limit:
            raise SegmentAudienceAllocationError(
                code="segment_audience_candidate_limit_exceeded",
                promotion_id=promotion_id,
                reason=(
                    "recommendation candidate batch exceeds the configured "
                    f"limit ({self._configured_candidate_limit})"
                ),
            )
        candidate_segment_ids = sorted(source.segment_id for source in sources)
        previews: list[dict[str, Any]] = []
        for size in range(1, len(sources) + 1):
            for combination in itertools.combinations(sources, size):
                self._materialize_winners(
                    project_id=project_id,
                    promotion_id=promotion_id,
                    sources=combination,
                )
                counts = self._winner_counts()
                per_segment = []
                for source in combination:
                    count = counts.get(source.segment_id, 0)
                    per_segment.append(
                        {
                            "segment_id": source.segment_id,
                            "allocated_user_count": count,
                            "targetable": count > 0,
                            "meets_min_sample_size": count >= source.min_sample_size,
                            "audience_status": _audience_status(
                                final_user_count=count,
                                min_sample_size=source.min_sample_size,
                            ),
                        }
                    )
                previews.append({
                    "selected_segment_ids": sorted(
                        source.segment_id for source in combination
                    ),
                    "candidate_batch_analysis_id": analysis_id,
                    "exclusion_revision": context.revision,
                    "preview_version": ALLOCATION_PREVIEW_VERSION,
                    "allocation_policy_version": ALLOCATION_POLICY_VERSION,
                    "allocation_policy_hash": ALLOCATION_POLICY_HASH,
                    "total_allocated_user_count": sum(counts.values()),
                    "per_segment": per_segment,
                })
        payload = {
            "preview_version": ALLOCATION_PREVIEW_VERSION,
            "candidate_batch_analysis_id": analysis_id,
            "candidate_segment_ids": candidate_segment_ids,
            "exclusion_revision": context.revision,
            "allocation_policy_version": ALLOCATION_POLICY_VERSION,
            "allocation_policy_hash": ALLOCATION_POLICY_HASH,
            "allocation_previews": previews,
        }
        self._db.execute(
            """
            UPDATE promotion_analyses
            SET output_json = jsonb_set(
                    coalesce(output_json, '{}'::jsonb),
                    '{audience_allocation_preview_context}',
                    %s::jsonb,
                    true
                ),
                updated_at = now()
            WHERE analysis_id = %s
              AND project_id = %s
              AND campaign_id = %s
              AND promotion_id = %s
            """,
            (payload, analysis_id, project_id, campaign_id, promotion_id),
        )
        return payload

    def release_reserved_target(
        self,
        *,
        target_analysis_id: str,
        segment_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        release_entire_plan: bool = False,
    ) -> PromotionAudienceExclusionContext:
        self._lock_promotion(project_id=project_id, promotion_id=promotion_id)
        row = self._db.fetchone(
            """
            SELECT
                target.audience_snapshot_id,
                target.allocation_plan_id,
                target.audience_reservation_state,
                plan.candidate_batch_analysis_id,
                plan.status,
                jsonb_array_length(plan.selected_segment_ids_json)
                    AS plan_segment_count,
                EXISTS (
                    SELECT 1
                    FROM promotion_run_target_bindings AS binding
                    WHERE binding.allocation_plan_id = target.allocation_plan_id
                ) AS run_bound,
                (SELECT count(*) FILTER (WHERE excluded.state = 'reserved')
                 FROM promotion_audience_exclusion_members AS excluded
                 WHERE excluded.allocation_plan_id = target.allocation_plan_id)
                    AS reserved_count,
                (SELECT count(*) FILTER (WHERE excluded.state = 'consumed')
                 FROM promotion_audience_exclusion_members AS excluded
                 WHERE excluded.allocation_plan_id = target.allocation_plan_id)
                    AS consumed_count
            FROM promotion_target_segments AS target
            JOIN segment_audience_allocation_plans AS plan
              ON plan.allocation_plan_id = target.allocation_plan_id
            WHERE target.analysis_id = %s
              AND target.project_id = %s
              AND target.promotion_id = %s
              AND target.segment_id = %s
            FOR UPDATE
            """,
            (target_analysis_id, project_id, promotion_id, segment_id),
        )
        if row is None:
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_binding_invalid",
                promotion_id=promotion_id,
                segment_id=segment_id,
                reason="reserved target binding does not exist",
            )
        plan_segment_count = int(row["plan_segment_count"])
        if plan_segment_count > 1 and not release_entire_plan:
            raise SegmentAudienceAllocationError(
                code="segment_audience_partial_release_unsupported",
                promotion_id=promotion_id,
                segment_id=segment_id,
                reason=(
                    "a multi-segment confirmation must be released as one "
                    "allocation plan"
                ),
            )
        if (
            bool(row["run_bound"])
            or int(row["consumed_count"]) > 0
            or row["status"] == "locked"
        ):
            raise SegmentAudienceAllocationError(
                code="segment_audience_allocation_locked",
                promotion_id=promotion_id,
                segment_id=segment_id,
                reason="run-bound or consumed audience cannot be released",
            )
        plan_id = str(row["allocation_plan_id"])
        segment_filter = (
            "" if release_entire_plan else "AND segment_id = %s"
        )
        state_params: tuple[Any, ...] = (
            (project_id, promotion_id, plan_id)
            if release_entire_plan
            else (project_id, promotion_id, plan_id, segment_id)
        )
        state = self._db.fetchone(
            f"""
            SELECT count(*) FILTER (WHERE state = 'reserved') AS reserved_count,
                   count(*) FILTER (WHERE state = 'consumed') AS consumed_count
            FROM {POSTGRES_EXCLUSION_RELATION}
            WHERE project_id = %s
              AND promotion_id = %s
              AND allocation_plan_id = %s
              {segment_filter}
            """,
            state_params,
        )
        if state is not None and int(state["consumed_count"]) > 0:
            raise SegmentAudienceAllocationError(
                code="segment_audience_allocation_locked",
                promotion_id=promotion_id,
                segment_id=segment_id,
                reason="consumed audience cannot be released",
            )
        previous = self._exclusion_reader.load_active_exclusion_context(
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
        )
        next_revision = self._advance_exclusion_revision(
            promotion_id=promotion_id,
            expected_revision=previous.revision,
        )
        self._db.execute(
            f"""
            UPDATE {POSTGRES_EXCLUSION_RELATION}
            SET state = 'released',
                released_at = now(),
                consumed_at = NULL,
                revision = %s
            WHERE project_id = %s
              AND promotion_id = %s
              AND allocation_plan_id = %s
              {segment_filter}
              AND state = 'reserved'
            """,
            (next_revision, *state_params),
        )
        target_filter = (
            "allocation_plan_id = %s"
            if release_entire_plan
            else "analysis_id = %s AND segment_id = %s"
        )
        target_params: tuple[Any, ...] = (
            (plan_id,)
            if release_entire_plan
            else (target_analysis_id, segment_id)
        )
        self._db.execute(
            f"""
            UPDATE promotion_target_segments
            SET audience_reservation_state = 'released'
            WHERE {target_filter}
            """,
            target_params,
        )
        self._db.execute(
            """
            UPDATE segment_audience_allocation_plans
            SET status = 'released',
                released_at = now(),
                locked_at = NULL
            WHERE allocation_plan_id = %s
              AND status = 'finalized'
            """,
            (plan_id,),
        )
        new_context = PromotionAudienceExclusionContext(
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            revision=next_revision,
            excluded_user_count=max(
                0,
                previous.excluded_user_count
                - (int(state["reserved_count"]) if state is not None else 0),
            ),
            projection_revision=previous.projection_revision,
        )
        self.refresh_recommendation_previews(
            analysis_id=str(row["candidate_batch_analysis_id"]),
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            context=new_context,
        )
        return new_context

    def _lock_promotion(self, *, project_id: str, promotion_id: str) -> None:
        self._db.execute(
            """
            SELECT pg_advisory_xact_lock(
                hashtext('promotion-audience-allocation-v1'),
                hashtext(%s || ':' || %s)
            )
            """,
            (project_id, promotion_id),
        )

    def _load_source_snapshots(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        segment_ids: Sequence[str],
        source_analysis_id: str | None = None,
    ) -> tuple[_SourceSnapshot, ...]:
        try:
            if source_analysis_id is None:
                rows = self._db.fetchall(
                    """
                WITH latest AS (
                    SELECT
                        suggestion.*,
                        row_number() OVER (
                            PARTITION BY suggestion.segment_id
                            ORDER BY suggestion.created_at DESC,
                                     suggestion.suggestion_id DESC
                        ) AS row_rank
                    FROM promotion_segment_suggestions AS suggestion
                    WHERE suggestion.project_id = %s
                      AND suggestion.campaign_id = %s
                      AND suggestion.promotion_id = %s
                      AND suggestion.segment_id = ANY(%s)
                      AND suggestion.status = 'suggested'
                )
                SELECT
                    latest.analysis_id AS source_analysis_id,
                    latest.segment_id,
                    latest.audience_snapshot_id AS snapshot_id,
                    snapshot.segment_vector_id,
                    snapshot.score_threshold,
                    snapshot.min_sample_size,
                    snapshot.metadata_json ->> 'candidate_type' AS candidate_type,
                    snapshot.metadata_json ->> 'semantic_margin' AS semantic_margin,
                    snapshot.snapshot_kind,
                    snapshot.source_snapshot_id AS parent_source_snapshot_id,
                    snapshot.allocation_plan_id,
                    snapshot.status AS snapshot_status,
                    snapshot.final_user_count,
                    (SELECT count(*)
                     FROM segment_audience_members AS member
                     WHERE member.snapshot_id = snapshot.snapshot_id)
                        AS actual_member_count
                FROM latest
                JOIN segment_audience_snapshots AS snapshot
                  ON snapshot.snapshot_id = latest.audience_snapshot_id
                WHERE latest.row_rank = 1
                ORDER BY latest.segment_id ASC
                """,
                    (project_id, campaign_id, promotion_id, list(segment_ids)),
                )
            else:
                rows = self._db.fetchall(
                    """
                    SELECT
                        snapshot.analysis_id AS source_analysis_id,
                        snapshot.segment_id,
                        snapshot.snapshot_id,
                        snapshot.segment_vector_id,
                        snapshot.score_threshold,
                        snapshot.min_sample_size,
                        snapshot.metadata_json ->> 'candidate_type'
                            AS candidate_type,
                        snapshot.metadata_json ->> 'semantic_margin'
                            AS semantic_margin,
                        snapshot.snapshot_kind,
                        snapshot.source_snapshot_id AS parent_source_snapshot_id,
                        snapshot.allocation_plan_id,
                        snapshot.status AS snapshot_status,
                        snapshot.final_user_count,
                        (SELECT count(*)
                         FROM segment_audience_members AS member
                         WHERE member.snapshot_id = snapshot.snapshot_id)
                            AS actual_member_count
                    FROM segment_audience_snapshots AS snapshot
                    WHERE snapshot.analysis_id = %s
                      AND snapshot.project_id = %s
                      AND snapshot.campaign_id = %s
                      AND snapshot.promotion_id = %s
                      AND snapshot.segment_id = ANY(%s)
                      AND snapshot.snapshot_kind = 'source'
                    ORDER BY snapshot.segment_id ASC
                    """,
                    (
                        source_analysis_id,
                        project_id,
                        campaign_id,
                        promotion_id,
                        list(segment_ids),
                    ),
                )
        except (errors.UndefinedTable, errors.UndefinedColumn) as exc:
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_contract_missing",
                promotion_id=promotion_id,
                reason="audience allocation Data Contract is missing",
            ) from exc
        if {str(row["segment_id"]) for row in rows} != set(segment_ids):
            raise SegmentAudienceAllocationError(
                code="segment_audience_snapshot_binding_required",
                promotion_id=promotion_id,
                reason="every selected segment requires a source snapshot",
            )
        sources = tuple(_source_snapshot_from_row(row, promotion_id) for row in rows)
        if any(
            row["snapshot_status"] != "completed"
            or str(row["snapshot_kind"]) != "source"
            or row["parent_source_snapshot_id"] is not None
            or row["allocation_plan_id"] is not None
            or int(row["actual_member_count"]) != int(row["final_user_count"])
            for row in rows
        ):
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_binding_invalid",
                promotion_id=promotion_id,
                reason="source snapshot is incomplete or has inconsistent members",
            )
        return sources

    def _load_unconfirmed_source_snapshots(
        self,
        *,
        analysis_id: str,
        project_id: str,
        promotion_id: str,
        exclude_segment_ids: Sequence[str] = (),
    ) -> tuple[_SourceSnapshot, ...]:
        rows = self._db.fetchall(
            """
            SELECT
                suggestion.analysis_id AS source_analysis_id,
                suggestion.segment_id,
                suggestion.audience_snapshot_id AS snapshot_id,
                snapshot.segment_vector_id,
                snapshot.score_threshold,
                snapshot.min_sample_size,
                snapshot.metadata_json ->> 'candidate_type' AS candidate_type,
                snapshot.metadata_json ->> 'semantic_margin' AS semantic_margin
            FROM promotion_segment_suggestions AS suggestion
            JOIN segment_audience_snapshots AS snapshot
              ON snapshot.snapshot_id = suggestion.audience_snapshot_id
            WHERE suggestion.analysis_id = %s
              AND suggestion.project_id = %s
              AND suggestion.promotion_id = %s
              AND suggestion.status = 'suggested'
              AND NOT (suggestion.segment_id = ANY(%s))
              AND NOT EXISTS (
                  SELECT 1
                  FROM promotion_target_segments AS target
                  WHERE target.project_id = suggestion.project_id
                    AND target.promotion_id = suggestion.promotion_id
                    AND target.segment_id = suggestion.segment_id
                    AND target.audience_snapshot_id IN (
                        SELECT final.snapshot_id
                        FROM segment_audience_snapshots AS final
                        WHERE final.source_snapshot_id = suggestion.audience_snapshot_id
                          AND final.snapshot_kind = 'final'
                    )
                    AND target.audience_reservation_state IN ('reserved', 'consumed')
              )
            ORDER BY suggestion.segment_id ASC
            """,
            (
                analysis_id,
                project_id,
                promotion_id,
                list(sorted(set(exclude_segment_ids))),
            ),
        )
        return tuple(_source_snapshot_from_row(row, promotion_id) for row in rows)

    def _find_idempotent_result(
        self,
        *,
        confirmation_analysis_id: str,
        project_id: str,
        promotion_id: str,
        sources: Sequence[_SourceSnapshot],
    ) -> ConfirmationAllocationResult | None:
        source_ids = sorted(source.snapshot_id for source in sources)
        rows = self._db.fetchall(
            """
            SELECT
                plan.allocation_plan_id,
                plan.candidate_batch_analysis_id AS source_analysis_id,
                plan.exclusion_revision,
                snapshot.segment_id,
                snapshot.source_snapshot_id,
                snapshot.snapshot_id AS final_snapshot_id,
                snapshot.final_user_count,
                snapshot.meets_min_sample_size,
                snapshot.audience_status
            FROM segment_audience_allocation_plans AS plan
            JOIN segment_audience_snapshots AS snapshot
              ON snapshot.allocation_plan_id = plan.allocation_plan_id
             AND snapshot.snapshot_kind = 'final'
            WHERE plan.promotion_id = %s
              AND plan.target_analysis_id = %s
              AND plan.status IN ('finalized', 'locked')
              AND plan.selected_segment_ids_json = %s::jsonb
              AND snapshot.source_snapshot_id = ANY(%s)
            ORDER BY snapshot.segment_id ASC
            """,
            (
                promotion_id,
                confirmation_analysis_id,
                json.dumps(sorted(source.segment_id for source in sources)),
                source_ids,
            ),
        )
        if not rows:
            return None
        if len(rows) != len(sources):
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_binding_invalid",
                promotion_id=promotion_id,
                reason="stored allocation retry has incomplete segment bindings",
            )
        allocations = {
            str(row["segment_id"]): FinalAudienceAllocation(
                segment_id=str(row["segment_id"]),
                source_analysis_id=str(row["source_analysis_id"]),
                source_snapshot_id=str(row["source_snapshot_id"]),
                final_snapshot_id=str(row["final_snapshot_id"]),
                allocation_plan_id=str(row["allocation_plan_id"]),
                final_user_count=int(row["final_user_count"]),
                meets_min_sample_size=bool(row["meets_min_sample_size"]),
                audience_status=str(row["audience_status"]),
                exclusion_revision=int(row["exclusion_revision"]),
            )
            for row in rows
        }
        first = rows[0]
        return ConfirmationAllocationResult(
            source_analysis_id=str(first["source_analysis_id"]),
            allocation_plan_id=str(first["allocation_plan_id"]),
            exclusion_revision=int(first["exclusion_revision"]),
            allocations=allocations,
        )

    def _raise_if_sources_already_confirmed(
        self,
        *,
        confirmation_analysis_id: str,
        project_id: str,
        promotion_id: str,
        sources: Sequence[_SourceSnapshot],
    ) -> None:
        source_ids = sorted(source.snapshot_id for source in sources)
        row = self._db.fetchone(
            """
            SELECT plan.target_analysis_id
            FROM segment_audience_allocation_plans AS plan
            JOIN segment_audience_snapshots AS snapshot
              ON snapshot.allocation_plan_id = plan.allocation_plan_id
             AND snapshot.snapshot_kind = 'final'
            WHERE plan.promotion_id = %s
              AND snapshot.source_snapshot_id = ANY(%s)
              AND plan.status IN ('finalized', 'locked')
              AND plan.target_analysis_id <> %s
            LIMIT 1
            """,
            (
                promotion_id,
                source_ids,
                confirmation_analysis_id,
            ),
        )
        if row is not None:
            raise SegmentAudienceAllocationError(
                code="segment_audience_source_already_confirmed",
                promotion_id=promotion_id,
                reason=(
                    "one or more source snapshots were already confirmed by "
                    + str(row["target_analysis_id"])
                ),
            )

    def _raise_if_segments_already_active(
        self,
        *,
        project_id: str,
        promotion_id: str,
        sources: Sequence[_SourceSnapshot],
    ) -> None:
        segment_ids = sorted(source.segment_id for source in sources)
        row = self._db.fetchone(
            """
            SELECT target.segment_id
            FROM promotion_target_segments AS target
            LEFT JOIN promotion_run_target_bindings AS binding
              ON binding.target_analysis_id = target.analysis_id
             AND binding.segment_id = target.segment_id
            LEFT JOIN promotion_runs AS run
              ON run.promotion_run_id = binding.promotion_run_id
            WHERE target.project_id = %s
              AND target.promotion_id = %s
              AND target.segment_id = ANY(%s)
              AND target.audience_snapshot_id IS NOT NULL
              AND (
                    (
                        target.audience_reservation_state = 'reserved'
                        AND binding.promotion_run_id IS NULL
                    )
                    OR (
                        target.audience_reservation_state = 'consumed'
                        AND run.status NOT IN (
                            'goal_met', 'goal_not_met', 'partial_goal_met',
                            'insufficient_data', 'stopped'
                        )
                    )
                  )
            ORDER BY target.segment_id ASC
            LIMIT 1
            """,
            (project_id, promotion_id, segment_ids),
        )
        if row is not None:
            segment_id = str(row["segment_id"])
            raise SegmentAudienceAllocationError(
                code="segment_audience_segment_already_confirmed",
                promotion_id=promotion_id,
                segment_id=segment_id,
                reason=(
                    "the segment already has an active confirmation or run "
                    "for this promotion"
                ),
            )

    def _materialize_winners(
        self,
        *,
        project_id: str,
        promotion_id: str,
        sources: Sequence[_SourceSnapshot],
    ) -> None:
        self._db.execute("DROP TABLE IF EXISTS audience_allocation_sources")
        self._db.execute("DROP TABLE IF EXISTS audience_allocation_winners")
        self._db.execute(
            """
            CREATE TEMP TABLE audience_allocation_sources (
                segment_id text PRIMARY KEY,
                source_snapshot_id text NOT NULL,
                priority integer NOT NULL,
                score_threshold numeric NOT NULL,
                semantic_margin numeric NOT NULL
            ) ON COMMIT DROP
            """
        )
        self._db.execute(
            """
            INSERT INTO audience_allocation_sources (
                segment_id,
                source_snapshot_id,
                priority,
                score_threshold,
                semantic_margin
            )
            SELECT *
            FROM unnest(
                %s::text[], %s::text[], %s::integer[], %s::numeric[], %s::numeric[]
            )
            """,
            (
                [source.segment_id for source in sources],
                [source.snapshot_id for source in sources],
                [
                    ALLOCATION_POLICY_PRIORITY[source.candidate_type]
                    for source in sources
                ],
                [source.score_threshold for source in sources],
                [source.semantic_margin for source in sources],
            ),
        )
        self._db.execute(
            f"""
            CREATE TEMP TABLE audience_allocation_winners
            ON COMMIT DROP
            AS
            WITH ranked AS (
                SELECT
                    member.user_id,
                    source.segment_id,
                    source.source_snapshot_id,
                    member.behavior_fit_score,
                    member.retrieval_source,
                    member.retrieval_rank,
                    source.score_threshold,
                    source.semantic_margin,
                    (member.behavior_fit_score - source.score_threshold)
                        / source.semantic_margin AS normalized_fit,
                    row_number() OVER (
                        PARTITION BY member.user_id
                        ORDER BY source.priority ASC,
                                 (member.behavior_fit_score - source.score_threshold)
                                    / source.semantic_margin DESC NULLS LAST,
                                 source.segment_id ASC
                    ) AS winner_rank
                FROM audience_allocation_sources AS source
                JOIN segment_audience_members AS member
                  ON member.snapshot_id = source.source_snapshot_id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM {POSTGRES_EXCLUSION_RELATION} AS excluded
                    WHERE excluded.project_id = %s
                      AND excluded.promotion_id = %s
                      AND excluded.user_id = member.user_id
                      AND excluded.state IN ('reserved', 'consumed')
                )
            )
            SELECT *
            FROM ranked
            WHERE winner_rank = 1
            """,
            (project_id, promotion_id),
        )
        self._db.execute(
            """
            CREATE UNIQUE INDEX audience_allocation_winners_user_idx
            ON audience_allocation_winners (user_id)
            """
        )

    def _winner_counts(self) -> dict[str, int]:
        rows = self._db.fetchall(
            """
            SELECT segment_id, count(*) AS allocated_user_count
            FROM audience_allocation_winners
            GROUP BY segment_id
            ORDER BY segment_id ASC
            """
        )
        return {
            str(row["segment_id"]): int(row["allocated_user_count"])
            for row in rows
        }

    def _ensure_target_rows(
        self,
        *,
        target_analysis_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        sources: Sequence[_SourceSnapshot],
    ) -> None:
        self._db.execute(
            """
            INSERT INTO promotion_target_segments (
                analysis_id, project_id, campaign_id, promotion_id,
                segment_id, segment_name, rule_json, profile_json,
                content_brief_json, data_evidence_json, segment_vector_id,
                estimated_size, priority, status
            )
            SELECT
                %s, %s, %s, %s,
                definition.segment_id, definition.segment_name,
                definition.rule_json, definition.profile_json,
                '{}'::jsonb, '{}'::jsonb, selected.segment_vector_id,
                0, NULL, 'planned'
            FROM unnest(%s::text[], %s::text[])
                AS selected(segment_id, segment_vector_id)
            JOIN segment_definitions AS definition
              ON definition.segment_id = selected.segment_id
            ON CONFLICT (analysis_id, segment_id) DO NOTHING
            """,
            (
                target_analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                [source.segment_id for source in sources],
                [source.segment_vector_id for source in sources],
            ),
        )

    def _insert_plan(
        self,
        *,
        plan_id: str,
        confirmation_analysis_id: str,
        source_analysis_id: str,
        promotion_id: str,
        plan_fingerprint: str,
        selected_segment_ids: Sequence[str],
        exclusion_revision: int,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO segment_audience_allocation_plans (
                allocation_plan_id,
                promotion_id,
                candidate_batch_analysis_id,
                target_analysis_id,
                selection_fingerprint,
                selected_segment_ids_json,
                exclusion_revision,
                allocation_policy_version,
                allocation_policy_hash,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, 'finalized')
            """,
            (
                plan_id,
                promotion_id,
                source_analysis_id,
                confirmation_analysis_id,
                plan_fingerprint,
                json.dumps(sorted(selected_segment_ids)),
                exclusion_revision,
                ALLOCATION_POLICY_VERSION,
                ALLOCATION_POLICY_HASH,
            ),
        )

    def _insert_final_snapshot(
        self,
        *,
        source: _SourceSnapshot,
        confirmation_analysis_id: str,
        plan_id: str,
        final_snapshot_id: str,
        final_user_count: int,
        min_sample_size: int,
        audience_status: str,
        context: PromotionAudienceExclusionContext,
    ) -> None:
        metadata_patch = {
            "snapshot_kind": "final",
            "source_snapshot_id": source.snapshot_id,
            "allocation_plan_id": plan_id,
            "allocation_policy_version": ALLOCATION_POLICY_VERSION,
            "allocation_policy_hash": ALLOCATION_POLICY_HASH,
            "source_exclusion_revision": context.revision,
            "targetable": final_user_count > 0,
        }
        self._db.execute(
            """
            INSERT INTO segment_audience_snapshots (
                snapshot_id, suggestion_id, analysis_id,
                project_id, campaign_id, promotion_id, segment_id,
                segment_vector_id, vector_generation_id,
                schema_version, vector_version, manifest_hash,
                audience_resolution_contract, segment_audience_spec_hash,
                query_vector_hash, query_compiler_version, query_compiler_hash,
                matcher_version, search_policy_version,
                calibration_version, calibration_hash, score_threshold,
                source_cutoff, window_start, window_end,
                eligible_user_count, behavior_match_count, final_user_count,
                min_sample_size, audience_status, selection_method,
                estimated_recall, recall_lower_bound, recall_target,
                input_fingerprint, meets_min_sample_size, status, metadata_json,
                snapshot_kind, source_snapshot_id, allocation_plan_id
            )
            SELECT
                %s, NULL, %s,
                project_id, campaign_id, promotion_id, segment_id,
                segment_vector_id, vector_generation_id,
                schema_version, vector_version, manifest_hash,
                audience_resolution_contract, segment_audience_spec_hash,
                query_vector_hash, query_compiler_version, query_compiler_hash,
                matcher_version, search_policy_version,
                calibration_version, calibration_hash, score_threshold,
                source_cutoff, window_start, window_end,
                eligible_user_count, behavior_match_count, %s,
                %s, %s, selection_method,
                estimated_recall, recall_lower_bound, recall_target,
                %s, %s, 'completed', metadata_json || %s::jsonb,
                'final', %s, %s
            FROM segment_audience_snapshots
            WHERE snapshot_id = %s
            """,
            (
                final_snapshot_id,
                confirmation_analysis_id,
                final_user_count,
                min_sample_size,
                audience_status,
                _final_snapshot_fingerprint(
                    plan_id=plan_id,
                    source_snapshot_id=source.snapshot_id,
                    segment_id=source.segment_id,
                    context=context,
                ),
                final_user_count >= min_sample_size,
                metadata_patch,
                source.snapshot_id,
                plan_id,
                source.snapshot_id,
            ),
        )

    def _insert_final_members(
        self,
        *,
        source: _SourceSnapshot,
        plan_id: str,
        final_snapshot_id: str,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO segment_audience_members (
                snapshot_id, user_id, behavior_fit_score,
                retrieval_source, retrieval_rank
            )
            SELECT
                %s,
                user_id,
                behavior_fit_score,
                retrieval_source,
                retrieval_rank
            FROM audience_allocation_winners
            WHERE segment_id = %s
            ORDER BY user_id ASC
            """,
            (final_snapshot_id, source.segment_id),
        )

    def _bind_target_audience(
        self,
        *,
        target_analysis_id: str,
        segment_id: str,
        plan_id: str,
        final_snapshot_id: str,
        final_user_count: int,
    ) -> None:
        self._db.execute(
            """
            UPDATE promotion_target_segments
            SET audience_snapshot_id = %s,
                allocation_plan_id = %s::uuid,
                audience_reservation_state = 'reserved',
                estimated_size = %s
            WHERE analysis_id = %s
              AND segment_id = %s
              AND audience_snapshot_id IS NULL
              AND allocation_plan_id IS NULL
              AND audience_reservation_state IS NULL
            """,
            (
                final_snapshot_id,
                plan_id,
                final_user_count,
                target_analysis_id,
                segment_id,
            ),
        )

    def _reserve_final_members(
        self,
        *,
        plan_id: str,
        target_analysis_id: str,
        project_id: str,
        promotion_id: str,
        previous: PromotionAudienceExclusionContext,
        reservation_revision: int,
    ) -> PromotionAudienceExclusionContext:
        winner_count = sum(self._winner_counts().values())
        try:
            reserved = self._db.fetchone(
                f"""
                WITH upserted AS (
                    INSERT INTO {POSTGRES_EXCLUSION_RELATION} (
                        project_id,
                        promotion_id,
                        user_id,
                        target_analysis_id,
                        segment_id,
                        allocation_plan_id,
                        final_snapshot_id,
                        state,
                        revision,
                        reserved_at,
                        consumed_at,
                        released_at
                    )
                    SELECT
                        %s, %s, winner.user_id, %s, winner.segment_id, %s::uuid,
                        final.snapshot_id, 'reserved', %s, now(), NULL, NULL
                    FROM audience_allocation_winners AS winner
                    JOIN segment_audience_snapshots AS final
                      ON final.allocation_plan_id = %s::uuid
                     AND final.segment_id = winner.segment_id
                     AND final.snapshot_kind = 'final'
                    ORDER BY winner.user_id ASC
                    ON CONFLICT (project_id, promotion_id, user_id)
                    DO UPDATE SET
                        target_analysis_id = EXCLUDED.target_analysis_id,
                        segment_id = EXCLUDED.segment_id,
                        allocation_plan_id = EXCLUDED.allocation_plan_id,
                        final_snapshot_id = EXCLUDED.final_snapshot_id,
                        state = 'reserved',
                        revision = EXCLUDED.revision,
                        reserved_at = EXCLUDED.reserved_at,
                        consumed_at = NULL,
                        released_at = NULL
                    WHERE {POSTGRES_EXCLUSION_RELATION}.state = 'released'
                    RETURNING user_id
                )
                SELECT count(*) AS reserved_count
                FROM upserted
                """,
                (
                    project_id,
                    promotion_id,
                    target_analysis_id,
                    plan_id,
                    reservation_revision,
                    plan_id,
                ),
            )
        except errors.UniqueViolation as exc:
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_conflict",
                promotion_id=promotion_id,
                reason="one or more users were reserved concurrently",
            ) from exc
        if reserved is None or int(reserved["reserved_count"]) != winner_count:
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_conflict",
                promotion_id=promotion_id,
                reason="one or more users already have an active reservation",
            )
        return PromotionAudienceExclusionContext(
            project_id=project_id,
            campaign_id=previous.campaign_id,
            promotion_id=promotion_id,
            revision=reservation_revision,
            excluded_user_count=previous.excluded_user_count + winner_count,
            projection_revision=previous.projection_revision,
        )

    def _advance_exclusion_revision(
        self,
        *,
        promotion_id: str,
        expected_revision: int,
    ) -> int:
        row = self._db.fetchone(
            """
            SELECT advance_promotion_audience_exclusion_revision(%s)
                AS revision
            """,
            (promotion_id,),
        )
        revision = int(row["revision"]) if row is not None else -1
        if revision != expected_revision + 1:
            raise SegmentAudienceAllocationError(
                code="segment_audience_exclusion_conflict",
                promotion_id=promotion_id,
                reason="promotion exclusion revision changed during allocation",
            )
        return revision


def _source_snapshot_from_row(
    row: Mapping[str, Any],
    promotion_id: str,
) -> _SourceSnapshot:
    candidate_type = str(row.get("candidate_type") or "")
    if candidate_type not in ALLOCATION_POLICY_PRIORITY:
        raise SegmentAudienceAllocationError(
            code="segment_audience_exclusion_binding_invalid",
            promotion_id=promotion_id,
            segment_id=str(row.get("segment_id") or ""),
            reason="source snapshot has an unsupported allocation template",
        )
    semantic_margin = Decimal(str(row.get("semantic_margin") or "0"))
    if semantic_margin <= 0:
        raise SegmentAudienceAllocationError(
            code="segment_audience_exclusion_binding_invalid",
            promotion_id=promotion_id,
            segment_id=str(row.get("segment_id") or ""),
            reason="source snapshot semantic margin must be positive",
        )
    return _SourceSnapshot(
        segment_id=str(row["segment_id"]),
        source_analysis_id=str(row["source_analysis_id"]),
        snapshot_id=str(row["snapshot_id"]),
        candidate_type=candidate_type,
        score_threshold=Decimal(str(row["score_threshold"])),
        semantic_margin=semantic_margin,
        min_sample_size=int(row["min_sample_size"]),
        segment_vector_id=str(row["segment_vector_id"]),
    )


def _plan_fingerprint(
    *,
    sources: Sequence[_SourceSnapshot],
    context: PromotionAudienceExclusionContext,
) -> str:
    return _sha256_json(
        {
            "source_snapshots": sorted(source.snapshot_id for source in sources),
            "segment_ids": sorted(source.segment_id for source in sources),
            "exclusion_revision": context.revision,
            "allocation_policy_version": ALLOCATION_POLICY_VERSION,
            "allocation_policy_hash": ALLOCATION_POLICY_HASH,
        }
    )


def _allocation_plan_id(*, promotion_id: str, plan_fingerprint: str) -> str:
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{promotion_id}:{plan_fingerprint}")
    )


def _final_snapshot_id(*, plan_id: str, segment_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{plan_id}:{segment_id}"))


def _final_snapshot_fingerprint(
    *,
    plan_id: str,
    source_snapshot_id: str,
    segment_id: str,
    context: PromotionAudienceExclusionContext,
) -> str:
    return _sha256_json(
        {
            "plan_id": plan_id,
            "source_snapshot_id": source_snapshot_id,
            "segment_id": segment_id,
            "exclusion_revision": context.revision,
            "allocation_policy_version": ALLOCATION_POLICY_VERSION,
            "allocation_policy_hash": ALLOCATION_POLICY_HASH,
        }
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audience_status(*, final_user_count: int, min_sample_size: int) -> str:
    if final_user_count <= 0:
        return "no_eligible_audience"
    if final_user_count < min_sample_size:
        return "insufficient_sample"
    return "targetable"
