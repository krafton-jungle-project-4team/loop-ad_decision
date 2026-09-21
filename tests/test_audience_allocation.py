from __future__ import annotations

from decimal import Decimal

import pytest
from psycopg import errors

from app.audience_allocation import (
    ALLOCATION_PREVIEW_VERSION,
    ALLOCATION_POLICY_HASH,
    ALLOCATION_POLICY_VERSION,
    AudienceAllocationService,
    PostgresAudienceAllocationRepository,
    SegmentAudienceAllocationError,
    _SourceSnapshot,
)
from app.audience_exclusions import PromotionAudienceExclusionContext


class _Db:
    def __init__(self) -> None:
        self.executed = []
        self.fetchall_rows = []
        self.fetchone_rows = []

    def execute(self, query, params=()):
        self.executed.append((query, params))

    def fetchall(self, query, params=()):
        self.executed.append((query, params))
        return list(self.fetchall_rows)

    def fetchone(self, query, params=()):
        self.executed.append((query, params))
        if self.fetchone_rows:
            return self.fetchone_rows.pop(0)
        return None


class _UnusedExclusionReader:
    def load_active_exclusion_context(self, **_kwargs):
        raise AssertionError("explicit test context must be used")


class _FixedExclusionReader:
    def __init__(self, context: PromotionAudienceExclusionContext) -> None:
        self.context = context

    def load_active_exclusion_context(self, **_kwargs):
        return self.context


class _MissingContractRepository:
    def confirm_selection(self, **_kwargs):
        raise errors.UndefinedTable("allocation contract missing")

    def refresh_recommendation_previews(self, **_kwargs):
        raise errors.UndefinedColumn("allocation contract incomplete")

    def release_reserved_target(self, **_kwargs):
        raise errors.UndefinedTable("allocation contract missing")


def test_allocation_service_maps_missing_data_contract_to_structured_error() -> None:
    service = AudienceAllocationService(_MissingContractRepository())

    for operation in (
        service.confirm_selection,
        service.refresh_recommendation_previews,
        service.release_reserved_target,
    ):
        with pytest.raises(SegmentAudienceAllocationError) as error:
            operation(promotion_id="promotion", segment_id="seg_a")

        assert error.value.code == "segment_audience_exclusion_contract_missing"
        assert error.value.promotion_id == "promotion"
        assert error.value.segment_id == "seg_a"


def _source(segment_id: str, candidate_type: str) -> _SourceSnapshot:
    return _SourceSnapshot(
        segment_id=segment_id,
        source_analysis_id="recommendation_analysis",
        snapshot_id=f"source_{segment_id}",
        candidate_type=candidate_type,
        score_threshold=Decimal("0.60"),
        semantic_margin=Decimal("0.20"),
        min_sample_size=10,
        segment_vector_id=f"vector_{segment_id}",
    )


def test_allocation_winner_sql_uses_template_priority_then_normalized_fit() -> None:
    db = _Db()
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )

    repository._materialize_winners(
        project_id="project",
        promotion_id="promotion",
        sources=(
            _source("seg_destination", "target_destination_affinity"),
            _source("seg_funnel", "funnel_recovery"),
        ),
    )

    winner_query = next(
        query
        for query, _params in db.executed
        if "CREATE TEMP TABLE audience_allocation_winners" in query
    )
    assert "source.priority ASC" in winner_query
    assert "member.behavior_fit_score - source.score_threshold" in winner_query
    assert "/ source.semantic_margin DESC NULLS LAST" in winner_query
    assert "source.segment_id ASC" in winner_query
    assert "promotion_audience_exclusion_members" in winner_query
    assert "NOT EXISTS" in winner_query
    assert "target.status <> 'stopped'" in winner_query
    assert "promotion_run_target_bindings" in winner_query
    assert "active_run.status NOT IN" in winner_query


def test_confirmation_reservation_uses_the_advanced_revision() -> None:
    db = _Db()
    db.fetchone_rows = [{"reserved_count": 5}]
    db.fetchall_rows = [
        {"segment_id": "seg_a", "allocated_user_count": 3},
        {"segment_id": "seg_b", "allocated_user_count": 2},
    ]
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )
    previous = PromotionAudienceExclusionContext(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        revision=5,
        excluded_user_count=20,
        projection_revision=5,
    )

    context = repository._reserve_final_members(
        plan_id="plan",
        target_analysis_id="confirmation",
        project_id="project",
        promotion_id="promotion",
        previous=previous,
        reservation_revision=6,
    )

    reservation_query = next(
        query
        for query, _params in db.executed
        if "INSERT INTO promotion_audience_exclusion_members" in query
    )
    assert "'reserved'" in reservation_query
    assert "target_analysis_id" in reservation_query
    assert "final_snapshot_id" in reservation_query
    assert "winner.segment_id" in reservation_query
    assert "ON CONFLICT (project_id, promotion_id, user_id)" in reservation_query
    assert "state = 'released'" in reservation_query
    assert "stale_target.status <> 'stopped'" in reservation_query
    assert "promotion_run_target_bindings" in reservation_query
    assert "active_run.status NOT IN" in reservation_query
    assert "'goal_met'" in reservation_query
    assert "'goal_not_met'" in reservation_query
    assert "stale_target.allocation_plan_id" in reservation_query
    assert "stale_target.audience_snapshot_id" in reservation_query
    assert "released_at = NULL" in reservation_query
    assert context.revision == 6
    assert context.excluded_user_count == 25


class _PreviewRepository(PostgresAudienceAllocationRepository):
    def __init__(self, db: _Db, sources, *, configured_candidate_limit: int = 3) -> None:
        super().__init__(
            postgres=db,
            exclusion_reader=_UnusedExclusionReader(),
            configured_candidate_limit=configured_candidate_limit,
        )
        self.sources = tuple(sources)
        self.current = ()

    def _load_unconfirmed_source_snapshots(self, **_kwargs):
        return self.sources

    def _materialize_winners(self, *, sources, **_kwargs):
        self.current = tuple(sources)

    def _winner_counts(self):
        return {
            source.segment_id: 10 - index
            for index, source in enumerate(self.current)
        }


def test_preview_metadata_is_one_lookup_per_current_selection_revision() -> None:
    db = _Db()
    repository = _PreviewRepository(
        db,
        (
            _source("seg_a", "target_destination_affinity"),
            _source("seg_b", "funnel_recovery"),
            _source("seg_c", "promotion_responsive"),
        ),
    )
    context = PromotionAudienceExclusionContext(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        revision=7,
        excluded_user_count=44,
        projection_revision=7,
    )

    payload = repository.refresh_recommendation_previews(
        analysis_id="recommendation_analysis",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        context=context,
    )

    assert payload["exclusion_revision"] == 7
    assert payload["preview_version"] == ALLOCATION_PREVIEW_VERSION
    assert payload["candidate_batch_analysis_id"] == "recommendation_analysis"
    assert payload["candidate_segment_ids"] == ["seg_a", "seg_b", "seg_c"]
    assert payload["allocation_policy_version"] == ALLOCATION_POLICY_VERSION
    assert payload["allocation_policy_hash"] == ALLOCATION_POLICY_HASH
    assert len(payload["allocation_previews"]) == 7
    for preview in payload["allocation_previews"]:
        selected = set(preview["selected_segment_ids"])
        assert {row["segment_id"] for row in preview["per_segment"]} == selected
        assert preview["candidate_batch_analysis_id"] == "recommendation_analysis"
        assert preview["exclusion_revision"] == 7
        assert preview["preview_version"] == ALLOCATION_PREVIEW_VERSION
        assert preview["allocation_policy_version"] == ALLOCATION_POLICY_VERSION
    stored_payload = db.executed[-1][1][0]
    assert stored_payload == payload


@pytest.mark.parametrize("configured_candidate_limit", [1, 2, 3])
def test_preview_respects_the_configured_candidate_batch_limit(
    configured_candidate_limit: int,
) -> None:
    db = _Db()
    all_sources = (
        _source("seg_a", "target_destination_affinity"),
        _source("seg_b", "funnel_recovery"),
        _source("seg_c", "promotion_responsive"),
    )
    repository = _PreviewRepository(
        db,
        all_sources[:configured_candidate_limit],
        configured_candidate_limit=configured_candidate_limit,
    )
    context = PromotionAudienceExclusionContext(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        revision=1,
        excluded_user_count=0,
        projection_revision=1,
    )

    payload = repository.refresh_recommendation_previews(
        analysis_id="recommendation_analysis",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        context=context,
    )

    assert payload["candidate_segment_ids"] == [
        source.segment_id for source in all_sources[:configured_candidate_limit]
    ]
    assert len(payload["allocation_previews"]) == (
        2**configured_candidate_limit - 1
    )


def test_preview_reader_explicitly_excludes_just_confirmed_sources() -> None:
    db = _Db()
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )

    assert repository._load_unconfirmed_source_snapshots(
        analysis_id="recommendation_analysis",
        project_id="project",
        promotion_id="promotion",
        exclude_segment_ids=("seg_b", "seg_a"),
    ) == ()

    query, params = db.executed[-1]
    assert "NOT (suggestion.segment_id = ANY(%s))" in query
    assert params[-1] == ["seg_a", "seg_b"]


class _ReservationConflictDb(_Db):
    def fetchone(self, query, params=()):
        if "INSERT INTO promotion_audience_exclusion_members" in query:
            self.executed.append((query, params))
            return {"reserved_count": 4}
        return super().fetchone(query, params)

    def fetchall(self, query, params=()):
        self.executed.append((query, params))
        return [
            {"segment_id": "seg_a", "allocated_user_count": 3},
            {"segment_id": "seg_b", "allocated_user_count": 2},
        ]


def test_concurrent_reservation_conflict_is_structured_and_not_silenced() -> None:
    db = _ReservationConflictDb()
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )
    previous = PromotionAudienceExclusionContext(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        revision=5,
        excluded_user_count=20,
        projection_revision=5,
    )

    with pytest.raises(SegmentAudienceAllocationError) as error:
        repository._reserve_final_members(
            plan_id="plan",
            target_analysis_id="confirmation",
            project_id="project",
            promotion_id="promotion",
            previous=previous,
            reservation_revision=6,
        )

    assert error.value.code == "segment_audience_exclusion_conflict"


def test_same_source_snapshot_cannot_be_confirmed_by_a_different_request() -> None:
    db = _Db()
    db.fetchone_rows = [{"target_analysis_id": "confirmation_existing"}]
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )

    with pytest.raises(SegmentAudienceAllocationError) as error:
        repository._raise_if_sources_already_confirmed(
            confirmation_analysis_id="confirmation_new",
            project_id="project",
            promotion_id="promotion",
            sources=(_source("seg_a", "funnel_recovery"),),
        )

    assert error.value.code == "segment_audience_source_already_confirmed"


def test_same_segment_cannot_be_reconfirmed_while_an_active_target_exists() -> None:
    db = _Db()
    db.fetchone_rows = [{"segment_id": "seg_a"}]
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )

    with pytest.raises(SegmentAudienceAllocationError) as error:
        repository._raise_if_segments_already_active(
            project_id="project",
            promotion_id="promotion",
            sources=(_source("seg_a", "funnel_recovery"),),
        )

    assert error.value.code == "segment_audience_segment_already_confirmed"


def test_reserved_target_release_advances_revision_and_refreshes_preview() -> None:
    db = _Db()
    db.fetchone_rows = [
        {
            "audience_snapshot_id": "final_a",
            "allocation_plan_id": "plan_a",
            "audience_reservation_state": "reserved",
            "candidate_batch_analysis_id": "recommendation_analysis",
            "status": "finalized",
            "plan_segment_count": 1,
            "run_bound": False,
            "reserved_count": 3,
            "consumed_count": 0,
        },
        {"reserved_count": 3, "consumed_count": 0},
        {"revision": 9},
    ]
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_FixedExclusionReader(
            PromotionAudienceExclusionContext(
                project_id="project",
                campaign_id="campaign",
                promotion_id="promotion",
                revision=8,
                excluded_user_count=10,
                projection_revision=8,
            )
        ),
    )

    context = repository.release_reserved_target(
        target_analysis_id="confirmation_analysis",
        segment_id="seg_a",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
    )

    sql = "\n".join(query for query, _params in db.executed)
    assert "SET state = 'released'" in sql
    assert "SET audience_reservation_state = 'released'" in sql
    assert "SET status = 'released'" in sql
    assert "audience_allocation_preview_context" in sql
    assert context.revision == 9
    assert context.excluded_user_count == 7


def test_consumed_target_cannot_be_released() -> None:
    db = _Db()
    db.fetchone_rows = [
        {
            "audience_snapshot_id": "final_a",
            "allocation_plan_id": "plan_a",
            "audience_reservation_state": "consumed",
            "candidate_batch_analysis_id": "recommendation_analysis",
            "status": "finalized",
            "plan_segment_count": 1,
            "run_bound": False,
            "reserved_count": 0,
            "consumed_count": 3,
        },
    ]
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )

    with pytest.raises(SegmentAudienceAllocationError) as error:
        repository.release_reserved_target(
            target_analysis_id="confirmation_analysis",
            segment_id="seg_a",
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
        )

    assert error.value.code == "segment_audience_allocation_locked"
    assert not any(
        "SET state = 'released'" in query for query, _params in db.executed
    )


def test_multi_target_confirmation_rejects_partial_release() -> None:
    db = _Db()
    db.fetchone_rows = [
        {
            "audience_snapshot_id": "final_a",
            "allocation_plan_id": "plan_ab",
            "audience_reservation_state": "reserved",
            "candidate_batch_analysis_id": "recommendation_analysis",
            "status": "finalized",
            "plan_segment_count": 2,
            "run_bound": False,
            "reserved_count": 5,
            "consumed_count": 0,
        }
    ]
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_UnusedExclusionReader(),
    )

    with pytest.raises(SegmentAudienceAllocationError) as error:
        repository.release_reserved_target(
            target_analysis_id="confirmation_analysis",
            segment_id="seg_a",
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
        )

    assert error.value.code == "segment_audience_partial_release_unsupported"
    assert not any(
        "SET state = 'released'" in query for query, _params in db.executed
    )


def test_multi_target_confirmation_can_release_the_entire_plan_before_run() -> None:
    db = _Db()
    db.fetchone_rows = [
        {
            "audience_snapshot_id": "final_a",
            "allocation_plan_id": "plan_ab",
            "audience_reservation_state": "reserved",
            "candidate_batch_analysis_id": "recommendation_analysis",
            "status": "finalized",
            "plan_segment_count": 2,
            "run_bound": False,
            "reserved_count": 5,
            "consumed_count": 0,
        },
        {"reserved_count": 5, "consumed_count": 0},
        {"revision": 9},
    ]
    repository = PostgresAudienceAllocationRepository(
        postgres=db,
        exclusion_reader=_FixedExclusionReader(
            PromotionAudienceExclusionContext(
                project_id="project",
                campaign_id="campaign",
                promotion_id="promotion",
                revision=8,
                excluded_user_count=10,
                projection_revision=8,
            )
        ),
    )

    context = repository.release_reserved_target(
        target_analysis_id="confirmation_analysis",
        segment_id="seg_a",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        release_entire_plan=True,
    )

    release_query = next(
        query for query, _params in db.executed if "SET state = 'released'" in query
    )
    target_query = next(
        query
        for query, _params in db.executed
        if "UPDATE promotion_target_segments" in query
    )
    assert "source_segment_id" not in release_query
    assert "WHERE allocation_plan_id = %s" in target_query
    assert any(
        "SET status = 'released'" in query
        for query, _params in db.executed
    )
    assert context.revision == 9
    assert context.excluded_user_count == 5
