from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.analysis.audience_search_repository import (
    PgClickHouseAudienceVectorSearchRepository,
    _hard_predicate_query,
)
from app.audience_exclusions import (
    EMPTY_EXCLUSION_HASH,
    PromotionAudienceExclusionContext,
    PromotionAudienceExclusionRepository,
    SegmentAudienceExclusionError,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def named_results(self):
        return iter(self._rows)


class _Postgres:
    def __init__(self, row=None) -> None:
        self.row = row
        self.calls = []

    def fetchone(self, query, params=()):
        self.calls.append((query, params))
        return self.row

    def execute(self, query, params=()):
        self.calls.append((query, params))


class _ClickHouse:
    def __init__(self, rows=()) -> None:
        self.rows = tuple(rows)
        self.calls = []

    def query(self, query, parameters=None):
        self.calls.append((query, parameters or {}))
        return _Result(self.rows)


class _ExclusionReader:
    def __init__(self, context: PromotionAudienceExclusionContext) -> None:
        self.context = context
        self.calls = []

    def load_active_exclusion_context(self, **kwargs):
        self.calls.append(kwargs)
        return self.context


def test_exclusion_context_requires_exact_clickhouse_revision_and_hash() -> None:
    postgres = _Postgres(
        {
            "revision": 4,
            "exclusion_hash": "sha256:revision-4",
            "excluded_user_count": 37,
        }
    )
    clickhouse = _ClickHouse(
        [
            {
                "revision": 4,
                "exclusion_hash": "sha256:revision-4",
                "status": "ready",
            }
        ]
    )
    repository = PromotionAudienceExclusionRepository(
        postgres=postgres,
        clickhouse=clickhouse,
    )

    context = repository.load_active_exclusion_context(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
    )

    assert context.revision == 4
    assert context.exclusion_hash == "sha256:revision-4"
    assert context.excluded_user_count == 37
    assert context.projection_revision == context.revision
    assert context.projection_hash == context.exclusion_hash


def test_stale_clickhouse_projection_fails_without_empty_fallback() -> None:
    repository = PromotionAudienceExclusionRepository(
        postgres=_Postgres(
            {
                "revision": 2,
                "exclusion_hash": "sha256:revision-2",
                "excluded_user_count": 3,
            }
        ),
        clickhouse=_ClickHouse(
            [
                {
                    "revision": 1,
                    "exclusion_hash": "sha256:revision-1",
                    "status": "ready",
                }
            ]
        ),
    )

    with pytest.raises(SegmentAudienceExclusionError) as error:
        repository.load_active_exclusion_context(
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
        )

    assert error.value.code == "segment_audience_exclusion_projection_not_ready"


def test_vector_population_count_anti_joins_same_promotion_exclusions() -> None:
    now = datetime(2026, 7, 17, tzinfo=UTC)
    postgres = _Postgres(
        {
            "vector_generation_id": "generation",
            "manifest_hash": "manifest",
            "source_cutoff": now,
            "source_revision_cutoff": now,
            "window_start": now,
            "corpus_user_count": 63,
        }
    )
    context = PromotionAudienceExclusionContext(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        revision=3,
        exclusion_hash="sha256:revision-3",
        excluded_user_count=12,
        projection_revision=3,
        projection_hash="sha256:revision-3",
    )
    exclusion_reader = _ExclusionReader(context)
    repository = PgClickHouseAudienceVectorSearchRepository(
        postgres=postgres,
        clickhouse=_ClickHouse(),
        exclusion_repository=exclusion_reader,
    )

    result = repository.get_context(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        vector_version="hotel_behavior.v2",
    )

    lock_query, lock_params = postgres.calls[0]
    query, params = postgres.calls[1]
    assert "pg_advisory_xact_lock" in lock_query
    assert lock_params == ("project", "promotion")
    assert "promotion_audience_exclusion_members" in query
    assert "NOT EXISTS" in query
    assert "reserved" in query and "consumed" in query
    assert params == ("project", "hotel_behavior.v2", "promotion")
    assert result.corpus_user_count == 63
    assert result.exclusion_context == context


def test_empty_exclusion_context_still_uses_versioned_empty_hash() -> None:
    repository = PromotionAudienceExclusionRepository(
        postgres=_Postgres(None),
        clickhouse=_ClickHouse(),
    )

    context = repository.load_active_exclusion_context(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
    )

    assert context.revision == 0
    assert context.exclusion_hash == EMPTY_EXCLUSION_HASH
    assert context.projection_hash == EMPTY_EXCLUSION_HASH


def test_clickhouse_hard_predicate_uses_revisioned_anti_join() -> None:
    query = _hard_predicate_query(
        ("booking_start_without_complete",),
        filter_user_ids=False,
        restrict_to_vector_population=True,
        exclude_promotion_users=True,
    )

    assert "LEFT ANTI JOIN" in query
    assert "promotion_audience_exclusion_members" in query
    assert "argMax(state, exclusion_revision)" in query
    assert "reserved" in query and "consumed" in query
    assert "NOT IN" not in query
