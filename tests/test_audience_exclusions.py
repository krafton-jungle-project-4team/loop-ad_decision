from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.analysis.audience_search_repository import (
    PgClickHouseAudienceVectorSearchRepository,
    _hard_predicate_query,
)
from app.audience_exclusions import (
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
    def __init__(self, row=None, rows=()) -> None:
        self.row = row
        self.rows = list(rows)
        self.calls = []

    def fetchone(self, query, params=()):
        self.calls.append((query, params))
        return self.row

    def fetchall(self, query, params=()):
        self.calls.append((query, params))
        return list(self.rows)

    def execute(self, query, params=()):
        self.calls.append((query, params))


class _ClickHouse:
    def __init__(self, rows=(), *, fail_member_insert=False) -> None:
        self.rows = tuple(rows)
        self.calls = []
        self.inserts = []
        self.fail_member_insert = fail_member_insert

    def query(self, query, parameters=None):
        self.calls.append((query, parameters or {}))
        return _Result(self.rows)

    def insert(self, table, data, column_names):
        if self.fail_member_insert and table == "promotion_audience_exclusion_projection":
            raise RuntimeError("member projection failed")
        self.inserts.append((table, list(data), tuple(column_names)))


class _ExclusionReader:
    def __init__(self, context: PromotionAudienceExclusionContext) -> None:
        self.context = context
        self.calls = []

    def load_active_exclusion_context(self, **kwargs):
        self.calls.append(kwargs)
        return self.context


def test_exclusion_context_requires_clickhouse_revision_at_least_postgres() -> None:
    postgres = _Postgres(
        {
            "revision": 4,
            "excluded_user_count": 37,
        }
    )
    clickhouse = _ClickHouse(
        [
            {
                "applied_revision": 4,
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
    assert context.excluded_user_count == 37
    assert context.projection_revision == context.revision


def test_stale_clickhouse_projection_is_repaired_before_use() -> None:
    now = datetime(2026, 7, 18, tzinfo=UTC)
    postgres = _Postgres(
        {
            "revision": 2,
            "excluded_user_count": 2,
        },
        rows=[
            {
                "user_id": "user_a",
                "state": "consumed",
                "revision": 2,
                "updated_at": now,
            },
            {
                "user_id": "user_b",
                "state": "reserved",
                "revision": 1,
                "updated_at": now,
            },
            {
                "user_id": "user_c",
                "state": "released",
                "revision": 2,
                "updated_at": now,
            },
        ],
    )
    clickhouse = _ClickHouse([{"applied_revision": 1}])
    repository = PromotionAudienceExclusionRepository(
        postgres=postgres,
        clickhouse=clickhouse,
    )

    context = repository.load_active_exclusion_context(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
    )

    assert context.projection_revision == 2
    assert [insert[0] for insert in clickhouse.inserts] == [
        "promotion_audience_exclusion_projection",
        "promotion_audience_exclusion_projection_status",
    ]
    assert clickhouse.inserts[0][1] == [
        ("project", "campaign", "promotion", "user_a", "consumed", 2, now),
        ("project", "campaign", "promotion", "user_b", "reserved", 1, now),
        ("project", "campaign", "promotion", "user_c", "released", 2, now),
    ]
    assert clickhouse.inserts[1][1][0][:3] == ("project", "promotion", 2)


def test_projection_checkpoint_is_not_advanced_when_member_write_fails() -> None:
    now = datetime(2026, 7, 18, tzinfo=UTC)
    clickhouse = _ClickHouse(
        [{"applied_revision": 1}],
        fail_member_insert=True,
    )
    repository = PromotionAudienceExclusionRepository(
        postgres=_Postgres(
            {"revision": 2, "excluded_user_count": 1},
            rows=[
                {
                    "user_id": "user_a",
                    "state": "reserved",
                    "revision": 2,
                    "updated_at": now,
                }
            ],
        ),
        clickhouse=clickhouse,
    )

    with pytest.raises(RuntimeError, match="member projection failed"):
        repository.load_active_exclusion_context(
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
        )

    assert clickhouse.inserts == []


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
        excluded_user_count=12,
        projection_revision=3,
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


def test_empty_exclusion_context_uses_zero_revisions() -> None:
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
    assert context.projection_revision == 0


def test_clickhouse_hard_predicate_uses_revisioned_anti_join() -> None:
    query = _hard_predicate_query(
        ("booking_start_without_complete",),
        filter_user_ids=False,
        restrict_to_vector_population=True,
        exclude_promotion_users=True,
    )

    assert "LEFT ANTI JOIN" in query
    assert "promotion_audience_exclusion_active" in query
    assert "reserved" in query and "consumed" in query
    assert "NOT IN" not in query
