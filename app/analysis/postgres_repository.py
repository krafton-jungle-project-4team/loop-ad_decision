from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

from app.analysis.models import SegmentAggregate, StoredSegment, UserPrimarySegmentCandidate
from app.analysis.segments import is_default_segment_key


class Cursor(Protocol):
    def execute(self, query: str, parameters: tuple[Any, ...] = ()) -> Any:
        ...

    def fetchone(self) -> tuple[Any, ...] | None:
        ...


class Connection(Protocol):
    def cursor(self) -> Any:
        ...


class PostgresAnalysisRepository:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def get_project_timezone(self, project_id: int) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT timezone FROM projects WHERE id = %s",
                (project_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise LookupError(f"project not found: {project_id}")
        return str(row[0])

    def upsert_segments(
        self,
        project_id: int,
        aggregates: list[SegmentAggregate],
        run_id: int | None,
    ) -> dict[str, StoredSegment]:
        stored_segments: dict[str, StoredSegment] = {}
        with self.connection.cursor() as cursor:
            for aggregate in aggregates:
                if is_default_segment_key(aggregate.segment_key):
                    continue
                cursor.execute(
                    UPSERT_SEGMENT_SQL,
                    (
                        project_id,
                        aggregate.segment_key,
                        aggregate.name,
                        build_segment_description(aggregate.dimensions),
                        json.dumps(aggregate.dimensions, ensure_ascii=False, sort_keys=True),
                        run_id,
                    ),
                )
                row = cursor.fetchone()
                if row is not None:
                    stored_segments[str(row[1])] = StoredSegment(id=int(row[0]), segment_key=str(row[1]))
        return stored_segments

    def upsert_segment_daily_metrics(
        self,
        project_id: int,
        analysis_date: date,
        aggregates: list[SegmentAggregate],
        stored_segments: dict[str, StoredSegment],
        run_id: int | None,
    ) -> int:
        metric_count = 0
        with self.connection.cursor() as cursor:
            for aggregate in aggregates:
                stored_segment = stored_segments.get(aggregate.segment_key)
                if stored_segment is None:
                    continue
                cursor.execute(
                    UPSERT_SEGMENT_DAILY_METRIC_SQL,
                    (
                        project_id,
                        stored_segment.id,
                        analysis_date,
                        aggregate.user_count,
                        aggregate.session_count,
                        aggregate.page_view_count,
                        aggregate.product_view_count,
                        aggregate.add_to_cart_count,
                        aggregate.checkout_start_count,
                        aggregate.purchase_count,
                        aggregate.ad_impression_count,
                        aggregate.ad_click_count,
                        aggregate.revenue,
                        aggregate.view_to_cart_rate,
                        aggregate.cart_to_checkout_rate,
                        aggregate.checkout_to_purchase_rate,
                        aggregate.view_to_purchase_rate,
                        aggregate.ctr,
                        aggregate.cvr,
                        None,
                        aggregate.target_view_to_purchase_rate,
                        json.dumps(
                            {
                                "segment_key": aggregate.segment_key,
                                "dimensions": aggregate.dimensions,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        run_id,
                    ),
                )
                metric_count += 1
        return metric_count

    def upsert_user_segment_memberships(
        self,
        project_id: int,
        analysis_date: date,
        candidates: list[UserPrimarySegmentCandidate],
        stored_segments: dict[str, StoredSegment],
        run_id: int | None,
    ) -> int:
        membership_count = 0
        with self.connection.cursor() as cursor:
            for candidate in candidates:
                stored_segment = stored_segments.get(candidate.segment_key)
                if stored_segment is None:
                    continue
                cursor.execute(
                    DELETE_STALE_PRIMARY_MEMBERSHIP_SQL,
                    (
                        project_id,
                        candidate.external_user_id,
                        analysis_date,
                        stored_segment.id,
                    ),
                )
                cursor.execute(
                    UPSERT_USER_SEGMENT_MEMBERSHIP_SQL,
                    (
                        project_id,
                        candidate.external_user_id,
                        stored_segment.id,
                        analysis_date,
                        True,
                        candidate.confidence,
                        json.dumps(
                            {
                                "segment_key": candidate.segment_key,
                                "dimensions": candidate.dimensions,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        run_id,
                    ),
                )
                membership_count += 1
        return membership_count


def build_segment_description(dimensions: Mapping[str, str]) -> str:
    return "Daily analysis segment: " + ", ".join(
        f"{key}={value}" for key, value in sorted(dimensions.items())
    )


UPSERT_SEGMENT_SQL = """
INSERT INTO segments (
    project_id,
    segment_key,
    name,
    description,
    rule_json,
    status,
    is_default,
    created_run_id
) VALUES (
    %s,
    %s,
    %s,
    %s,
    %s::jsonb,
    'active',
    false,
    %s
)
ON CONFLICT (project_id, segment_key) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    rule_json = EXCLUDED.rule_json,
    status = 'active',
    is_default = false,
    created_run_id = COALESCE(segments.created_run_id, EXCLUDED.created_run_id),
    updated_at = now()
RETURNING id, segment_key
""".strip()


UPSERT_SEGMENT_DAILY_METRIC_SQL = """
INSERT INTO segment_daily_metrics (
    project_id,
    segment_id,
    analysis_date,
    user_count,
    session_count,
    page_view_count,
    product_view_count,
    add_to_cart_count,
    checkout_start_count,
    purchase_count,
    ad_impression_count,
    ad_click_count,
    revenue,
    view_to_cart_rate,
    cart_to_checkout_rate,
    checkout_to_purchase_rate,
    view_to_purchase_rate,
    ctr,
    cvr,
    baseline_view_to_purchase_rate,
    target_view_to_purchase_rate,
    metric_json,
    created_run_id
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
)
ON CONFLICT (project_id, segment_id, analysis_date) DO UPDATE SET
    user_count = EXCLUDED.user_count,
    session_count = EXCLUDED.session_count,
    page_view_count = EXCLUDED.page_view_count,
    product_view_count = EXCLUDED.product_view_count,
    add_to_cart_count = EXCLUDED.add_to_cart_count,
    checkout_start_count = EXCLUDED.checkout_start_count,
    purchase_count = EXCLUDED.purchase_count,
    ad_impression_count = EXCLUDED.ad_impression_count,
    ad_click_count = EXCLUDED.ad_click_count,
    revenue = EXCLUDED.revenue,
    view_to_cart_rate = EXCLUDED.view_to_cart_rate,
    cart_to_checkout_rate = EXCLUDED.cart_to_checkout_rate,
    checkout_to_purchase_rate = EXCLUDED.checkout_to_purchase_rate,
    view_to_purchase_rate = EXCLUDED.view_to_purchase_rate,
    ctr = EXCLUDED.ctr,
    cvr = EXCLUDED.cvr,
    baseline_view_to_purchase_rate = EXCLUDED.baseline_view_to_purchase_rate,
    target_view_to_purchase_rate = EXCLUDED.target_view_to_purchase_rate,
    metric_json = EXCLUDED.metric_json,
    created_run_id = EXCLUDED.created_run_id
""".strip()


DELETE_STALE_PRIMARY_MEMBERSHIP_SQL = """
DELETE FROM user_segment_memberships
WHERE project_id = %s
  AND external_user_id = %s
  AND analysis_date = %s
  AND is_primary = true
  AND segment_id <> %s
""".strip()


UPSERT_USER_SEGMENT_MEMBERSHIP_SQL = """
INSERT INTO user_segment_memberships (
    project_id,
    external_user_id,
    segment_id,
    analysis_date,
    is_primary,
    confidence,
    reason_json,
    created_run_id
) VALUES (
    %s, %s, %s, %s, %s, %s, %s::jsonb, %s
)
ON CONFLICT (project_id, external_user_id, segment_id, analysis_date) DO UPDATE SET
    is_primary = EXCLUDED.is_primary,
    confidence = EXCLUDED.confidence,
    reason_json = EXCLUDED.reason_json,
    created_run_id = EXCLUDED.created_run_id
""".strip()
