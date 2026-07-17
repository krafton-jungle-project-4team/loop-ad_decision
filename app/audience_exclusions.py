from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from psycopg import errors


POSTGRES_EXCLUSION_RELATION = "promotion_audience_exclusion_members"
CLICKHOUSE_EXCLUSION_RELATION = "promotion_audience_exclusion_active"
EXCLUSION_REVISION_RELATION = "promotion_audience_exclusion_state"
CLICKHOUSE_PROJECTION_REVISION_RELATION = (
    "promotion_audience_exclusion_projection_status"
)


class SegmentAudienceExclusionError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        promotion_id: str,
        reason: str,
        segment_id: str | None = None,
    ) -> None:
        self.code = code
        self.promotion_id = promotion_id
        self.segment_id = segment_id
        self.reason = reason
        super().__init__(reason)

    def to_detail(self) -> dict[str, str]:
        detail = {
            "code": self.code,
            "promotion_id": self.promotion_id,
            "reason": self.reason,
        }
        if self.segment_id is not None:
            detail["segment_id"] = self.segment_id
        return detail


@dataclass(frozen=True, slots=True)
class PromotionAudienceExclusionContext:
    project_id: str
    campaign_id: str
    promotion_id: str
    revision: int
    excluded_user_count: int
    postgres_relation: str = POSTGRES_EXCLUSION_RELATION
    clickhouse_relation: str = CLICKHOUSE_EXCLUSION_RELATION
    projection_revision: int = 0

    def require_projection_ready(self) -> None:
        if self.projection_revision < self.revision:
            raise SegmentAudienceExclusionError(
                code="segment_audience_exclusion_projection_not_ready",
                promotion_id=self.promotion_id,
                reason=(
                    "ClickHouse exclusion projection must be caught up to the "
                    "PostgreSQL exclusion revision"
                ),
            )


class PostgresExecutor(Protocol):
    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        ...


class ClickHouseClient(Protocol):
    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class PromotionAudienceExclusionReader(Protocol):
    def load_active_exclusion_context(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionAudienceExclusionContext:
        ...


class PromotionAudienceExclusionRepository:
    """Read the promotion-scoped exclusion revision shared by PG and CH.

    The schema is owned by Data Contract. A missing relation is a contract
    failure; this repository never interprets it as an empty legacy audience.
    """

    def __init__(
        self,
        *,
        postgres: PostgresExecutor,
        clickhouse: ClickHouseClient,
    ) -> None:
        self._postgres = postgres
        self._clickhouse = clickhouse

    def load_active_exclusion_context(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionAudienceExclusionContext:
        try:
            row = self._postgres.fetchone(
                f"""
                SELECT
                    state.revision,
                    (
                        SELECT count(*)
                        FROM {POSTGRES_EXCLUSION_RELATION} AS member
                        WHERE member.project_id = %s
                          AND member.promotion_id = state.promotion_id
                          AND member.state IN ('reserved', 'consumed')
                    ) AS excluded_user_count
                FROM {EXCLUSION_REVISION_RELATION} AS state
                WHERE state.promotion_id = %s
                """,
                (project_id, promotion_id),
            )
        except (errors.UndefinedTable, errors.UndefinedColumn) as exc:
            raise SegmentAudienceExclusionError(
                code="segment_audience_exclusion_contract_missing",
                promotion_id=promotion_id,
                reason="promotion audience exclusion PostgreSQL contract is missing",
            ) from exc

        revision = int(row["revision"]) if row is not None else 0
        excluded_user_count = (
            int(row["excluded_user_count"]) if row is not None else 0
        )
        projection = self._load_projection_revision(
            project_id=project_id,
            promotion_id=promotion_id,
        )
        context = PromotionAudienceExclusionContext(
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            revision=revision,
            excluded_user_count=excluded_user_count,
            projection_revision=projection,
        )
        context.require_projection_ready()
        return context

    def _load_projection_revision(
        self,
        *,
        project_id: str,
        promotion_id: str,
    ) -> int:
        try:
            result = self._clickhouse.query(
                f"""
                SELECT applied_revision
                FROM {CLICKHOUSE_PROJECTION_REVISION_RELATION}
                WHERE project_id = {{project_id:String}}
                  AND promotion_id = {{promotion_id:String}}
                ORDER BY applied_revision DESC
                LIMIT 1
                """,
                parameters={
                    "project_id": project_id,
                    "promotion_id": promotion_id,
                },
            )
        except Exception as exc:
            if _is_missing_clickhouse_contract(exc):
                raise SegmentAudienceExclusionError(
                    code="segment_audience_exclusion_contract_missing",
                    promotion_id=promotion_id,
                    reason=(
                        "promotion audience exclusion ClickHouse projection "
                        "contract is missing"
                    ),
                ) from exc
            raise
        rows = (
            list(result.named_results())
            if hasattr(result, "named_results")
            else list(result.result_rows)
        )
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, Mapping):
            return int(row["applied_revision"])
        return int(row[0])


def _is_missing_clickhouse_contract(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in {47, 60}:
        return True
    message = str(exc).lower()
    return "unknown table" in message or "unknown identifier" in message
