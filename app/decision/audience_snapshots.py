from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol, Sequence

from app.audience_contract import (
    LEGACY_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_CONTRACT,
    SegmentAudienceContractError,
    SegmentDefinitionAudienceAdapter,
)
from app.analysis.semantic_selection import (
    compile_registered_segment_audience,
    semantic_query_vector_hash,
)


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


@dataclass(frozen=True, slots=True)
class AudienceSnapshotMember:
    user_id: str
    segment_id: str
    behavior_fit_score: Decimal


@dataclass(frozen=True, slots=True)
class AudienceSnapshotSet:
    analysis_id: str
    segment_ids: tuple[str, ...]
    vector_version: str
    member_count: int
    snapshot_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TargetAudienceResolution:
    analysis_id: str
    segment_ids: tuple[str, ...]
    contract: str


class AudienceSnapshotReader(Protocol):
    def resolve_target_contract(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> TargetAudienceResolution:
        ...

    def require_complete_set(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> AudienceSnapshotSet:
        ...

    def materialize_winning_members(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> None:
        ...

    def list_winning_members(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
        after_user_id: str | None,
        limit: int,
    ) -> list[AudienceSnapshotMember]:
        ...


class AudienceSnapshotContractError(RuntimeError):
    pass


class AudienceSnapshotRepository:
    """Read immutable analysis audiences created by the analysis pipeline.

    This repository intentionally contains no fallback to live vectors. Missing
    snapshots must stop experiment launch rather than silently changing users.
    """

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db
        self._adapter = SegmentDefinitionAudienceAdapter()

    def resolve_target_contract(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> TargetAudienceResolution:
        expected = tuple(sorted(set(segment_ids)))
        if not expected:
            raise AudienceSnapshotContractError("target segments are required")
        rows = self._db.fetchall(
            """
            SELECT segment_id, rule_json, audience_snapshot_id
            FROM promotion_target_segments
            WHERE analysis_id = %s
              AND segment_id = ANY(%s)
            ORDER BY segment_id ASC
            """,
            (analysis_id, list(expected)),
        )
        found = tuple(str(row["segment_id"]) for row in rows)
        if found != expected:
            missing = sorted(set(expected) - set(found))
            raise AudienceSnapshotContractError(
                "target audience contract is missing: " + ", ".join(missing)
            )
        contracts: list[str] = []
        for row in rows:
            segment_id = str(row["segment_id"])
            try:
                resolution = self._adapter.resolve(
                    segment_id=segment_id,
                    rule_json=row["rule_json"],
                )
            except SegmentAudienceContractError as exc:
                raise AudienceSnapshotContractError(
                    f"{exc.code}: {exc.segment_id}: {exc.reason}"
                ) from exc
            contract = (
                SEGMENT_AUDIENCE_CONTRACT
                if resolution.is_v2
                else LEGACY_AUDIENCE_CONTRACT
            )
            snapshot_id = row["audience_snapshot_id"]
            if contract == SEGMENT_AUDIENCE_CONTRACT and snapshot_id is None:
                raise AudienceSnapshotContractError(
                    "segment_audience.v1 target requires audience_snapshot_id: "
                    + segment_id
                )
            if contract == LEGACY_AUDIENCE_CONTRACT and snapshot_id is not None:
                raise AudienceSnapshotContractError(
                    "legacy target must not bind an audience snapshot: " + segment_id
                )
            contracts.append(contract)
        if len(set(contracts)) != 1:
            raise AudienceSnapshotContractError(
                "legacy and segment_audience.v1 targets cannot be mixed"
            )
        return TargetAudienceResolution(
            analysis_id=analysis_id,
            segment_ids=expected,
            contract=contracts[0],
        )

    def require_complete_set(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> AudienceSnapshotSet:
        expected = tuple(sorted(set(segment_ids)))
        if not expected:
            raise AudienceSnapshotContractError("snapshot segments are required")
        rows = self._db.fetchall(
            """
            SELECT
                target.segment_id,
                target.rule_json,
                target.audience_snapshot_id,
                snapshot.vector_version,
                snapshot.schema_version,
                snapshot.manifest_hash,
                snapshot.calibration_version,
                snapshot.calibration_hash,
                snapshot.audience_resolution_contract,
                snapshot.segment_audience_spec_hash,
                snapshot.query_vector_hash,
                snapshot.query_compiler_version,
                snapshot.query_compiler_hash,
                snapshot.score_threshold,
                snapshot.matcher_version,
                snapshot.search_policy_version,
                snapshot.metadata_json,
                snapshot.status AS snapshot_status,
                snapshot.final_user_count,
                snapshot.audience_status,
                snapshot.project_id = target.project_id
                    AND snapshot.campaign_id = target.campaign_id
                    AND snapshot.promotion_id = target.promotion_id
                    AND snapshot.segment_id = target.segment_id AS identity_matches,
                generation.status AS generation_status,
                generation.is_active AS generation_is_active,
                generation.project_id = target.project_id
                    AND generation.vector_version = snapshot.vector_version
                    AND generation.manifest_hash = snapshot.manifest_hash
                    AND generation.window_start = snapshot.window_start
                    AND generation.window_end = snapshot.window_end
                    AS generation_matches,
                (SELECT count(*) FROM segment_audience_members AS member
                 WHERE member.snapshot_id = snapshot.snapshot_id)
                    AS actual_member_count
            FROM promotion_target_segments AS target
            LEFT JOIN segment_audience_snapshots AS snapshot
              ON snapshot.snapshot_id = target.audience_snapshot_id
            LEFT JOIN user_behavior_vector_search_generations AS generation
              ON generation.vector_generation_id = snapshot.vector_generation_id
            WHERE target.analysis_id = %s
              AND target.segment_id = ANY(%s)
            ORDER BY target.segment_id ASC
            """,
            (analysis_id, list(expected)),
        )
        found = tuple(str(row["segment_id"]) for row in rows)
        if found != expected:
            missing = sorted(set(expected) - set(found))
            raise AudienceSnapshotContractError(
                "completed audience snapshot is required for every segment: "
                + ", ".join(missing)
            )
        binding_values = [row["audience_snapshot_id"] for row in rows]
        if any(value is None for value in binding_values):
            if not all(value is None for value in binding_values):
                raise AudienceSnapshotContractError(
                    "snapshot V2 and legacy target segments cannot be mixed"
                )
            raise AudienceSnapshotContractError(
                "explicit audience snapshot binding is required for every segment"
            )
        if any(
            str(row["snapshot_status"]) != "completed"
            or str(row["generation_status"]) not in {"activated", "superseded"}
            or not bool(row["identity_matches"])
            or not bool(row["generation_matches"])
            or int(row["actual_member_count"]) != int(row["final_user_count"])
            for row in rows
        ):
            raise AudienceSnapshotContractError(
                "audience snapshot contract validation failed"
            )
        if any(
            int(row["final_user_count"]) <= 0
            or str(row["audience_status"]) == "no_eligible_audience"
            for row in rows
        ):
            raise AudienceSnapshotContractError(
                "no_eligible_audience snapshot cannot start assignment"
            )
        for row in rows:
            segment_id = str(row["segment_id"])
            try:
                compiled = compile_registered_segment_audience(
                    segment_id=segment_id,
                    rule_json=row["rule_json"],
                )
            except SegmentAudienceContractError as exc:
                raise AudienceSnapshotContractError(
                    f"{exc.code}: {exc.segment_id}: {exc.reason}"
                ) from exc
            if not _snapshot_row_matches_compiled(row, compiled=compiled):
                raise AudienceSnapshotContractError(
                    "segment_audience_snapshot_semantic_mismatch: " + segment_id
                )
        versions = {str(row["vector_version"]) for row in rows}
        if len(versions) != 1:
            raise AudienceSnapshotContractError(
                "audience snapshots must use one vector version"
            )
        shared_semantic_versions = {
            (
                str(row["schema_version"]),
                str(row["manifest_hash"]),
                str(row["audience_resolution_contract"]),
                str(row["query_compiler_version"]),
                str(row["query_compiler_hash"]),
                str(row["matcher_version"]),
                str(row["search_policy_version"]),
            )
            for row in rows
        }
        segment_semantic_values = [
            (
                str(row["calibration_version"]),
                str(row["calibration_hash"]),
                str(row["segment_audience_spec_hash"]),
                str(row["query_vector_hash"]),
            )
            for row in rows
        ]
        if len(shared_semantic_versions) != 1 or any(
            not value for value in next(iter(shared_semantic_versions))
        ) or any(
            not value
            for segment_values in segment_semantic_values
            for value in segment_values
        ):
            raise AudienceSnapshotContractError(
                "audience snapshots must use one complete semantic contract"
            )
        return AudienceSnapshotSet(
            analysis_id=analysis_id,
            segment_ids=expected,
            vector_version=next(iter(versions)),
            member_count=sum(int(row["final_user_count"]) for row in rows),
            snapshot_ids=tuple(str(value) for value in binding_values),
        )

    def materialize_winning_members(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> None:
        expected = list(sorted(set(segment_ids)))
        self._db.execute(
            """
            CREATE TEMP TABLE resolved_audience_members
            ON COMMIT DROP
            AS
            SELECT DISTINCT ON (member.user_id)
                member.user_id,
                target.segment_id,
                member.behavior_fit_score
            FROM promotion_target_segments AS target
            JOIN segment_audience_members AS member
              ON member.snapshot_id = target.audience_snapshot_id
            WHERE target.analysis_id = %s
              AND target.segment_id = ANY(%s)
            ORDER BY member.user_id ASC,
                     member.behavior_fit_score DESC,
                     target.segment_id ASC
            """,
            (analysis_id, expected),
        )
        self._db.execute(
            """
            CREATE UNIQUE INDEX resolved_audience_members_user_idx
            ON resolved_audience_members (user_id)
            """
        )

    def list_winning_members(
        self,
        *,
        analysis_id: str,
        segment_ids: Sequence[str],
        after_user_id: str | None,
        limit: int,
    ) -> list[AudienceSnapshotMember]:
        if limit <= 0:
            raise ValueError("snapshot page limit must be positive")
        rows = self._db.fetchall(
            """
            SELECT user_id, segment_id, behavior_fit_score
            FROM resolved_audience_members
            WHERE (%s::text IS NULL OR user_id > %s)
            ORDER BY user_id ASC
            LIMIT %s
            """,
            (
                after_user_id,
                after_user_id,
                limit,
            ),
        )
        return [
            AudienceSnapshotMember(
                user_id=str(row["user_id"]),
                segment_id=str(row["segment_id"]),
                behavior_fit_score=Decimal(str(row["behavior_fit_score"])),
            )
            for row in rows
        ]


def _snapshot_row_matches_compiled(
    row: Mapping[str, Any],
    *,
    compiled: Any,
) -> bool:
    metadata = row.get("metadata_json")
    if not isinstance(metadata, Mapping):
        return False
    expected = (
        compiled.schema_version,
        compiled.vector_version,
        compiled.manifest_hash,
        compiled.calibration_version,
        compiled.calibration_hash,
        compiled.audience_resolution_contract,
        compiled.segment_audience_spec_hash,
        semantic_query_vector_hash(compiled),
        compiled.query_compiler_version,
        compiled.query_compiler_hash,
        Decimal(str(compiled.score_threshold)),
        compiled.template_id,
        compiled.template_version,
        compiled.template_semantic_hash,
        list(compiled.hard_predicate_keys),
        {
            key: list(value)
            for key, value in compiled.predicate_parameters.items()
        },
        compiled.semantic_selection_policy_id,
        compiled.semantic_anchor_policy_id,
        compiled.semantic_anchor_hash,
        Decimal(str(compiled.semantic_margin)),
        compiled.semantic_selection_status,
        compiled.business_lift_status,
        compiled.user_vectorizer_version,
        compiled.user_vectorizer_semantic_hash,
    )
    try:
        semantic_margin = Decimal(str(metadata.get("semantic_margin")))
    except (InvalidOperation, ValueError):
        return False
    actual = (
        str(row["schema_version"]),
        str(row["vector_version"]),
        str(row["manifest_hash"]),
        str(row["calibration_version"]),
        str(row["calibration_hash"]),
        str(row["audience_resolution_contract"]),
        str(row["segment_audience_spec_hash"]),
        str(row["query_vector_hash"]),
        str(row["query_compiler_version"]),
        str(row["query_compiler_hash"]),
        Decimal(str(row["score_threshold"])),
        metadata.get("template_id"),
        metadata.get("template_version"),
        metadata.get("template_semantic_hash"),
        metadata.get("hard_predicate_keys"),
        metadata.get("predicate_parameters"),
        metadata.get("semantic_selection_policy_id"),
        metadata.get("semantic_anchor_policy_id"),
        metadata.get("semantic_anchor_hash"),
        semantic_margin,
        metadata.get("semantic_selection_status"),
        metadata.get("business_lift_status"),
        metadata.get("user_vectorizer_version"),
        metadata.get("user_vectorizer_semantic_hash"),
    )
    return actual == expected
