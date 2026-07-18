from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol, Sequence

from app.audience_contract import (
    LEGACY_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_CONTRACT,
    SegmentAudienceContractError,
    SegmentDefinitionAudienceAdapter,
    contract_score_threshold,
)
from app.analysis.semantic_selection import (
    compile_registered_segment_audience,
    semantic_query_vector_hash,
)
from app.logging import log


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
    behavior_fit_score: Decimal | None


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


@dataclass(frozen=True, slots=True)
class RunAudienceTargetBindingWrite:
    target_analysis_id: str
    segment_id: str
    allocation_plan_id: str
    final_snapshot_id: str


class RunAudienceBindingWriter(Protocol):
    def bind_run_targets(
        self,
        *,
        promotion_run_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        bindings: Sequence[RunAudienceTargetBindingWrite],
    ) -> None:
        ...

    def resolve_run_contract(
        self,
        *,
        promotion_run_id: str,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> TargetAudienceResolution:
        ...

    def require_run_binding_set(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
    ) -> AudienceSnapshotSet:
        ...


class AudienceSnapshotReader(Protocol):
    def resolve_run_contract(
        self,
        *,
        promotion_run_id: str,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> TargetAudienceResolution:
        ...

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

    def require_run_binding_set(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
    ) -> AudienceSnapshotSet:
        ...

    def consume_run_members(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
    ) -> None:
        ...

    def list_run_members(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
        after_user_id: str | None,
        limit: int,
    ) -> list[AudienceSnapshotMember]:
        ...


class AudienceSnapshotContractError(RuntimeError):
    pass


class AudienceSnapshotTargetAlreadyBoundError(AudienceSnapshotContractError):
    code = "segment_audience_target_already_run_bound"

    def __init__(self, *, segment_id: str, reason: str) -> None:
        self.segment_id = segment_id
        self.reason = reason
        super().__init__(reason)


class AudienceSnapshotRepository:
    """Read immutable analysis audiences created by the analysis pipeline.

    This repository intentionally contains no fallback to live vectors. Missing
    snapshots must stop experiment launch rather than silently changing users.
    """

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db
        self._adapter = SegmentDefinitionAudienceAdapter()

    def bind_run_targets(
        self,
        *,
        promotion_run_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        bindings: Sequence[RunAudienceTargetBindingWrite],
    ) -> None:
        expected = tuple(sorted({binding.segment_id for binding in bindings}))
        if not expected or len(expected) != len(bindings):
            raise AudienceSnapshotContractError(
                "run audience bindings require unique target segments"
            )
        pending: list[RunAudienceTargetBindingWrite] = []
        for binding in sorted(bindings, key=lambda value: value.segment_id):
            row = self._load_binding_target(
                binding=binding,
                project_id=project_id,
                campaign_id=campaign_id,
                promotion_id=promotion_id,
            )
            self._validate_binding_target(row=row, binding=binding)
            stored = self._db.fetchone(
                """
                SELECT promotion_run_id, target_analysis_id, allocation_plan_id,
                       final_snapshot_id
                FROM promotion_run_target_bindings
                WHERE target_analysis_id = %s AND segment_id = %s
                """,
                (binding.target_analysis_id, binding.segment_id),
            )
            if stored is not None:
                if str(stored["promotion_run_id"]) != promotion_run_id:
                    raise AudienceSnapshotTargetAlreadyBoundError(
                        segment_id=binding.segment_id,
                        reason=(
                            "V2 target is already bound to promotion run "
                            + str(stored["promotion_run_id"])
                        ),
                    )
                if (
                    str(stored["allocation_plan_id"]),
                    str(stored["final_snapshot_id"]),
                ) != (binding.allocation_plan_id, binding.final_snapshot_id):
                    raise AudienceSnapshotContractError(
                        "segment_audience_run_binding_conflict: "
                        + binding.segment_id
                    )
                continue
            if str(row["audience_reservation_state"]) != "reserved":
                raise AudienceSnapshotContractError(
                    "segment_audience_run_binding_reservation_invalid: "
                    + binding.segment_id
                )
            pending.append(binding)

        revision = self._advance_exclusion_revision(promotion_id) if pending else 0
        for binding in pending:
            self._consume_binding_reservation(
                promotion_run_id=promotion_run_id,
                project_id=project_id,
                promotion_id=promotion_id,
                revision=revision,
                binding=binding,
            )
        self.require_run_binding_set(
            promotion_run_id=promotion_run_id,
            segment_ids=expected,
        )

    def _load_binding_target(
        self,
        *,
        binding: RunAudienceTargetBindingWrite,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> Mapping[str, Any]:
        row = self._db.fetchone(
            """
            SELECT target.audience_reservation_state,
                   plan.status AS plan_status,
                   snapshot.status AS snapshot_status,
                   snapshot.snapshot_kind,
                   snapshot.source_snapshot_id,
                   snapshot.allocation_plan_id AS snapshot_allocation_plan_id,
                   snapshot.final_user_count,
                   (SELECT count(*) FROM segment_audience_members AS member
                    WHERE member.snapshot_id = snapshot.snapshot_id)
                       AS actual_member_count,
                   (SELECT count(*)
                    FROM promotion_audience_exclusion_members AS excluded
                    WHERE excluded.project_id = target.project_id
                      AND excluded.promotion_id = target.promotion_id
                      AND excluded.target_analysis_id = target.analysis_id
                      AND excluded.segment_id = target.segment_id
                      AND excluded.allocation_plan_id = target.allocation_plan_id
                      AND excluded.final_snapshot_id = target.audience_snapshot_id
                      AND excluded.state = target.audience_reservation_state)
                       AS reservation_count,
                   NOT EXISTS (
                       SELECT 1 FROM segment_audience_members AS member
                       WHERE member.snapshot_id = target.audience_snapshot_id
                         AND NOT EXISTS (
                             SELECT 1
                             FROM promotion_audience_exclusion_members AS excluded
                             WHERE excluded.project_id = target.project_id
                               AND excluded.promotion_id = target.promotion_id
                               AND excluded.user_id = member.user_id
                               AND excluded.target_analysis_id = target.analysis_id
                               AND excluded.segment_id = target.segment_id
                               AND excluded.allocation_plan_id = target.allocation_plan_id
                               AND excluded.final_snapshot_id = target.audience_snapshot_id
                               AND excluded.state = target.audience_reservation_state
                         )
                   ) AS every_member_reserved
            FROM promotion_target_segments AS target
            JOIN segment_audience_allocation_plans AS plan
              ON plan.allocation_plan_id = target.allocation_plan_id
            JOIN segment_audience_snapshots AS snapshot
              ON snapshot.snapshot_id = target.audience_snapshot_id
            WHERE target.analysis_id = %s
              AND target.project_id = %s
              AND target.campaign_id = %s
              AND target.promotion_id = %s
              AND target.segment_id = %s
              AND target.audience_snapshot_id = %s
              AND target.allocation_plan_id = %s
            FOR UPDATE OF target, plan
            """,
            (
                binding.target_analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                binding.segment_id,
                binding.final_snapshot_id,
                binding.allocation_plan_id,
            ),
        )
        if row is None:
            raise AudienceSnapshotContractError(
                "segment_audience_run_binding_target_invalid: "
                + binding.segment_id
            )
        return row

    @staticmethod
    def _validate_binding_target(
        *,
        row: Mapping[str, Any],
        binding: RunAudienceTargetBindingWrite,
    ) -> None:
        final_user_count = int(row["final_user_count"])
        actual_member_count = int(row["actual_member_count"])
        reservation_count = int(row["reservation_count"])
        checks = {
            "plan_status": str(row["plan_status"]) in {"finalized", "locked"},
            "snapshot_status": str(row["snapshot_status"]) == "completed",
            "snapshot_kind": str(row["snapshot_kind"]) == "final",
            "source_snapshot": bool(row["source_snapshot_id"]),
            "allocation_plan": (
                str(row["snapshot_allocation_plan_id"])
                == binding.allocation_plan_id
            ),
            "final_user_count": final_user_count > 0,
            "actual_member_count": actual_member_count == final_user_count,
            "reservation_count": reservation_count == final_user_count,
            "every_member_reserved": bool(row["every_member_reserved"]),
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if not failed_checks:
            return

        log.warn(
            "segment_audience_exclusion_binding_invalid",
            {
                "segmentId": binding.segment_id,
                "failedChecks": failed_checks,
                "planStatus": str(row["plan_status"]),
                "snapshotStatus": str(row["snapshot_status"]),
                "snapshotKind": str(row["snapshot_kind"]),
                "audienceReservationState": str(
                    row["audience_reservation_state"]
                ),
                "hasSourceSnapshot": bool(row["source_snapshot_id"]),
                "allocationPlanMatches": checks["allocation_plan"],
                "finalUserCount": final_user_count,
                "actualMemberCount": actual_member_count,
                "reservationCount": reservation_count,
                "everyMemberReserved": bool(row["every_member_reserved"]),
            },
        )
        raise AudienceSnapshotContractError(
            "segment_audience_exclusion_binding_invalid: " + binding.segment_id
        )

    def _advance_exclusion_revision(self, promotion_id: str) -> int:
        row = self._db.fetchone(
            "SELECT advance_promotion_audience_exclusion_revision(%s) AS revision",
            (promotion_id,),
        )
        if row is None:
            raise AudienceSnapshotContractError(
                "segment_audience_exclusion_revision_advance_failed"
            )
        return int(row["revision"])

    def _consume_binding_reservation(
        self,
        *,
        promotion_run_id: str,
        project_id: str,
        promotion_id: str,
        revision: int,
        binding: RunAudienceTargetBindingWrite,
    ) -> None:
        self._db.execute(
            """
            UPDATE promotion_audience_exclusion_members
            SET state = 'consumed', revision = %s,
                consumed_at = coalesce(consumed_at, now()), released_at = NULL
            WHERE project_id = %s AND promotion_id = %s
              AND target_analysis_id = %s AND segment_id = %s
              AND allocation_plan_id = %s AND final_snapshot_id = %s
              AND state = 'reserved'
            """,
            (
                revision,
                project_id,
                promotion_id,
                binding.target_analysis_id,
                binding.segment_id,
                binding.allocation_plan_id,
                binding.final_snapshot_id,
            ),
        )
        self._db.execute(
            """
            UPDATE promotion_target_segments
            SET audience_reservation_state = 'consumed'
            WHERE analysis_id = %s AND segment_id = %s
              AND allocation_plan_id = %s AND audience_snapshot_id = %s
              AND audience_reservation_state = 'reserved'
            """,
            (
                binding.target_analysis_id,
                binding.segment_id,
                binding.allocation_plan_id,
                binding.final_snapshot_id,
            ),
        )
        self._db.execute(
            """
            INSERT INTO promotion_run_target_bindings (
                promotion_run_id, target_analysis_id, segment_id,
                allocation_plan_id, final_snapshot_id, bound_at
            ) VALUES (%s, %s, %s, %s, %s, now())
            """,
            (
                promotion_run_id,
                binding.target_analysis_id,
                binding.segment_id,
                binding.allocation_plan_id,
                binding.final_snapshot_id,
            ),
        )
        self._db.execute(
            """
            UPDATE segment_audience_allocation_plans
            SET status = 'locked', locked_at = coalesce(locked_at, now())
            WHERE allocation_plan_id = %s AND status = 'finalized'
            """,
            (binding.allocation_plan_id,),
        )

    def resolve_run_contract(
        self,
        *,
        promotion_run_id: str,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> TargetAudienceResolution:
        expected = tuple(sorted(set(segment_ids)))
        rows = self._db.fetchall(
            """
            SELECT segment_id
            FROM promotion_run_target_bindings
            WHERE promotion_run_id = %s
            ORDER BY segment_id ASC
            """,
            (promotion_run_id,),
        )
        bound = tuple(str(row["segment_id"]) for row in rows)
        if bound:
            if bound != expected:
                raise AudienceSnapshotContractError(
                    "segment_audience_run_binding_set_incomplete"
                )
            return TargetAudienceResolution(
                analysis_id=analysis_id,
                segment_ids=expected,
                contract=SEGMENT_AUDIENCE_CONTRACT,
            )
        resolution = self.resolve_target_contract(
            analysis_id=analysis_id,
            segment_ids=expected,
        )
        if resolution.contract == SEGMENT_AUDIENCE_CONTRACT:
            raise AudienceSnapshotContractError(
                "segment_audience_run_binding_required"
            )
        return resolution

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
            SELECT segment_id, rule_json, audience_snapshot_id,
                   allocation_plan_id, audience_reservation_state
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
            allocation_plan_id = row["allocation_plan_id"]
            reservation_state = row["audience_reservation_state"]
            if contract == SEGMENT_AUDIENCE_CONTRACT and (
                snapshot_id is None
                or allocation_plan_id is None
                or str(reservation_state) not in {"reserved", "consumed"}
            ):
                raise AudienceSnapshotContractError(
                    "segment_audience.v1 target requires an active final audience: "
                    + segment_id
                )
            if contract == LEGACY_AUDIENCE_CONTRACT and any(
                value is not None
                for value in (snapshot_id, allocation_plan_id, reservation_state)
            ):
                raise AudienceSnapshotContractError(
                    "legacy target must not bind a V2 audience: " + segment_id
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
                snapshot.snapshot_kind,
                snapshot.source_snapshot_id,
                snapshot.allocation_plan_id AS snapshot_allocation_plan_id,
                snapshot.final_user_count,
                snapshot.audience_status,
                target.allocation_plan_id AS target_allocation_plan_id,
                target.audience_reservation_state,
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
            or str(row["snapshot_kind"]) != "final"
            or not row["source_snapshot_id"]
            or str(row["snapshot_allocation_plan_id"])
            != str(row["target_allocation_plan_id"])
            or str(row["audience_reservation_state"])
            not in {"reserved", "consumed"}
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

    def require_run_binding_set(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
    ) -> AudienceSnapshotSet:
        expected = tuple(sorted(set(segment_ids)))
        if not expected:
            raise AudienceSnapshotContractError(
                "run audience bindings require target segments"
            )
        rows = self._db.fetchall(
            """
            SELECT
                binding.segment_id,
                binding.final_snapshot_id,
                binding.allocation_plan_id,
                binding.target_analysis_id,
                plan.status AS plan_status,
                snapshot.status AS snapshot_status,
                snapshot.vector_version,
                snapshot.final_user_count,
                snapshot.audience_status,
                snapshot.snapshot_kind,
                snapshot.source_snapshot_id,
                snapshot.allocation_plan_id AS snapshot_allocation_plan_id,
                target.audience_reservation_state,
                (SELECT count(*)
                 FROM segment_audience_members AS member
                 WHERE member.snapshot_id = binding.final_snapshot_id)
                    AS actual_member_count,
                (SELECT count(*)
                 FROM promotion_audience_exclusion_members AS excluded
                 WHERE excluded.target_analysis_id = binding.target_analysis_id
                   AND excluded.segment_id = binding.segment_id
                   AND excluded.allocation_plan_id = binding.allocation_plan_id
                   AND excluded.final_snapshot_id = binding.final_snapshot_id
                   AND excluded.state = 'consumed')
                    AS active_reservation_count,
                NOT EXISTS (
                    SELECT 1
                    FROM segment_audience_members AS member
                    WHERE member.snapshot_id = binding.final_snapshot_id
                      AND NOT EXISTS (
                          SELECT 1
                          FROM promotion_audience_exclusion_members AS excluded
                          WHERE excluded.user_id = member.user_id
                            AND excluded.target_analysis_id = binding.target_analysis_id
                            AND excluded.segment_id = binding.segment_id
                            AND excluded.allocation_plan_id = binding.allocation_plan_id
                            AND excluded.final_snapshot_id = binding.final_snapshot_id
                            AND excluded.state = 'consumed'
                      )
                ) AS every_member_reserved
            FROM promotion_run_target_bindings AS binding
            JOIN segment_audience_allocation_plans AS plan
              ON plan.allocation_plan_id = binding.allocation_plan_id
            JOIN segment_audience_snapshots AS snapshot
              ON snapshot.snapshot_id = binding.final_snapshot_id
            JOIN promotion_target_segments AS target
              ON target.analysis_id = binding.target_analysis_id
             AND target.segment_id = binding.segment_id
             AND target.allocation_plan_id = binding.allocation_plan_id
             AND target.audience_snapshot_id = binding.final_snapshot_id
            WHERE binding.promotion_run_id = %s
            ORDER BY binding.segment_id ASC
            """,
            (promotion_run_id,),
        )
        found = tuple(str(row["segment_id"]) for row in rows)
        if found != expected:
            raise AudienceSnapshotContractError(
                "segment_audience_run_binding_set_incomplete"
            )
        for row in rows:
            if (
                str(row["plan_status"]) != "locked"
                or str(row["snapshot_status"]) != "completed"
                or str(row["snapshot_kind"]) != "final"
                or str(row["snapshot_allocation_plan_id"])
                != str(row["allocation_plan_id"])
                or not row["source_snapshot_id"]
                or str(row["audience_reservation_state"]) != "consumed"
                or int(row["final_user_count"]) <= 0
                or str(row["audience_status"]) == "no_eligible_audience"
                or int(row["actual_member_count"]) != int(row["final_user_count"])
                or int(row["active_reservation_count"])
                != int(row["final_user_count"])
                or not bool(row["every_member_reserved"])
            ):
                raise AudienceSnapshotContractError(
                    "segment_audience_run_binding_invalid: "
                    + str(row["segment_id"])
                )
        duplicate = self._db.fetchone(
            """
            SELECT member.user_id
            FROM promotion_run_target_bindings AS binding
            JOIN segment_audience_members AS member
              ON member.snapshot_id = binding.final_snapshot_id
            WHERE binding.promotion_run_id = %s
            GROUP BY member.user_id
            HAVING count(*) > 1
            LIMIT 1
            """,
            (promotion_run_id,),
        )
        if duplicate is not None:
            raise AudienceSnapshotContractError(
                "segment_audience_final_snapshots_overlap"
            )
        versions = {str(row["vector_version"]) for row in rows}
        if len(versions) != 1:
            raise AudienceSnapshotContractError(
                "run final snapshots must use one vector version"
            )
        return AudienceSnapshotSet(
            analysis_id=promotion_run_id,
            segment_ids=expected,
            vector_version=next(iter(versions)),
            member_count=sum(int(row["final_user_count"]) for row in rows),
            snapshot_ids=tuple(
                str(row["final_snapshot_id"]) for row in rows
            ),
        )

    def consume_run_members(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
    ) -> None:
        self.require_run_binding_set(
            promotion_run_id=promotion_run_id,
            segment_ids=segment_ids,
        )

    def list_run_members(
        self,
        *,
        promotion_run_id: str,
        segment_ids: Sequence[str],
        after_user_id: str | None,
        limit: int,
    ) -> list[AudienceSnapshotMember]:
        if limit <= 0:
            raise ValueError("snapshot page limit must be positive")
        rows = self._db.fetchall(
            """
            SELECT
                member.user_id,
                binding.segment_id,
                member.behavior_fit_score
            FROM promotion_run_target_bindings AS binding
            JOIN segment_audience_members AS member
              ON member.snapshot_id = binding.final_snapshot_id
            WHERE binding.promotion_run_id = %s
              AND binding.segment_id = ANY(%s)
              AND (%s::text IS NULL OR member.user_id > %s)
            ORDER BY member.user_id ASC
            LIMIT %s
            """,
            (
                promotion_run_id,
                list(sorted(set(segment_ids))),
                after_user_id,
                after_user_id,
                limit,
            ),
        )
        return [
            AudienceSnapshotMember(
                user_id=str(row["user_id"]),
                segment_id=str(row["segment_id"]),
                behavior_fit_score=(
                    Decimal(str(row["behavior_fit_score"]))
                    if row["behavior_fit_score"] is not None
                    else None
                ),
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
        contract_score_threshold(compiled.score_threshold),
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
