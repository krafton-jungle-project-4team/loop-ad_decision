from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Protocol, Sequence

from app.analysis.audience_search import AudienceSearchMethod, AudienceSearchResult
from app.analysis.behavior_vector_schema import CandidateBehaviorSpec
from app.analysis.semantic_selection import semantic_query_vector_hash
from app.audience_exclusions import PromotionAudienceExclusionContext


SCORE_THRESHOLD_QUANTUM = Decimal("0.000001")


class AudienceSnapshotBindingError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        code: str = "segment_audience_snapshot_binding_invalid",
        segment_id: str = "",
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.segment_id = segment_id
        self.reason = reason

    def to_detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "segment_id": self.segment_id,
            "reason": self.reason,
        }


class PostgresExecutor(Protocol):
    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        ...

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class AudienceSnapshotWrite:
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    segment_id: str
    segment_vector_id: str
    vector_generation_id: str
    source_cutoff: datetime
    window_start: datetime
    window_end: datetime
    spec: CandidateBehaviorSpec
    search_result: AudienceSearchResult
    min_sample_size: int
    suggestion_id: str | None = None
    exclusion_context: PromotionAudienceExclusionContext | None = None


@dataclass(frozen=True, slots=True)
class BoundAudienceSnapshot:
    snapshot_id: str
    segment_vector_id: str
    vector_generation_id: str
    source_cutoff: datetime
    window_start: datetime
    window_end: datetime
    eligible_user_count: int
    behavior_match_count: int
    final_user_count: int
    selection_method: str
    estimated_recall: float
    recall_lower_bound: float
    recall_target: float
    meets_min_sample_size: bool
    promotion_exclusion_revision: int | None = None
    excluded_user_count: int = 0


class AudienceSnapshotRepository:
    MEMBER_BATCH_SIZE = 1000
    MATERIALIZED_MEMBER_RELATIONS = {
        "audience_exact_members",
        "audience_ann_members",
    }

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def save_completed(self, write: AudienceSnapshotWrite) -> str:
        if write.window_start >= write.window_end:
            raise ValueError("audience snapshot window must be increasing")
        if write.search_result.hard_match_user_count > write.search_result.corpus_user_count:
            raise ValueError("hard match count must not exceed eligible count")
        snapshot_id = _snapshot_id(write)
        audit = write.search_result.recall_audit
        input_fingerprint = _input_fingerprint(write)
        final_user_count = write.search_result.final_user_count
        audience_status = _audience_status(
            final_user_count=final_user_count,
            min_sample_size=write.min_sample_size,
        )
        existing = self._db.fetchone(
            """
            SELECT input_fingerprint, final_user_count,
                   (SELECT count(*) FROM segment_audience_members AS member
                    WHERE member.snapshot_id = snapshot.snapshot_id)
                       AS actual_member_count
            FROM segment_audience_snapshots AS snapshot
            WHERE snapshot_id = %s
            """,
            (snapshot_id,),
        )
        if existing is not None:
            if (
                str(existing["input_fingerprint"]) == input_fingerprint
                and int(existing["final_user_count"])
                == final_user_count
                and int(existing["actual_member_count"])
                == final_user_count
            ):
                return snapshot_id
            raise RuntimeError(
                "audience snapshot retry conflicts with stored semantic fingerprint"
            )
        self._db.execute(
            """
            INSERT INTO segment_audience_snapshots (
                snapshot_id,
                suggestion_id,
                analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                segment_id,
                segment_vector_id,
                vector_generation_id,
                schema_version,
                vector_version,
                manifest_hash,
                audience_resolution_contract,
                segment_audience_spec_hash,
                query_vector_hash,
                query_compiler_version,
                query_compiler_hash,
                matcher_version,
                search_policy_version,
                calibration_version,
                calibration_hash,
                score_threshold,
                source_cutoff,
                window_start,
                window_end,
                eligible_user_count,
                behavior_match_count,
                final_user_count,
                min_sample_size,
                audience_status,
                selection_method,
                estimated_recall,
                recall_lower_bound,
                recall_target,
                input_fingerprint,
                meets_min_sample_size,
                status,
                metadata_json,
                snapshot_kind,
                source_snapshot_id,
                allocation_plan_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                'source', NULL, NULL
            )
            """,
            (
                snapshot_id,
                write.suggestion_id,
                write.analysis_id,
                write.project_id,
                write.campaign_id,
                write.promotion_id,
                write.segment_id,
                write.segment_vector_id,
                write.vector_generation_id,
                write.spec.schema_version,
                write.spec.vector_version,
                write.spec.manifest_hash,
                write.spec.audience_resolution_contract,
                write.spec.segment_audience_spec_hash,
                _query_vector_hash(write.spec),
                write.spec.query_compiler_version,
                write.spec.query_compiler_hash,
                "exact_cosine_rerank.v2",
                write.search_result.policy_version,
                write.spec.calibration_version,
                write.spec.calibration_hash,
                _contract_score_threshold(write.spec.score_threshold),
                write.source_cutoff,
                write.window_start,
                write.window_end,
                write.search_result.corpus_user_count,
                write.search_result.hard_match_user_count,
                final_user_count,
                write.min_sample_size,
                audience_status,
                write.search_result.method.value,
                Decimal(str(audit.estimated_recall)) if audit else Decimal("1"),
                Decimal(str(audit.recall_lower_bound)) if audit else Decimal("1"),
                Decimal(str(audit.target_recall)) if audit else Decimal("1"),
                input_fingerprint,
                final_user_count > 0
                and final_user_count >= write.min_sample_size,
                "completed",
                {
                    "candidate_type": write.spec.candidate_type,
                    "snapshot_kind": "source",
                    "spec_fingerprint": _spec_fingerprint(write.spec),
                    "hard_predicate_keys": list(write.spec.hard_predicate_keys),
                    "predicate_parameters": {
                        key: list(value)
                        for key, value in write.spec.predicate_parameters.items()
                    },
                    "active_blocks": list(write.spec.active_blocks),
                    "block_weights": dict(write.spec.block_weights),
                    "requested_k": write.search_result.requested_k,
                    "recall_confidence": audit.confidence if audit else 1.0,
                    "calibration_hash": write.spec.calibration_hash,
                    "min_sample_size": write.min_sample_size,
                    "targetable": final_user_count > 0,
                    "audience_resolution_contract": (
                        write.spec.audience_resolution_contract
                    ),
                    "segment_audience_spec_hash": (
                        write.spec.segment_audience_spec_hash
                    ),
                    "query_compiler_version": write.spec.query_compiler_version,
                    "query_compiler_hash": write.spec.query_compiler_hash,
                    "template_id": write.spec.template_id,
                    "template_version": write.spec.template_version,
                    "template_semantic_hash": (
                        write.spec.template_semantic_hash
                    ),
                    "semantic_selection_policy_id": (
                        write.spec.semantic_selection_policy_id
                    ),
                    "semantic_anchor_policy_id": (
                        write.spec.semantic_anchor_policy_id
                    ),
                    "semantic_anchor_hash": write.spec.semantic_anchor_hash,
                    "semantic_margin": write.spec.semantic_margin,
                    "semantic_selection_status": (
                        write.spec.semantic_selection_status
                    ),
                    "business_lift_status": write.spec.business_lift_status,
                    "user_vectorizer_version": (
                        write.spec.user_vectorizer_version
                    ),
                    "user_vectorizer_semantic_hash": (
                        write.spec.user_vectorizer_semantic_hash
                    ),
                    "exclusion_revision": (
                        write.exclusion_context.revision
                        if write.exclusion_context is not None
                        else None
                    ),
                    "excluded_user_count": (
                        write.exclusion_context.excluded_user_count
                        if write.exclusion_context is not None
                        else 0
                    ),
                    "exclusion_projection_revision": (
                        write.exclusion_context.projection_revision
                        if write.exclusion_context is not None
                        else None
                    ),
                },
            ),
        )
        if write.search_result.members_relation is not None:
            relation = write.search_result.members_relation
            if relation not in self.MATERIALIZED_MEMBER_RELATIONS:
                raise ValueError("unsupported materialized audience member relation")
            retrieval_source = (
                "ann"
                if write.search_result.method == AudienceSearchMethod.ANN
                else "exact"
            )
            self._db.execute(
                f"""
                INSERT INTO segment_audience_members (
                    snapshot_id,
                    user_id,
                    behavior_fit_score,
                    retrieval_source,
                    retrieval_rank
                )
                SELECT
                    %s,
                    user_id,
                    behavior_fit_score,
                    %s,
                    retrieval_rank
                FROM {relation}
                ORDER BY user_id ASC
                """,
                (snapshot_id, retrieval_source),
            )
            self._require_member_count(
                snapshot_id=snapshot_id,
                expected_count=final_user_count,
            )
            return snapshot_id
        members = write.search_result.members
        for offset in range(0, len(members), self.MEMBER_BATCH_SIZE):
            chunk = members[offset : offset + self.MEMBER_BATCH_SIZE]
            self._db.execute(
                """
                INSERT INTO segment_audience_members (
                    snapshot_id,
                    user_id,
                    behavior_fit_score,
                    retrieval_source,
                    retrieval_rank
                )
                SELECT *
                FROM unnest(
                    %s::text[],
                    %s::text[],
                    %s::numeric[],
                    %s::text[],
                    %s::integer[]
                )
                """,
                (
                    [snapshot_id] * len(chunk),
                    [member.user_id for member in chunk],
                    [
                        Decimal(str(member.behavior_fit_score))
                        for member in chunk
                    ],
                    [
                        "ann"
                        if write.search_result.method == AudienceSearchMethod.ANN
                        else "exact"
                    ] * len(chunk),
                    [member.retrieval_rank for member in chunk],
                ),
            )
        self._require_member_count(
            snapshot_id=snapshot_id,
            expected_count=final_user_count,
        )
        return snapshot_id

    def _require_member_count(
        self,
        *,
        snapshot_id: str,
        expected_count: int,
    ) -> None:
        row = self._db.fetchone(
            """
            SELECT count(*) AS actual_member_count
            FROM segment_audience_members
            WHERE snapshot_id = %s
            """,
            (snapshot_id,),
        )
        actual_count = int(row["actual_member_count"]) if row is not None else -1
        if actual_count != expected_count:
            raise RuntimeError(
                "audience snapshot member insert did not preserve final user count"
            )

    def require_binding(
        self,
        *,
        snapshot_id: str,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        segment_id: str,
        spec: CandidateBehaviorSpec,
    ) -> BoundAudienceSnapshot:
        row = self._db.fetchone(
            """
            SELECT
                snapshot.snapshot_id, snapshot.segment_vector_id,
                snapshot.vector_generation_id, snapshot.source_cutoff,
                snapshot.window_start, snapshot.window_end,
                snapshot.eligible_user_count, snapshot.behavior_match_count,
                snapshot.final_user_count, snapshot.selection_method,
                snapshot.estimated_recall, snapshot.recall_lower_bound,
                snapshot.recall_target, snapshot.meets_min_sample_size,
                snapshot.status AS snapshot_status,
                snapshot.project_id, snapshot.campaign_id,
                snapshot.promotion_id, snapshot.segment_id,
                snapshot.schema_version, snapshot.vector_version,
                snapshot.manifest_hash, snapshot.calibration_version,
                snapshot.calibration_hash,
                snapshot.audience_resolution_contract,
                snapshot.segment_audience_spec_hash,
                snapshot.query_vector_hash,
                snapshot.query_compiler_version,
                snapshot.query_compiler_hash,
                snapshot.score_threshold, snapshot.metadata_json,
                generation.status AS generation_status,
                generation.is_active AS generation_is_active,
                (SELECT count(*) FROM segment_audience_members AS member
                 WHERE member.snapshot_id = snapshot.snapshot_id)
                    AS actual_member_count
            FROM segment_audience_snapshots AS snapshot
            JOIN user_behavior_vector_search_generations AS generation
              ON generation.vector_generation_id = snapshot.vector_generation_id
            WHERE snapshot.snapshot_id = %s
            """,
            (snapshot_id,),
        )
        if row is None:
            raise AudienceSnapshotBindingError(
                "audience snapshot binding does not exist"
            )
        expected_identity = (
            project_id,
            campaign_id,
            promotion_id,
            segment_id,
            spec.schema_version,
            spec.vector_version,
            spec.manifest_hash,
            spec.calibration_version,
            spec.calibration_hash,
            spec.audience_resolution_contract,
            spec.segment_audience_spec_hash,
            _query_vector_hash(spec),
            spec.query_compiler_version,
            spec.query_compiler_hash,
            _contract_score_threshold(spec.score_threshold),
            _spec_fingerprint(spec),
        )
        metadata = row["metadata_json"]
        if not isinstance(metadata, Mapping):
            raise AudienceSnapshotBindingError("audience snapshot metadata is invalid")
        actual_identity = (
            str(row["project_id"]),
            str(row["campaign_id"]),
            str(row["promotion_id"]),
            str(row["segment_id"]),
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
            _contract_score_threshold(row["score_threshold"]),
            str(metadata.get("spec_fingerprint", "")),
        )
        if actual_identity != expected_identity:
            raise AudienceSnapshotBindingError(
                "audience snapshot binding has a different fingerprint"
            )
        if str(row["snapshot_status"]) != "completed":
            raise AudienceSnapshotBindingError("audience snapshot binding is incomplete")
        if str(row["generation_status"]) not in {"activated", "superseded"}:
            raise AudienceSnapshotBindingError(
                "audience snapshot vector generation was never activated"
            )
        if int(row["actual_member_count"]) != int(row["final_user_count"]):
            raise AudienceSnapshotBindingError(
                "audience snapshot member count is inconsistent"
            )
        return BoundAudienceSnapshot(
            snapshot_id=str(row["snapshot_id"]),
            segment_vector_id=str(row["segment_vector_id"]),
            vector_generation_id=str(row["vector_generation_id"]),
            source_cutoff=row["source_cutoff"],
            window_start=row["window_start"],
            window_end=row["window_end"],
            eligible_user_count=int(row["eligible_user_count"]),
            behavior_match_count=int(row["behavior_match_count"]),
            final_user_count=int(row["final_user_count"]),
            selection_method=str(row["selection_method"]),
            estimated_recall=float(row["estimated_recall"]),
            recall_lower_bound=float(row["recall_lower_bound"]),
            recall_target=float(row["recall_target"]),
            meets_min_sample_size=bool(row["meets_min_sample_size"]),
            promotion_exclusion_revision=(
                int(metadata["exclusion_revision"])
                if metadata.get("exclusion_revision") is not None
                else None
            ),
            excluded_user_count=int(metadata.get("excluded_user_count") or 0),
        )


def _snapshot_id(write: AudienceSnapshotWrite) -> str:
    fingerprint = _input_fingerprint(write)
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            ":".join(
                (
                    write.analysis_id,
                    write.segment_id,
                    fingerprint,
                )
            ),
        )
    )


def _input_fingerprint(write: AudienceSnapshotWrite) -> str:
    payload = {
        "segment_id": write.segment_id,
        "audience_resolution_contract": write.spec.audience_resolution_contract,
        "segment_audience_spec_hash": write.spec.segment_audience_spec_hash,
        "query_vector_hash": _query_vector_hash(write.spec),
        "hard_predicate_keys": list(write.spec.hard_predicate_keys),
        "predicate_parameters": {
            key: list(value) for key, value in write.spec.predicate_parameters.items()
        },
        "score_threshold": write.spec.score_threshold,
        "vector_generation_id": write.vector_generation_id,
        "schema_version": write.spec.schema_version,
        "vector_version": write.spec.vector_version,
        "manifest_hash": write.spec.manifest_hash,
        "calibration_version": write.spec.calibration_version,
        "calibration_hash": write.spec.calibration_hash,
        "query_compiler_version": write.spec.query_compiler_version,
        "query_compiler_hash": write.spec.query_compiler_hash,
        "template_id": write.spec.template_id,
        "template_version": write.spec.template_version,
        "template_semantic_hash": write.spec.template_semantic_hash,
        "semantic_selection_policy_id": (
            write.spec.semantic_selection_policy_id
        ),
        "semantic_anchor_policy_id": write.spec.semantic_anchor_policy_id,
        "semantic_anchor_hash": write.spec.semantic_anchor_hash,
        "semantic_margin": write.spec.semantic_margin,
        "semantic_selection_status": write.spec.semantic_selection_status,
        "business_lift_status": write.spec.business_lift_status,
        "user_vectorizer_version": write.spec.user_vectorizer_version,
        "user_vectorizer_semantic_hash": (
            write.spec.user_vectorizer_semantic_hash
        ),
        "search_policy_version": write.search_result.policy_version,
        "source_cutoff": write.source_cutoff.isoformat(),
        "window_start": write.window_start.isoformat(),
        "window_end": write.window_end.isoformat(),
        "exclusion_revision": (
            write.exclusion_context.revision
            if write.exclusion_context is not None
            else None
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _query_vector_hash(spec: CandidateBehaviorSpec) -> str:
    return semantic_query_vector_hash(spec)


def _contract_score_threshold(value: float | Decimal) -> Decimal:
    """Normalize to the NUMERIC(10, 6) precision defined by the Data Contract."""
    return Decimal(str(value)).quantize(
        SCORE_THRESHOLD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _spec_fingerprint(spec: CandidateBehaviorSpec) -> str:
    payload = {
        "candidate_type": spec.candidate_type,
        "query_vector": list(spec.query_vector),
        "hard_predicate_keys": list(spec.hard_predicate_keys),
        "predicate_parameters": {
            key: list(value) for key, value in spec.predicate_parameters.items()
        },
        "score_threshold": spec.score_threshold,
        "schema_version": spec.schema_version,
        "vector_version": spec.vector_version,
        "manifest_hash": spec.manifest_hash,
        "calibration_version": spec.calibration_version,
        "calibration_hash": spec.calibration_hash,
        "audience_resolution_contract": spec.audience_resolution_contract,
        "segment_audience_spec_hash": spec.segment_audience_spec_hash,
        "query_compiler_version": spec.query_compiler_version,
        "query_compiler_hash": spec.query_compiler_hash,
        "template_id": spec.template_id,
        "template_version": spec.template_version,
        "template_semantic_hash": spec.template_semantic_hash,
        "semantic_selection_policy_id": spec.semantic_selection_policy_id,
        "semantic_anchor_policy_id": spec.semantic_anchor_policy_id,
        "semantic_anchor_hash": spec.semantic_anchor_hash,
        "semantic_margin": spec.semantic_margin,
        "semantic_selection_status": spec.semantic_selection_status,
        "business_lift_status": spec.business_lift_status,
        "user_vectorizer_version": spec.user_vectorizer_version,
        "user_vectorizer_semantic_hash": spec.user_vectorizer_semantic_hash,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _audience_status(*, final_user_count: int, min_sample_size: int) -> str:
    if final_user_count <= 0:
        return "no_eligible_audience"
    if final_user_count < min_sample_size:
        return "insufficient_sample"
    return "targetable"
