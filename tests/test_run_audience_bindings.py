from __future__ import annotations

from decimal import Decimal

import pytest

from app.decision.audience_snapshots import (
    AudienceSnapshotContractError,
    AudienceSnapshotRepository,
    AudienceSnapshotTargetAlreadyBoundError,
    RunAudienceTargetBindingWrite,
)


class _Db:
    def __init__(
        self,
        *,
        fetchone_rows=(),
        fetchall_rows=(),
    ) -> None:
        self.fetchone_rows = list(fetchone_rows)
        self.fetchall_rows = list(fetchall_rows)
        self.executed = []

    def fetchone(self, query, params=()):
        self.executed.append((query, params))
        if self.fetchone_rows:
            return self.fetchone_rows.pop(0)
        return None

    def fetchall(self, query, params=()):
        self.executed.append((query, params))
        return list(self.fetchall_rows)

    def execute(self, query, params=()):
        self.executed.append((query, params))


def _target_binding_row():
    return {
        "audience_reservation_state": "reserved",
        "plan_status": "finalized",
        "snapshot_status": "completed",
        "snapshot_kind": "final",
        "source_snapshot_id": "source_a",
        "snapshot_allocation_plan_id": "plan_a",
        "final_user_count": 2,
        "actual_member_count": 2,
        "reservation_count": 2,
        "every_member_reserved": True,
    }


def _stored_binding_row():
    return {
        "promotion_run_id": "run",
        "target_analysis_id": "confirmation_a",
        "allocation_plan_id": "plan_a",
        "final_snapshot_id": "final_a",
    }


def _complete_binding_row():
    return {
        "segment_id": "seg_a",
        "final_snapshot_id": "final_a",
        "allocation_plan_id": "plan_a",
        "target_analysis_id": "confirmation_a",
        "plan_status": "locked",
        "snapshot_status": "completed",
        "vector_version": "hotel_behavior.v2",
        "final_user_count": 2,
        "audience_status": "insufficient_sample",
        "snapshot_kind": "final",
        "source_snapshot_id": "source_a",
        "snapshot_allocation_plan_id": "plan_a",
        "audience_reservation_state": "consumed",
        "actual_member_count": 2,
        "active_reservation_count": 2,
        "every_member_reserved": True,
    }


def test_run_binding_locks_final_snapshot_and_reservation_provenance() -> None:
    db = _Db(
        fetchone_rows=(_target_binding_row(), None, {"revision": 4}, None),
        fetchall_rows=(_complete_binding_row(),),
    )
    repository = AudienceSnapshotRepository(db)

    repository.bind_run_targets(
        promotion_run_id="run",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        bindings=(
            RunAudienceTargetBindingWrite(
                target_analysis_id="confirmation_a",
                segment_id="seg_a",
                allocation_plan_id="plan_a",
                final_snapshot_id="final_a",
            ),
        ),
    )

    sql = "\n".join(query for query, _params in db.executed)
    assert "INSERT INTO promotion_run_target_bindings" in sql
    assert "SET state = 'consumed'" in sql
    assert "revision = %s" in sql
    assert "SET status = 'locked'" in sql
    assert "DISTINCT ON" not in sql
    assert "behavior_fit_score DESC" not in sql


def test_assignment_consumption_is_already_complete_at_run_binding() -> None:
    db = _Db(
        fetchone_rows=(None, None),
        fetchall_rows=(_complete_binding_row(),),
    )
    repository = AudienceSnapshotRepository(db)

    repository.consume_run_members(
        promotion_run_id="run",
        segment_ids=("seg_a",),
    )

    assert not any(
        "SET state = 'consumed'" in query for query, _params in db.executed
    )


def test_run_member_reader_reads_final_binding_without_search_or_winner() -> None:
    db = _Db(
        fetchall_rows=(
            {
                "user_id": "user_1",
                "segment_id": "seg_a",
                "behavior_fit_score": "0.81",
            },
            {
                "user_id": "user_2",
                "segment_id": "seg_a",
                "behavior_fit_score": None,
            },
        )
    )
    repository = AudienceSnapshotRepository(db)

    members = repository.list_run_members(
        promotion_run_id="run",
        segment_ids=("seg_a",),
        after_user_id=None,
        limit=100,
    )

    assert [
        (member.user_id, member.segment_id, member.behavior_fit_score)
        for member in members
    ] == [
        ("user_1", "seg_a", Decimal("0.81")),
        ("user_2", "seg_a", None),
    ]
    query = db.executed[0][0]
    assert "promotion_run_target_bindings" in query
    assert "final_snapshot_id" in query
    assert "row_number()" not in query
    assert "cosine" not in query.lower()


def test_run_binding_allows_one_card_from_a_multi_target_allocation_plan() -> None:
    db = _Db(
        fetchone_rows=(_target_binding_row(), None, {"revision": 4}, None),
        fetchall_rows=(_complete_binding_row(),),
    )
    repository = AudienceSnapshotRepository(db)

    repository.bind_run_targets(
        promotion_run_id="run_a",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        bindings=(
            RunAudienceTargetBindingWrite(
                target_analysis_id="confirmation_a",
                segment_id="seg_a",
                allocation_plan_id="plan_a",
                final_snapshot_id="final_a",
            ),
        ),
    )

    assert any(
        "SET status = 'locked'" in query for query, _params in db.executed
    )
    assert not any(
        "segment_audience_allocation_plan_segments" in query
        for query, _params in db.executed
    )


def test_run_binding_reports_target_already_bound_to_another_run() -> None:
    db = _Db(
        fetchone_rows=(
            _target_binding_row(),
            {
                "promotion_run_id": "run_existing",
                "target_analysis_id": "confirmation_a",
                "allocation_plan_id": "plan_a",
                "final_snapshot_id": "final_a",
            },
        ),
    )
    repository = AudienceSnapshotRepository(db)

    with pytest.raises(AudienceSnapshotTargetAlreadyBoundError) as error:
        repository.bind_run_targets(
            promotion_run_id="run_new",
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
            bindings=(
                RunAudienceTargetBindingWrite(
                    target_analysis_id="confirmation_a",
                    segment_id="seg_a",
                    allocation_plan_id="plan_a",
                    final_snapshot_id="final_a",
                ),
            ),
        )

    assert error.value.code == "segment_audience_target_already_run_bound"
    assert error.value.segment_id == "seg_a"


def test_locked_plan_allows_a_later_unbound_card_to_create_its_own_run() -> None:
    target_row = {
        **_target_binding_row(),
        "source_snapshot_id": "source_b",
        "plan_status": "locked",
    }
    complete_row = {
        **_complete_binding_row(),
        "segment_id": "seg_b",
        "final_snapshot_id": "final_b",
        "source_snapshot_id": "source_b",
        "target_analysis_id": "confirmation_ab",
        "snapshot_allocation_plan_id": "plan_a",
    }
    db = _Db(
        fetchone_rows=(target_row, None, {"revision": 5}, None),
        fetchall_rows=(complete_row,),
    )
    repository = AudienceSnapshotRepository(db)

    repository.bind_run_targets(
        promotion_run_id="run_b",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        bindings=(
            RunAudienceTargetBindingWrite(
                target_analysis_id="confirmation_ab",
                segment_id="seg_b",
                allocation_plan_id="plan_a",
                final_snapshot_id="final_b",
            ),
        ),
    )

    assert any(
        "INSERT INTO promotion_run_target_bindings" in query
        for query, _params in db.executed
    )
