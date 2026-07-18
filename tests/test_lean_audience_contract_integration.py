from __future__ import annotations

import hashlib
import json
import uuid

import psycopg
from psycopg.types.json import Jsonb

from app.analysis.repositories import PsycopgPostgresExecutor
from app.audience_allocation import PostgresAudienceAllocationRepository
from app.audience_exclusions import PromotionAudienceExclusionContext
from app.decision.audience_snapshots import (
    AudienceSnapshotRepository,
    RunAudienceTargetBindingWrite,
)


class _InitialExclusionReader:
    def load_active_exclusion_context(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionAudienceExclusionContext:
        return PromotionAudienceExclusionContext(
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            revision=0,
            excluded_user_count=0,
            projection_revision=0,
        )


def test_lean_contract_allocation_and_run_binding_lifecycle(
    loopad_test_postgres_dsn: str,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    project_id = f"project_lean_{suffix}"
    campaign_id = f"campaign_lean_{suffix}"
    promotion_id = f"promotion_lean_{suffix}"
    source_analysis_id = f"analysis_source_{suffix}"
    target_analysis_id = f"analysis_target_{suffix}"
    generation_id = f"generation_lean_{suffix}"
    promotion_run_id = f"run_lean_{suffix}"
    segment_id = f"segment_lean_{suffix}"
    segment_vector_id = f"segment_vector_lean_{suffix}"
    vector_generation_id = f"vector_generation_lean_{suffix}"
    source_snapshot_id = f"snapshot_source_{suffix}"
    member_ids = (f"user_lean_{suffix}_1", f"user_lean_{suffix}_2")
    zero_vector = [0.0] * 64

    connection = psycopg.connect(loopad_test_postgres_dsn)
    database = PsycopgPostgresExecutor(connection)
    try:
        database.execute(
            """
            INSERT INTO projects (
                project_id, project_name, domain, write_key
            ) VALUES (%s, %s, %s, %s)
            """,
            (project_id, "Lean contract test", "hotel", f"write_{suffix}"),
        )
        database.execute(
            """
            INSERT INTO campaigns (campaign_id, project_id, name)
            VALUES (%s, %s, %s)
            """,
            (campaign_id, project_id, "Lean campaign"),
        )
        database.execute(
            """
            INSERT INTO promotions (
                promotion_id, project_id, campaign_id, channel,
                goal_metric, goal_target_value, goal_basis, min_sample_size
            ) VALUES (%s, %s, %s, 'onsite_banner',
                      'booking_conversion_rate', 0.1, 'all_segments', 1)
            """,
            (promotion_id, project_id, campaign_id),
        )
        for analysis_id in (source_analysis_id, target_analysis_id):
            database.execute(
                """
                INSERT INTO promotion_analyses (
                    analysis_id, project_id, campaign_id, promotion_id, status
                ) VALUES (%s, %s, %s, %s, 'completed')
                """,
                (analysis_id, project_id, campaign_id, promotion_id),
            )
        database.execute(
            """
            INSERT INTO segment_definitions (
                segment_id, project_id, campaign_id, promotion_id,
                segment_name, source, rule_json
            ) VALUES (%s, %s, %s, %s, %s, 'ai_suggested', %s)
            """,
            (
                segment_id,
                project_id,
                campaign_id,
                promotion_id,
                "Lean V2 segment",
                {"audience_resolution_contract": "segment_audience.v1"},
            ),
        )
        database.execute(
            """
            INSERT INTO user_behavior_vector_search_generations (
                vector_generation_id, project_id, vector_version,
                manifest_hash, window_start, window_end,
                source_revision_cutoff, expected_user_count,
                synced_user_count, status, is_active, activated_at
            ) VALUES (
                %s, %s, 'v2', 'manifest-lean-v2',
                now() - interval '30 days', now(), now(), 2, 2,
                'activated', true, now()
            )
            """,
            (vector_generation_id, project_id),
        )
        database.execute(
            """
            INSERT INTO segment_vectors (
                segment_vector_id, project_id, segment_id, promotion_id,
                analysis_id, vector_values, embedding, vector_version, source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector, 'v2',
                      'behavior_query')
            """,
            (
                segment_vector_id,
                project_id,
                segment_id,
                promotion_id,
                source_analysis_id,
                Jsonb(zero_vector),
                json.dumps(zero_vector),
            ),
        )
        database.execute(
            """
            INSERT INTO segment_audience_snapshots (
                snapshot_id, analysis_id, project_id, campaign_id,
                promotion_id, segment_id, segment_vector_id,
                vector_generation_id, schema_version, vector_version,
                manifest_hash, audience_resolution_contract,
                segment_audience_spec_hash, query_vector_hash,
                query_compiler_version, query_compiler_hash, matcher_version,
                search_policy_version, calibration_version, calibration_hash,
                score_threshold, source_cutoff, window_start, window_end,
                eligible_user_count, behavior_match_count, final_user_count,
                min_sample_size, audience_status, selection_method,
                estimated_recall, recall_lower_bound, recall_target,
                input_fingerprint, meets_min_sample_size, status,
                metadata_json, snapshot_kind
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                'hotel_behavior.v2', 'v2', 'manifest-lean-v2',
                'segment_audience.v1', 'spec-hash', 'query-hash',
                'compiler-v1', 'compiler-hash', 'matcher-v1', 'search-v1',
                'selection-v1', 'selection-hash', 0.65,
                now(), now() - interval '30 days', now(),
                2, 2, 2, 1, 'targetable', 'exact',
                1.0, 1.0, 0.95, 'source-fingerprint', true, 'completed',
                %s, 'source'
            )
            """,
            (
                source_snapshot_id,
                source_analysis_id,
                project_id,
                campaign_id,
                promotion_id,
                segment_id,
                segment_vector_id,
                vector_generation_id,
                {
                    "candidate_type": "promotion_responsive",
                    "semantic_margin": "0.10",
                },
            ),
        )
        for rank, user_id in enumerate(member_ids, start=1):
            database.execute(
                """
                INSERT INTO segment_audience_members (
                    snapshot_id, user_id, behavior_fit_score,
                    retrieval_source, retrieval_rank
                ) VALUES (%s, %s, 0.8, 'exact', %s)
                """,
                (source_snapshot_id, user_id, rank),
            )

        allocation_repository = PostgresAudienceAllocationRepository(
            postgres=database,
            exclusion_reader=_InitialExclusionReader(),
        )
        allocation = allocation_repository.confirm_selection(
            confirmation_analysis_id=target_analysis_id,
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            segment_ids=[segment_id],
            min_sample_size=1,
            source_analysis_id=source_analysis_id,
        )
        final = allocation.allocations[segment_id]

        # Exercise every deferred lifecycle validator on the real contract.
        database.execute("SET CONSTRAINTS ALL IMMEDIATE")
        stored = database.fetchone(
            """
            SELECT plan.status, target.audience_reservation_state,
                   snapshot.snapshot_kind, snapshot.source_snapshot_id,
                   snapshot.final_user_count,
                   count(excluded.user_id) AS reserved_count
            FROM segment_audience_allocation_plans AS plan
            JOIN promotion_target_segments AS target
              ON target.allocation_plan_id = plan.allocation_plan_id
            JOIN segment_audience_snapshots AS snapshot
              ON snapshot.snapshot_id = target.audience_snapshot_id
            JOIN promotion_audience_exclusion_members AS excluded
              ON excluded.allocation_plan_id = plan.allocation_plan_id
             AND excluded.segment_id = target.segment_id
             AND excluded.state = 'reserved'
            WHERE plan.allocation_plan_id = %s::uuid
            GROUP BY plan.status, target.audience_reservation_state,
                     snapshot.snapshot_kind, snapshot.source_snapshot_id,
                     snapshot.final_user_count
            """,
            (allocation.allocation_plan_id,),
        )
        assert stored == {
            "status": "finalized",
            "audience_reservation_state": "reserved",
            "snapshot_kind": "final",
            "source_snapshot_id": source_snapshot_id,
            "final_user_count": 2,
            "reserved_count": 2,
        }

        database.execute("SET CONSTRAINTS ALL DEFERRED")
        database.execute(
            """
            INSERT INTO generation_runs (
                generation_id, analysis_id, project_id, campaign_id,
                promotion_id, status, started_at, finished_at
            ) VALUES (%s, %s, %s, %s, %s, 'completed', now(), now())
            """,
            (
                generation_id,
                target_analysis_id,
                project_id,
                campaign_id,
                promotion_id,
            ),
        )
        canonical_scope = json.dumps([segment_id], separators=(",", ":"))
        scope_fingerprint = hashlib.sha256(canonical_scope.encode()).hexdigest()
        database.execute(
            """
            INSERT INTO promotion_runs (
                promotion_run_id, project_id, campaign_id, promotion_id,
                analysis_id, generation_id, segment_scope_json,
                segment_scope_fingerprint
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                promotion_run_id,
                project_id,
                campaign_id,
                promotion_id,
                target_analysis_id,
                generation_id,
                Jsonb([segment_id]),
                scope_fingerprint,
            ),
        )
        snapshot_repository = AudienceSnapshotRepository(database)
        snapshot_repository.bind_run_targets(
            promotion_run_id=promotion_run_id,
            project_id=project_id,
            campaign_id=campaign_id,
            promotion_id=promotion_id,
            bindings=[
                RunAudienceTargetBindingWrite(
                    target_analysis_id=target_analysis_id,
                    segment_id=segment_id,
                    allocation_plan_id=allocation.allocation_plan_id,
                    final_snapshot_id=final.final_snapshot_id,
                )
            ],
        )
        database.execute("SET CONSTRAINTS ALL IMMEDIATE")

        bound = database.fetchone(
            """
            SELECT plan.status, target.audience_reservation_state,
                   count(excluded.user_id) AS consumed_count
            FROM promotion_run_target_bindings AS binding
            JOIN segment_audience_allocation_plans AS plan
              ON plan.allocation_plan_id = binding.allocation_plan_id
            JOIN promotion_target_segments AS target
              ON target.analysis_id = binding.target_analysis_id
             AND target.segment_id = binding.segment_id
            JOIN promotion_audience_exclusion_members AS excluded
              ON excluded.allocation_plan_id = binding.allocation_plan_id
             AND excluded.segment_id = binding.segment_id
             AND excluded.state = 'consumed'
            WHERE binding.promotion_run_id = %s
            GROUP BY plan.status, target.audience_reservation_state
            """,
            (promotion_run_id,),
        )
        assert bound == {
            "status": "locked",
            "audience_reservation_state": "consumed",
            "consumed_count": 2,
        }
    finally:
        connection.rollback()
        connection.close()
