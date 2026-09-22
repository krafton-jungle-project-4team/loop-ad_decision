"""Synthetic V2 input; production allocation builds final snapshots/reservations."""
from __future__ import annotations

import json
from psycopg import sql
from psycopg.types.json import Jsonb
from app.analysis.repositories import PsycopgPostgresExecutor
from app.analysis.segment_audience_templates import RegisteredSegmentAudienceBinder
from app.analysis.semantic_selection import compile_registered_segment_audience, semantic_query_vector_hash
from app.audience_contract import contract_score_threshold
from app.audience_allocation import PostgresAudienceAllocationRepository
from app.audience_exclusions import PromotionAudienceExclusionContext

PROJECT = 'rcg_synthetic_project'
CAMPAIGN = 'rcg_synthetic_campaign'
PROMOTION = 'rcg_synthetic_promotion'
SOURCE = 'rcg_synthetic_source'
ANALYSIS = 'rcg_synthetic_analysis'
GENERATION = 'rcg_synthetic_generation'
SEGMENTS = ['rcg_segment_a', 'rcg_segment_b', 'rcg_segment_c']
REQUEST = dict(analysis_id=ANALYSIS, generation_id=GENERATION, segment_ids=SEGMENTS[:2], loop_count=1)


class InitialExclusions:
    """Only used to prepare a brand new synthetic promotion before any run."""
    def load_active_exclusion_context(self, *, project_id, campaign_id, promotion_id):
        return PromotionAudienceExclusionContext(project_id=project_id, campaign_id=campaign_id,
            promotion_id=promotion_id, revision=0, excluded_user_count=0, projection_revision=0)


def insert(connection, table, **row):
    values = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in row.values()]
    connection.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
        sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, row)),
        sql.SQL(',').join(sql.Placeholder() for _ in row)), values)


def seed(connection):
    identity = dict(project_id=PROJECT, campaign_id=CAMPAIGN, promotion_id=PROMOTION)
    insert(connection, 'projects', project_id=PROJECT, project_name='Synthetic Gate',
           domain='hotel', write_key='synthetic-not-a-credential')
    insert(connection, 'campaigns', campaign_id=CAMPAIGN, project_id=PROJECT, name='Synthetic Gate')
    insert(connection, 'promotions', **identity, channel='onsite_banner',
           goal_metric='booking_conversion_rate', goal_target_value=0.1, goal_basis='all_segments', min_sample_size=1)
    for analysis in (SOURCE, ANALYSIS):
        insert(connection, 'promotion_analyses', **identity, analysis_id=analysis, status='completed',
               created_at='2026-01-31T00:00:00Z' if analysis == SOURCE else '2026-02-01T00:00:00Z')
    rule = dict(audience_resolution_contract='segment_audience.v1',
                segment_audience_spec=dict(RegisteredSegmentAudienceBinder().bind(candidate_type='intent_matched')))
    compiled = compile_registered_segment_audience(segment_id=SEGMENTS[0], rule_json=rule)
    insert(connection, 'user_behavior_vector_search_generations',
           vector_generation_id='rcg_vector_generation', project_id=PROJECT,
           vector_version=compiled.vector_version, manifest_hash=compiled.manifest_hash,
           window_start='2026-01-01T00:00:00Z', window_end='2026-01-31T00:00:00Z',
           source_revision_cutoff='2026-01-31T00:00:00Z', expected_user_count=6, synced_user_count=6,
           status='activated', is_active=True, activated_at='2026-01-31T00:00:00Z')
    for segment in SEGMENTS:
        compiled = compile_registered_segment_audience(segment_id=segment, rule_json=rule)
        insert(connection, 'segment_definitions', **identity, segment_id=segment,
               segment_name='Synthetic ' + segment, source='ai_suggested', rule_json=rule)
        insert(connection, 'segment_vectors', project_id=PROJECT, promotion_id=PROMOTION,
               segment_id=segment, segment_vector_id=segment+'_vector', analysis_id=SOURCE,
               vector_values=list(compiled.query_vector), embedding=json.dumps(list(compiled.query_vector)),
               vector_version=compiled.vector_version, source='behavior_query')
        metadata = {key: getattr(compiled, key) for key in (
            'template_id', 'template_version', 'template_semantic_hash', 'semantic_selection_policy_id',
            'semantic_anchor_policy_id', 'semantic_anchor_hash', 'semantic_margin', 'semantic_selection_status',
            'business_lift_status', 'user_vectorizer_version', 'user_vectorizer_semantic_hash')}
        metadata.update(candidate_type='intent_matched', hard_predicate_keys=list(compiled.hard_predicate_keys),
                        predicate_parameters={k:list(v) for k,v in compiled.predicate_parameters.items()})
        insert(connection, 'segment_audience_snapshots', **identity, snapshot_id=segment+'_source',
               analysis_id=SOURCE, segment_id=segment, segment_vector_id=segment+'_vector',
               vector_generation_id='rcg_vector_generation', schema_version=compiled.schema_version,
               vector_version=compiled.vector_version, manifest_hash=compiled.manifest_hash,
               audience_resolution_contract=compiled.audience_resolution_contract,
               segment_audience_spec_hash=compiled.segment_audience_spec_hash,
               query_vector_hash=semantic_query_vector_hash(compiled), query_compiler_version=compiled.query_compiler_version,
               query_compiler_hash=compiled.query_compiler_hash, matcher_version='synthetic-exact-v1',
               search_policy_version='synthetic-exact-v1', calibration_version=compiled.calibration_version,
               calibration_hash=compiled.calibration_hash, score_threshold=contract_score_threshold(compiled.score_threshold),
               source_cutoff='2026-01-31T00:00:00Z', window_start='2026-01-01T00:00:00Z',
               window_end='2026-01-31T00:00:00Z', eligible_user_count=2, behavior_match_count=2,
               final_user_count=2, min_sample_size=1, audience_status='targetable', selection_method='exact',
               estimated_recall=1.0, recall_lower_bound=1.0, recall_target=0.95,
               input_fingerprint=segment+'_synthetic', meets_min_sample_size=True, status='completed',
               metadata_json=metadata, snapshot_kind='source')
        for rank in (1, 2):
            insert(connection, 'segment_audience_members', snapshot_id=segment+'_source',
                   user_id=segment+'_synthetic_user_'+str(rank), behavior_fit_score=0.8,
                   retrieval_source='exact', retrieval_rank=rank)
    PostgresAudienceAllocationRepository(postgres=PsycopgPostgresExecutor(connection),
        exclusion_reader=InitialExclusions()).confirm_selection(confirmation_analysis_id=ANALYSIS,
            **identity, segment_ids=SEGMENTS, min_sample_size=1, source_analysis_id=SOURCE)
    connection.execute("UPDATE promotion_target_segments SET status = 'approved'")
    insert(connection, 'generation_runs', **identity, generation_id=GENERATION, analysis_id=ANALYSIS,
           status='completed', started_at='2026-02-01T00:00:00Z', finished_at='2026-02-01T00:01:00Z',
           input_json={'target_segment_ids': SEGMENTS})
    for segment in SEGMENTS:
        insert(connection, 'content_candidates', **identity, generation_id=GENERATION, analysis_id=ANALYSIS,
               segment_id=segment, content_id=segment+'_content', content_option_id=segment+'_option',
               channel='onsite_banner', status='approved')
    connection.commit()
