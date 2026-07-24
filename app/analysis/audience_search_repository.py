from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Sequence

from app.analysis.audience_search import SearchCandidate
from app.audience_contract import CUSTOM_STRUCTURED_PROPERTY_KEYS
from app.analysis.behavior_manifest import clickhouse_canonical_destination_sql
from app.analysis.repositories import ClickHouseClient, PostgresExecutor
from app.audience_exclusions import (
    CLICKHOUSE_EXCLUSION_RELATION,
    POSTGRES_EXCLUSION_RELATION,
    PromotionAudienceExclusionContext,
    PromotionAudienceExclusionReader,
    PromotionAudienceExclusionRepository,
)


VECTOR_DIM = 64
PREDICATE_CHUNK_SIZE = 10_000
MATERIALIZED_RELATIONS = {
    "audience_exact_candidates",
    "audience_exact_members",
    "audience_ann_retrieval",
    "audience_ann_members",
}


@dataclass(frozen=True, slots=True)
class AudienceSearchContext:
    vector_generation_id: str
    manifest_hash: str
    source_cutoff: datetime
    source_revision_cutoff: datetime
    window_start: datetime
    corpus_user_count: int
    exclusion_context: PromotionAudienceExclusionContext | None = None


@dataclass(frozen=True, slots=True)
class HardMatchAggregateRequest:
    segment_id: str
    hard_predicate_keys: tuple[str, ...]
    predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]]


class PgClickHouseAudienceVectorSearchRepository:
    """pgvector retrieval with ClickHouse raw-event truth filtering."""

    def __init__(
        self,
        *,
        postgres: PostgresExecutor,
        clickhouse: ClickHouseClient,
        exclusion_repository: PromotionAudienceExclusionReader | None = None,
    ) -> None:
        self._postgres = postgres
        self._clickhouse = clickhouse
        self._exclusion_repository = (
            exclusion_repository
            or PromotionAudienceExclusionRepository(
                postgres=postgres,
                clickhouse=clickhouse,
            )
        )
        self._exclusions_by_generation: dict[
            str,
            PromotionAudienceExclusionContext,
        ] = {}

    def get_context(
        self,
        *,
        project_id: str,
        vector_version: str,
        campaign_id: str | None = None,
        promotion_id: str | None = None,
    ) -> AudienceSearchContext:
        exclusion_context = None
        exclusion_join = ""
        exclusion_params: tuple[Any, ...] = ()
        if campaign_id is not None and promotion_id is not None:
            self._postgres.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtext('promotion-audience-allocation-v1'),
                    hashtext(%s || ':' || %s)
                )
                """,
                (project_id, promotion_id),
            )
            exclusion_context = (
                self._exclusion_repository.load_active_exclusion_context(
                    project_id=project_id,
                    campaign_id=campaign_id,
                    promotion_id=promotion_id,
                )
            )
            exclusion_join = f"""
              AND NOT EXISTS (
                  SELECT 1
                  FROM {POSTGRES_EXCLUSION_RELATION} AS excluded
                  JOIN promotion_target_segments AS target
                    ON target.analysis_id = excluded.target_analysis_id
                   AND target.segment_id = excluded.segment_id
                   AND target.allocation_plan_id = excluded.allocation_plan_id
                   AND target.audience_snapshot_id = excluded.final_snapshot_id
                  WHERE excluded.project_id = generation.project_id
                    AND excluded.promotion_id = %s
                    AND excluded.user_id = search.user_id
                    AND excluded.state IN ('reserved', 'consumed')
                    AND target.status <> 'stopped'
              )
            """
            exclusion_params = (promotion_id,)
        row = self._postgres.fetchone(
            f"""
            SELECT
                generation.vector_generation_id,
                generation.manifest_hash,
                generation.window_end AS source_cutoff,
                generation.source_revision_cutoff,
                generation.window_start,
                count(search.user_id) AS corpus_user_count
            FROM user_behavior_vector_search_generations AS generation
            JOIN user_behavior_vector_search AS search
              ON search.vector_generation_id = generation.vector_generation_id
            WHERE generation.project_id = %s
              AND generation.vector_version = %s
              AND generation.status = 'activated'
              AND generation.is_active = true
              {exclusion_join}
            GROUP BY generation.vector_generation_id, generation.manifest_hash,
                     generation.window_end, generation.source_revision_cutoff,
                     generation.window_start
            """,
            (project_id, vector_version, *exclusion_params),
        )
        if row is None:
            raise RuntimeError("completed user vector search sync is required")
        context = AudienceSearchContext(
            vector_generation_id=str(row["vector_generation_id"]),
            manifest_hash=str(row["manifest_hash"]),
            source_cutoff=row["source_cutoff"],
            source_revision_cutoff=row["source_revision_cutoff"],
            window_start=row["window_start"],
            corpus_user_count=int(row["corpus_user_count"]),
            exclusion_context=exclusion_context,
        )
        if exclusion_context is not None:
            self._exclusions_by_generation[context.vector_generation_id] = (
                exclusion_context
            )
        return context

    def count_hard_matches(
        self,
        *,
        project_id: str,
        vector_version: str,
        source_revision_cutoff: datetime,
        window_start: datetime,
        vector_window_start: datetime,
        window_end: datetime,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        exclusion_context: PromotionAudienceExclusionContext | None = None,
    ) -> int:
        exclusion_context = exclusion_context or self._context_for_window(
            project_id=project_id,
            vector_version=vector_version,
            window_end=window_end,
        )
        result = self._clickhouse.query(
            "SELECT count() AS matching_user_count FROM ("
            + _hard_predicate_query(
                hard_predicate_keys,
                filter_user_ids=False,
                restrict_to_vector_population=True,
                exclude_promotion_users=exclusion_context is not None,
                predicate_parameters=predicate_parameters,
            )
            + ")",
            parameters={
                "project_id": project_id,
                "vector_version": vector_version,
                "source_revision_cutoff": _datetime_string(
                    source_revision_cutoff
                ),
                "raw_event_received_cutoff": _datetime_string(
                    source_revision_cutoff
                ),
                "window_start": _datetime_string(window_start),
                "vector_window_start": _datetime_string(vector_window_start),
                "window_end": _datetime_string(window_end),
                "destinations": list(predicate_parameters.get("destinations", ())),
                "season_months": list(predicate_parameters.get("season_months", ())),
                "benefit_keys": list(predicate_parameters.get("benefit_keys", ())),
                **_structured_query_parameters(predicate_parameters),
                **_clickhouse_exclusion_parameters(exclusion_context),
            },
        )
        rows = (
            list(result.named_results())
            if hasattr(result, "named_results")
            else list(result.result_rows)
        )
        if not rows:
            return 0
        row = rows[0]
        return int(
            row["matching_user_count"] if isinstance(row, Mapping) else row[0]
        )

    def count_hard_matches_batch(
        self,
        *,
        project_id: str,
        vector_version: str,
        source_revision_cutoff: datetime,
        window_start: datetime,
        window_end: datetime,
        requests: Sequence[HardMatchAggregateRequest],
        exclusion_context: PromotionAudienceExclusionContext | None = None,
    ) -> Mapping[str, int]:
        if not requests:
            return {}
        exclusion_context = exclusion_context or self._context_for_window(
            project_id=project_id,
            vector_version=vector_version,
            window_end=window_end,
        )
        query, predicate_parameters = _hard_predicate_batch_query(
            requests,
            exclude_promotion_users=exclusion_context is not None,
        )
        result = self._clickhouse.query(
            query,
            parameters={
                "project_id": project_id,
                "vector_version": vector_version,
                "source_revision_cutoff": _datetime_string(
                    source_revision_cutoff
                ),
                "raw_event_received_cutoff": _datetime_string(
                    source_revision_cutoff
                ),
                "window_start": _datetime_string(window_start),
                "window_end": _datetime_string(window_end),
                **predicate_parameters,
                **_clickhouse_exclusion_parameters(exclusion_context),
            },
        )
        rows = (
            list(result.named_results())
            if hasattr(result, "named_results")
            else list(result.result_rows)
        )
        if not rows:
            return {request.segment_id: 0 for request in requests}
        row = rows[0]
        return {
            request.segment_id: int(
                row[f"match_{index}"]
                if isinstance(row, Mapping)
                else row[index]
            )
            for index, request in enumerate(requests)
        }

    def estimate_score_pass_rate(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_revision_cutoff: datetime,
        source_cutoff: datetime,
        window_start: datetime,
        vector_window_start: datetime,
        query_vector: Sequence[float],
        score_threshold: float,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        sample_size: int = 10_000,
        exclusion_context: PromotionAudienceExclusionContext | None = None,
    ) -> float:
        exclusion_context = exclusion_context or self._exclusions_by_generation.get(
            vector_generation_id
        )
        result = self._clickhouse.query(
            _hard_predicate_query(
                hard_predicate_keys,
                filter_user_ids=False,
                restrict_to_vector_population=True,
                deterministic_sample=True,
                exclude_promotion_users=exclusion_context is not None,
                predicate_parameters=predicate_parameters,
            )
            + " LIMIT {sample_size:UInt32}",
            parameters={
                "project_id": project_id,
                "vector_version": vector_version,
                "source_revision_cutoff": _datetime_string(
                    source_revision_cutoff
                ),
                "raw_event_received_cutoff": _datetime_string(
                    source_revision_cutoff
                ),
                "window_start": _datetime_string(window_start),
                "vector_window_start": _datetime_string(vector_window_start),
                "window_end": _datetime_string(source_cutoff),
                "destinations": list(predicate_parameters.get("destinations", ())),
                "season_months": list(predicate_parameters.get("season_months", ())),
                "benefit_keys": list(predicate_parameters.get("benefit_keys", ())),
                "sample_seed": (
                    f"{project_id}:{vector_generation_id}:{source_cutoff.isoformat()}"
                ),
                "sample_size": sample_size,
                **_structured_query_parameters(predicate_parameters),
                **_clickhouse_exclusion_parameters(exclusion_context),
            },
        )
        rows = (
            list(result.named_results())
            if hasattr(result, "named_results")
            else list(result.result_rows)
        )
        hard_match_user_ids = [
            str(row["user_id"] if isinstance(row, Mapping) else row[0])
            for row in rows
        ]
        if not hard_match_user_ids:
            return 0.0
        self._replace_temp_user_ids(
            table_name="audience_hard_match_sample",
            user_ids=hard_match_user_ids,
        )
        row = self._postgres.fetchone(
            """
            SELECT
                count(*) AS sampled_count,
                count(*) FILTER (
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                ) AS passed_count
            FROM (
                SELECT search.embedding
                FROM user_behavior_vector_search AS search
                JOIN audience_hard_match_sample AS sample USING (user_id)
                WHERE search.project_id = %s
                  AND search.vector_version = %s
                  AND search.window_end = %s
                  AND search.vector_generation_id = %s
                ORDER BY search.user_id
            ) AS deterministic_sample
            """,
            (
                _vector_literal(query_vector),
                score_threshold,
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
            ),
        )
        if row is None or int(row["sampled_count"]) == 0:
            return 0.0
        return int(row["passed_count"]) / int(row["sampled_count"])

    def materialize_exact_members(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        query_vector: Sequence[float],
        score_threshold: float,
        apply_score_threshold: bool = True,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
    ) -> int:
        candidate_relation = "audience_exact_candidates"
        exclusion_sql, exclusion_params = self._postgres_exclusion_clause(
            vector_generation_id=vector_generation_id,
            user_expression="search.user_id",
            project_expression="search.project_id",
        )
        self._drop_temp_relation(candidate_relation)
        score_filter_sql = (
            "WHERE scored.behavior_fit_score >= %s"
            if apply_score_threshold
            else ""
        )
        score_filter_params = (score_threshold,) if apply_score_threshold else ()
        self._postgres.execute(
            f"""
            CREATE TEMP TABLE {candidate_relation}
            ON COMMIT DROP
            AS
            SELECT
                scored.user_id,
                scored.behavior_fit_score,
                row_number() OVER (
                    ORDER BY scored.behavior_fit_score DESC, scored.user_id ASC
                )::integer AS retrieval_rank
            FROM (
                SELECT
                    search.user_id,
                    COALESCE(
                        1 - (search.embedding <=> %s::vector),
                        -1.0
                    ) AS behavior_fit_score
                FROM user_behavior_vector_search AS search
                WHERE search.project_id = %s
                  AND search.vector_version = %s
                  AND search.vector_dim = 64
                  AND search.window_end = %s
                  AND search.vector_generation_id = %s
                  {exclusion_sql}
            ) AS scored
            {score_filter_sql}
            """,
            (
                _vector_literal(query_vector),
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
                *exclusion_params,
                *score_filter_params,
            ),
        )
        self._index_temp_candidates(candidate_relation)
        return self._materialize_hard_filtered_relation(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=vector_version,
            source_cutoff=source_cutoff,
            hard_predicate_keys=hard_predicate_keys,
            predicate_parameters=predicate_parameters,
            candidate_relation=candidate_relation,
            member_relation="audience_exact_members",
            score_threshold=None,
        )

    def materialize_ann_candidates(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        query_vector: Sequence[float],
        limit: int,
    ) -> int:
        relation = "audience_ann_retrieval"
        exclusion_sql, exclusion_params = self._postgres_exclusion_clause(
            vector_generation_id=vector_generation_id,
            user_expression="search.user_id",
            project_expression="search.project_id",
        )
        self._postgres.execute("SET LOCAL hnsw.ef_search = 100")
        self._postgres.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        self._postgres.execute("SET LOCAL hnsw.max_scan_tuples = 20000")
        self._drop_temp_relation(relation)
        self._postgres.execute(
            f"""
            CREATE TEMP TABLE {relation}
            ON COMMIT DROP
            AS
            SELECT
                retrieved.user_id,
                retrieved.behavior_fit_score,
                row_number() OVER (
                    ORDER BY retrieved.behavior_fit_score DESC,
                             retrieved.user_id ASC
                )::integer AS retrieval_rank
            FROM (
                SELECT
                    search.user_id,
                    1 - (search.embedding <=> %s::vector)
                        AS behavior_fit_score
                FROM user_behavior_vector_search AS search
                WHERE search.project_id = %s
                  AND search.vector_version = %s
                  AND search.vector_dim = 64
                  AND search.window_end = %s
                  AND search.vector_generation_id = %s
                  {exclusion_sql}
                ORDER BY search.embedding <=> %s::vector
                LIMIT %s
            ) AS retrieved
            """,
            (
                _vector_literal(query_vector),
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
                *exclusion_params,
                _vector_literal(query_vector),
                limit,
            ),
        )
        self._index_temp_candidates(relation)
        return self._count_temp_relation(relation)

    def materialize_ann_members(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        score_threshold: float,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
    ) -> int:
        return self._materialize_hard_filtered_relation(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=vector_version,
            source_cutoff=source_cutoff,
            hard_predicate_keys=hard_predicate_keys,
            predicate_parameters=predicate_parameters,
            candidate_relation="audience_ann_retrieval",
            member_relation="audience_ann_members",
            score_threshold=score_threshold,
        )

    def audit_materialized_nonretrieved(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        query_vector: Sequence[float],
        score_threshold: float,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        sample_size: int,
    ) -> tuple[int, int]:
        exclusion_sql, exclusion_params = self._postgres_exclusion_clause(
            vector_generation_id=vector_generation_id,
            user_expression="search.user_id",
            project_expression="search.project_id",
        )
        rows = self._postgres.fetchall(
            f"""
            SELECT
                search.user_id,
                1 - (search.embedding <=> %s::vector) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              {exclusion_sql}
              AND NOT EXISTS (
                  SELECT 1
                  FROM audience_ann_retrieval AS retrieved
                  WHERE retrieved.user_id = search.user_id
              )
            ORDER BY md5(search.user_id || %s) ASC, search.user_id ASC
            LIMIT %s
            """,
            (
                _vector_literal(query_vector),
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
                *exclusion_params,
                f"{project_id}:{vector_generation_id}:{source_cutoff}",
                sample_size,
            ),
        )
        score_passed = [
            row for row in rows
            if float(row["behavior_fit_score"]) >= score_threshold
        ]
        missed = self._filter_hard_predicates(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=vector_version,
            source_cutoff=source_cutoff,
            hard_predicate_keys=hard_predicate_keys,
            predicate_parameters=predicate_parameters,
            candidates=_rows_to_candidates(score_passed),
        )
        return len(rows), len(missed)

    def compare_materialized_members(
        self,
        *,
        authoritative_relation: str,
        shadow_relation: str,
    ) -> tuple[int, int]:
        _require_materialized_relation(authoritative_relation)
        _require_materialized_relation(shadow_relation)
        row = self._postgres.fetchone(
            f"""
            SELECT
                count(*) FILTER (WHERE shadow.user_id IS NOT NULL)
                    AS retrieved_positive_count,
                count(*) FILTER (WHERE shadow.user_id IS NULL)
                    AS missed_positive_count
            FROM {authoritative_relation} AS authoritative
            LEFT JOIN {shadow_relation} AS shadow USING (user_id)
            """
        )
        if row is None:
            return 0, 0
        return (
            int(row["retrieved_positive_count"]),
            int(row["missed_positive_count"]),
        )

    def exact_search(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        query_vector: Sequence[float],
        score_threshold: float,
        apply_score_threshold: bool = True,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
    ) -> list[SearchCandidate]:
        exclusion_sql, exclusion_params = self._postgres_exclusion_clause(
            vector_generation_id=vector_generation_id,
            user_expression="search.user_id",
            project_expression="search.project_id",
        )
        score_filter_sql = (
            "AND COALESCE(1 - (search.embedding <=> %s::vector), -1.0) >= %s"
            if apply_score_threshold
            else ""
        )
        score_filter_params = (
            (_vector_literal(query_vector), score_threshold)
            if apply_score_threshold
            else ()
        )
        rows = self._postgres.fetchall(
            f"""
            SELECT
                search.user_id,
                COALESCE(
                    1 - (search.embedding <=> %s::vector),
                    -1.0
                ) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              {score_filter_sql}
              {exclusion_sql}
            ORDER BY behavior_fit_score DESC, search.user_id ASC
            """,
            (
                _vector_literal(query_vector),
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
                *score_filter_params,
                *exclusion_params,
            ),
        )
        candidates = _rows_to_candidates(rows)
        return self._filter_hard_predicates(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=vector_version,
            source_cutoff=source_cutoff,
            hard_predicate_keys=hard_predicate_keys,
            predicate_parameters=predicate_parameters,
            candidates=candidates,
        )

    def ann_search(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[SearchCandidate]:
        exclusion_sql, exclusion_params = self._postgres_exclusion_clause(
            vector_generation_id=vector_generation_id,
            user_expression="search.user_id",
            project_expression="search.project_id",
        )
        self._postgres.execute("SET LOCAL hnsw.ef_search = 100")
        self._postgres.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        self._postgres.execute("SET LOCAL hnsw.max_scan_tuples = 20000")
        rows = self._postgres.fetchall(
            f"""
            SELECT
                search.user_id,
                1 - (search.embedding <=> %s::vector) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              {exclusion_sql}
            ORDER BY search.embedding <=> %s::vector
            LIMIT %s
            """,
            (
                _vector_literal(query_vector),
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
                *exclusion_params,
                _vector_literal(query_vector),
                limit,
            ),
        )
        return _rows_to_candidates(rows)

    def exact_filter_candidates(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        query_vector: Sequence[float],
        score_threshold: float,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        user_ids: Sequence[str],
    ) -> list[SearchCandidate]:
        if not user_ids:
            return []
        exclusion_sql, exclusion_params = self._postgres_exclusion_clause(
            vector_generation_id=vector_generation_id,
            user_expression="search.user_id",
            project_expression="search.project_id",
        )
        self._replace_temp_user_ids(
            table_name="audience_ann_candidates",
            user_ids=user_ids,
        )
        rows = self._postgres.fetchall(
            f"""
            SELECT
                user_id,
                1 - (embedding <=> %s::vector) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            JOIN audience_ann_candidates AS candidate USING (user_id)
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              AND 1 - (embedding <=> %s::vector) >= %s
              {exclusion_sql}
            ORDER BY behavior_fit_score DESC, search.user_id ASC
            """,
            (
                _vector_literal(query_vector),
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
                _vector_literal(query_vector),
                score_threshold,
                *exclusion_params,
            ),
        )
        return self._filter_hard_predicates(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=vector_version,
            source_cutoff=source_cutoff,
            hard_predicate_keys=hard_predicate_keys,
            predicate_parameters=predicate_parameters,
            candidates=_rows_to_candidates(rows),
        )

    def audit_nonretrieved(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        query_vector: Sequence[float],
        score_threshold: float,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        excluded_user_ids: Sequence[str],
        sample_size: int,
    ) -> tuple[int, int]:
        exclusion_sql, exclusion_params = self._postgres_exclusion_clause(
            vector_generation_id=vector_generation_id,
            user_expression="search.user_id",
            project_expression="search.project_id",
        )
        self._replace_temp_user_ids(
            table_name="audience_ann_retrieved",
            user_ids=excluded_user_ids,
        )
        rows = self._postgres.fetchall(
            f"""
            SELECT
                user_id,
                1 - (embedding <=> %s::vector) AS behavior_fit_score
            FROM user_behavior_vector_search AS search
            WHERE search.project_id = %s
              AND search.vector_version = %s
              AND search.vector_dim = 64
              AND search.window_end = %s
              AND search.vector_generation_id = %s
              {exclusion_sql}
              AND NOT EXISTS (
                  SELECT 1 FROM audience_ann_retrieved AS retrieved
                  WHERE retrieved.user_id = search.user_id
              )
            ORDER BY md5(search.user_id || %s) ASC, search.user_id ASC
            LIMIT %s
            """,
            (
                _vector_literal(query_vector),
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
                *exclusion_params,
                f"{project_id}:{vector_generation_id}:{source_cutoff}",
                sample_size,
            ),
        )
        sample = list(rows)
        score_passed = [
            row for row in sample if float(row["behavior_fit_score"]) >= score_threshold
        ]
        missed = self._filter_hard_predicates(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=vector_version,
            source_cutoff=source_cutoff,
            hard_predicate_keys=hard_predicate_keys,
            predicate_parameters=predicate_parameters,
            candidates=_rows_to_candidates(score_passed),
        )
        return len(sample), len(missed)

    def _materialize_hard_filtered_relation(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        candidate_relation: str,
        member_relation: str,
        score_threshold: float | None,
    ) -> int:
        _require_materialized_relation(candidate_relation)
        _require_materialized_relation(member_relation)
        self._drop_temp_relation(member_relation)
        self._postgres.execute(
            f"""
            CREATE TEMP TABLE {member_relation} (
                user_id text PRIMARY KEY,
                behavior_fit_score double precision NOT NULL,
                retrieval_rank integer NOT NULL
            ) ON COMMIT DROP
            """
        )
        after_user_id: str | None = None
        while True:
            rows = self._postgres.fetchall(
                f"""
                SELECT user_id, behavior_fit_score, retrieval_rank
                FROM {candidate_relation}
                WHERE (%s::text IS NULL OR user_id > %s)
                  AND (%s::double precision IS NULL OR behavior_fit_score >= %s)
                ORDER BY user_id ASC
                LIMIT %s
                """,
                (
                    after_user_id,
                    after_user_id,
                    score_threshold,
                    score_threshold,
                    PREDICATE_CHUNK_SIZE,
                ),
            )
            if not rows:
                break
            candidates = [
                SearchCandidate(
                    user_id=str(row["user_id"]),
                    behavior_fit_score=float(row["behavior_fit_score"]),
                    retrieval_rank=int(row["retrieval_rank"]),
                )
                for row in rows
            ]
            matched = self._filter_hard_predicates(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                vector_version=vector_version,
                source_cutoff=source_cutoff,
                hard_predicate_keys=hard_predicate_keys,
                predicate_parameters=predicate_parameters,
                candidates=candidates,
            )
            if matched:
                self._postgres.execute(
                    f"""
                    INSERT INTO {member_relation} (
                        user_id, behavior_fit_score, retrieval_rank
                    )
                    SELECT *
                    FROM unnest(
                        %s::text[],
                        %s::double precision[],
                        %s::integer[]
                    )
                    """,
                    (
                        [member.user_id for member in matched],
                        [member.behavior_fit_score for member in matched],
                        [member.retrieval_rank for member in matched],
                    ),
                )
            after_user_id = str(rows[-1]["user_id"])
            if len(rows) < PREDICATE_CHUNK_SIZE:
                break
        return self._count_temp_relation(member_relation)

    def _drop_temp_relation(self, relation: str) -> None:
        _require_materialized_relation(relation)
        self._postgres.execute(f"DROP TABLE IF EXISTS {relation}")

    def _index_temp_candidates(self, relation: str) -> None:
        _require_materialized_relation(relation)
        self._postgres.execute(
            f"CREATE UNIQUE INDEX {relation}_user_idx ON {relation} (user_id)"
        )

    def _count_temp_relation(self, relation: str) -> int:
        _require_materialized_relation(relation)
        row = self._postgres.fetchone(
            f"SELECT count(*) AS row_count FROM {relation}"
        )
        return int(row["row_count"]) if row is not None else 0

    def _replace_temp_user_ids(
        self,
        *,
        table_name: str,
        user_ids: Sequence[str],
    ) -> None:
        supported = {
            "audience_hard_match_sample",
            "audience_ann_candidates",
            "audience_ann_retrieved",
        }
        if table_name not in supported:
            raise ValueError("unsupported temporary candidate relation")
        self._postgres.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {table_name} ("
            "user_id text PRIMARY KEY) ON COMMIT DROP"
        )
        self._postgres.execute(f"TRUNCATE {table_name}")
        for offset in range(0, len(user_ids), PREDICATE_CHUNK_SIZE):
            chunk = user_ids[offset : offset + PREDICATE_CHUNK_SIZE]
            self._postgres.execute(
                f"INSERT INTO {table_name} (user_id) "
                "SELECT DISTINCT rows.user_id "
                "FROM unnest(%s::text[]) AS rows(user_id) "
                "ON CONFLICT (user_id) DO NOTHING",
                (list(chunk),),
            )

    def _filter_hard_predicates(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        candidates: Sequence[SearchCandidate],
    ) -> list[SearchCandidate]:
        if not candidates:
            return []
        generation_window_start, raw_event_received_cutoff = self._generation_window(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=vector_version,
            source_cutoff=source_cutoff,
        )
        window_start = _observation_window_start(
            generation_window_start=generation_window_start,
            source_cutoff=source_cutoff,
            predicate_parameters=predicate_parameters,
        )
        exclusion_context = self._exclusions_by_generation.get(
            vector_generation_id
        )
        matched_user_ids: set[str] = set()
        for offset in range(0, len(candidates), PREDICATE_CHUNK_SIZE):
            chunk = candidates[offset : offset + PREDICATE_CHUNK_SIZE]
            result = self._clickhouse.query(
                _hard_predicate_query(
                    hard_predicate_keys,
                    exclude_promotion_users=exclusion_context is not None,
                    predicate_parameters=predicate_parameters,
                ),
                parameters={
                    "project_id": project_id,
                    "user_ids": [candidate.user_id for candidate in chunk],
                    "window_start": _datetime_string(window_start),
                    "window_end": _datetime_string(source_cutoff),
                    "raw_event_received_cutoff": _datetime_string(
                        raw_event_received_cutoff
                    ),
                    "destinations": list(
                        predicate_parameters.get("destinations", ())
                    ),
                    "season_months": list(
                        predicate_parameters.get("season_months", ())
                    ),
                    "benefit_keys": list(
                        predicate_parameters.get("benefit_keys", ())
                    ),
                    **_structured_query_parameters(predicate_parameters),
                    **_clickhouse_exclusion_parameters(exclusion_context),
                },
            )
            rows = (
                list(result.named_results())
                if hasattr(result, "named_results")
                else list(result.result_rows)
            )
            for row in rows:
                matched_user_ids.add(
                    str(row["user_id"] if isinstance(row, Mapping) else row[0])
                )
        return [
            candidate
            for candidate in candidates
            if candidate.user_id in matched_user_ids
        ]

    def _postgres_exclusion_clause(
        self,
        *,
        vector_generation_id: str,
        user_expression: str,
        project_expression: str,
    ) -> tuple[str, tuple[Any, ...]]:
        context = self._exclusions_by_generation.get(vector_generation_id)
        if context is None:
            return "", ()
        return (
            f"""
            AND NOT EXISTS (
                SELECT 1
                FROM {POSTGRES_EXCLUSION_RELATION} AS excluded
                JOIN promotion_target_segments AS target
                  ON target.analysis_id = excluded.target_analysis_id
                 AND target.segment_id = excluded.segment_id
                 AND target.allocation_plan_id = excluded.allocation_plan_id
                 AND target.audience_snapshot_id = excluded.final_snapshot_id
                WHERE excluded.project_id = {project_expression}
                  AND excluded.promotion_id = %s
                  AND excluded.user_id = {user_expression}
                  AND excluded.state IN ('reserved', 'consumed')
                  AND target.status <> 'stopped'
            )
            """,
            (context.promotion_id,),
        )

    def _context_for_window(
        self,
        *,
        project_id: str,
        vector_version: str,
        window_end: str | datetime,
    ) -> PromotionAudienceExclusionContext | None:
        _ = (project_id, vector_version, window_end)
        if len(self._exclusions_by_generation) != 1:
            return None
        return next(iter(self._exclusions_by_generation.values()))

    def _generation_window(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str | datetime,
    ) -> tuple[str | datetime, str | datetime]:
        row = self._postgres.fetchone(
            """
            SELECT window_start, source_revision_cutoff
            FROM user_behavior_vector_search_generations
            WHERE project_id = %s
              AND vector_version = %s
              AND window_end = %s
              AND vector_generation_id = %s
            """,
            (
                project_id,
                vector_version,
                source_cutoff,
                vector_generation_id,
            ),
        )
        if (
            row is None
            or row["window_start"] is None
            or row["source_revision_cutoff"] is None
        ):
            raise RuntimeError("search vector window is unavailable")
        return row["window_start"], row["source_revision_cutoff"]


def _require_materialized_relation(relation: str) -> None:
    if relation not in MATERIALIZED_RELATIONS:
        raise ValueError("unsupported materialized audience relation")


def hard_predicates_support_batch_aggregate(keys: Sequence[str]) -> bool:
    return set(keys).issubset(
        {
            "hotel_product_interest",
            "target_destination_affinity",
            "booking_start_without_complete",
            "benefit_interest",
            "promotion_response",
            "general_destination_exploration",
            "recent_destination_search",
            "season_match",
        }
    )


def _hard_predicate_batch_query(
    requests: Sequence[HardMatchAggregateRequest],
    *,
    exclude_promotion_users: bool = False,
) -> tuple[str, dict[str, Any]]:
    if not requests:
        raise ValueError("hard match aggregate requests are required")
    if any(
        not hard_predicates_support_batch_aggregate(request.hard_predicate_keys)
        for request in requests
    ):
        raise ValueError("hard predicate cannot be represented by batch aggregate")

    parameters: dict[str, Any] = {}
    match_expressions: list[str] = []
    for index, request in enumerate(requests):
        destination_parameter = f"batch_{index}_destinations"
        season_parameter = f"batch_{index}_season_months"
        benefit_parameter = f"batch_{index}_benefit_keys"
        parameters[destination_parameter] = list(
            request.predicate_parameters.get("destinations", ())
        )
        parameters[season_parameter] = list(
            request.predicate_parameters.get("season_months", ())
        )
        parameters[benefit_parameter] = list(
            request.predicate_parameters.get("benefit_keys", ())
        )
        conditions: list[str] = []
        for key in request.hard_predicate_keys:
            if key == "hotel_product_interest":
                conditions.append("hotel_interest_count > 0")
            elif key == "target_destination_affinity":
                conditions.append(
                    "arrayCount(value -> value IN "
                    f"{{{destination_parameter}:Array(String)}}, "
                    "destination_values) >= 2"
                )
            elif key == "recent_destination_search":
                conditions.append(
                    "arrayExists(value -> value IN "
                    f"{{{destination_parameter}:Array(String)}}, "
                    "destination_values)"
                )
            elif key == "booking_start_without_complete":
                conditions.append(
                    "booking_start_count > booking_complete_count "
                    "AND booking_start_count > 0"
                )
            elif key == "benefit_interest":
                conditions.append(
                    f"(empty({{{benefit_parameter}:Array(String)}}) AND "
                    "(deal_count + price_count + free_cancellation_count + "
                    "breakfast_count) > 0) OR "
                    "(arrayExists(value -> value IN ('discount','early_booking'), "
                    f"{{{benefit_parameter}:Array(String)}}) AND "
                    "(deal_count + price_count) > 0) OR "
                    f"(has({{{benefit_parameter}:Array(String)}}, "
                    "'free_cancellation') AND "
                    "free_cancellation_count > 0) OR "
                    f"(has({{{benefit_parameter}:Array(String)}}, "
                    "'breakfast_included') AND breakfast_count > 0)"
                )
            elif key == "promotion_response":
                conditions.append("promotion_response_count > 0")
            elif key == "general_destination_exploration":
                conditions.append("destination_count >= 2")
            elif key == "season_match":
                conditions.append(
                    "arrayExists(value -> value IN "
                    f"{{{season_parameter}:Array(UInt8)}}, checkin_months)"
                )
        combined = " AND ".join(f"({condition})" for condition in conditions)
        match_expressions.append(f"countIf({combined or '1'}) AS match_{index}")

    destination_value = clickhouse_canonical_destination_sql(
        """
        coalesce(
            nullIf(JSONExtractString(properties_json, 'destination_id'), ''),
            nullIf(JSONExtractString(properties_json, 'destination_name'), ''),
            nullIf(JSONExtractString(properties_json, 'hotel_city'), ''),
            ''
        )
        """.strip()
    )
    query = f"""
        WITH per_user AS (
            SELECT
                user_id,
                countIf(event_name IN (
                    'hotel_search','hotel_click','hotel_detail_view'
                )) AS hotel_interest_count,
                countIf(event_name = 'booking_start') AS booking_start_count,
                countIf(event_name = 'booking_complete') AS booking_complete_count,
                countIf(event_name IN (
                    'promotion_click','campaign_landing'
                )) AS promotion_response_count,
                uniqExactIf(
                    {destination_value},
                    event_name IN ('hotel_search','hotel_click','hotel_detail_view')
                    AND {destination_value} != ''
                ) AS destination_count,
                groupArrayIf({destination_value}, {destination_value} != '')
                    AS destination_values,
                groupArrayIf(
                    toMonth(parseDateTimeBestEffortOrNull(
                        JSONExtractString(properties_json, 'checkin_date')
                    )),
                    parseDateTimeBestEffortOrNull(
                        JSONExtractString(properties_json, 'checkin_date')
                    ) IS NOT NULL
                ) AS checkin_months,
                countIf(toUInt8OrZero(
                    JSONExtractString(properties_json, 'deal')
                ) = 1) AS deal_count,
                countIf(nullIf(
                    JSONExtractString(properties_json, 'price'), ''
                ) IS NOT NULL) AS price_count,
                countIf(toUInt8OrZero(JSONExtractString(
                    properties_json, 'free_cancellation'
                )) = 1) AS free_cancellation_count,
                countIf(toUInt8OrZero(JSONExtractString(
                    properties_json, 'breakfast_included'
                )) = 1) AS breakfast_count
            FROM raw_events
            {_clickhouse_exclusion_join(exclude_promotion_users)}
            WHERE project_id = {{project_id:String}}
              AND user_id IN (
                  SELECT user_id
                  FROM user_behavior_vector_revisions
                  WHERE project_id = {{project_id:String}}
                    AND vector_version = {{vector_version:String}}
                    AND window_start = toDateTime64(
                        parseDateTimeBestEffort({{window_start:String}}), 3, 'UTC'
                    )
                    AND window_end = toDateTime64(
                        parseDateTimeBestEffort({{window_end:String}}), 3, 'UTC'
                    )
                    AND ingested_at <= parseDateTime64BestEffort(
                        {{source_revision_cutoff:String}}, 6, 'UTC'
                    )
                  GROUP BY user_id
              )
              AND validation_status = 'valid'
              AND received_at <= parseDateTime64BestEffort(
                  {{raw_event_received_cutoff:String}}, 6, 'UTC'
              )
              AND event_time >= toDateTime64(
                  parseDateTimeBestEffort({{window_start:String}}), 3, 'UTC'
              )
              AND event_time < toDateTime64(
                  parseDateTimeBestEffort({{window_end:String}}), 3, 'UTC'
              )
            GROUP BY user_id
        )
        SELECT {', '.join(match_expressions)}
        FROM per_user
    """
    return query, parameters


_DESTINATION_ALIAS_GROUPS = (
    ("제주", "jeju"),
    ("오키나와", "okinawa"),
    ("삿포로", "sapporo"),
    ("도쿄", "tokyo"),
    ("오사카", "osaka"),
    ("부산", "busan"),
    ("서울", "seoul"),
    ("다낭", "da nang", "danang"),
)


def _structured_having_conditions(
    predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
) -> list[str]:
    conditions = _structured_conditions(predicate_parameters)
    result: list[str] = []
    destination_text = "concat(" + ", ' ', ".join(
        (
            "ifNull(JSONExtractString(properties_json, 'destination_id'), '')",
            "ifNull(JSONExtractString(properties_json, 'destination_name'), '')",
            "ifNull(JSONExtractString(properties_json, 'hotel_city'), '')",
            "ifNull(JSONExtractString(properties_json, 'hotel_country'), '')",
        )
    ) + ")"
    for index, condition in enumerate(conditions):
        prefix = f"custom_{index}"
        event_predicates = [
            f"event_name = {{{prefix}_event_name:String}}",
        ]
        if condition.get("destination"):
            event_predicates.append(
                "arrayExists(term -> positionCaseInsensitiveUTF8("
                f"{destination_text}, term) > 0, "
                f"{{{prefix}_destination_terms:Array(String)}})"
            )
        if condition.get("checkin_months"):
            event_predicates.append(
                "toMonth(parseDateTimeBestEffortOrNull(nullIf("
                "JSONExtractString(properties_json, 'checkin_date'), ''))) "
                f"IN {{{prefix}_checkin_months:Array(UInt8)}}"
            )
        for filter_index, property_filter in enumerate(
            condition.get("property_filters", ())
        ):
            event_predicates.append(
                _structured_property_predicate(
                    property_filter,
                    parameter_name=f"{prefix}_property_{filter_index}",
                )
            )
        event_match = " AND ".join(
            f"({predicate})" for predicate in event_predicates
        )
        count_expression = f"countIf({event_match})"
        bounds = [f"{count_expression} >= {{{prefix}_minimum_count:UInt32}}"]
        if condition.get("maximum_count") is not None:
            bounds.append(
                f"{count_expression} <= {{{prefix}_maximum_count:UInt32}}"
            )
        result.append(" AND ".join(f"({bound})" for bound in bounds))
    return result


def _structured_query_parameters(
    predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
) -> dict[str, Any]:
    if "structured_conditions_json" not in predicate_parameters:
        return {}
    parameters: dict[str, Any] = {}
    if "base_user_ids" in predicate_parameters:
        parameters["base_user_ids"] = list(
            predicate_parameters.get("base_user_ids", ())
        )
    for index, condition in enumerate(_structured_conditions(predicate_parameters)):
        prefix = f"custom_{index}"
        parameters[f"{prefix}_event_name"] = str(condition["event_name"])
        parameters[f"{prefix}_minimum_count"] = int(condition["minimum_count"])
        if condition.get("maximum_count") is not None:
            parameters[f"{prefix}_maximum_count"] = int(condition["maximum_count"])
        if condition.get("destination"):
            parameters[f"{prefix}_destination_terms"] = list(
                _destination_search_terms(str(condition["destination"]))
            )
        if condition.get("checkin_months"):
            parameters[f"{prefix}_checkin_months"] = [
                int(month) for month in condition["checkin_months"]
            ]
        for filter_index, property_filter in enumerate(
            condition.get("property_filters", ())
        ):
            operator = str(property_filter["operator"])
            if operator == "exists":
                continue
            raw_value = str(property_filter["value"])
            parameter_name = f"{prefix}_property_{filter_index}"
            if operator == "in":
                parameters[parameter_name] = list(
                    _structured_property_values(raw_value)
                )
                continue
            if (
                operator == "equals"
                and property_filter["key"]
                in {"free_cancellation", "breakfast_included"}
                and raw_value.lower() in {"true", "false", "1", "0"}
            ):
                parameters[parameter_name] = (
                    "1" if raw_value.lower() in {"true", "1"} else "0"
                )
            else:
                parameters[parameter_name] = (
                    float(raw_value)
                    if operator in {"gte", "lte"}
                    else raw_value
                )
    return parameters


def _structured_conditions(
    predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
) -> tuple[Mapping[str, Any], ...]:
    serialized_values = predicate_parameters.get("structured_conditions_json", ())
    if len(serialized_values) != 1 or not isinstance(serialized_values[0], str):
        raise ValueError("structured conditions payload is missing")
    try:
        payload = json.loads(serialized_values[0])
    except json.JSONDecodeError as exc:
        raise ValueError("structured conditions payload is invalid") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("structured conditions payload must be a non-empty array")
    return tuple(
        condition
        for condition in payload
        if isinstance(condition, Mapping)
    )


def _structured_property_predicate(
    property_filter: Mapping[str, Any],
    *,
    parameter_name: str,
) -> str:
    key = str(property_filter["key"])
    if key not in CUSTOM_STRUCTURED_PROPERTY_KEYS:
        raise ValueError(f"unsupported structured property key: {key}")
    operator = str(property_filter["operator"])
    extracted = f"ifNull(JSONExtractString(properties_json, '{key}'), '')"
    if operator == "exists":
        return f"nullIf({extracted}, '') IS NOT NULL"
    if operator == "contains":
        return (
            f"positionCaseInsensitiveUTF8({extracted}, "
            f"{{{parameter_name}:String}}) > 0"
        )
    if operator == "in":
        return (
            f"has({{{parameter_name}:Array(String)}}, "
            f"lowerUTF8({extracted}))"
        )
    if operator in {"gte", "lte"}:
        comparison = ">=" if operator == "gte" else "<="
        return (
            f"toFloat64OrNull(nullIf({extracted}, '')) {comparison} "
            f"{{{parameter_name}:Float64}}"
        )
    if operator != "equals":
        raise ValueError(f"unsupported structured property operator: {operator}")
    if key in {"free_cancellation", "breakfast_included"}:
        return (
            f"toUInt8OrZero({extracted}) = "
            f"toUInt8OrZero({{{parameter_name}:String}})"
        )
    return f"lowerUTF8({extracted}) = lowerUTF8({{{parameter_name}:String}})"


def _structured_property_values(value: str) -> tuple[str, ...]:
    normalized = value.replace("，", ",").replace("/", ",").replace("·", ",")
    normalized = normalized.replace("또는", ",").replace("혹은", ",")
    return tuple(
        sorted(
            {
                item.strip().casefold()
                for item in normalized.split(",")
                if item.strip()
            }
        )
    )


def _destination_search_terms(destination: str) -> tuple[str, ...]:
    normalized_values = [
        value.strip().lower()
        for value in destination.replace("，", ",")
        .replace("/", ",")
        .replace("·", ",")
        .split(",")
        if value.strip()
    ]
    terms: set[str] = set()
    for value in normalized_values:
        aliases = next(
            (group for group in _DESTINATION_ALIAS_GROUPS if value in group),
            (value,),
        )
        terms.update(aliases)
    return tuple(sorted(terms))


def _hard_predicate_query(
    keys: Sequence[str],
    *,
    filter_user_ids: bool = True,
    restrict_to_vector_population: bool = False,
    deterministic_sample: bool = False,
    exclude_promotion_users: bool = False,
    predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]] | None = None,
) -> str:
    supported = {
        "hotel_product_interest",
        "target_destination_affinity",
        "booking_start_without_complete",
        "benefit_interest",
        "promotion_response",
        "general_destination_exploration",
        "recent_destination_search",
        "season_match",
        "source_audience_membership",
        "structured_conditions",
    }
    unknown = sorted(set(keys) - supported)
    if unknown:
        raise ValueError("unsupported hard predicates: " + ", ".join(unknown))
    conditions: list[str] = []
    interest = "countIf(event_name IN ('hotel_search','hotel_click','hotel_detail_view'))"
    destination_value = clickhouse_canonical_destination_sql(
        """
        coalesce(
            nullIf(JSONExtractString(properties_json, 'destination_id'), ''),
            nullIf(JSONExtractString(properties_json, 'destination_name'), ''),
            nullIf(JSONExtractString(properties_json, 'hotel_city'), ''),
            ''
        )
        """.strip()
    )
    destination = f"""
        countIf(
            {destination_value} IN {{destinations:Array(String)}}
        )
    """
    for key in keys:
        if key == "hotel_product_interest":
            conditions.append(f"{interest} > 0")
        elif key == "target_destination_affinity":
            conditions.append(f"{destination} >= 2")
        elif key == "recent_destination_search":
            conditions.append(f"{destination} > 0")
        elif key == "booking_start_without_complete":
            conditions.append(
                "countIf(event_name = 'booking_start') > "
                "countIf(event_name = 'booking_complete')"
            )
        elif key == "benefit_interest":
            conditions.append(
                "(empty({benefit_keys:Array(String)}) AND countIf("
                "toUInt8OrZero(JSONExtractString(properties_json, 'deal')) = 1 "
                "OR toUInt8OrZero(JSONExtractString(properties_json, 'free_cancellation')) = 1 "
                "OR toUInt8OrZero(JSONExtractString(properties_json, 'breakfast_included')) = 1 "
                "OR nullIf(JSONExtractString(properties_json, 'price'), '') IS NOT NULL"
                ") > 0) OR "
                "(arrayExists(value -> value IN ('discount','early_booking'), "
                "{benefit_keys:Array(String)}) AND countIf("
                "toUInt8OrZero(JSONExtractString(properties_json, 'deal')) = 1 "
                "OR nullIf(JSONExtractString(properties_json, 'price'), '') IS NOT NULL"
                ") > 0) OR "
                "(has({benefit_keys:Array(String)}, 'free_cancellation') AND "
                "countIf(toUInt8OrZero(JSONExtractString(properties_json, "
                "'free_cancellation')) = 1) > 0) OR "
                "(has({benefit_keys:Array(String)}, 'breakfast_included') AND "
                "countIf(toUInt8OrZero(JSONExtractString(properties_json, "
                "'breakfast_included')) = 1) > 0)"
            )
        elif key == "promotion_response":
            conditions.append(
                "countIf(event_name IN ('promotion_click','campaign_landing')) > 0"
            )
        elif key == "general_destination_exploration":
            conditions.append(
                "uniqExactIf("
                f"{destination_value}, "
                "event_name IN ('hotel_search','hotel_click','hotel_detail_view') "
                f"AND {destination_value} != ''"
                ") >= 2"
            )
        elif key == "season_match":
            conditions.append(
                "countIf(toMonth(parseDateTimeBestEffortOrNull("
                "JSONExtractString(properties_json, 'checkin_date'))) "
                "IN {season_months:Array(UInt8)}) > 0"
            )
        elif key == "source_audience_membership":
            continue
        elif key == "structured_conditions":
            conditions.extend(
                _structured_having_conditions(predicate_parameters or {})
            )
    having = " AND ".join(f"({condition})" for condition in conditions) or "1"
    user_filter = (
        "AND user_id IN {user_ids:Array(String)}" if filter_user_ids else ""
    )
    source_user_filter = (
        "AND user_id IN {base_user_ids:Array(String)}"
        if "source_audience_membership" in keys
        else ""
    )
    vector_population_filter = ""
    if restrict_to_vector_population:
        vector_population_filter = """
          AND user_id IN (
              SELECT user_id
              FROM user_behavior_vector_revisions
              WHERE project_id = {project_id:String}
                AND vector_version = {vector_version:String}
                AND window_start = toDateTime64(
                    parseDateTimeBestEffort({vector_window_start:String}), 3, 'UTC'
                )
                AND window_end = toDateTime64(
                    parseDateTimeBestEffort({window_end:String}), 3, 'UTC'
                )
                AND ingested_at <= parseDateTime64BestEffort(
                    {source_revision_cutoff:String}, 6, 'UTC'
                )
              GROUP BY user_id
          )
        """
    order_by = (
        "ORDER BY cityHash64(concat(user_id, {sample_seed:String})) ASC, "
        "user_id ASC"
        if deterministic_sample
        else "ORDER BY user_id ASC"
    )
    return f"""
        SELECT user_id
        FROM raw_events
        {_clickhouse_exclusion_join(exclude_promotion_users)}
        WHERE project_id = {{project_id:String}}
          {user_filter}
          {source_user_filter}
          {vector_population_filter}
          AND validation_status = 'valid'
          AND received_at <= parseDateTime64BestEffort(
              {{raw_event_received_cutoff:String}}, 6, 'UTC'
          )
          AND event_time >= toDateTime64(
              parseDateTimeBestEffort({{window_start:String}}), 3, 'UTC'
          )
          AND event_time < toDateTime64(
              parseDateTimeBestEffort({{window_end:String}}), 3, 'UTC'
          )
        GROUP BY user_id
        HAVING {having}
        {order_by}
    """


def _clickhouse_exclusion_join(enabled: bool) -> str:
    if not enabled:
        return ""
    return f"""
      LEFT ANTI JOIN (
          SELECT user_id
          FROM {CLICKHOUSE_EXCLUSION_RELATION}
          WHERE project_id = {{project_id:String}}
            AND promotion_id = {{exclusion_promotion_id:String}}
            AND exclusion_revision <= {{exclusion_revision:UInt64}}
          GROUP BY user_id
          HAVING argMax(state, exclusion_revision) IN ('reserved', 'consumed')
      ) AS promotion_excluded USING (user_id)
    """


def _clickhouse_exclusion_parameters(
    context: PromotionAudienceExclusionContext | None,
) -> dict[str, Any]:
    if context is None:
        return {}
    return {
        "exclusion_promotion_id": context.promotion_id,
        "exclusion_revision": context.revision,
    }


def _rows_to_candidates(rows: Sequence[Mapping[str, Any]]) -> list[SearchCandidate]:
    return [
        SearchCandidate(
            user_id=str(row["user_id"]),
            behavior_fit_score=float(row["behavior_fit_score"]),
            retrieval_rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]


def _vector_literal(values: Sequence[float]) -> str:
    if len(values) != VECTOR_DIM:
        raise ValueError("query vector must contain 64 values")
    return "[" + ",".join(format(float(value), ".17g") for value in values) + "]"


def _datetime_string(value: str | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")
    return value


def _observation_window_start(
    *,
    generation_window_start: str | datetime,
    source_cutoff: str | datetime,
    predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
) -> datetime:
    generation_start = _utc_datetime(generation_window_start)
    window_values = predicate_parameters.get("observation_window_days", ())
    if not window_values:
        return generation_start
    if len(window_values) != 1:
        raise ValueError("observation_window_days must contain exactly one value")
    window_days = window_values[0]
    if (
        not isinstance(window_days, int)
        or isinstance(window_days, bool)
        or not 1 <= window_days <= 365
    ):
        raise ValueError("observation_window_days must be between 1 and 365")
    requested_start = _utc_datetime(source_cutoff) - timedelta(days=window_days)
    return max(generation_start, requested_start)


def _utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
