from __future__ import annotations

import json
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from app.decision.models import (
    ActionCatalogItem,
    Experiment,
    ExperimentVariant,
    GeneratedContent,
    RecommendationAction,
    RecommendationResult,
    RootCauseCandidate,
    SegmentAnomaly,
    VariantPerformance,
)


class QueryClient(Protocol):
    def query(self, sql: str, parameters: dict[str, Any]) -> Any: ...


class PostgresDecisionRepository:
    """PostgreSQL writer repository for the schema.sql contract.

    The connection is expected to be a DB-API compatible object. SQL uses
    psycopg/psycopg2-style %s placeholders, but the module does not import a
    concrete driver so service tests can stay dependency-light.
    """

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def list_detected_anomalies(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> list[SegmentAnomaly]:
        return [
            self._anomaly(row)
            for row in self._fetchall(
                """
                SELECT *
                FROM segment_anomalies
                WHERE project_id = %s
                  AND analysis_date = %s
                  AND status = 'detected'
                ORDER BY impact_score DESC, id
                """,
                (project_id, analysis_date),
            )
        ]

    def list_root_causes(self, *, anomaly_id: int) -> list[RootCauseCandidate]:
        return [
            self._root_cause(row)
            for row in self._fetchall(
                """
                SELECT *
                FROM root_cause_candidates
                WHERE anomaly_id = %s
                ORDER BY rank_no ASC, impact_score DESC, id
                """,
                (anomaly_id,),
            )
        ]

    def get_active_action_catalog(self, *, action_key: str) -> ActionCatalogItem | None:
        row = self._fetchone(
            """
            SELECT *
            FROM action_catalog
            WHERE action_key = %s
              AND is_active = true
            """,
            (action_key,),
        )
        return self._action_catalog(row) if row is not None else None

    def upsert_recommendation_result(
        self,
        *,
        project_id: int,
        segment_id: int,
        anomaly_id: int,
        primary_root_cause_id: int | None,
        analysis_date: date,
        summary: str,
        status: str,
        recommendation_json: dict[str, Any],
        run_id: int,
    ) -> RecommendationResult:
        row = self._fetchone(
            """
            INSERT INTO recommendation_results (
                project_id,
                segment_id,
                anomaly_id,
                primary_root_cause_id,
                analysis_date,
                summary,
                status,
                recommendation_json,
                created_run_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (project_id, segment_id, analysis_date, anomaly_id)
            WHERE anomaly_id IS NOT NULL
            DO UPDATE
            SET
                primary_root_cause_id = EXCLUDED.primary_root_cause_id,
                summary = EXCLUDED.summary,
                status = CASE
                    WHEN recommendation_results.status IN ('pending_content', 'no_action')
                    THEN EXCLUDED.status
                    ELSE recommendation_results.status
                END,
                recommendation_json = EXCLUDED.recommendation_json,
                updated_at = now()
            RETURNING *
            """,
            (
                project_id,
                segment_id,
                anomaly_id,
                primary_root_cause_id,
                analysis_date,
                summary,
                status,
                self._json(recommendation_json),
                run_id,
            ),
        )
        return self._recommendation_result(row)

    def upsert_recommendation_action(
        self,
        *,
        recommendation_result_id: int,
        project_id: int,
        segment_id: int,
        action_catalog_id: int | None,
        action_key: str,
        title: str,
        description: str | None,
        priority: int,
        expected_effect_metric: str,
        expected_effect_direction: str,
        expected_effect_value: Decimal | None,
        status: str,
        metadata: dict[str, Any],
    ) -> RecommendationAction:
        row = self._fetchone(
            """
            INSERT INTO recommendation_actions (
                recommendation_result_id,
                project_id,
                segment_id,
                action_catalog_id,
                action_key,
                title,
                description,
                priority,
                expected_effect_metric,
                expected_effect_direction,
                expected_effect_value,
                status,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (recommendation_result_id, action_key)
            DO UPDATE
            SET
                action_catalog_id = EXCLUDED.action_catalog_id,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                priority = EXCLUDED.priority,
                expected_effect_metric = EXCLUDED.expected_effect_metric,
                expected_effect_direction = EXCLUDED.expected_effect_direction,
                expected_effect_value = EXCLUDED.expected_effect_value,
                status = CASE
                    WHEN recommendation_actions.status = 'recommended'
                    THEN EXCLUDED.status
                    ELSE recommendation_actions.status
                END,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING *
            """,
            (
                recommendation_result_id,
                project_id,
                segment_id,
                action_catalog_id,
                action_key,
                title,
                description,
                priority,
                expected_effect_metric,
                expected_effect_direction,
                expected_effect_value,
                status,
                self._json(metadata),
            ),
        )
        return self._recommendation_action(row)

    def list_actions_for_experiment_sync(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> list[RecommendationAction]:
        return [
            self._recommendation_action(row)
            for row in self._fetchall(
                """
                SELECT ra.*
                FROM recommendation_actions ra
                JOIN recommendation_results rr
                  ON rr.id = ra.recommendation_result_id
                WHERE ra.project_id = %s
                  AND rr.analysis_date = %s
                  AND ra.status IN (
                    'recommended',
                    'content_generated',
                    'experiment_created',
                    'running'
                  )
                ORDER BY ra.id
                """,
                (project_id, analysis_date),
            )
        ]

    def get_recommendation_result(self, *, result_id: int) -> RecommendationResult | None:
        row = self._fetchone("SELECT * FROM recommendation_results WHERE id = %s", (result_id,))
        return self._recommendation_result(row) if row is not None else None

    def find_action_content(
        self,
        *,
        project_id: int,
        recommendation_action_id: int,
        variant_key: str,
        statuses: tuple[str, ...],
    ) -> GeneratedContent | None:
        row = self._fetchone(
            """
            SELECT *
            FROM generated_contents
            WHERE project_id = %s
              AND recommendation_action_id = %s
              AND variant_key = %s
              AND generation_status = ANY(%s)
            ORDER BY (generation_status = 'approved') DESC, updated_at DESC, id DESC
            LIMIT 1
            """,
            (project_id, recommendation_action_id, variant_key, list(statuses)),
        )
        return self._generated_content(row) if row is not None else None

    def find_project_default_content(self, *, project_id: int) -> GeneratedContent | None:
        row = self._fetchone(
            """
            SELECT *
            FROM generated_contents
            WHERE project_id = %s
              AND recommendation_action_id IS NULL
              AND variant_key = 'default'
              AND generation_status IN ('generated', 'approved')
            ORDER BY (generation_status = 'approved') DESC, updated_at DESC, id DESC
            LIMIT 1
            """,
            (project_id,),
        )
        return self._generated_content(row) if row is not None else None

    def get_experiment_by_recommendation_action(
        self,
        *,
        project_id: int,
        recommendation_action_id: int,
    ) -> Experiment | None:
        row = self._fetchone(
            """
            SELECT *
            FROM experiments
            WHERE project_id = %s
              AND recommendation_action_id = %s
            """,
            (project_id, recommendation_action_id),
        )
        return self._experiment(row) if row is not None else None

    def upsert_experiment(
        self,
        *,
        project_id: int,
        segment_id: int,
        recommendation_action_id: int,
        name: str,
        objective_metric: str,
        target_value: Decimal,
        allocation_policy: str,
        status: str,
        start_date: date,
        run_id: int,
    ) -> Experiment:
        row = self._fetchone(
            """
            INSERT INTO experiments (
                project_id,
                segment_id,
                recommendation_action_id,
                name,
                objective_metric,
                target_value,
                allocation_policy,
                status,
                start_date,
                created_run_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_id, recommendation_action_id)
            WHERE recommendation_action_id IS NOT NULL
            DO UPDATE
            SET
                segment_id = EXCLUDED.segment_id,
                name = EXCLUDED.name,
                objective_metric = EXCLUDED.objective_metric,
                target_value = EXCLUDED.target_value,
                allocation_policy = EXCLUDED.allocation_policy,
                status = CASE
                    WHEN experiments.status = 'winner_selected'
                    THEN experiments.status
                    ELSE EXCLUDED.status
                END,
                updated_at = now()
            RETURNING *
            """,
            (
                project_id,
                segment_id,
                recommendation_action_id,
                name,
                objective_metric,
                target_value,
                allocation_policy,
                status,
                start_date,
                run_id,
            ),
        )
        return self._experiment(row)

    def upsert_experiment_variant(
        self,
        *,
        experiment_id: int,
        project_id: int,
        variant_key: str,
        name: str,
        generated_content_id: int | None,
        is_control: bool,
        traffic_weight: Decimal,
        status: str,
    ) -> ExperimentVariant:
        row = self._fetchone(
            """
            INSERT INTO experiment_variants (
                experiment_id,
                project_id,
                variant_key,
                name,
                generated_content_id,
                is_control,
                traffic_weight,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id, variant_key)
            DO UPDATE
            SET
                name = EXCLUDED.name,
                generated_content_id = EXCLUDED.generated_content_id,
                is_control = EXCLUDED.is_control,
                traffic_weight = EXCLUDED.traffic_weight,
                status = EXCLUDED.status,
                updated_at = now()
            RETURNING *
            """,
            (
                experiment_id,
                project_id,
                variant_key,
                name,
                generated_content_id,
                is_control,
                traffic_weight,
                status,
            ),
        )
        return self._experiment_variant(row)

    def deactivate_mappings_for_experiment(self, *, experiment_id: int) -> int:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                UPDATE segment_ad_mappings
                SET
                    is_active = false,
                    traffic_weight = 0,
                    is_winner = false,
                    updated_at = now()
                WHERE experiment_id = %s
                """,
                (experiment_id,),
            )
            return cursor.rowcount
        finally:
            cursor.close()

    def upsert_segment_ad_mapping(
        self,
        *,
        project_id: int,
        segment_id: int,
        placement_key: str,
        experiment_id: int,
        experiment_variant_id: int,
        generated_content_id: int,
        traffic_weight: Decimal,
        is_active: bool,
        is_winner: bool,
        priority: int,
        run_id: int,
    ):
        row = self._fetchone(
            """
            INSERT INTO segment_ad_mappings (
                project_id,
                segment_id,
                placement_key,
                experiment_id,
                experiment_variant_id,
                generated_content_id,
                traffic_weight,
                is_active,
                is_winner,
                priority,
                created_run_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_id, segment_id, placement_key, experiment_variant_id)
            DO UPDATE
            SET
                experiment_id = EXCLUDED.experiment_id,
                generated_content_id = EXCLUDED.generated_content_id,
                traffic_weight = EXCLUDED.traffic_weight,
                is_active = EXCLUDED.is_active,
                is_winner = EXCLUDED.is_winner,
                priority = EXCLUDED.priority,
                updated_at = now()
            RETURNING *
            """,
            (
                project_id,
                segment_id,
                placement_key,
                experiment_id,
                experiment_variant_id,
                generated_content_id,
                traffic_weight,
                is_active,
                is_winner,
                priority,
                run_id,
            ),
        )
        return row

    def update_recommendation_result_status(self, *, result_id: int, status: str):
        self._execute(
            """
            UPDATE recommendation_results
            SET status = %s, updated_at = now()
            WHERE id = %s
            """,
            (status, result_id),
        )

    def update_recommendation_action_status(self, *, action_id: int, status: str):
        self._execute(
            """
            UPDATE recommendation_actions
            SET status = %s, updated_at = now()
            WHERE id = %s
            """,
            (status, action_id),
        )

    def get_project_timezone(self, *, project_id: int) -> str:
        row = self._fetchone("SELECT timezone FROM projects WHERE id = %s", (project_id,))
        return row["timezone"] if row is not None else "Asia/Seoul"

    def list_experiments_by_status(self, *, project_id: int, status: str) -> list[Experiment]:
        return [
            self._experiment(row)
            for row in self._fetchall(
                """
                SELECT *
                FROM experiments
                WHERE project_id = %s
                  AND status = %s
                ORDER BY id
                """,
                (project_id, status),
            )
        ]

    def list_experiment_variants(self, *, experiment_id: int) -> list[ExperimentVariant]:
        return [
            self._experiment_variant(row)
            for row in self._fetchall(
                """
                SELECT *
                FROM experiment_variants
                WHERE experiment_id = %s
                ORDER BY variant_key
                """,
                (experiment_id,),
            )
        ]

    def update_experiment_variant_results(
        self,
        *,
        variant_id: int,
        impression_count: int,
        click_count: int,
        conversion_count: int,
        ctr: Decimal,
        conversion_rate: Decimal,
    ) -> ExperimentVariant:
        row = self._fetchone(
            """
            UPDATE experiment_variants
            SET
                impression_count = %s,
                click_count = %s,
                conversion_count = %s,
                ctr = %s,
                conversion_rate = %s,
                updated_at = now()
            WHERE id = %s
            RETURNING *
            """,
            (
                impression_count,
                click_count,
                conversion_count,
                ctr,
                conversion_rate,
                variant_id,
            ),
        )
        return self._experiment_variant(row)

    def set_experiment_winner(
        self,
        *,
        experiment: Experiment,
        variants: list[ExperimentVariant],
        winner_variant: ExperimentVariant,
    ) -> None:
        action_status = "lost" if winner_variant.variant_key == "control" else "won"
        self._execute(
            """
            UPDATE experiments
            SET status = 'winner_selected',
                winner_variant_id = %s,
                updated_at = now()
            WHERE id = %s
              AND status = 'running'
            """,
            (winner_variant.id, experiment.id),
        )
        self._execute(
            """
            UPDATE recommendation_results
            SET status = 'winner_selected',
                updated_at = now()
            WHERE id = (
                SELECT recommendation_result_id
                FROM recommendation_actions
                WHERE id = %s
            )
            """,
            (experiment.recommendation_action_id,),
        )
        self._execute(
            """
            UPDATE recommendation_actions
            SET status = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (action_status, experiment.recommendation_action_id),
        )
        for variant in variants:
            is_winner = variant.id == winner_variant.id
            self._execute(
                """
                UPDATE experiment_variants
                SET status = %s,
                    traffic_weight = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                ("winner" if is_winner else "loser", Decimal("1") if is_winner else Decimal("0"), variant.id),
            )
            self._execute(
                """
                UPDATE segment_ad_mappings
                SET traffic_weight = %s,
                    is_active = %s,
                    is_winner = %s,
                    updated_at = now()
                WHERE experiment_id = %s
                  AND experiment_variant_id = %s
                """,
                (
                    Decimal("1") if is_winner else Decimal("0"),
                    is_winner,
                    is_winner,
                    experiment.id,
                    variant.id,
                ),
            )

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params)
        finally:
            cursor.close()

    def _fetchone(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cursor, row)
        finally:
            cursor.close()

    def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params)
            return [self._row_to_dict(cursor, row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def _row_to_dict(self, cursor: Any, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row, strict=True))

    def _json(self, value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _decimal(self, value: Any, default: str = "0") -> Decimal:
        if value is None:
            return Decimal(default)
        return Decimal(str(value))

    def _anomaly(self, row: dict[str, Any]) -> SegmentAnomaly:
        return SegmentAnomaly(
            id=row["id"],
            project_id=row["project_id"],
            segment_id=row["segment_id"],
            analysis_date=row["analysis_date"],
            metric_name=row["metric_name"],
            severity=row["severity"],
            impact_score=self._decimal(row["impact_score"]),
            status=row["status"],
            evidence_json=row.get("evidence_json") or {},
        )

    def _root_cause(self, row: dict[str, Any]) -> RootCauseCandidate:
        return RootCauseCandidate(
            id=row["id"],
            anomaly_id=row["anomaly_id"],
            cause_type=row["cause_type"],
            cause_key=row["cause_key"],
            title=row["title"],
            description=row.get("description"),
            confidence_score=self._decimal(row["confidence_score"]),
            impact_score=self._decimal(row["impact_score"]),
            rank_no=row["rank_no"],
            evidence_json=row.get("evidence_json") or {},
        )

    def _action_catalog(self, row: dict[str, Any]) -> ActionCatalogItem:
        return ActionCatalogItem(
            id=row["id"],
            action_key=row["action_key"],
            name=row["name"],
            description=row.get("description"),
            target_funnel_step=row.get("target_funnel_step"),
            default_channel=row["default_channel"],
            template_json=row.get("template_json") or {},
        )

    def _recommendation_result(self, row: dict[str, Any]) -> RecommendationResult:
        return RecommendationResult(
            id=row["id"],
            project_id=row["project_id"],
            segment_id=row["segment_id"],
            anomaly_id=row.get("anomaly_id"),
            primary_root_cause_id=row.get("primary_root_cause_id"),
            analysis_date=row["analysis_date"],
            summary=row["summary"],
            status=row["status"],
            recommendation_json=row.get("recommendation_json") or {},
        )

    def _recommendation_action(self, row: dict[str, Any]) -> RecommendationAction:
        return RecommendationAction(
            id=row["id"],
            recommendation_result_id=row["recommendation_result_id"],
            project_id=row["project_id"],
            segment_id=row["segment_id"],
            action_catalog_id=row.get("action_catalog_id"),
            action_key=row["action_key"],
            title=row["title"],
            description=row.get("description"),
            priority=row["priority"],
            expected_effect_metric=row["expected_effect_metric"],
            expected_effect_direction=row["expected_effect_direction"],
            expected_effect_value=(
                self._decimal(row["expected_effect_value"])
                if row.get("expected_effect_value") is not None
                else None
            ),
            status=row["status"],
            metadata=row.get("metadata") or {},
        )

    def _generated_content(self, row: dict[str, Any]) -> GeneratedContent:
        return GeneratedContent(
            id=row["id"],
            project_id=row["project_id"],
            segment_id=row["segment_id"],
            recommendation_action_id=row.get("recommendation_action_id"),
            variant_key=row["variant_key"],
            generation_status=row["generation_status"],
        )

    def _experiment(self, row: dict[str, Any]) -> Experiment:
        return Experiment(
            id=row["id"],
            project_id=row["project_id"],
            segment_id=row["segment_id"],
            recommendation_action_id=row["recommendation_action_id"],
            name=row["name"],
            objective_metric=row["objective_metric"],
            target_value=self._decimal(row["target_value"]),
            allocation_policy=row["allocation_policy"],
            status=row["status"],
            start_date=row["start_date"],
            winner_variant_id=row.get("winner_variant_id"),
        )

    def _experiment_variant(self, row: dict[str, Any]) -> ExperimentVariant:
        return ExperimentVariant(
            id=row["id"],
            experiment_id=row["experiment_id"],
            project_id=row["project_id"],
            variant_key=row["variant_key"],
            name=row["name"],
            generated_content_id=row.get("generated_content_id"),
            is_control=row["is_control"],
            traffic_weight=self._decimal(row["traffic_weight"], "0.5"),
            impression_count=row["impression_count"],
            click_count=row["click_count"],
            conversion_count=row["conversion_count"],
            ctr=self._decimal(row.get("ctr")),
            conversion_rate=self._decimal(row.get("conversion_rate")),
            status=row["status"],
        )


class ClickHouseExperimentResultRepository:
    """Adapter that expects a ClickHouse client with a query(sql, parameters) method."""

    def __init__(self, client: QueryClient, *, events_table: str = "events") -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", events_table):
            raise ValueError("events_table must be a safe SQL identifier")
        self.client = client
        self.events_table = events_table

    def fetch_variant_results(
        self,
        *,
        project_id: int,
        experiment: Experiment,
        variants: list[ExperimentVariant],
        window_start: datetime,
        window_end: datetime,
    ) -> dict[int, VariantPerformance]:
        if not variants:
            return {}
        variants_by_id = {variant.id: variant for variant in variants}
        result = self.client.query(
            f"""
            SELECT
                experiment_variant_id,
                generated_content_id,
                countIf(event_name = 'ad_impression') AS ad_impression_count,
                countIf(event_name = 'ad_click') AS ad_click_count,
                countIf(event_name = 'purchase') AS purchase_count
            FROM {self.events_table}
            WHERE project_id = %(project_id)s
              AND experiment_id = %(experiment_id)s
              AND experiment_variant_id IN %(variant_ids)s
              AND event_time >= %(window_start)s
              AND event_time < %(window_end)s
            GROUP BY experiment_variant_id, generated_content_id
            """,
            {
                "project_id": project_id,
                "experiment_id": experiment.id,
                "variant_ids": tuple(variants_by_id),
                "window_start": window_start,
                "window_end": window_end,
            },
        )
        rows = getattr(result, "result_rows", result)
        by_variant = {
            variant.id: VariantPerformance(
                experiment_variant_id=variant.id,
                ad_impression_count=0,
                ad_click_count=0,
                attributed_purchase_count=0,
            )
            for variant in variants
        }
        for row in rows:
            variant_id = int(row[0])
            if variant_id not in variants_by_id:
                continue
            generated_content_id = row[1]
            current = by_variant[variant_id]
            variant = variants_by_id[variant_id]
            purchase_count = int(row[4])
            attributed_purchase_count = current.attributed_purchase_count
            if variant.generated_content_id is None or generated_content_id == variant.generated_content_id:
                attributed_purchase_count += purchase_count
            by_variant[variant_id] = VariantPerformance(
                experiment_variant_id=variant_id,
                ad_impression_count=current.ad_impression_count + int(row[2]),
                ad_click_count=current.ad_click_count + int(row[3]),
                attributed_purchase_count=attributed_purchase_count,
            )
        return by_variant
