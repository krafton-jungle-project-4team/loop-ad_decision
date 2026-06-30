from __future__ import annotations

from app.contents.postgres_repository import (
    PostgresContentRepository,
    advisory_lock_key,
)
from app.contents.repository import GenerationLockUnavailable
from app.contents.types import (
    ACTION_STATUS_CONTENT_GENERATED,
    ACTION_STATUS_FAILED,
    GENERATION_STATUS_GENERATED,
    GeneratedContentDraft,
)


class FakeCursor:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.rowcount = len(self.rows)

    def fetchone(self):
        if not self.rows:
            return None
        return self.rows.pop(0)

    def fetchall(self):
        rows = self.rows
        self.rows = []
        return rows


class FakeConnection:
    def __init__(self, rows_by_call: list[list[dict[str, object]]] | None = None) -> None:
        self.rows_by_call = rows_by_call or []
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def execute(self, query: str, params: dict[str, object] | None = None) -> FakeCursor:
        self.calls.append((query, params))
        rows = self.rows_by_call.pop(0) if self.rows_by_call else []
        return FakeCursor(rows)


def make_draft() -> GeneratedContentDraft:
    return GeneratedContentDraft(
        project_id=1,
        segment_id=1,
        recommendation_action_id=10,
        variant_key="control",
        content_type="banner",
        title="Today deal",
        body="Check this offer.",
        cta_label="Shop now",
        landing_url="/collections/fresh",
        image_prompt="fresh food ecommerce banner",
        generation_model="mock",
        generation_status=GENERATION_STATUS_GENERATED,
        created_run_id=77,
        metadata={"generator": "mock"},
    )


def generated_content_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": 1,
        "project_id": 1,
        "recommendation_action_id": 10,
        "segment_id": 1,
        "variant_key": "control",
        "generation_status": GENERATION_STATUS_GENERATED,
        "created_run_id": 77,
        "metadata": {"generator": "mock"},
    }
    row.update(overrides)
    return row


def test_list_generation_targets_uses_anomalous_non_default_recommended_actions() -> None:
    connection = FakeConnection(
        rows_by_call=[
            [
                {
                    "recommendation_action_id": 10,
                    "project_id": 1,
                    "recommendation_result_id": 100,
                    "action_key": "highlight_benefit_banner",
                    "action_status": "recommended",
                    "action_type": "banner",
                    "action_title": "Highlight benefit",
                    "action_description": "Show benefit",
                    "content_type": "banner",
                    "action_metadata": {"landing_url": "/collections/fresh"},
                    "analysis_date": "2021-01-04",
                    "segment_id": 1,
                    "segment_key": "fresh_kakao_30m",
                    "segment_name": "Fresh food kakao male 30s",
                    "segment_is_default": False,
                    "segment_description": "Fresh food interest segment",
                    "segment_attributes": {"category": "fresh"},
                    "metrics": {"view_to_purchase_rate": 0.03},
                    "root_cause": {"cause_key": "view_to_cart"},
                }
            ]
        ]
    )
    repository = PostgresContentRepository(connection)

    targets = list(
        repository.list_generation_targets(
            project_id=1,
            analysis_date="2021-01-04",
            eligible_statuses=("recommended",),
        )
    )

    query, params = connection.calls[0]
    assert "FROM recommendation_actions ra" in query
    assert "JOIN recommendation_results rr" in query
    assert "JOIN segments s" in query
    assert "LEFT JOIN action_catalog ac" in query
    assert "s.name AS segment_name" in query
    assert "s.rule_json" in query
    assert "ra.action_type" not in query
    assert "s.segment_name" not in query
    assert "s.attributes" not in query
    assert "s.is_default = false" in query
    assert "rr.anomaly_id IS NOT NULL" in query
    assert params == {
        "project_id": 1,
        "analysis_date": "2021-01-04",
        "eligible_statuses": ["recommended"],
    }
    assert targets[0].id == 10
    assert targets[0].segment.is_default is False
    assert targets[0].root_cause["cause_key"] == "view_to_cart"


def test_generation_lock_uses_stable_project_action_advisory_lock_key() -> None:
    connection = FakeConnection(rows_by_call=[[{"acquired": True}]])
    repository = PostgresContentRepository(connection)

    with repository.generation_lock(project_id=1, recommendation_action_id=10):
        pass

    acquire_query, acquire_params = connection.calls[0]
    release_query, release_params = connection.calls[1]
    assert "pg_try_advisory_lock" in acquire_query
    assert "pg_advisory_unlock" in release_query
    assert acquire_params == {"lock_key": advisory_lock_key(1, 10)}
    assert release_params == acquire_params
    assert advisory_lock_key(1, 10) == advisory_lock_key(1, 10)
    assert advisory_lock_key(1, 10) != advisory_lock_key(1, 11)


def test_generation_lock_raises_when_another_worker_holds_lock() -> None:
    connection = FakeConnection(rows_by_call=[[{"acquired": False}]])
    repository = PostgresContentRepository(connection)

    try:
        with repository.generation_lock(project_id=1, recommendation_action_id=10):
            raise AssertionError("lock body should not run")
    except GenerationLockUnavailable as exc:
        assert "recommendation_action_id=10" in str(exc)
    else:
        raise AssertionError("GenerationLockUnavailable was not raised")


def test_force_false_inserts_generated_content_without_overwriting_existing_rows() -> None:
    connection = FakeConnection(rows_by_call=[[generated_content_row()]])
    repository = PostgresContentRepository(connection)

    record = repository.upsert_generated_content(draft=make_draft(), force=False)

    query, params = connection.calls[0]
    assert "DO NOTHING" in query
    assert "DO UPDATE" not in query
    assert "recommendation_action_id IS NOT NULL" in query
    assert "CAST(%(metadata)s AS jsonb)" in query
    assert "created_run_id" in query
    assert params is not None
    assert params["metadata"] == '{"generator": "mock"}'
    assert params["created_run_id"] == 77
    assert record.id == 1
    assert record.created_run_id == 77


def test_force_false_returns_existing_generated_content_after_insert_conflict() -> None:
    connection = FakeConnection(rows_by_call=[[], [generated_content_row(id=22)]])
    repository = PostgresContentRepository(connection)

    record = repository.upsert_generated_content(draft=make_draft(), force=False)

    assert "DO NOTHING" in connection.calls[0][0]
    assert "FROM generated_contents" in connection.calls[1][0]
    assert record.id == 22
    assert record.created_run_id == 77


def test_force_true_updates_ai_generated_content_but_never_targets_null_action_rows() -> None:
    connection = FakeConnection(rows_by_call=[[generated_content_row(id=3)]])
    repository = PostgresContentRepository(connection)

    record = repository.upsert_generated_content(draft=make_draft(), force=True)

    query, _ = connection.calls[0]
    assert "DO UPDATE SET" in query
    assert "recommendation_action_id IS NOT NULL" in query
    assert "title = EXCLUDED.title" in query
    assert "created_run_id = EXCLUDED.created_run_id" in query
    assert record.id == 3
    assert record.created_run_id == 77


def test_mark_action_failed_writes_schema_status_and_metadata_error() -> None:
    connection = FakeConnection()
    repository = PostgresContentRepository(connection)

    repository.mark_action_failed(
        recommendation_action_id=10,
        error_type="content_generation_failed",
        error_message="boom",
    )

    query, params = connection.calls[0]
    assert "UPDATE recommendation_actions" in query
    assert params is not None
    assert params["status"] == ACTION_STATUS_FAILED
    assert params["metadata_patch"] == (
        '{"error_message": "boom", "error_type": "content_generation_failed"}'
    )


def test_mark_action_content_generated_uses_content_generated_status() -> None:
    connection = FakeConnection()
    repository = PostgresContentRepository(connection)

    repository.mark_action_content_generated(recommendation_action_id=10)

    _, params = connection.calls[0]
    assert params is not None
    assert params["status"] == ACTION_STATUS_CONTENT_GENERATED
