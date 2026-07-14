from datetime import UTC, datetime
from uuid import UUID

import pytest

from app.generation.repositories import (
    CONTENT_CANDIDATE_COLUMNS,
    GENERATION_RUN_COLUMNS,
    ContentCandidateRecord,
    ContentCandidateRepository,
    GenerationInputRepository,
    GenerationRunRecord,
    GenerationRunRepository,
    _content_brief_json,
)
from app.generation.schemas import ContentChannel, GenerationRequest


class FakeCursor:
    def __init__(
        self,
        *,
        fetchone_result: dict[str, object] | None = None,
        fetchone_results: list[dict[str, object] | None] | None = None,
        fetchall_result: list[dict[str, object]] | None = None,
    ) -> None:
        self.fetchone_result = fetchone_result
        self.fetchone_results = list(fetchone_results or [])
        self.fetchall_result = fetchall_result or []
        self.executed: list[tuple[str, dict[str, object] | None]] = []
        self._last_query = ""

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: dict[str, object] | None = None) -> None:
        self._last_query = query
        self.executed.append((query, params))

    def fetchone(self) -> dict[str, object] | None:
        if self.fetchone_results:
            return self.fetchone_results.pop(0)
        return self.fetchone_result

    def fetchall(self) -> list[dict[str, object]]:
        rows = list(self.fetchall_result)
        if "pts.status = 'approved'" in self._last_query:
            rows = [row for row in rows if row.get("status") == "approved"]
        if "pts.segment_id = any(%(segment_ids)s)" in self._last_query.lower():
            params = self.executed[-1][1] or {}
            requested_ids = set(params["segment_ids"])
            rows = [row for row in rows if row.get("segment_id") in requested_ids]
        return rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor
        self.row_factories: list[object] = []

    def cursor(self, *, row_factory: object = None) -> FakeCursor:
        self.row_factories.append(row_factory)
        return self.cursor_instance


LEASE_TOKEN = UUID("00000000-0000-0000-0000-000000000247")


def generation_run_record(**overrides: object) -> GenerationRunRecord:
    values: dict[str, object] = {
        "generation_id": "generation_banner_001",
        "analysis_id": "analysis_banner_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "content_option_count": 2,
        "operator_instruction": None,
        "input_json": {
            "schema_version": "generation.request.v1",
            "target_segments": [{"segment_id": "seg_repeat_hotel_no_booking"}],
        },
        "output_json": None,
        "generation_report_json": {},
        "status": "requested",
        "idempotency_key": "generation:banner:001",
        "request_fingerprint": "a" * 64,
    }
    values.update(overrides)
    return GenerationRunRecord(**values)  # type: ignore[arg-type]


def content_candidate_record(**overrides: object) -> ContentCandidateRecord:
    values: dict[str, object] = {
        "content_id": "content_banner_001",
        "content_option_id": "banner_option_001",
        "generation_id": "generation_banner_001",
        "analysis_id": "analysis_banner_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "segment_id": "seg_repeat_hotel_no_booking",
        "channel": ContentChannel.ONSITE_BANNER,
        "title": "Book this weekend's rooms",
        "body": "Compare refundable summer offers before rooms run out.",
        "cta": "View hotel deals",
        "image_prompt": "bright modern hotel room, summer travel banner",
        "image_url": "https://cdn.example.test/content_banner_001.png",
        "landing_url": "https://demo-stay.example.com/summer",
        "creative_format": "banner_html",
        "image_generation_status": "completed",
        "artifact_status": "published",
        "artifact_storage_key": "genai/project/content_banner_001/banner.html",
        "artifact_public_url": "https://cdn.example.test/content_banner_001.html",
        "artifact_sha256": "b" * 64,
        "artifact_content_type": "text/html; charset=utf-8",
    }
    values.update(overrides)
    return ContentCandidateRecord(**values)  # type: ignore[arg-type]


def test_generation_run_repository_columns_match_data_source_contract() -> None:
    assert GENERATION_RUN_COLUMNS == (
        "generation_id",
        "analysis_id",
        "project_id",
        "campaign_id",
        "promotion_id",
        "content_option_count",
        "operator_instruction",
        "input_json",
        "output_json",
        "generation_report_json",
        "status",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "retry_count",
        "next_retry_at",
        "last_error_code",
        "last_error_message",
        "worker_id",
        "lease_token",
        "heartbeat_at",
        "lease_expires_at",
        "idempotency_key",
        "request_fingerprint",
    )


def test_content_candidate_repository_columns_match_data_source_contract() -> None:
    assert CONTENT_CANDIDATE_COLUMNS == (
        "content_id",
        "content_option_id",
        "generation_id",
        "analysis_id",
        "project_id",
        "campaign_id",
        "promotion_id",
        "segment_id",
        "channel",
        "subject",
        "preheader",
        "title",
        "body",
        "cta",
        "message",
        "image_prompt",
        "image_url",
        "landing_url",
        "generation_prompt",
        "reason_summary",
        "data_evidence_json",
        "message_strategy",
        "metadata_json",
        "status",
        "created_at",
        "updated_at",
        "creative_format",
        "image_generation_status",
        "artifact_status",
        "artifact_storage_key",
        "artifact_public_url",
        "artifact_sha256",
        "artifact_content_type",
        "artifact_error_code",
        "artifact_published_at",
    )


def test_v2_content_brief_does_not_merge_legacy_data_evidence() -> None:
    content_brief = {
        "schema_version": "content_brief.v2",
        "fallback_guidance": {
            "message_direction": "Use a hotel booking reminder.",
            "keywords": ["refundable stay"],
        },
        "audience_evidence": {
            "primary_signals": ["same_hotel_repeat_view"],
        },
    }

    result = _content_brief_json(
        {
            "content_brief_json": content_brief,
            "data_evidence_json": {
                "booking_conversion_rate": "0.018",
                "comparison_group_conversion_rate": "0.034",
                "top_common_features": ["must_not_merge"],
                "keywords": ["must_not_merge"],
            },
        }
    )

    assert result == content_brief


def test_generation_run_repository_create_executes_insert() -> None:
    cursor = FakeCursor(fetchone_result={"generation_id": "generation_banner_001"})
    repository = GenerationRunRepository(FakeConnection(cursor))

    result = repository.create(
        GenerationRunRecord(
            generation_id="generation_banner_001",
            analysis_id="analysis_banner_001",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            content_option_count=2,
            operator_instruction=None,
            input_json={"analysis_id": "analysis_banner_001"},
            output_json={"content_candidate_ids": ["content_banner_001"]},
            generation_report_json={"content_candidate_count": 2},
            status="completed",
        )
    )

    assert result == {"generation_id": "generation_banner_001"}
    query, params = cursor.executed[0]
    assert "INSERT INTO generation_runs" in query
    assert params is not None
    assert params["generation_id"] == "generation_banner_001"
    assert params["input_json"].obj == {"analysis_id": "analysis_banner_001"}
    assert params["output_json"].obj == {
        "content_candidate_ids": ["content_banner_001"]
    }
    assert params["generation_report_json"].obj == {"content_candidate_count": 2}
    assert "idempotency_key" in query
    assert "request_fingerprint" in query
    assert params["retry_count"] == 0


def test_generation_run_repository_create_or_get_idempotent_creates_once() -> None:
    created_row = {
        "generation_id": "generation_banner_001",
        "request_fingerprint": "a" * 64,
    }
    cursor = FakeCursor(fetchone_result=created_row)
    repository = GenerationRunRepository(FakeConnection(cursor))

    result, created = repository.create_or_get_idempotent(
        generation_run_record()
    )

    assert result == created_row
    assert created is True
    assert len(cursor.executed) == 1
    query, params = cursor.executed[0]
    assert "ON CONFLICT (project_id, idempotency_key)" in query
    assert "WHERE idempotency_key IS NOT NULL" in query
    assert params is not None
    assert params["idempotency_key"] == "generation:banner:001"


def test_generation_run_repository_create_or_get_idempotent_returns_existing() -> None:
    existing_row = {
        "generation_id": "generation_banner_001",
        "request_fingerprint": "a" * 64,
    }
    cursor = FakeCursor(fetchone_results=[None, existing_row])
    repository = GenerationRunRepository(FakeConnection(cursor))

    result, created = repository.create_or_get_idempotent(
        generation_run_record()
    )

    assert result == existing_row
    assert created is False
    assert len(cursor.executed) == 2
    query, params = cursor.executed[1]
    assert "project_id = %(project_id)s" in query
    assert "idempotency_key = %(idempotency_key)s" in query
    assert params == {
        "project_id": "hotel-client-a",
        "idempotency_key": "generation:banner:001",
    }


def test_generation_run_repository_rejects_reused_key_with_new_fingerprint() -> None:
    cursor = FakeCursor(
        fetchone_results=[
            None,
            {
                "generation_id": "generation_banner_001",
                "request_fingerprint": "b" * 64,
            },
        ]
    )
    repository = GenerationRunRepository(FakeConnection(cursor))

    with pytest.raises(ValueError, match="different request"):
        repository.create_or_get_idempotent(generation_run_record())


def test_generation_run_repository_claims_due_job_with_new_lease() -> None:
    claimed_row = {"generation_id": "generation_banner_001", "status": "running"}
    cursor = FakeCursor(fetchone_result=claimed_row)
    repository = GenerationRunRepository(FakeConnection(cursor))

    result = repository.claim_next(
        worker_id="decision-task-1",
        lease_token=LEASE_TOKEN,
        lease_seconds=180,
    )

    assert result == claimed_row
    query, params = cursor.executed[0]
    assert "FOR UPDATE SKIP LOCKED" in query
    assert "status = 'requested'" in query
    assert "next_retry_at <= now()" in query
    assert "status = 'running'" in query
    assert "started_at = COALESCE(run.started_at, now())" in query
    assert params == {
        "worker_id": "decision-task-1",
        "lease_token": LEASE_TOKEN,
        "lease_seconds": 180,
    }


def test_generation_run_repository_heartbeat_is_fenced_and_requires_live_lease() -> None:
    cursor = FakeCursor(fetchone_result={"generation_id": "generation_banner_001"})
    repository = GenerationRunRepository(FakeConnection(cursor))

    assert repository.heartbeat(
        generation_id="generation_banner_001",
        worker_id="decision-task-1",
        lease_token=LEASE_TOKEN,
        lease_seconds=180,
    )

    query, params = cursor.executed[0]
    assert "worker_id = %(worker_id)s" in query
    assert "lease_token = %(lease_token)s" in query
    assert "lease_expires_at > now()" in query
    assert params is not None
    assert params["lease_seconds"] == 180


def test_generation_run_repository_recovers_expired_leases_with_retry_budget() -> None:
    rows = [
        {"generation_id": "generation_retry", "status": "requested"},
        {"generation_id": "generation_failed", "status": "failed"},
    ]
    cursor = FakeCursor(fetchall_result=rows)
    repository = GenerationRunRepository(FakeConnection(cursor))

    result = repository.recover_expired(
        max_retries=3,
        retry_backoff_seconds=(60, 300, 900),
        limit=25,
    )

    assert result == rows
    query, params = cursor.executed[0]
    assert "lease_expires_at <= now()" in query
    assert "FOR UPDATE SKIP LOCKED" in query
    assert "retry_count + 1" in query
    assert "generation_lease_expired" in query
    assert "worker_id = NULL" in query
    assert params == {
        "max_retries": 3,
        "retry_backoff_seconds": [60, 300, 900],
        "limit": 25,
    }


def test_generation_run_repository_schedules_retry_with_fencing() -> None:
    cursor = FakeCursor(fetchone_result={"generation_id": "generation_banner_001"})
    repository = GenerationRunRepository(FakeConnection(cursor))
    retry_at = datetime(2026, 7, 14, 7, 1, tzinfo=UTC)

    assert repository.schedule_retry_fenced(
        generation_id="generation_banner_001",
        worker_id="decision-task-1",
        lease_token=LEASE_TOKEN,
        next_retry_at=retry_at,
        error_code="provider_rate_limited",
        error_message="provider temporarily unavailable",
    )

    query, params = cursor.executed[0]
    assert "status = 'requested'" in query
    assert "retry_count = retry_count + 1" in query
    assert "lease_expires_at > now()" in query
    assert "lease_token = NULL" in query
    assert params is not None
    assert params["next_retry_at"] == retry_at


def test_generation_run_repository_marks_terminal_failure_with_fencing() -> None:
    cursor = FakeCursor(fetchone_result={"generation_id": "generation_banner_001"})
    repository = GenerationRunRepository(FakeConnection(cursor))

    assert repository.mark_failed_fenced(
        generation_id="generation_banner_001",
        worker_id="decision-task-1",
        lease_token=LEASE_TOKEN,
        error_code="invalid_generation_input",
        error_message="required target segment is missing",
    )

    query, _params = cursor.executed[0]
    assert "status = 'failed'" in query
    assert "finished_at = now()" in query
    assert "next_retry_at = NULL" in query
    assert "lease_expires_at > now()" in query


def test_generation_run_repository_completes_only_after_strict_readiness() -> None:
    completed_row = {
        "generation_id": "generation_banner_001",
        "status": "completed",
    }
    cursor = FakeCursor(fetchone_result=completed_row)
    repository = GenerationRunRepository(FakeConnection(cursor))

    result = repository.complete_if_ready_fenced(
        generation_id="generation_banner_001",
        worker_id="decision-task-1",
        lease_token=LEASE_TOKEN,
        output_json={"status": "completed"},
        generation_report_json={"candidate_count": 2},
    )

    assert result == completed_row
    query, params = cursor.executed[0]
    assert "FOR UPDATE" in query
    assert "lease_expires_at > now()" in query
    assert "generation.request.v1" in query
    assert "count(DISTINCT segment_id)" in query
    assert "<> run.content_option_count" in query
    assert "candidate.channel = 'sms'" in query
    assert "candidate.channel IN ('email', 'onsite_banner')" in query
    assert "candidate.image_generation_status = 'completed'" in query
    assert "candidate.artifact_status = 'published'" in query
    assert "candidate.artifact_published_at <= now()" in query
    assert "status = 'completed'" in query
    assert params is not None
    assert params["output_json"].obj == {"status": "completed"}


def test_generation_run_repository_lists_ids_by_promotion() -> None:
    cursor = FakeCursor(
        fetchall_result=[
            {"generation_id": "generation_banner_001"},
            {"generation_id": "generation_banner_001_run_2"},
        ]
    )
    repository = GenerationRunRepository(FakeConnection(cursor))

    result = repository.list_ids_by_promotion("promo_banner_001")

    assert result == ["generation_banner_001", "generation_banner_001_run_2"]
    query, params = cursor.executed[0]
    assert "FROM generation_runs" in query
    assert "promotion_id = %(promotion_id)s" in query
    assert params == {"promotion_id": "promo_banner_001"}


def test_content_candidate_repository_create_executes_insert() -> None:
    cursor = FakeCursor(fetchone_result={"content_id": "content_banner_001"})
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.create(
        ContentCandidateRecord(
            content_id="content_banner_001",
            content_option_id="banner_option_001",
            generation_id="generation_banner_001",
            analysis_id="analysis_banner_001",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            segment_id="seg_repeat_hotel_no_booking",
            channel=ContentChannel.ONSITE_BANNER,
            title="Book this weekend's rooms",
            body="Compare refundable summer offers before rooms run out.",
            cta="View hotel deals",
            image_prompt="bright modern hotel room, summer travel banner",
            image_url="https://gen-ai.asset.dev.loop-ad.org/generated-assets/content_banner_001.png",
            landing_url="https://demo-stay.example.com/summer",
            generation_prompt="Create an onsite banner.",
            data_evidence_json={"segment_id": "seg_repeat_hotel_no_booking"},
            metadata_json={"content_id": "content_banner_001"},
        )
    )

    assert result == {"content_id": "content_banner_001"}
    query, params = cursor.executed[0]
    assert "INSERT INTO content_candidates" in query
    assert params is not None
    assert params["content_id"] == "content_banner_001"
    assert params["channel"] == "onsite_banner"
    assert params["image_url"] == (
        "https://gen-ai.asset.dev.loop-ad.org/generated-assets/content_banner_001.png"
    )
    assert params["data_evidence_json"].obj == {
        "segment_id": "seg_repeat_hotel_no_booking"
    }
    assert params["metadata_json"].obj == {"content_id": "content_banner_001"}
    assert "creative_format" in query
    assert "artifact_public_url" in query
    assert params["creative_format"] is None
    assert params["artifact_public_url"] is None


def test_content_candidate_repository_upserts_all_contract_fields_with_fencing() -> None:
    stored_row = {
        "content_id": "content_banner_001",
        "artifact_status": "published",
    }
    cursor = FakeCursor(fetchone_result=stored_row)
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.upsert_fenced(
        content_candidate_record(),
        worker_id="decision-task-1",
        lease_token=LEASE_TOKEN,
    )

    assert result == stored_row
    query, params = cursor.executed[0]
    assert "FROM generation_runs" in query
    assert "status = 'running'" in query
    assert "lease_expires_at > now()" in query
    assert "FOR UPDATE" in query
    assert "ON CONFLICT (generation_id, segment_id, content_option_id)" in query
    assert "content_candidates.content_id = EXCLUDED.content_id" in query
    assert "WHEN %(artifact_status)s::varchar = 'published' THEN now()" in query
    assert "artifact_sha256 = EXCLUDED.artifact_sha256" in query
    assert params is not None
    assert params["worker_id"] == "decision-task-1"
    assert params["creative_format"] == "banner_html"
    assert params["image_generation_status"] == "completed"
    assert params["artifact_status"] == "published"
    assert params["artifact_public_url"] == (
        "https://cdn.example.test/content_banner_001.html"
    )
    assert params["artifact_sha256"] == "b" * 64


def test_content_candidate_repository_updates_image_url() -> None:
    cursor = FakeCursor(fetchone_result={"content_id": "content_banner_001"})
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.update_image_url(
        content_id="content_banner_001",
        image_url="https://gen-ai.asset.dev.loop-ad.org/generated/content_banner_001.png",
    )

    assert result == {"content_id": "content_banner_001"}
    query, params = cursor.executed[0]
    assert "UPDATE content_candidates" in query
    assert "image_generation_status" in query
    assert params == {
        "content_id": "content_banner_001",
        "image_url": (
            "https://gen-ai.asset.dev.loop-ad.org/generated/content_banner_001.png"
        ),
    }


def test_content_candidate_repository_marks_image_generation_failed() -> None:
    cursor = FakeCursor(fetchone_result={"content_id": "content_banner_001"})
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.mark_image_generation_failed(
        content_id="content_banner_001",
        error_code="image_generation_failed",
    )

    assert result == {"content_id": "content_banner_001"}
    query, params = cursor.executed[0]
    assert "UPDATE content_candidates" in query
    assert "image_generation_status" in query
    assert params == {
        "content_id": "content_banner_001",
        "error_code": "image_generation_failed",
    }


def test_content_candidate_repository_lists_by_generation() -> None:
    rows = [{"content_id": "content_banner_001"}]
    cursor = FakeCursor(fetchall_result=rows)
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.list_by_generation("generation_banner_001")

    assert result == rows
    query, params = cursor.executed[0]
    assert "FROM content_candidates" in query
    assert "ORDER BY segment_id, content_option_id, content_id" in query
    assert "FOR UPDATE" not in query
    assert params == {"generation_id": "generation_banner_001"}


def test_content_candidate_repository_locks_entire_generation_in_stable_order() -> None:
    rows = [{"content_id": "content_banner_001"}]
    cursor = FakeCursor(fetchall_result=rows)
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.list_by_generation_for_update("generation_banner_001")

    assert result == rows
    query, params = cursor.executed[0]
    assert "FROM content_candidates" in query
    assert "ORDER BY segment_id, content_option_id, content_id" in query
    assert "FOR UPDATE" in query
    assert params == {"generation_id": "generation_banner_001"}


def test_generation_input_repository_reads_confirmed_target_segments() -> None:
    cursor = FakeCursor(
        fetchone_result={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "channel": "onsite_banner",
            "goal_metric": "booking_conversion_rate",
            "goal_target_value": "0.030000",
            "goal_basis": "all_segments",
            "message_brief": "Drive hotel booking conversion.",
            "landing_url": "https://demo-stay.example.com/summer",
        },
        fetchall_result=[
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_ai_repeat_hotel",
                "segment_name": "AI suggested repeat hotel viewers",
                "content_brief_json": {
                    "message_direction": "Highlight refundable hotel stays.",
                    "keywords": ["refundable stays", "hotel deals"],
                },
                "data_evidence_json": {
                    "source": "ai_suggested",
                    "booking_conversion_rate": "0.018",
                    "top_common_features": ["same_hotel_repeat_view"],
                    "sample_ratio": "0.018000",
                },
                "segment_vector_id": "segvec_ai_repeat_hotel_v1",
                "estimated_size": 1342,
                "priority": "high",
                "status": "approved",
                "segment_source": "ai_suggested",
                "query_preview_id": "seg_query_preview_001",
                "natural_language_query": "repeat hotel viewers without booking",
                "generated_sql": "SELECT user_id FROM hotel_detail_events",
                "segment_sample_size": 1342,
                "segment_sample_ratio": "0.018000",
            },
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_manual_family_trip",
                "segment_name": "Manual family trip segment",
                "content_brief_json": {
                    "message_direction": "Promote family-friendly rooms.",
                    "keywords": ["family rooms"],
                },
                "data_evidence_json": {"source": "manual_rule"},
                "segment_vector_id": "segvec_manual_family_trip_v1",
                "estimated_size": 820,
                "priority": "medium",
                "status": "content_ready",
                "segment_source": "manual_rule",
                "query_preview_id": None,
                "natural_language_query": "family hotel trip planners",
                "generated_sql": None,
                "segment_sample_size": 820,
                "segment_sample_ratio": "0.011000",
            },
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_running_experiment",
                "segment_name": "Running experiment segment",
                "content_brief_json": {
                    "message_direction": "Do not regenerate running segments.",
                    "keywords": ["running experiment"],
                },
                "data_evidence_json": {"source": "ai_suggested"},
                "segment_vector_id": "segvec_running_experiment_v1",
                "estimated_size": 910,
                "priority": "medium",
                "status": "running",
                "segment_source": "ai_suggested",
                "query_preview_id": None,
                "natural_language_query": "running experiment segment",
                "generated_sql": None,
                "segment_sample_size": 910,
                "segment_sample_ratio": "0.012000",
            },
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_planned_not_selected",
                "segment_name": "Planned but not selected segment",
                "content_brief_json": {
                    "message_direction": "Do not generate this planned segment.",
                    "keywords": ["planned only"],
                },
                "data_evidence_json": {"source": "ai_suggested"},
                "segment_vector_id": "segvec_planned_not_selected_v1",
                "estimated_size": 640,
                "priority": "low",
                "status": "planned",
                "segment_source": "ai_suggested",
                "query_preview_id": None,
                "natural_language_query": "planned segment candidate",
                "generated_sql": None,
                "segment_sample_size": 640,
                "segment_sample_ratio": "0.009000",
            },
        ],
    )
    repository = GenerationInputRepository(FakeConnection(cursor))
    request = GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=2,
        operator_instruction=None,
    )

    promotion = repository.get_promotion_input(request)
    target_segments = repository.list_target_segment_inputs(request)

    assert promotion is not None
    assert promotion.channel == ContentChannel.ONSITE_BANNER
    assert promotion.landing_url == "https://demo-stay.example.com/summer"
    assert [segment.segment_id for segment in target_segments] == [
        "seg_ai_repeat_hotel",
    ]
    assert [segment.status for segment in target_segments] == ["approved"]
    assert target_segments[0].source == "ai_suggested"
    assert target_segments[0].content_brief_json["booking_conversion_rate"] == "0.018"
    assert target_segments[0].natural_language_query == (
        "repeat hotel viewers without booking"
    )
    assert target_segments[0].generated_sql == (
        "SELECT user_id FROM hotel_detail_events"
    )
    assert target_segments[0].query_preview_id == "seg_query_preview_001"

    executed_sql = "\n".join(query for query, _params in cursor.executed)
    assert "FROM promotion_target_segments" in executed_sql
    assert "pts.status" in executed_sql
    assert "pts.status = 'approved'" in executed_sql
    assert "LEFT JOIN segment_definitions" in executed_sql
    assert "promotion_segment_suggestions" not in executed_sql
    target_segment_params = next(
        params
        for query, params in cursor.executed
        if "FROM promotion_target_segments" in query
    )
    assert target_segment_params == {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
    }


def test_generation_input_repository_limits_confirmed_targets_to_requested_segment_ids() -> None:
    cursor = FakeCursor(
        fetchall_result=[
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_family_trip",
                "segment_name": "Family trip planners",
                "content_brief_json": {"message_direction": "Family stays."},
                "data_evidence_json": {},
                "segment_vector_id": "segvec_family_trip_v1",
                "estimated_size": 1200,
                "priority": "high",
                "status": "approved",
                "segment_source": "manual_rule",
                "query_preview_id": None,
                "natural_language_query": None,
                "generated_sql": None,
                "segment_sample_size": 1200,
                "segment_sample_ratio": "0.010000",
            },
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_mobile_user",
                "segment_name": "Mobile hotel users",
                "content_brief_json": {"message_direction": "Mobile booking."},
                "data_evidence_json": {},
                "segment_vector_id": "segvec_mobile_user_v1",
                "estimated_size": 900,
                "priority": "medium",
                "status": "approved",
                "segment_source": "manual_rule",
                "query_preview_id": None,
                "natural_language_query": None,
                "generated_sql": None,
                "segment_sample_size": 900,
                "segment_sample_ratio": "0.008000",
            },
        ]
    )
    repository = GenerationInputRepository(FakeConnection(cursor))
    request = GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        segment_ids=["seg_mobile_user"],
        content_option_count=1,
        operator_instruction=None,
    )

    target_segments = repository.list_target_segment_inputs(request)

    assert [segment.segment_id for segment in target_segments] == ["seg_mobile_user"]
    query, params = next(
        (query, params)
        for query, params in cursor.executed
        if "FROM promotion_target_segments" in query
    )
    assert "pts.segment_id = ANY(%(segment_ids)s)" in query
    assert params is not None
    assert params["segment_ids"] == ["seg_mobile_user"]


def test_generation_input_repository_focus_read_bypasses_confirmed_status_filter() -> None:
    cursor = FakeCursor(
        fetchall_result=[
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_failed_planned",
                "segment_name": "Failed planned focus segment",
                "content_brief_json": {
                    "message_direction": "Refine the failed hotel message.",
                    "keywords": ["hotel retry"],
                },
                "data_evidence_json": {"source": "next_loop"},
                "segment_vector_id": "segvec_failed_planned_v1",
                "estimated_size": 320,
                "priority": "high",
                "status": "planned",
                "segment_source": "ai_suggested",
                "query_preview_id": None,
                "natural_language_query": "failed segment from previous loop",
                "generated_sql": None,
                "segment_sample_size": 320,
                "segment_sample_ratio": "0.004000",
            },
        ],
    )
    repository = GenerationInputRepository(FakeConnection(cursor))
    request = GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=1,
        operator_instruction=None,
    )

    target_segments = repository.list_focus_target_segment_inputs(request)

    assert [segment.segment_id for segment in target_segments] == [
        "seg_failed_planned"
    ]
    assert target_segments[0].status == "planned"

    executed_sql = "\n".join(query for query, _params in cursor.executed)
    assert "FROM promotion_target_segments" in executed_sql
    assert "pts.status = 'approved'" not in executed_sql
