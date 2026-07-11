from __future__ import annotations

import pytest

from app.content_brief import normalize_content_brief


@pytest.mark.parametrize(
    ("audience_evidence", "level", "missing_sections"),
    (
        (
            {
                "primary_signals": ["recent_destination_search"],
                "score_components": {"final_score": 0.8},
            },
            "evidence_ready",
            [],
        ),
        (
            {"primary_signals": ["recent_destination_search"]},
            "partial",
            ["score_components"],
        ),
        (
            {"score_components": {"final_score": 0.8}},
            "partial",
            ["primary_signals"],
        ),
        (
            {
                "promotion_vector_basis": {"goal_metric": "inflow_rate"},
                "promotion_matched_features": ["destination affinity"],
            },
            "fallback_only",
            ["primary_signals", "score_components"],
        ),
        ({}, "fallback_only", ["primary_signals", "score_components"]),
        (
            {
                "primary_signals": "not-a-sequence",
                "score_components": ["not-a-mapping"],
                "promotion_vector_basis": {},
                "promotion_matched_features": [],
            },
            "fallback_only",
            ["primary_signals", "score_components"],
        ),
    ),
)
def test_normalize_content_brief_recomputes_readiness_from_supported_evidence(
    audience_evidence: dict[str, object],
    level: str,
    missing_sections: list[str],
) -> None:
    normalized = normalize_content_brief(
        {
            "schema_version": "content_brief.v2",
            "readiness": {"level": "evidence_ready"},
            "audience_evidence": audience_evidence,
        }
    )

    assert normalized.readiness["level"] == level
    assert normalized.readiness["missing_sections"] == missing_sections


def test_normalize_content_brief_keeps_legacy_briefs_fallback_only() -> None:
    normalized = normalize_content_brief(
        {
            "message_direction": "Highlight refundable hotel stays.",
            "keywords": ["refundable"],
        }
    )

    assert normalized.schema_version == "content_brief.v1"
    assert normalized.readiness["level"] == "fallback_only"
    assert normalized.fallback_guidance_present is True


def test_normalize_content_brief_does_not_count_default_fallback_as_present() -> None:
    normalized = normalize_content_brief({"schema_version": "content_brief.v2"})

    assert normalized.message_direction
    assert normalized.fallback_guidance_present is False


def test_normalize_content_brief_preserves_operator_instruction() -> None:
    normalized = normalize_content_brief(
        {
            "schema_version": "content_brief.v2",
            "operator_instruction": "  Emphasize booking confidence.  ",
        }
    )

    assert normalized.operator_instruction == "Emphasize booking confidence."
