from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.generation.prompt_builder import GenerationPromptInput, PromptBuildResult
from app.generation.schemas import GenerationStatus


GENERATION_REPORT_VERSION = "dec-c4.v1"
MAX_SQL_SUMMARY_LENGTH = 240


@dataclass(frozen=True)
class CandidateGenerationReport:
    reason_summary: str
    message_strategy: str
    data_evidence_json: dict[str, Any]
    metadata_json: dict[str, Any]


class GenerationReportBuilder:
    def build_candidate_report(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        content_id: str,
        content_option_id: str,
        content_generator_version: str,
        content_values: Mapping[str, str | None],
        status: str,
    ) -> CandidateGenerationReport:
        target_segment = prompt_input.target_segment
        reason_summary = prompt_result.reason_summary
        message_strategy = prompt_result.message_strategy
        generated_sql_summary = _generated_sql_summary(target_segment.generated_sql)
        data_evidence = _data_evidence(prompt_input)

        metadata = {
            "report_version": GENERATION_REPORT_VERSION,
            "prompt_builder_version": prompt_result.metadata_json.get(
                "prompt_builder_version"
            ),
            "content_generator_version": content_generator_version,
            "reason_summary": reason_summary,
            "data_evidence": data_evidence,
            "message_strategy": message_strategy,
            "operator_instruction": prompt_input.request.operator_instruction,
            "source_segment_definition_id": target_segment.segment_id,
            "source_query_preview_id": target_segment.query_preview_id,
            "generated_sql_summary": generated_sql_summary,
            "content_id": content_id,
            "content_option_id": content_option_id,
            "segment_id": target_segment.segment_id,
            "segment_name": target_segment.segment_name,
            "channel": prompt_input.promotion.channel.value,
            **content_values,
            "status": status,
        }

        return CandidateGenerationReport(
            reason_summary=reason_summary,
            message_strategy=message_strategy,
            data_evidence_json=data_evidence,
            metadata_json=metadata,
        )

    def build_run_output(
        self,
        *,
        status: GenerationStatus,
        target_segment_count: int,
        content_candidate_metadata: Sequence[Mapping[str, Any]],
        error_code: str | None = None,
    ) -> dict[str, Any]:
        content_candidate_ids = [
            str(metadata["content_id"])
            for metadata in content_candidate_metadata
            if metadata.get("content_id")
        ]
        output_json: dict[str, Any] = {
            "report_version": GENERATION_REPORT_VERSION,
            "content_candidate_ids": content_candidate_ids,
            "generation_summary": {
                "status": status.value,
                "content_candidate_count": len(content_candidate_ids),
                "target_segment_count": target_segment_count,
            },
            "segment_summaries": _segment_summaries(content_candidate_metadata),
            "content_report_summaries": [
                _content_report_summary(metadata)
                for metadata in content_candidate_metadata
            ],
        }
        if error_code:
            output_json["error_code"] = error_code
        return output_json


def _data_evidence(prompt_input: GenerationPromptInput) -> dict[str, Any]:
    promotion = prompt_input.promotion
    target_segment = prompt_input.target_segment
    content_brief = target_segment.content_brief_json
    keywords = _string_list(content_brief.get("keywords"))
    top_common_features = _string_list(content_brief.get("top_common_features"))
    goal_target_value = _optional_float(promotion.goal_target_value)

    return {
        "analysis_id": prompt_input.request.analysis_id,
        "promotion_id": promotion.promotion_id,
        "segment_id": target_segment.segment_id,
        "segment_name": target_segment.segment_name,
        "segment_vector_id": target_segment.segment_vector_id,
        "sample_size": target_segment.estimated_size,
        "sample_ratio": _optional_float(target_segment.sample_ratio),
        "booking_conversion_rate": _optional_float(
            content_brief.get("booking_conversion_rate")
        ),
        "comparison_group_conversion_rate": _optional_float(
            content_brief.get("comparison_group_conversion_rate")
        ),
        "top_common_features": top_common_features or keywords,
        "priority": target_segment.priority,
        "target_segment_status": target_segment.status,
        "goal_metric": promotion.goal_metric,
        "goal_basis": promotion.goal_basis,
        "goal_target_value": goal_target_value
        if goal_target_value is not None
        else promotion.goal_target_value,
        "content_brief_keywords": keywords,
    }


def _segment_summaries(
    content_candidate_metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for metadata in content_candidate_metadata:
        segment_id = str(metadata.get("segment_id", ""))
        if not segment_id:
            continue

        summary = summaries.setdefault(
            segment_id,
            {
                "segment_id": segment_id,
                "segment_name": metadata.get("segment_name"),
                "content_candidate_ids": [],
                "content_option_ids": [],
                "reason_summary": metadata.get("reason_summary"),
                "data_evidence": metadata.get("data_evidence", {}),
                "message_strategy": metadata.get("message_strategy"),
                "operator_instruction": metadata.get("operator_instruction"),
                "source_segment_definition_id": metadata.get(
                    "source_segment_definition_id"
                ),
                "source_query_preview_id": metadata.get("source_query_preview_id"),
                "generated_sql_summary": metadata.get("generated_sql_summary"),
            },
        )
        summary["content_candidate_ids"].append(metadata.get("content_id"))
        summary["content_option_ids"].append(metadata.get("content_option_id"))

    return list(summaries.values())


def _content_report_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "content_id": metadata.get("content_id"),
        "content_option_id": metadata.get("content_option_id"),
        "segment_id": metadata.get("segment_id"),
        "channel": metadata.get("channel"),
        "reason_summary": metadata.get("reason_summary"),
        "data_evidence": metadata.get("data_evidence", {}),
        "message_strategy": metadata.get("message_strategy"),
        "operator_instruction": metadata.get("operator_instruction"),
        "source_query_preview_id": metadata.get("source_query_preview_id"),
        "generated_sql_summary": metadata.get("generated_sql_summary"),
    }


def _generated_sql_summary(value: str | None) -> str | None:
    if not value:
        return None
    compacted = " ".join(value.split())
    if len(compacted) <= MAX_SQL_SUMMARY_LENGTH:
        return compacted
    return compacted[: MAX_SQL_SUMMARY_LENGTH - 1].rstrip() + "."


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
