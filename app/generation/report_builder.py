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
        data_evidence = dict(prompt_result.data_evidence_json)

        metadata = {
            "report_version": GENERATION_REPORT_VERSION,
            "prompt_builder_version": prompt_result.metadata_json.get(
                "prompt_builder_version"
            ),
            "content_generator_version": content_generator_version,
            "reason_summary": reason_summary,
            "data_evidence": data_evidence,
            "message_strategy": message_strategy,
            "content_brief_schema_version": data_evidence.get(
                "content_brief_schema_version"
            ),
            "content_brief_readiness": data_evidence.get(
                "content_brief_readiness"
            ),
            "fallback_guidance_present": prompt_result.fallback_guidance_present,
            "fallback_guidance_used": prompt_result.fallback_guidance_used,
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
