from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

from app.audience_contract import SEGMENT_AUDIENCE_CONTRACT
from app.analysis.repositories import (
    PromotionRecord,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
)
from app.config import Settings
from app.generation.adapters import (
    OPENAI_RESPONSES_URL,
    JsonTransport,
    _parse_output_json,
    _post_json,
)
from app.logging import duration_ms, log, log_context_scope


REPORT_GENERATOR_VERSION = "dec.segment-report.v3"

FORBIDDEN_REPORT_TERMS = (
    "벡터",
    "군집",
    "클러스터",
    "centroid",
    "유사도",
    "cosine",
    "Rank",
    "랭크",
)


class SegmentSuggestionReportGenerator(Protocol):
    def generate_report(
        self,
        report_input: "SegmentSuggestionReportInput",
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class SegmentSuggestionReportInput:
    promotion: PromotionRecord
    segment: SegmentDefinitionRecord
    target_segment: PromotionTargetSegmentWrite
    display_copy: Mapping[str, Any]
    primary_signals: Sequence[Mapping[str, str]]
    score_json: Mapping[str, Any]
    reason_json: Mapping[str, Any]


class DeterministicSegmentSuggestionReportGenerator:
    @log_context_scope
    def generate_report(
        self,
        report_input: SegmentSuggestionReportInput,
    ) -> dict[str, Any]:
        return _fallback_report(report_input=report_input, source="deterministic")


class OpenAISegmentSuggestionReportGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = OPENAI_RESPONSES_URL,
        timeout_seconds: float = 20.0,
        fallback_generator: SegmentSuggestionReportGenerator | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._fallback_generator = (
            fallback_generator or DeterministicSegmentSuggestionReportGenerator()
        )
        self._transport = transport or _post_json

    def generate_report(
        self,
        report_input: SegmentSuggestionReportInput,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _system_instruction(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _user_instruction(report_input),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "segment_suggestion_report",
                    "strict": True,
                    "schema": _report_schema(),
                }
            },
            "max_output_tokens": 900,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started_at = perf_counter()
        log.assign_context(
            {
                "promotionId": report_input.promotion.promotion_id,
                "segmentId": report_input.segment.segment_id,
                "provider": "openai",
                "model": self._model,
            }
        )
        log.info(
            "provider_request_prepared",
            {
                "providerOperation": "segment_suggestion_report",
                "endpoint": self._endpoint,
            },
        )
        try:
            response_payload = self._transport(
                self._endpoint,
                headers,
                payload,
                self._timeout_seconds,
            )
            report = dict(_parse_output_json(response_payload))
        except Exception as exc:
            log.warn(
                "provider_request_failed",
                {
                    "providerOperation": "segment_suggestion_report",
                    "endpoint": self._endpoint,
                    "err": exc,
                    "durationMs": duration_ms(started_at),
                    "fallback": "deterministic",
                },
            )
            return self._fallback_generator.generate_report(report_input)

        log.info(
            "provider_request_completed",
            {
                "providerOperation": "segment_suggestion_report",
                "endpoint": self._endpoint,
                "durationMs": duration_ms(started_at),
            },
        )
        return _sanitize_report(report, report_input=report_input, source="openai")


def build_segment_suggestion_report_generator(
    settings: Settings,
) -> SegmentSuggestionReportGenerator:
    if settings.env == "test" or _is_placeholder_api_key(settings.openai_api_key):
        return DeterministicSegmentSuggestionReportGenerator()
    return OpenAISegmentSuggestionReportGenerator(
        api_key=settings.openai_api_key,
        model=settings.openai_content_model,
    )


def _is_placeholder_api_key(api_key: str) -> bool:
    normalized = api_key.strip().lower()
    return (
        not normalized
        or normalized.startswith("replace-with")
        or normalized in {"changeme", "placeholder"}
    )


def _system_instruction() -> str:
    return (
        "당신은 숙박 예약 플랫폼의 기획자가 읽을 수 있는 마케팅 리포트를 작성합니다. "
        "한국어로만 답하고, 비전문가가 이해하기 쉬운 표현을 사용하세요. "
        "고객 행동을 설명할 때는 전문적이고 구체적인 표현을 사용하고, 쳐다보다 같은 구어체나 모호한 표현은 쓰지 마세요. "
        "벡터, 군집, 클러스터, centroid, cosine, 유사도 같은 기술 용어는 절대 쓰지 마세요. "
        "데이터로 확인된 사실만 말하고, 과장하지 말고, 실행 가능한 마케팅 판단을 돕는 문장으로 작성하세요."
    )


def _user_instruction(report_input: SegmentSuggestionReportInput) -> str:
    promotion = report_input.promotion
    display_copy = report_input.display_copy
    evidence = report_input.target_segment.data_evidence_json
    signal_chips = display_copy.get("signal_chips", [])
    performance_estimate = _mapping_value(display_copy.get("performance_estimate"))
    audience_count_label = (
        "실험 대상 사용자 수"
        if _is_v2_final_audience(report_input)
        else "표본 수"
    )
    return "\n".join(
        [
            "아래 세그먼트 추천 결과를 대시보드 리포트로 정리하세요.",
            "",
            f"- 프로모션 채널: {promotion.channel}",
            f"- 목표 지표: {promotion.goal_metric}",
            f"- 목표값: {_format_goal_value(promotion.goal_target_value)}",
            f"- 랜딩 URL: {promotion.landing_url or '-'}",
            f"- 프로모션 설명: {promotion.message_brief or '-'}",
            f"- 추천 고객군 이름: {display_copy.get('title', report_input.target_segment.segment_name)}",
            f"- 추천 전략: {display_copy.get('strategy_role', display_copy.get('rank_role', '-'))}",
            f"- 추천 분류: {display_copy.get('recommendation_tier_label', '주요 추천')}",
            f"- 추천 분류 근거: {display_copy.get('recommendation_tier_reason', '-')}",
            f"- 분석 대상 요약: {display_copy.get('audience_summary', '-')}",
            f"- 주요 행동 신호: {', '.join(map(str, signal_chips)) or '-'}",
            f"- {audience_count_label}: {evidence.get('sample_size', report_input.target_segment.estimated_size)}",
            f"- 전체 분석 대상 수: {evidence.get('total_eligible_user_count', '-')}",
            f"- 예상 목표 성과: {performance_estimate.get('label', '-')} {performance_estimate.get('formatted', '-')}",
            f"- 예상 기준: {performance_estimate.get('window_label', performance_estimate.get('basis_label', '-'))}",
            f"- 예측 신뢰도: {performance_estimate.get('confidence_label', '-')}",
            f"- 예측 신뢰도 근거: {performance_estimate.get('confidence_reason', '-')}",
            f"- 후보의 강점: {display_copy.get('strength_summary', '-')}",
            f"- 선택 시 고려사항: {display_copy.get('tradeoff_summary', '-')}",
            "",
            "JSON 필드 설명:",
            "- title: 카드 제목으로 쓸 짧은 고객군 이름",
            "- summary: 이 고객군이 어떤 사람들인지 한 문장",
            "- promotion_interpretation: 프로모션 조건을 사용자가 이해할 수 있게 해석한 문장 2개",
            "- why_recommended: 추천 이유 2~3개",
            "- evidence: 판단 근거 2~3개",
            "- candidate_strengths: 이 후보 자체의 강점 1~2개",
            "- selection_considerations: 이 후보를 선택할 때 고려할 점 1~2개",
            "- action_hint: 이 프로모션에서 어떻게 활용하면 좋은지",
            "- caution: 제공된 예측 신뢰도 근거만 사용하고 표본 크기를 임의로 평가하지 말 것",
            "- confidence_label: high, medium, low 중 하나",
        ],
    )


def _report_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "title",
            "summary",
            "promotion_interpretation",
            "why_recommended",
            "evidence",
            "candidate_strengths",
            "selection_considerations",
            "action_hint",
            "caution",
            "confidence_label",
        ],
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "promotion_interpretation": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            },
            "why_recommended": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            },
            "candidate_strengths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
            "selection_considerations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
            "action_hint": {"type": "string"},
            "caution": {"type": "string"},
            "confidence_label": {"type": "string", "enum": ["high", "medium", "low"]},
        },
    }


def _sanitize_report(
    report: Mapping[str, Any],
    *,
    report_input: SegmentSuggestionReportInput,
    source: str,
) -> dict[str, Any]:
    fallback = _fallback_report(report_input=report_input, source=source)
    computed_strength = _safe_text(
        report_input.display_copy.get("strength_summary")
    )
    computed_tradeoff = _safe_text(
        report_input.display_copy.get("tradeoff_summary")
    )
    generated_strengths = _safe_text_list(report.get("candidate_strengths"))
    generated_considerations = _safe_text_list(
        report.get("selection_considerations")
    )
    candidate_strengths = _merge_report_facts(
        computed_strength,
        generated_strengths,
        fallback["candidate_strengths"],
    )
    selection_considerations = _merge_report_facts(
        computed_tradeoff,
        generated_considerations,
        fallback["selection_considerations"],
    )
    performance_estimate = _mapping_value(
        report_input.display_copy.get("performance_estimate")
    )
    computed_confidence = _confidence_label(
        performance_estimate.get("confidence_label")
    )
    if (
        _is_v2_final_audience(report_input)
        and _report_sample_size(report_input)
        < report_input.promotion.min_sample_size
    ):
        computed_confidence = "low"
    sanitized = {
        "version": REPORT_GENERATOR_VERSION,
        "source": source,
        "title": _safe_text(report.get("title")) or fallback["title"],
        "summary": _safe_text(report.get("summary")) or fallback["summary"],
        "promotion_interpretation": _safe_text_list(
            report.get("promotion_interpretation")
        )
        or fallback["promotion_interpretation"],
        "why_recommended": _safe_text_list(report.get("why_recommended"))
        or fallback["why_recommended"],
        "evidence": _safe_text_list(report.get("evidence")) or fallback["evidence"],
        "candidate_strengths": candidate_strengths,
        "selection_considerations": selection_considerations,
        "action_hint": _safe_text(report.get("action_hint"))
        or fallback["action_hint"],
        "caution": _verified_caution(report_input),
        "confidence_label": computed_confidence
        or _confidence_label(report.get("confidence_label"))
        or fallback["confidence_label"],
    }
    if _contains_forbidden_terms(sanitized):
        return _fallback_report(report_input=report_input, source="fallback")
    if _is_v2_final_audience(report_input):
        return _replace_legacy_sample_wording(sanitized)
    return sanitized


def _fallback_report(
    *,
    report_input: SegmentSuggestionReportInput,
    source: str,
) -> dict[str, Any]:
    display_copy = report_input.display_copy
    evidence = report_input.target_segment.data_evidence_json
    signal_chips = [
        str(chip)
        for chip in display_copy.get("signal_chips", [])
        if str(chip).strip()
    ]
    audience_summary = str(display_copy.get("audience_summary", "")).strip()
    title = str(display_copy.get("title", "")).strip()
    if not title:
        title = report_input.target_segment.segment_name
    evidence_items = [
        audience_summary or _audience_summary(report_input),
        _signals_sentence(signal_chips),
    ]
    message_brief = (report_input.promotion.message_brief or "").strip()
    if message_brief:
        evidence_items.append(f"프로모션 설명에 맞춰 '{message_brief[:80]}' 흐름을 반영했습니다.")
    promotion_interpretation = [
        _promotion_goal_sentence(report_input.promotion),
        _promotion_message_sentence(report_input.promotion),
    ]
    strength_summary = str(display_copy.get("strength_summary", "")).strip()
    tradeoff_summary = str(display_copy.get("tradeoff_summary", "")).strip()
    performance_estimate = _mapping_value(display_copy.get("performance_estimate"))
    performance_confidence = _confidence_label(
        performance_estimate.get("confidence_label")
    )
    sample_size = _report_sample_size(report_input)
    fallback_confidence = _fallback_confidence_label(
        sample_size=sample_size,
        min_sample_size=report_input.promotion.min_sample_size,
    )
    if (
        _is_v2_final_audience(report_input)
        and sample_size < report_input.promotion.min_sample_size
    ):
        confidence_label = "low"
    else:
        confidence_label = performance_confidence or fallback_confidence
    report = {
        "version": REPORT_GENERATOR_VERSION,
        "source": source,
        "title": title,
        "summary": "이번 프로모션 목표와 맞는 행동을 보인 고객군입니다.",
        "promotion_interpretation": promotion_interpretation,
        "why_recommended": [
            str(display_copy.get("reason", "")).strip()
            or "예약 전환에 가까운 행동이 확인되었습니다.",
            _signals_sentence(signal_chips),
        ],
        "evidence": evidence_items[:3],
        "candidate_strengths": [
            strength_summary
            or "프로모션 목표와 연결되는 행동 조건이 확인된 고객군입니다."
        ],
        "selection_considerations": [
            tradeoff_summary
            or (
                "예상 성과와 실험 대상 사용자 규모를 함께 확인해 선택하세요."
                if _is_v2_final_audience(report_input)
                else "예상 성과와 대표 표본 규모를 함께 확인해 선택하세요."
            )
        ],
        "action_hint": str(display_copy.get("action_hint", "")).strip()
        or "이 고객군을 우선 타겟으로 테스트해보는 것이 좋습니다.",
        "caution": _verified_caution(report_input),
        "confidence_label": confidence_label,
    }
    if _is_v2_final_audience(report_input):
        return _replace_legacy_sample_wording(report)
    return report


def _audience_summary(report_input: SegmentSuggestionReportInput) -> str:
    evidence = report_input.target_segment.data_evidence_json
    sample_size = int(evidence.get("sample_size", 0) or 0)
    total_users = int(evidence.get("total_eligible_user_count", 0) or 0)
    if _is_v2_final_audience(report_input):
        matching_user_count = int(evidence.get("matching_user_count", 0) or 0)
        return (
            f"분석 가능 사용자 {total_users}명 · "
            f"행동 조건 부합 {matching_user_count}명 · "
            f"실험 대상 사용자 {sample_size}명"
        )
    return f"분석 대상 {total_users}명 중 {sample_size}명이 이 고객군에 해당합니다."


def _signals_sentence(signal_chips: Sequence[str]) -> str:
    if not signal_chips:
        return "호텔 예약 관심 행동이 확인되었습니다."
    return "주요 행동 신호는 " + ", ".join(signal_chips[:3]) + "입니다."


def _promotion_goal_sentence(promotion: PromotionRecord) -> str:
    if promotion.goal_metric == "booking_conversion_rate":
        return "이번 프로모션은 숙소 예약 전환을 늘리는 것이 목표입니다."
    if promotion.goal_metric == "inflow_rate":
        return "이번 프로모션은 랜딩과 숙소 탐색 유입을 늘리는 것이 목표입니다."
    return "이번 프로모션은 다음 퍼널 단계로 이동하는 고객을 늘리는 것이 목표입니다."


def _promotion_message_sentence(promotion: PromotionRecord) -> str:
    message_brief = (promotion.message_brief or "").strip()
    if not message_brief:
        return "채널과 목표 지표를 기준으로 확인 가능한 행동 신호를 연결했습니다."
    return f"프로모션 설명의 핵심 메시지는 '{message_brief[:80]}'입니다."


def _caution_text(*, sample_size: int, min_sample_size: int) -> str:
    if sample_size < min_sample_size:
        return "표본이 적어 첫 실험 결과를 빠르게 확인한 뒤 다음 타겟을 조정하는 것이 좋습니다."
    return "첫 발송 후 랜딩과 예약 시작 지표를 함께 확인하면 다음 액션을 더 잘 정할 수 있습니다."


def _verified_caution(report_input: SegmentSuggestionReportInput) -> str:
    performance_estimate = _mapping_value(
        report_input.display_copy.get("performance_estimate")
    )
    confidence_label = _confidence_label(
        performance_estimate.get("confidence_label")
    )
    if _is_v2_final_audience(report_input):
        sample_size = _report_sample_size(report_input)
        if sample_size < report_input.promotion.min_sample_size:
            return (
                "실험 대상 사용자가 최소 평가 인원보다 적어 첫 실험 결과는 "
                "insufficient_data로 평가됩니다."
            )
        if confidence_label == "low":
            return (
                "예상 성과의 신뢰도가 제한적인 후보이므로 실제 캠페인 "
                "성과와 함께 비교해 활용하세요."
            )
        if confidence_label in {"high", "medium"}:
            return (
                "예상값은 과거 행동을 바탕으로 한 참고 지표이며 실제 캠페인 "
                "성과와 함께 활용하세요."
            )
        return _caution_text(
            sample_size=sample_size,
            min_sample_size=report_input.promotion.min_sample_size,
        )
    if confidence_label == "low":
        return (
            "대표 표본이 제한적인 후보이므로 실제 캠페인 성과와 함께 "
            "비교해 활용하세요."
        )
    if confidence_label in {"high", "medium"}:
        return (
            "예상값은 과거 행동을 바탕으로 한 참고 지표이며 실제 캠페인 "
            "성과와 함께 활용하세요."
        )

    evidence = report_input.target_segment.data_evidence_json
    return _caution_text(
        sample_size=int(evidence.get("sample_size", 0) or 0),
        min_sample_size=report_input.promotion.min_sample_size,
    )


def _is_v2_final_audience(report_input: SegmentSuggestionReportInput) -> bool:
    evidence = report_input.target_segment.data_evidence_json
    return (
        evidence.get("audience_resolution_contract")
        == SEGMENT_AUDIENCE_CONTRACT
        and evidence.get("selected_user_role") == "final_experiment_audience"
    )


def _report_sample_size(report_input: SegmentSuggestionReportInput) -> int:
    evidence = report_input.target_segment.data_evidence_json
    return int(
        evidence.get("sample_size", report_input.target_segment.estimated_size)
        or 0
    )


def _replace_legacy_sample_wording(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("대표 표본", "실험 대상 사용자")
    if isinstance(value, list):
        return [_replace_legacy_sample_wording(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_legacy_sample_wording(item)
            for key, item in value.items()
        }
    return value


def _fallback_confidence_label(*, sample_size: int, min_sample_size: int) -> str:
    if sample_size < min_sample_size:
        return "low"
    if sample_size < min_sample_size * 3:
        return "medium"
    return "high"


def _format_goal_value(value: Decimal) -> str:
    return f"{float(value) * 100:g}%"


def _mapping_value(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_text_list(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [text for item in value if (text := _safe_text(item))]


def _merge_report_facts(
    computed_fact: str | None,
    generated_facts: Sequence[str],
    fallback_facts: Sequence[str],
) -> list[str]:
    facts: list[str] = []
    if computed_fact:
        facts.append(computed_fact)
    facts.extend(fact for fact in generated_facts if fact not in facts)
    if not facts:
        facts.extend(fallback_facts)
    return facts[:2]


def _confidence_label(value: object) -> str | None:
    text = _safe_text(value)
    if text in {"high", "medium", "low"}:
        return text
    return None


def _contains_forbidden_terms(report: Mapping[str, Any]) -> bool:
    values: list[str] = []
    for value in report.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            values.extend(str(item) for item in value)
    joined = " ".join(values).lower()
    return any(term.lower() in joined for term in FORBIDDEN_REPORT_TERMS)
