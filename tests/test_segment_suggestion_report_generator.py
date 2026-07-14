from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from typing import Any, Mapping

from app.analysis.repositories import (
    PromotionRecord,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
)
from app.analysis.report_generator import (
    OpenAISegmentSuggestionReportGenerator,
    SegmentSuggestionReportInput,
)


def test_openai_segment_report_rejects_technical_terms() -> None:
    def transport(
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        return {
            "output_text": json.dumps(
                {
                    "title": "벡터 유사도가 높은 군집",
                    "summary": "프로모션 벡터와 centroid가 가깝습니다.",
                    "why_recommended": ["군집 중심과 유사도가 높습니다.", "예약 신호가 있습니다."],
                    "evidence": ["cosine 점수가 높습니다.", "표본이 있습니다."],
                    "action_hint": "이 군집을 타겟으로 쓰세요.",
                    "caution": "클러스터 크기를 확인하세요.",
                },
                ensure_ascii=False,
            )
        }

    generator = OpenAISegmentSuggestionReportGenerator(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    report = generator.generate_report(_report_input())

    assert report["source"] == "fallback"
    assert report["title"] == "예약 가능성이 높은 고객"
    assert all(
        forbidden not in str(report)
        for forbidden in ("벡터", "군집", "클러스터", "centroid", "유사도", "cosine")
    )


def test_openai_segment_report_preserves_computed_candidate_facts() -> None:
    def transport(
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        return {
            "output_text": json.dumps(
                {
                    "title": "예약 관심 고객",
                    "summary": "예약 행동이 확인된 고객입니다.",
                    "promotion_interpretation": ["예약 전환이 목표입니다.", "혜택을 안내합니다."],
                    "why_recommended": ["예약 시작 행동이 있습니다.", "숙소를 탐색했습니다."],
                    "evidence": ["6명이 포함됐습니다.", "예약 시작 신호가 있습니다."],
                    "candidate_strengths": ["근거 없이 가장 좋은 후보입니다."],
                    "selection_considerations": ["대상 범위를 확인하세요."],
                    "action_hint": "예약 혜택을 안내하세요.",
                    "caution": "첫 발송 결과를 확인하세요.",
                    "confidence_label": "high",
                },
                ensure_ascii=False,
            )
        }

    base_input = _report_input()
    computed_strength = "예약 단계까지 진입한 고객을 회수하는 전략입니다."
    computed_tradeoff = "목적지 조건의 직접 일치 정도를 함께 확인해야 합니다."
    report_input = replace(
        base_input,
        display_copy={
            **dict(base_input.display_copy),
            "strength_summary": computed_strength,
            "tradeoff_summary": computed_tradeoff,
            "performance_estimate": {
                "label": "예상 예약 전환율",
                "formatted": "18.0%",
                "confidence_label": "low",
            },
        },
    )
    generator = OpenAISegmentSuggestionReportGenerator(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    report = generator.generate_report(report_input)

    assert report["candidate_strengths"][0] == computed_strength
    assert report["selection_considerations"][0] == computed_tradeoff
    assert "rank" not in str(report).lower()
    assert report["confidence_label"] == "low"


def test_openai_segment_report_replaces_unverified_sample_caution() -> None:
    def transport(
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        return {
            "output_text": json.dumps(
                {
                    "title": "프로모션 반응 고객",
                    "summary": "프로모션에 반응한 고객입니다.",
                    "promotion_interpretation": [
                        "예약 전환이 목표입니다.",
                        "혜택을 안내합니다.",
                    ],
                    "why_recommended": [
                        "캠페인 랜딩 행동이 있습니다.",
                        "프로모션 클릭 행동이 있습니다.",
                    ],
                    "evidence": [
                        "160명을 분석했습니다.",
                        "클릭 행동이 확인됐습니다.",
                    ],
                    "candidate_strengths": ["확장형 후보입니다."],
                    "selection_considerations": ["실제 성과를 함께 확인하세요."],
                    "action_hint": "혜택 메시지를 안내하세요.",
                    "caution": "표본 수가 160명으로 작아 신뢰도가 낮습니다.",
                    "confidence_label": "low",
                },
                ensure_ascii=False,
            )
        }

    confidence_reason = (
        "표본은 충분하지만 예측에 영향을 주는 일부 행동 신호가 "
        "학습 범위를 벗어나 분포 제한을 적용했습니다."
    )
    base_input = _report_input()
    report_input = replace(
        base_input,
        display_copy={
            **dict(base_input.display_copy),
            "performance_estimate": {
                "label": "예상 예약 전환율",
                "formatted": "5.1%",
                "confidence_label": "medium",
                "confidence_reason": confidence_reason,
                "prediction_adjustment": {
                    "candidate_sample_size": 160,
                    "prior_user_count": 30.0,
                },
            },
        },
    )
    generator = OpenAISegmentSuggestionReportGenerator(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    report = generator.generate_report(report_input)

    assert report["confidence_label"] == "medium"
    assert report["caution"] == (
        "예상값은 과거 행동을 바탕으로 한 참고 지표이며 실제 캠페인 "
        "성과와 함께 활용하세요."
    )
    assert "표본 수가 160명으로 작아" not in report["caution"]
    assert "학습 범위" not in report["caution"]
    assert "분포 제한" not in report["caution"]


def _report_input() -> SegmentSuggestionReportInput:
    promotion = PromotionRecord(
        project_id="demo_project",
        campaign_id="camp_demo",
        promotion_id="promo_demo",
        channel="email",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.1"),
        goal_basis="promotion_average",
        min_sample_size=3,
        landing_url="https://demo-shoppingmall.dev.loop-ad.org/search?deal=summer",
        message_brief="여름 특가 세일을 기존 유저에게 안내한다.",
    )
    segment = SegmentDefinitionRecord(
        segment_id="seg_ai_demo",
        project_id="demo_project",
        segment_name="예약 가능성이 높은 고객",
        source="ai_suggested",
        query_preview_id=None,
        natural_language_query=None,
        generated_sql=None,
        rule_json={"candidate_user_ids": ["user_1", "user_2", "user_3", "user_4"]},
        profile_json={},
        sample_size=4,
        total_eligible_user_count=8,
        sample_ratio=Decimal("0.5"),
        status="active",
        campaign_id="camp_demo",
        promotion_id="promo_demo",
    )
    target_segment = PromotionTargetSegmentWrite(
        analysis_id="analysis_demo",
        project_id="demo_project",
        campaign_id="camp_demo",
        promotion_id="promo_demo",
        segment_id=segment.segment_id,
        segment_name=segment.segment_name,
        rule_json=segment.rule_json,
        profile_json=segment.profile_json,
        content_brief_json={},
        data_evidence_json={
            "sample_size": 4,
            "sample_ratio": "0.5",
            "total_eligible_user_count": 8,
        },
        segment_vector_id="segvec_demo",
        estimated_size=4,
        priority="high",
        status="planned",
    )
    return SegmentSuggestionReportInput(
        promotion=promotion,
        segment=segment,
        target_segment=target_segment,
        display_copy={
            "title": "예약 가능성이 높은 고객",
            "audience_summary": "분석 대상 8명 중 4명 · 50%",
            "signal_chips": ["예약 완료 경험", "프로모션 반응"],
            "reason": "예약 전환 목표에 가까운 행동 패턴을 보인 고객군입니다.",
            "action_hint": "이메일 예약 혜택 메시지의 우선 타겟으로 적합합니다.",
        },
        primary_signals=[
            {"key": "booking_complete", "chip": "예약 완료 경험"},
            {"key": "promotion_engaged", "chip": "프로모션 반응"},
        ],
        score_json={},
        reason_json={},
    )
