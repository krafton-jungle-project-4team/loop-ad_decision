from __future__ import annotations

import json
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
        transport=transport,
    )

    report = generator.generate_report(_report_input())

    assert report["source"] == "fallback"
    assert report["title"] == "예약 가능성이 높은 고객"
    assert all(
        forbidden not in str(report)
        for forbidden in ("벡터", "군집", "클러스터", "centroid", "유사도", "cosine")
    )


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
