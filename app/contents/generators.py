from __future__ import annotations

import json
from typing import Any, Protocol

from app.contents.prompt_builder import ContentPromptBuilder
from app.contents.types import (
    GENERATION_STATUS_GENERATED,
    GeneratedContentDraft,
    RecommendationActionTarget,
)


class ContentGenerator(Protocol):
    def generate(
        self,
        *,
        target: RecommendationActionTarget,
        variant_key: str,
    ) -> GeneratedContentDraft:
        ...


class PartialContentGenerationError(Exception):
    def __init__(self, message: str, draft: GeneratedContentDraft | None = None) -> None:
        super().__init__(message)
        self.draft = draft


class MockContentGenerator:
    def __init__(self, generation_model: str = "mock") -> None:
        self.generation_model = generation_model

    def generate(
        self,
        *,
        target: RecommendationActionTarget,
        variant_key: str,
    ) -> GeneratedContentDraft:
        content_type = _content_type_for(target)
        segment_name = target.segment.name or target.segment.segment_key
        root_cause_key = str(target.root_cause.get("cause_key") or "")
        category = str(
            target.segment.attributes.get("primary_category")
            or target.segment.attributes.get("category")
            or "추천 상품"
        )

        if variant_key == "control":
            title = f"{segment_name}을 위한 오늘의 추천"
            body = "지금 인기 상품과 혜택을 확인해보세요."
            cta_label = "상품 보러가기"
        elif "coupon" in target.action_key:
            title = "구매 전 마지막 혜택을 확인하세요"
            body = f"{segment_name} 고객을 위해 준비한 쿠폰 혜택을 놓치지 마세요."
            cta_label = "쿠폰 확인하기"
        elif target.action_key == "alternative_product_banner" or "stock" in root_cause_key:
            title = f"{category} 대체 추천 상품"
            body = "품절되기 쉬운 상품 대신 바로 구매 가능한 추천 상품을 만나보세요."
            cta_label = "대체 상품 보기"
        else:
            title = f"{category} 혜택을 더 잘 보이게"
            body = "장바구니에 담기 전, 핵심 혜택과 추천 상품을 확인해보세요."
            cta_label = "혜택 상품 보기"

        landing_url = str(
            target.metadata.get("landing_url")
            or target.segment.attributes.get("landing_url")
            or f"/segments/{target.segment.segment_key}"
        )
        image_prompt = (
            f"clean ecommerce banner for {category}, segment {target.segment.segment_key}, "
            f"variant {variant_key}, no text, no logo, product-focused"
        )

        return GeneratedContentDraft(
            project_id=target.project_id,
            recommendation_action_id=target.id,
            segment_id=target.segment.id,
            variant_key=variant_key,
            content_type=content_type,
            title=title,
            body=body,
            cta_label=cta_label,
            landing_url=landing_url,
            image_prompt=image_prompt,
            generation_model=self.generation_model,
            generation_status=GENERATION_STATUS_GENERATED,
            metadata={
                "generator": "mock",
                "action_key": target.action_key,
                "recommendation_result_id": target.recommendation_result_id,
            },
        )


class OpenAIContentGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: Any | None = None,
        prompt_builder: ContentPromptBuilder | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("OpenAI model must come from env/config and must not be empty")
        self.model = model
        self.prompt_builder = prompt_builder or ContentPromptBuilder()
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package is required for OpenAIContentGenerator") from exc
            client = OpenAI(api_key=api_key)
        self.client = client

    def generate(
        self,
        *,
        target: RecommendationActionTarget,
        variant_key: str,
    ) -> GeneratedContentDraft:
        prompt_payload = self.prompt_builder.build(target, variant_key)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Create concise ecommerce ad content as JSON only. "
                        "Do not infer metrics, do not include personal data, and do not include raw events."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
        )
        payload = _load_response_json(response)
        draft = GeneratedContentDraft(
            project_id=target.project_id,
            recommendation_action_id=target.id,
            segment_id=target.segment.id,
            variant_key=variant_key,
            content_type=str(payload.get("content_type") or _content_type_for(target)),
            title=str(payload.get("title") or ""),
            body=str(payload.get("body") or ""),
            cta_label=str(payload.get("cta_label") or ""),
            landing_url=str(payload.get("landing_url") or target.metadata.get("landing_url") or "/"),
            image_prompt=str(payload.get("image_prompt") or ""),
            generation_model=self.model,
            generation_status=GENERATION_STATUS_GENERATED,
            metadata={
                "generator": "openai",
                "action_key": target.action_key,
                "recommendation_result_id": target.recommendation_result_id,
            },
        )
        if not draft.has_required_fields():
            raise PartialContentGenerationError("OpenAI content response is missing required fields", draft)
        return draft


def _load_response_json(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if output_text is None and isinstance(response, dict):
        output_text = response.get("output_text")
    if not output_text:
        raise ValueError("OpenAI response did not include output_text")
    parsed = json.loads(output_text)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI response JSON must be an object")
    return parsed


def _content_type_for(target: RecommendationActionTarget) -> str:
    if target.action_key in {"cart_coupon_banner", "checkout_coupon_banner"}:
        return "coupon_banner"
    if target.action_key == "alternative_product_banner":
        return "product_recommendation_banner"
    return "banner"
