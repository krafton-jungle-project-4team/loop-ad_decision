from typing import Any


class CreativeBriefService:
    def create_brief(
        self,
        *,
        recommendation_action: Any,
        action_catalog_item: Any,
        ad_creative: Any,
        recommendation_result_id: int,
        generation_id: str,
    ) -> dict[str, Any]:
        execution_hint = getattr(recommendation_action, "execution_hint_json", None) or {}
        creative_payload = getattr(ad_creative, "payload_json", None) or {}

        return {
            "generation_id": generation_id,
            "recommendation_result_id": recommendation_result_id,
            "project_id": getattr(recommendation_action, "project_id", None),
            "action": {
                "action_id": getattr(recommendation_action, "action_id", None),
                "action_type": getattr(recommendation_action, "action_type", None),
                "title": getattr(recommendation_action, "title", None)
                or getattr(action_catalog_item, "title", None),
                "description": getattr(recommendation_action, "description", None)
                or getattr(action_catalog_item, "description", None),
                "target_step": getattr(recommendation_action, "target_step", None)
                or getattr(action_catalog_item, "target_step", None),
                "expected_impact": getattr(recommendation_action, "expected_impact", None)
                or getattr(action_catalog_item, "expected_impact", None),
                "execution_hint": execution_hint,
            },
            "creative": {
                "creative_id": str(getattr(ad_creative, "id", "")),
                "creative_type": getattr(ad_creative, "creative_type", None),
                "title": getattr(ad_creative, "title", None),
                "message": getattr(ad_creative, "message", None),
                "landing_url": getattr(ad_creative, "landing_url", None),
                "payload": creative_payload,
            },
            "format": {
                "width": 1200,
                "height": 628,
                "final_type": "image/png",
                "source_type": "image/svg+xml",
            },
            "style": {
                "mood": "clean bright ecommerce fresh grocery delivery",
                "background_requirements": [
                    "no text",
                    "no letters",
                    "no logo",
                    "no watermark",
                    "empty left side for headline and CTA",
                ],
            },
        }
