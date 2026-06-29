from typing import Any


class CopyGenerationService:
    def generate_copy(self, brief: dict[str, Any]) -> dict[str, str]:
        action = brief["action"]
        creative = brief["creative"]
        action_id = str(action.get("action_id") or "")

        if "free_shipping" in action_id:
            headline = "Fresh food ships free"
            subcopy = "A delivery benefit for your fresh grocery cart."
            cta = "Claim free shipping"
            badge = "FREE SHIPPING"
        elif "coupon" in action_id:
            headline = "Fresh picks, better price"
            subcopy = "Limited coupon support for shoppers ready to buy."
            cta = "Use coupon"
            badge = "COUPON"
        elif "stock" in action_id or "recovery" in action_id:
            headline = "Fresh favorites are back"
            subcopy = "Guide shoppers to available fresh food alternatives."
            cta = "Shop available items"
            badge = "RESTOCK"
        else:
            headline = str(creative.get("title") or action.get("title") or "Fresh offer")
            subcopy = str(
                creative.get("message")
                or action.get("expected_impact")
                or action.get("description")
                or "A timely offer for your next order."
            )
            cta = "Shop now"
            badge = "TODAY"

        return {
            "headline": headline,
            "subcopy": subcopy,
            "cta": cta,
            "badge": badge,
            "brand_name": "LoopAd Market",
        }
