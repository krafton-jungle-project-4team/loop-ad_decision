from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from app.generation.prompt_builder import PromptBuildResult


MAX_RICH_IMAGE_PROMPT_LENGTH = 1200


class RichImagePromptBuilder:
    def build(
        self,
        prompt_result: PromptBuildResult,
        *,
        provider_visual_concept: str | None = None,
    ) -> str:
        context = prompt_result.generation_context
        strategy_plan = prompt_result.strategy_plan
        if context is None or strategy_plan is None:
            return _compact(
                "Property-agnostic hotel booking advertisement image. "
                "Keep the scene generic and within the accommodation booking domain. "
                + _image_guardrails([]),
                max_length=MAX_RICH_IMAGE_PROMPT_LENGTH,
            )

        objective = context.promotion_objective
        visual_context = _verified_visual_context(context.hotel_profile)
        constraints = _string_list(context.generation_constraints.get("do_not_claim"))
        lines = [
            "Hotel booking advertisement image.",
            (
                "Promotion context: "
                f"channel={objective.get('channel', 'unknown')}, "
                f"goal_metric={objective.get('goal_metric', 'unknown')}."
            ),
            (
                "Audience focus: "
                f"{', '.join(strategy_plan.audience_focus) or 'general hotel audience'}."
            ),
            (
                "Strategy visual direction: "
                f"{', '.join(strategy_plan.visual_direction)}."
            ),
        ]
        if visual_context:
            lines.append(
                "Verified hotel visual context: "
                + json.dumps(
                    visual_context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "."
            )
            if provider_visual_concept and provider_visual_concept.strip():
                lines.append(
                    "Provider visual concept, style guidance only: "
                    f"{provider_visual_concept.strip()}."
                )
        else:
            lines.append(
                "Verified hotel visual context: none. Keep the property depiction "
                "generic and property-agnostic."
            )
        lines.append(_image_guardrails(constraints))
        return _compact(
            " ".join(lines),
            max_length=MAX_RICH_IMAGE_PROMPT_LENGTH,
        )


def _verified_visual_context(
    hotel_profile: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if hotel_profile is None:
        return {}
    value = hotel_profile.get("visual_context")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if item is not None
        and not (isinstance(item, str) and not item.strip())
        and not (
            isinstance(item, (Mapping, Sequence))
            and not isinstance(item, str)
            and not item
        )
    }


def _image_guardrails(constraints: Sequence[str]) -> str:
    constraint_text = ", ".join(constraints) or "no additional verified claims"
    return (
        "Guardrails: no visible text, letters, numbers, logos, or typography; "
        "stay in the hotel and accommodation booking domain; do not depict "
        "discounts, prices, room inventory, booking policies, amenities, or "
        "facilities unless present in verified hotel visual context; "
        f"generation constraints={constraint_text}."
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _compact(value: str, *, max_length: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_length:
        return compacted
    return compacted[: max_length - 1].rstrip() + "."
