from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from app.generation.prompt_builder import PromptBuildResult


MAX_RICH_IMAGE_PROMPT_LENGTH = 1200
MAX_GENERATION_CONSTRAINT_TEXT_LENGTH = 220
_HEX_COLOUR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}\b")
_DESIGN_METADATA_PATTERN = re.compile(
    r"\b(?:color\s+palette|palette|color\s+swatches?|swatches?|"
    r"contact\s+sheet|mood\s*board|collage|split[-\s]+panel|"
    r"style\s+guide|reference\s+board)\b",
    re.IGNORECASE,
)


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
            return _compose_prompt(
                [
                    "Property-agnostic hotel booking advertisement image.",
                    (
                        "Keep the scene generic and within the accommodation "
                        "booking domain."
                    ),
                ],
                guardrails=_image_guardrails([]),
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
            encoded_visual_context = _safe_scene_guidance(
                json.dumps(
                    visual_context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            lines.append(
                "Verified hotel visual context: "
                + encoded_visual_context
                + "."
            )
            if provider_visual_concept and provider_visual_concept.strip():
                visual_concept = _safe_scene_guidance(provider_visual_concept)
                if visual_concept:
                    lines.append(
                        "Provider visual concept, style guidance only: "
                        f"{visual_concept}."
                    )
        else:
            lines.append(
                "Verified hotel visual context: none. Keep the property depiction "
                "generic and property-agnostic."
            )
        brand_visual_context = _brand_visual_context(context.brand_context)
        if brand_visual_context:
            encoded_brand_context = _safe_scene_guidance(
                json.dumps(
                    brand_visual_context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            lines.append(
                "Approved brand visual direction: "
                + encoded_brand_context
                + ". Follow the approved photographic style and asset direction. "
                "Treat brand color guidance only as a subtle scene mood, never as "
                "a palette or design reference."
            )
        return _compose_prompt(
            lines,
            guardrails=_image_guardrails(constraints),
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


def _brand_visual_context(brand_context: object) -> Mapping[str, object]:
    if brand_context is None:
        return {}
    guardrails = getattr(brand_context, "guardrails", None)
    documents = getattr(brand_context, "documents", ())
    selected_asset_id = getattr(brand_context, "selected_asset_id", None)
    context: dict[str, object] = {}
    approved_colours = getattr(guardrails, "approved_colours", ())
    if approved_colours:
        context["brand_color_guidance"] = "subtle photographic color mood"
    approved_styles = getattr(guardrails, "approved_image_styles", ())
    if approved_styles:
        context["approved_image_styles"] = list(approved_styles)
    for document in documents:
        if (
            getattr(document, "source_kind", None) == "brand_asset"
            and getattr(document, "source_id", None) == selected_asset_id
        ):
            description = " ".join(
                str(getattr(document, "document_text", "")).split()
            )
            if description:
                context["asset_direction"] = description[:300]
            break
    return context


def _image_guardrails(constraints: Sequence[str]) -> str:
    constraint_text = _compact(
        _without_colour_codes(", ".join(constraints))
        or "no additional verified claims",
        max_length=MAX_GENERATION_CONSTRAINT_TEXT_LENGTH,
    )
    return (
        "Guardrails: create one coherent, natural hotel or travel photograph; "
        "do not render a color palette, color swatches, contact sheet, mood board, "
        "collage, split panel, comparison grid, style guide, reference board, "
        "color code, asset ID, or reference label; no visible text, letters, "
        "numbers, logos, or typography; do not add people unless the scene needs "
        "them; when people appear, depict only clearly adult travelers aged 20 to "
        "39, never children, teenagers, middle-aged people, or elderly people; "
        "stay in the hotel and accommodation booking domain; do not depict "
        "discounts, prices, room inventory, booking policies, amenities, or "
        "facilities unless present in verified hotel visual context; "
        f"generation constraints={constraint_text}."
    )


def _compose_prompt(lines: Sequence[str], *, guardrails: str) -> str:
    required_suffix = " ".join(guardrails.split())
    if len(required_suffix) >= MAX_RICH_IMAGE_PROMPT_LENGTH:
        return _compact(
            required_suffix,
            max_length=MAX_RICH_IMAGE_PROMPT_LENGTH,
        )
    prefix_budget = MAX_RICH_IMAGE_PROMPT_LENGTH - len(required_suffix) - 1
    prefix = _compact(
        " ".join(lines),
        max_length=prefix_budget,
    )
    return f"{prefix} {required_suffix}"


def _without_colour_codes(value: str) -> str:
    return " ".join(_HEX_COLOUR_PATTERN.sub("", value).split())


def _safe_scene_guidance(value: str) -> str:
    without_metadata = _DESIGN_METADATA_PATTERN.sub("", value)
    return _without_colour_codes(without_metadata)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _compact(value: str, *, max_length: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_length:
        return compacted
    return compacted[: max_length - 1].rstrip() + "."
