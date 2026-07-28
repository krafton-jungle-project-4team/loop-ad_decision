from __future__ import annotations

import pytest

from scripts.finalize_ann_scale_scenarios import (
    selected_scenario_ids,
    validate_selected_pairs,
)


def test_selected_ids_exclude_unvalidated_fallbacks() -> None:
    selected, fallbacks = selected_scenario_ids(
        {
            "pairs": [
                {
                    "validated": True,
                    "tuning_scenario_id": "t",
                    "confirmation_scenario_id": "c",
                },
                {"validated": False, "fallback_reason": "missing"},
            ]
        }
    )
    assert selected == {"t", "c"}
    assert len(fallbacks) == 1


def test_selected_pair_definitions_must_be_distinct() -> None:
    scenarios = [
        {"scenario_id": "t", "scenario_set": "tuning", "definition_sha256": "x"},
        {
            "scenario_id": "c",
            "scenario_set": "confirmation",
            "definition_sha256": "x",
        },
    ]
    with pytest.raises(ValueError, match="not distinct"):
        validate_selected_pairs(
            scenarios,
            {
                "pairs": [
                    {
                        "validated": True,
                        "tuning_scenario_id": "t",
                        "confirmation_scenario_id": "c",
                    }
                ]
            },
        )
