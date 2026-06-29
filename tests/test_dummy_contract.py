import re
from decimal import Decimal
from pathlib import Path


def test_dummy_sample_experiment_does_not_preselect_winner() -> None:
    dummy = Path("dummy.sql").read_text(encoding="utf-8")
    values = re.findall(
        r"0\.5000,\s+410,\s+(\d+),\s+(\d+),\s+0\.\d+,\s+(0\.\d+),\s+'active'",
        dummy,
    )
    assert values == [("28", "1", "0.035714"), ("38", "1", "0.026316")]

    configs = [
        {"minimum_impressions": 100, "minimum_clicks": 30, "target_value": Decimal("0.05")},
        {"minimum_impressions": 10, "minimum_clicks": 3, "target_value": Decimal("0.05")},
    ]
    for config in configs:
        candidates = [
            variant
            for variant in values
            if 410 >= config["minimum_impressions"]
            and int(variant[0]) >= config["minimum_clicks"]
            and Decimal(variant[2]) >= config["target_value"]
        ]
        assert candidates == []
