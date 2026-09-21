from __future__ import annotations

import argparse
import pytest

from scripts.prepare_ann_scale_postgres import (
    cohort_database_name,
    positive_ints,
    sql_literal,
)


def test_cohort_database_names_are_deterministic_and_bounded() -> None:
    assert cohort_database_name("loopad_ann_v2", 50_000) == "loopad_ann_v2_50k"
    assert cohort_database_name("loopad_ann_v2", 1_000_000) == "loopad_ann_v2_1000k"
    with pytest.raises(ValueError, match="unsafe"):
        cohort_database_name("loopad;drop", 50_000)


def test_positive_cohort_sizes_reject_duplicates() -> None:
    assert positive_ints("50000,100000") == (50_000, 100_000)
    with pytest.raises(argparse.ArgumentTypeError):
        positive_ints("50000,50000")


def test_postgres_literal_escapes_quotes() -> None:
    assert sql_literal("a'b") == "'a''b'"
