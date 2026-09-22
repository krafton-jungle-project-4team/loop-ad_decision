from __future__ import annotations

import pytest

from scripts.prepare_ann_scale_cohorts import require_expected_prefix


def test_prepared_cohort_must_equal_frozen_membership_prefix() -> None:
    digest = "a" * 64
    require_expected_prefix(
        cohort_size=50_000,
        observed_hash=digest,
        expected_hashes={"50000": digest},
    )
    with pytest.raises(RuntimeError, match="frozen membership"):
        require_expected_prefix(
            cohort_size=50_000,
            observed_hash="b" * 64,
            expected_hashes={"50000": digest},
        )
