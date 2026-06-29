from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.bandit.service import get_action_probabilities
from app.persistence.repository import BanditArmServingRow


MISSING = object()


class FakeBanditRepository:
    def __init__(
        self,
        *,
        policy: SimpleNamespace | None | object = MISSING,
        rows: list[BanditArmServingRow] | None = None,
    ) -> None:
        self.policy = SimpleNamespace(id=1) if policy is MISSING else policy
        self.rows = rows or []
        self.created_decisions = 0

    def get_bandit_policy(self, bandit_policy_id: int) -> SimpleNamespace | None:
        if self.policy is None or self.policy.id != bandit_policy_id:
            return None
        return self.policy

    def list_bandit_arm_serving_rows(self, bandit_policy_id: int) -> list[BanditArmServingRow]:
        return self.rows

    def create_bandit_decision(self, **values: object) -> None:
        self.created_decisions += 1


def arm(
    arm_id: int,
    *,
    action_id: str,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=arm_id,
        action_id=action_id,
        action_type="PRODUCT",
        alpha=alpha,
        beta=beta,
        impressions=10,
        conversions=2,
    )


def test_bandit_probability_single_arm_is_one_and_read_only() -> None:
    repository = FakeBanditRepository(
        rows=[
            BanditArmServingRow(
                arm=arm(1, action_id="recommend_alternative_product"),
                mapping=SimpleNamespace(id=100, creative_id=200),
                creative=SimpleNamespace(image_url="https://cdn.example/ad.png"),
            )
        ]
    )

    response = get_action_probabilities(
        repository=repository,
        bandit_policy_id=1,
    )

    assert response.items[0].probability == 1.0
    assert response.items[0].mapping_id == 100
    assert response.items[0].creative_id == 200
    assert response.items[0].content_url == "https://cdn.example/ad.png"
    assert repository.created_decisions == 0


def test_bandit_probability_multi_arm_sum_is_one() -> None:
    repository = FakeBanditRepository(
        rows=[
            BanditArmServingRow(arm=arm(1, action_id="a", alpha=5.0, beta=2.0)),
            BanditArmServingRow(arm=arm(2, action_id="b", alpha=2.0, beta=5.0)),
            BanditArmServingRow(arm=arm(3, action_id="c", alpha=3.0, beta=3.0)),
        ]
    )

    response = get_action_probabilities(
        repository=repository,
        bandit_policy_id=1,
        samples=1000,
        seed=7,
    )

    assert len(response.items) == 3
    assert round(sum(item.probability for item in response.items), 6) == 1.0
    assert repository.created_decisions == 0


def test_bandit_probability_missing_active_mapping_returns_none_fields() -> None:
    repository = FakeBanditRepository(
        rows=[BanditArmServingRow(arm=arm(1, action_id="a"))]
    )

    response = get_action_probabilities(
        repository=repository,
        bandit_policy_id=1,
    )

    assert response.items[0].mapping_id is None
    assert response.items[0].creative_id is None
    assert response.items[0].content_url is None


def test_bandit_probability_missing_policy_returns_404() -> None:
    repository = FakeBanditRepository(policy=None)

    with pytest.raises(HTTPException) as exc_info:
        get_action_probabilities(repository=repository, bandit_policy_id=999)

    assert exc_info.value.status_code == 404
