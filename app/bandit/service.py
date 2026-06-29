import random

from fastapi import HTTPException, status

from app.bandit.schemas import BanditActionProbabilityItem, BanditActionProbabilityResponse
from app.persistence.repository import BanditArmServingRow, PostgresRepository


def get_action_probabilities(
    *,
    repository: PostgresRepository,
    bandit_policy_id: int,
    samples: int = 10000,
    seed: int = 42,
) -> BanditActionProbabilityResponse:
    if samples < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="samples must be greater than 0",
        )
    policy = repository.get_bandit_policy(bandit_policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="bandit policy not found",
        )

    rows = repository.list_bandit_arm_serving_rows(bandit_policy_id)
    if not rows:
        return BanditActionProbabilityResponse(bandit_policy_id=bandit_policy_id, items=[])

    probabilities = calculate_thompson_probabilities(rows, samples=samples, seed=seed)
    return BanditActionProbabilityResponse(
        bandit_policy_id=bandit_policy_id,
        items=[
            BanditActionProbabilityItem(
                bandit_arm_id=row.arm.id,
                action_id=row.arm.action_id,
                action_type=row.arm.action_type,
                probability=probabilities[index],
                alpha=row.arm.alpha,
                beta=row.arm.beta,
                impressions=row.arm.impressions,
                conversions=row.arm.conversions,
                mapping_id=getattr(row.mapping, "id", None),
                creative_id=getattr(row.mapping, "creative_id", None),
                content_url=getattr(row.creative, "image_url", None),
            )
            for index, row in enumerate(rows)
        ],
    )


def calculate_thompson_probabilities(
    rows: list[BanditArmServingRow],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if len(rows) == 1:
        return [1.0]

    rng = random.Random(seed)
    wins = [0 for _ in rows]
    for _ in range(samples):
        sampled_values = [
            rng.betavariate(max(row.arm.alpha, 1e-9), max(row.arm.beta, 1e-9))
            for row in rows
        ]
        winner_index = max(
            range(len(sampled_values)),
            key=lambda index: sampled_values[index],
        )
        wins[winner_index] += 1

    probabilities = [round(win_count / samples, 6) for win_count in wins]
    if probabilities:
        probabilities[-1] = round(1.0 - sum(probabilities[:-1]), 6)
    return probabilities
