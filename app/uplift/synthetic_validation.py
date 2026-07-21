from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from app.uplift.contracts import UpliftTrainingExample
from app.uplift.metrics import evaluate_uplift_predictions
from app.uplift.model import (
    fit_transformed_outcome_ridge,
    signed_cate_summary,
)
from app.uplift.validation import (
    collecting_data_metadata,
    experiment_cluster_bootstrap_cate_ci,
)


def run_synthetic_validation(
    *,
    sample_size: int = 12_000,
    seed: int = 20260721,
) -> dict[str, Any]:
    if sample_size < 300:
        raise ValueError("synthetic validation requires at least 300 examples")
    train = _synthetic_examples(sample_size=sample_size, seed=seed, split="train")
    validation = _synthetic_examples(
        sample_size=sample_size,
        seed=seed + 1,
        split="validation",
    )
    model = fit_transformed_outcome_ridge(train, ridge_strength=1.0)
    scores = model.predict_many(validation)
    by_effect: dict[str, list[float]] = defaultdict(list)
    for example, score in zip(validation, scores, strict=True):
        by_effect[_effect_label(example.features[0])].append(score)
    recovered = {
        label: sum(values) / len(values)
        for label, values in sorted(by_effect.items())
    }
    metrics = evaluate_uplift_predictions(validation, scores)
    ci = experiment_cluster_bootstrap_cate_ci(
        validation,
        scores,
        iterations=1000,
        seed=seed,
    )
    metadata = collecting_data_metadata(model_version=model.model_version)
    return {
        "artifact_type": "synthetic_uplift_pipeline_validation",
        "sample_size": sample_size,
        "seed": seed,
        "known_effects": {
            "negative": -0.12,
            "zero": 0.0,
            "positive": 0.12,
        },
        "recovered_signed_cate": recovered,
        "signed_cate_summary": dict(signed_cate_summary(scores)),
        "metrics": metrics,
        "cate_confidence_interval": ci.to_json(),
        "model_metadata": metadata.to_json(),
        "serving_activation_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Uplift pipeline with known synthetic effects."
    )
    parser.add_argument("--sample-size", type=int, default=12_000)
    parser.add_argument("--sample-seed", type=int, default=20260721)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_synthetic_validation(
        sample_size=args.sample_size,
        seed=args.sample_seed,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _synthetic_examples(
    *,
    sample_size: int,
    seed: int,
    split: str,
) -> list[UpliftTrainingExample]:
    rng = random.Random(seed)
    examples: list[UpliftTrainingExample] = []
    effect_values = (-1.0, 0.0, 1.0)
    for index in range(sample_size):
        effect_feature = effect_values[index % len(effect_values)]
        context = rng.uniform(-1.0, 1.0)
        treatment = 1 if rng.random() < 0.5 else 0
        baseline = 0.25 + 0.04 * context
        treatment_effect = 0.12 * effect_feature
        probability = max(
            0.01,
            min(0.99, baseline + treatment * treatment_effect),
        )
        outcome = 1 if rng.random() < probability else 0
        experiment_index = index % 6
        examples.append(
            UpliftTrainingExample(
                experiment_unit_id=f"synthetic_{split}_{index}",
                project_id="synthetic",
                promotion_run_id=f"synthetic_run_{experiment_index}",
                ad_experiment_id=f"synthetic_experiment_{experiment_index}",
                segment_id=f"effect_{_effect_label(effect_feature)}",
                user_id=f"synthetic_user_{split}_{index}",
                audience_snapshot_id=f"synthetic_snapshot_{experiment_index}",
                vector_generation_id=f"synthetic_generation_{experiment_index}",
                features=(effect_feature, context),
                treatment=treatment,
                outcome=outcome,
                treatment_probability=0.5,
            )
        )
    return examples


def _effect_label(effect_feature: float) -> str:
    if effect_feature < 0:
        return "negative"
    if effect_feature > 0:
        return "positive"
    return "zero"


if __name__ == "__main__":
    raise SystemExit(main())
