from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from app.uplift.contracts import UpliftTrainingExample
from app.uplift.metrics import evaluate_uplift_predictions
from app.uplift.model import fit_transformed_outcome_ridge, signed_cate_summary
from app.uplift.validation import (
    external_validation_metadata,
    predicted_cate_cluster_variability_interval,
)


_NON_FEATURE_COLUMNS = frozenset(
    {"treatment", "conversion", "visit", "exposure"}
)
CRITEO_SPLIT_POLICY_VERSION = "stable-row-hash-stratified-70-30.v1"


def load_criteo_examples(
    *,
    input_path: Path,
    max_rows: int,
    sample_seed: int,
) -> tuple[list[UpliftTrainingExample], dict[str, Any]]:
    if max_rows < 2:
        raise ValueError("max_rows must be at least two")
    if not input_path.is_file():
        raise ValueError(f"Criteo input file does not exist: {input_path}")

    rng = random.Random(sample_seed)
    reservoir: list[tuple[int, Mapping[str, str]]] = []
    source_row_count = 0
    with _open_text(input_path) as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        _validate_columns(fieldnames)
        for row_index, row in enumerate(reader):
            source_row_count += 1
            item = (row_index, dict(row))
            if len(reservoir) < max_rows:
                reservoir.append(item)
                continue
            replacement_index = rng.randint(0, row_index)
            if replacement_index < max_rows:
                reservoir[replacement_index] = item
    if not reservoir:
        raise ValueError("Criteo input contains no data rows")
    reservoir.sort(key=lambda item: item[0])
    feature_columns = tuple(
        column for column in fieldnames if column not in _NON_FEATURE_COLUMNS
    )
    treatment_count = sum(_binary(row["treatment"]) for _index, row in reservoir)
    treatment_probability = treatment_count / len(reservoir)
    if not 0 < treatment_probability < 1:
        raise ValueError("sampled Criteo rows require treatment and control")
    outcome_column = "conversion" if "conversion" in fieldnames else "visit"
    examples = [
        UpliftTrainingExample(
            experiment_unit_id=f"criteo_{row_index}",
            project_id="external_criteo",
            promotion_run_id="external_criteo_validation",
            ad_experiment_id="external_criteo_validation",
            segment_id="external_criteo_population",
            user_id=f"criteo_user_{row_index}",
            audience_snapshot_id="external_criteo_snapshot",
            vector_generation_id="external_criteo_features",
            features=tuple(
                _feature_value(row[column]) for column in feature_columns
            ),
            treatment=_binary(row["treatment"]),
            outcome=_binary(row[outcome_column]),
            treatment_probability=treatment_probability,
        )
        for row_index, row in reservoir
    ]
    metadata = {
        "input_path": str(input_path),
        "source_row_count": source_row_count,
        "sampled_row_count": len(examples),
        "max_rows": max_rows,
        "sample_seed": sample_seed,
        "feature_columns": list(feature_columns),
        "outcome_column": outcome_column,
        "sample_fingerprint": _sample_fingerprint(examples),
    }
    return examples, metadata


def validate_criteo_pipeline(
    *,
    input_path: Path,
    max_rows: int,
    sample_seed: int,
) -> dict[str, Any]:
    examples, adapter_metadata = load_criteo_examples(
        input_path=input_path,
        max_rows=max_rows,
        sample_seed=sample_seed,
    )
    train_examples, test_examples = _stable_train_test_split(
        examples,
        sample_seed=sample_seed,
    )
    adapter_metadata.update(
        {
            "split_policy_version": CRITEO_SPLIT_POLICY_VERSION,
            "train_count": len(train_examples),
            "test_count": len(test_examples),
            "train_sample_fingerprint": _sample_fingerprint(train_examples),
            "test_sample_fingerprint": _sample_fingerprint(test_examples),
        }
    )
    model = fit_transformed_outcome_ridge(train_examples, ridge_strength=1.0)
    cate_scores = model.predict_many(test_examples)
    metrics = evaluate_uplift_predictions(test_examples, cate_scores)
    variability_interval = predicted_cate_cluster_variability_interval(
        test_examples,
        cate_scores,
        iterations=1000,
        seed=sample_seed,
    )
    metadata = external_validation_metadata(
        model_version=model.model_version,
        dataset="criteo_uplift",
    )
    return {
        "artifact_type": "uplift_external_pipeline_validation",
        "adapter": adapter_metadata,
        "treatment_control_balance": {
            "treatment_count": metrics["treatment_count"],
            "control_count": metrics["control_count"],
            "actual_treatment_ratio": (
                metrics["treatment_count"] / metrics["observation_count"]
            ),
        },
        "metrics": metrics,
        "signed_cate_summary": dict(signed_cate_summary(cate_scores)),
        "predicted_cate_cluster_variability_interval": (
            variability_interval.to_json()
        ),
        "model_metadata": metadata.to_json(),
        "domain_claim": "external_pipeline_only_not_hotel_performance_evidence",
        "serving_activation_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Uplift pipeline with a local Criteo dataset."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_criteo_pipeline(
        input_path=args.input_path,
        max_rows=args.max_rows,
        sample_seed=args.sample_seed,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _validate_columns(fieldnames: Sequence[str]) -> None:
    if "treatment" not in fieldnames:
        raise ValueError("Criteo input requires a treatment column")
    if "conversion" not in fieldnames and "visit" not in fieldnames:
        raise ValueError("Criteo input requires conversion or visit outcome")
    if not any(column not in _NON_FEATURE_COLUMNS for column in fieldnames):
        raise ValueError("Criteo input requires at least one feature column")


def _binary(value: str) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise ValueError(f"expected a binary value, got {value!r}")


def _feature_value(value: str) -> float:
    normalized = str(value).strip()
    if not normalized:
        return 0.0
    try:
        return float(normalized)
    except ValueError:
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        return bucket * 2.0 - 1.0


def _stable_train_test_split(
    examples: Sequence[UpliftTrainingExample],
    *,
    sample_seed: int,
) -> tuple[list[UpliftTrainingExample], list[UpliftTrainingExample]]:
    by_arm = {
        arm: [example for example in examples if example.treatment == arm]
        for arm in (0, 1)
    }
    if any(len(arm_examples) < 2 for arm_examples in by_arm.values()):
        raise ValueError(
            "Criteo train/test split requires at least two treatment and control rows"
        )

    train_ids: set[str] = set()
    for arm in (0, 1):
        ordered = sorted(
            by_arm[arm],
            key=lambda example: (
                _split_hash(example.experiment_unit_id, sample_seed),
                example.experiment_unit_id,
            ),
        )
        raw_train_count = int(len(ordered) * 0.7 + 0.5)
        train_count = min(max(raw_train_count, 1), len(ordered) - 1)
        train_ids.update(
            example.experiment_unit_id for example in ordered[:train_count]
        )

    train = [
        example
        for example in examples
        if example.experiment_unit_id in train_ids
    ]
    test = [
        example
        for example in examples
        if example.experiment_unit_id not in train_ids
    ]
    return train, test


def _split_hash(experiment_unit_id: str, sample_seed: int) -> str:
    payload = (
        f"{CRITEO_SPLIT_POLICY_VERSION}\x00{sample_seed}\x00"
        f"{experiment_unit_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_fingerprint(examples: Iterable[UpliftTrainingExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.experiment_unit_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
