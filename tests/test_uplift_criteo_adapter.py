import json
from pathlib import Path

import app.uplift.criteo_adapter as criteo_adapter
from app.uplift.criteo_adapter import main, validate_criteo_pipeline


def test_criteo_adapter_is_deterministic_and_never_serving_eligible(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "criteo.csv"
    input_path.write_text(
        "treatment,conversion,visit,f0,f1\n"
        + "\n".join(
            f"{index % 2},{int(index % 5 == 0)},{int(index % 3 == 0)},"
            f"{index / 10},category_{index % 4}"
            for index in range(100)
        )
        + "\n",
        encoding="utf-8",
    )

    first = validate_criteo_pipeline(
        input_path=input_path,
        max_rows=40,
        sample_seed=17,
    )
    second = validate_criteo_pipeline(
        input_path=input_path,
        max_rows=40,
        sample_seed=17,
    )

    assert first["adapter"]["sample_fingerprint"] == (
        second["adapter"]["sample_fingerprint"]
    )
    assert first["adapter"]["sampled_row_count"] == 40
    assert first["adapter"]["split_policy_version"] == (
        "stable-row-hash-stratified-70-30.v1"
    )
    assert first["adapter"]["train_count"] + first["adapter"]["test_count"] == 40
    assert first["adapter"]["train_sample_fingerprint"] == (
        second["adapter"]["train_sample_fingerprint"]
    )
    assert first["adapter"]["test_sample_fingerprint"] == (
        second["adapter"]["test_sample_fingerprint"]
    )
    assert first["metrics"]["observation_count"] == first["adapter"]["test_count"]
    assert "ate" in first["metrics"]
    assert "auuc" in first["metrics"]
    assert "qini" in first["metrics"]
    assert "uplift_at_top_k" in first["metrics"]
    assert first["model_metadata"] == {
        "model_lifecycle_status": "candidate",
        "validation_scope": "external_pipeline_validation",
        "dataset": "criteo_uplift",
        "serving_eligible": False,
        "model_version": "transformed-outcome-ridge.v1",
        "validation_policy_version": None,
    }
    assert first["serving_activation_evidence"] is False
    assert first["predicted_cate_cluster_variability_interval"][
        "reference_only"
    ] is True


def test_criteo_cli_writes_report_to_explicit_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "criteo.csv"
    output_path = tmp_path / "report.json"
    input_path.write_text(
        "treatment,conversion,f0\n"
        "1,1,0.9\n"
        "0,0,0.1\n"
        "1,0,0.8\n"
        "0,1,0.2\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input-path",
            str(input_path),
            "--max-rows",
            "4",
            "--sample-seed",
            "7",
            "--output-path",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["adapter"]["input_path"] == str(input_path)
    assert report["model_metadata"]["serving_eligible"] is False


def test_criteo_model_fits_train_and_evaluates_disjoint_test_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "criteo.csv"
    input_path.write_text(
        "treatment,conversion,f0\n"
        + "\n".join(
            f"{index % 2},{int(index % 7 == 0)},{index / 100}"
            for index in range(100)
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}
    real_fit = criteo_adapter.fit_transformed_outcome_ridge
    real_evaluate = criteo_adapter.evaluate_uplift_predictions

    def capture_fit(examples, **kwargs):
        captured["train_ids"] = {
            example.experiment_unit_id for example in examples
        }
        return real_fit(examples, **kwargs)

    def capture_evaluate(examples, scores, **kwargs):
        captured["test_ids"] = {
            example.experiment_unit_id for example in examples
        }
        return real_evaluate(examples, scores, **kwargs)

    monkeypatch.setattr(
        criteo_adapter,
        "fit_transformed_outcome_ridge",
        capture_fit,
    )
    monkeypatch.setattr(
        criteo_adapter,
        "evaluate_uplift_predictions",
        capture_evaluate,
    )

    report = validate_criteo_pipeline(
        input_path=input_path,
        max_rows=80,
        sample_seed=23,
    )

    assert captured["train_ids"].isdisjoint(captured["test_ids"])
    assert len(captured["train_ids"]) == report["adapter"]["train_count"]
    assert len(captured["test_ids"]) == report["adapter"]["test_count"]
    assert report["metrics"]["observation_count"] == len(captured["test_ids"])
