from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.analysis.segment_performance import ContextualBookingHeuristicPredictor
from offline_evaluation.external_backtest import (
    ExternalBacktestConfig,
    ExternalBacktestError,
)
from offline_evaluation.external_datasets import (
    EXTERNAL_DEVELOPMENT_ROLE,
    EXTERNAL_SEALED_FINAL_ROLE,
    ExternalAdapterConfig,
)
from offline_evaluation.external_final_test import (
    EXTERNAL_COHORT_MODULO,
    EXTERNAL_DEVELOPMENT_REMAINDERS,
    EXTERNAL_FINAL_REMAINDERS,
    build_external_sealed_final_test_manifest,
    load_external_sealed_final_test_manifest,
    reserve_external_sealed_final_test_execution,
    run_external_sealed_final_test,
    verify_external_sealed_final_test_runtime,
    write_external_sealed_final_test_artifacts,
    write_external_sealed_final_test_manifest,
)


def test_external_development_and_final_user_cohorts_are_disjoint() -> None:
    development = ExternalAdapterConfig(
        sample_modulo=EXTERNAL_COHORT_MODULO,
        sample_remainders=EXTERNAL_DEVELOPMENT_REMAINDERS,
        evaluation_role=EXTERNAL_DEVELOPMENT_ROLE,
    )
    final = ExternalAdapterConfig(
        sample_modulo=EXTERNAL_COHORT_MODULO,
        sample_remainder=EXTERNAL_FINAL_REMAINDERS[0],
        sample_remainders=EXTERNAL_FINAL_REMAINDERS,
        evaluation_role=EXTERNAL_SEALED_FINAL_ROLE,
    )
    subjects = {f"user-{index}" for index in range(1000)}
    development_subjects = {
        subject for subject in subjects if development.includes_subject(subject)
    }
    final_subjects = {
        subject for subject in subjects if final.includes_subject(subject)
    }

    assert development_subjects
    assert final_subjects
    assert development_subjects.isdisjoint(final_subjects)
    assert development_subjects | final_subjects == subjects


def test_sealing_hashes_booking_outcome_file_without_parsing_it(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "booking"
    source_dir.mkdir()
    (source_dir / "test_set.csv").write_text("not,a,valid,test\n", encoding="utf-8")
    (source_dir / "ground_truth.csv").write_text(
        "not,a,valid,outcome\n",
        encoding="utf-8",
    )
    model_path = _model_file(tmp_path)

    manifest = _build_manifest(source_dir, model_path)

    assert manifest.source["selection_uses_outcomes"] is False
    assert manifest.model["trained_on_external_data"] is False
    assert manifest.outcome_contract["prediction_error_comparable"] is False
    assert manifest.partition_contract["final"]["disjoint_from_development"] is True


def test_external_manifest_integrity_and_runtime_fingerprint_are_enforced(
    tmp_path: Path,
) -> None:
    source_dir = _booking_fixture(tmp_path)
    model_path = _model_file(tmp_path)
    manifest = _build_manifest(source_dir, model_path)
    manifest_path = tmp_path / "external-final.json"
    write_external_sealed_final_test_manifest(manifest, manifest_path)

    loaded = load_external_sealed_final_test_manifest(manifest_path)
    verify_external_sealed_final_test_runtime(
        loaded,
        source_dir=source_dir,
        model_path=model_path,
        model_metadata=_model_metadata(),
        code_commit="commit-1",
        code_tree="tree-1",
    )
    with (source_dir / "ground_truth.csv").open("a", encoding="utf-8") as output:
        output.write("trip-new,999,Z\n")
    with pytest.raises(ValueError, match="source data changed"):
        verify_external_sealed_final_test_runtime(
            loaded,
            source_dir=source_dir,
            model_path=model_path,
            model_metadata=_model_metadata(),
            code_commit="commit-1",
            code_tree="tree-1",
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["adapter_config"]["profile_pool_limit"] = 999999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        load_external_sealed_final_test_manifest(manifest_path)


def test_execution_marker_blocks_second_external_final_run(tmp_path: Path) -> None:
    source_dir = _booking_fixture(tmp_path)
    model_path = _model_file(tmp_path)
    manifest = _build_manifest(source_dir, model_path)
    manifest_path = tmp_path / "external-final.json"
    write_external_sealed_final_test_manifest(manifest, manifest_path)

    marker = reserve_external_sealed_final_test_execution(
        manifest_path,
        manifest,
        code_commit="commit-1",
    )

    assert marker.is_file()
    with pytest.raises(ExternalBacktestError, match="already started"):
        reserve_external_sealed_final_test_execution(
            manifest_path,
            manifest,
            code_commit="commit-1",
        )


def test_external_final_run_uses_official_booking_ground_truth_and_writes_result(
    tmp_path: Path,
) -> None:
    source_dir = _booking_fixture(tmp_path)
    model_path = _model_file(tmp_path)
    predictor = ContextualBookingHeuristicPredictor()
    manifest = _build_manifest(
        source_dir,
        model_path,
        model_metadata=predictor.metadata(),
    )

    result = run_external_sealed_final_test(
        manifest=manifest,
        source_dir=source_dir,
        performance_predictor=predictor,
    )
    artifacts = write_external_sealed_final_test_artifacts(
        result,
        manifest=manifest,
        output_dir=tmp_path / "result",
        model_metadata=predictor.metadata(),
    )

    assert result.run.results
    assert result.dataset_manifest.evaluation_role == EXTERNAL_SEALED_FINAL_ROLE
    assert result.run.summary["prediction_error_comparable"] is False
    assert (
        result.run.summary["mean_absolute_prediction_error_percentage_points"]
        is None
    )
    assert artifacts["summary"].is_file()
    assert artifacts["report"].is_file()


def _build_manifest(
    source_dir: Path,
    model_path: Path,
    *,
    model_metadata: dict[str, object] | None = None,
):
    return build_external_sealed_final_test_manifest(
        dataset_id="booking-com",
        source_dir=source_dir,
        model_path=model_path,
        model_metadata=model_metadata or _model_metadata(),
        adapter_config=_final_adapter_config(),
        backtest_config=ExternalBacktestConfig(
            max_suggested_segments=3,
            min_sample_size=1,
            prediction_error_comparable=False,
            prediction_error_comparability_reason="different outcome",
        ),
        code_commit="commit-1",
        code_tree="tree-1",
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _final_adapter_config() -> ExternalAdapterConfig:
    return ExternalAdapterConfig(
        profile_pool_limit=20,
        max_scenarios=1,
        min_scenario_users=1,
        sample_modulo=1,
        sample_remainder=0,
        sample_remainders=(0,),
        evaluation_role=EXTERNAL_SEALED_FINAL_ROLE,
        include_checksum=True,
    )


def _booking_fixture(tmp_path: Path) -> Path:
    source_dir = tmp_path / "booking"
    source_dir.mkdir()
    test_rows: list[dict[str, str]] = []
    ground_truth_rows: list[dict[str, str]] = []
    for index in range(1, 7):
        trip_id = f"trip-{index}"
        test_rows.extend(
            [
                _booking_row(trip_id, "2016-01-01", "10"),
                _booking_row(
                    trip_id,
                    "2016-01-02",
                    "10" if index <= 4 else str(20 + index),
                ),
                _booking_row(trip_id, "2016-01-03", "0"),
            ]
        )
        ground_truth_rows.append(
            {
                "utrip_id": trip_id,
                "city_id": "10" if index in {1, 2, 5} else "99",
                "hotel_country": "A",
            }
        )
    _write_csv(source_dir / "test_set.csv", test_rows)
    _write_csv(source_dir / "ground_truth.csv", ground_truth_rows)
    return source_dir


def _booking_row(trip_id: str, checkin: str, city_id: str) -> dict[str, str]:
    return {
        "user_id": trip_id,
        "checkin": checkin,
        "checkout": checkin,
        "device_class": "desktop",
        "affiliate_id": "1",
        "booker_country": "A",
        "utrip_id": trip_id,
        "city_id": city_id,
        "hotel_country": "B" if city_id != "0" else "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.json"
    path.write_text('{"model": "fixture"}\n', encoding="utf-8")
    return path


def _model_metadata() -> dict[str, object]:
    return {
        "model_version": "fixture.v1",
        "training_dataset": "expedia",
    }
