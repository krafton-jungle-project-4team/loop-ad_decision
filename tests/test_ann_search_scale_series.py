from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from offline_evaluation.ann_search_scale_artifacts import (
    ActualScalePoint,
    AppendOnlyJsonl,
    ScaleArtifactValidator,
    checkpoint_payload,
    sha256_file,
    write_actual_only_reports,
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (
    COHORT_MEMBERSHIP_RELATION,
    SCALE_EXPERIMENT_VERSION,
    SCALE_OUTPUT_ROOT,
    ExpectedMemberBucket,
    HardMatchBucket,
    ScaleCohortScope,
    ScenarioCensusRecord,
    ScenarioSet,
    ScreeningCandidate,
    bind_query_to_cohort,
    build_environment_manifest,
    build_fingerprint,
    build_resource_manifest,
    cohort_bound_hard_match_query,
    cohort_bound_reference_sample_query,
    confirmation_baseline_k,
    decide_artifact_reuse,
    expected_member_bucket,
    hard_match_bucket,
    inspect_fingerprint,
    membership_population_sql,
    old_50k_reuse_decision,
    rank_cohort_members,
    scale_manifest_template,
    select_scenario_pairs,
    select_survivors,
    validate_cohort_bound_users,
    validate_exact_result_sets,
    validate_nested_prefixes,
)
from offline_evaluation.ann_search_benchmark import (
    BenchmarkManifest,
    BenchmarkScenario,
    LiveAnnSearchBenchmark,
)
from offline_evaluation.ann_search_experiment import HnswSettings


HASH = "a" * 64


def fingerprint(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "scale_series_id": "expedia-full-2015-v2",
        "project_id": "expedia-ann-scale-v2",
        "vector_version": "hotel_behavior.v2",
        "vector_manifest_hash": HASH,
        "vector_generation_id": "generation-v2",
        "window_start": "2013-01-01T00:00:00+00:00",
        "window_end": "2015-01-01T00:00:00+00:00",
        "source_revision_cutoff": "2026-07-18T00:00:00+00:00",
        "source_user_count": 1_000_000,
        "source_vector_revision_count": 1_000_000,
        "raw_event_count": 50_000_000,
        "cohort_seed": "ann-scale-v2",
        "membership_sha256": HASH,
        "scenario_manifest_sha256": HASH,
        "environment_sha256": HASH,
        "resource_manifest_sha256": HASH,
        "code_revision": "deadbeef",
    }
    payload.update(overrides)
    return payload


def candidate(
    *,
    scenario_id: str = "target-a",
    scenario_set: ScenarioSet = ScenarioSet.TUNING,
    k: int = 5_000,
    ann_p95: float = 115.0,
    exact_all_p95: float = 100.0,
    filter_p95: float | None = 60.0,
    filter_equal: bool = True,
) -> ScreeningCandidate:
    return ScreeningCandidate(
        scenario_id=scenario_id,
        candidate_type="target_destination_affinity",
        scenario_set=scenario_set,
        corpus_user_count=50_000,
        requested_k=k,
        exact_positive_count=1_000,
        recall=0.95,
        ann_p95_ms=ann_p95,
        exact_all_p95_ms=exact_all_p95,
        filter_first_p95_ms=filter_p95,
        filter_first_results_equal=filter_equal,
        hnsw_index_used=True,
        temp_spill=False,
        oom=False,
    )


def test_v2_manifest_is_separate_and_fixed() -> None:
    payload = scale_manifest_template()
    assert payload["experiment_version"] == SCALE_EXPERIMENT_VERSION
    assert payload["output_root"] == str(SCALE_OUTPUT_ROOT)
    assert payload["vector_version"] == "hotel_behavior.v2"
    assert payload["window_start"] == "2013-01-01T00:00:00+00:00"
    assert payload["window_end"] == "2015-01-01T00:00:00+00:00"
    assert payload["candidate_policy_generated"] is False
    assert payload["goal2_executed"] is False


def test_cohort_order_and_every_smaller_cohort_are_exact_prefixes() -> None:
    memberships = rank_cohort_members(
        scale_series_id="series",
        cohort_seed="seed",
        user_ids=(f"user-{index}" for index in range(20)),
    )
    repeated = rank_cohort_members(
        scale_series_id="series",
        cohort_seed="seed",
        user_ids=reversed([f"user-{index}" for index in range(20)]),
    )
    assert memberships == repeated
    hashes = validate_nested_prefixes(memberships, (5, 10, 20))
    assert set(hashes) == {5, 10, 20}
    assert len(set(hashes.values())) == 3


def test_cohort_bound_queries_join_frozen_signals_to_membership() -> None:
    h_query = cohort_bound_hard_match_query(("hotel_product_interest",))
    p_query = cohort_bound_reference_sample_query(("hotel_product_interest",))
    for query in (h_query, p_query):
        assert "FROM ann_benchmark_user_signals" in query
        assert f"INNER JOIN {COHORT_MEMBERSHIP_RELATION}" in query
        assert "cohort_rank <= {cohort_size:UInt64}" in query
        assert "FROM raw_events" not in query
    assert "SHA256(concat({reference_sample_seed:String}, '|', user_id))" in p_query
    assert "LIMIT least({reference_sample_size:UInt32}, {cohort_size:UInt64})" in p_query
    with pytest.raises(ValueError, match="one raw_events"):
        bind_query_to_cohort("SELECT 1")


def test_membership_population_uses_exact_frozen_vector_order() -> None:
    query = membership_population_sql()
    assert "GROUP BY user_id" in query
    assert "ingested_at <= {source_revision_cutoff" in query
    assert "SHA256(concat({cohort_seed:String}, '|', user_id))" in query
    assert "ORDER BY cohort_rank" in query


def test_outside_users_are_rejected_for_h_p_and_filter_first() -> None:
    validate_cohort_bound_users(
        cohort_user_ids=("a", "b"),
        hard_match_user_ids=("a",),
        reference_sample_user_ids=("b",),
        filter_first_user_ids=("a", "b"),
    )
    with pytest.raises(ValueError, match="P contains users outside"):
        validate_cohort_bound_users(
            cohort_user_ids=("a", "b"),
            reference_sample_user_ids=("c",),
        )


def test_exact_all_and_filter_first_must_have_identical_sets() -> None:
    validate_exact_result_sets(("a", "b"), ("b", "a"))
    with pytest.raises(ValueError, match="differs from exact_all"):
        validate_exact_result_sets(("a", "b"), ("a", "c"))


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, HardMatchBucket.LE_005),
        (0.05, HardMatchBucket.LE_005),
        (0.0500001, HardMatchBucket.GT_005_LE_020),
        (0.20, HardMatchBucket.GT_005_LE_020),
        (0.200001, HardMatchBucket.GT_020),
        (1.0, HardMatchBucket.GT_020),
    ],
)
def test_hard_match_ratio_boundaries(
    ratio: float,
    expected: HardMatchBucket,
) -> None:
    assert hard_match_bucket(ratio) == expected


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, ExpectedMemberBucket.LE_001),
        (0.01, ExpectedMemberBucket.LE_001),
        (0.010001, ExpectedMemberBucket.GT_001_LE_005),
        (0.05, ExpectedMemberBucket.GT_001_LE_005),
        (0.050001, ExpectedMemberBucket.GT_005_LE_010),
        (0.10, ExpectedMemberBucket.GT_005_LE_010),
        (0.100001, ExpectedMemberBucket.GT_010_LE_025),
        (0.25, ExpectedMemberBucket.GT_010_LE_025),
        (0.250001, ExpectedMemberBucket.GT_025),
        (1.0, ExpectedMemberBucket.GT_025),
    ],
)
def test_expected_member_ratio_boundaries(
    ratio: float,
    expected: ExpectedMemberBucket,
) -> None:
    assert expected_member_bucket(ratio) == expected


def test_scenario_pair_requires_distinct_tuning_and_confirmation() -> None:
    base = {
        "candidate_type": "intent_matched",
        "corpus_user_count": 50_000,
        "hard_match_user_count": 2_000,
        "estimated_member_count": 100.0,
        "exact_positive_count": 100,
    }
    records = (
        ScenarioCensusRecord(
            scenario_id="intent-tuning",
            definition_sha256="1" * 64,
            scenario_set=ScenarioSet.TUNING,
            **base,
        ),
        ScenarioCensusRecord(
            scenario_id="intent-confirmation",
            definition_sha256="2" * 64,
            scenario_set=ScenarioSet.CONFIRMATION,
            **base,
        ),
    )
    pair = select_scenario_pairs(records)[0]
    assert pair.validated is True
    assert pair.tuning_scenario_id == "intent-tuning"
    assert pair.confirmation_scenario_id == "intent-confirmation"
    fallback = select_scenario_pairs(records[:1])[0]
    assert fallback.validated is False


def test_scale_survivor_uses_exact_all_policy_uses_fastest_valid_exact() -> None:
    scale_only = candidate()
    invalid_filter = candidate(
        scenario_id="target-b",
        k=10_000,
        filter_equal=False,
    )
    selected = select_survivors((scale_only, invalid_filter))
    assert [item.scenario_id for item in selected.scale] == ["target-a", "target-b"]
    assert [item.scenario_id for item in selected.policy] == ["target-b"]
    assert {item.identity for item in selected.goal2_union} == (
        {item.identity for item in selected.scale}
        | {item.identity for item in selected.policy}
    )


def test_confirmation_never_selects_a_survivor() -> None:
    confirmation = candidate(
        scenario_id="target-confirmation",
        scenario_set=ScenarioSet.CONFIRMATION,
    )
    selected = select_survivors((confirmation,))
    assert selected.scale == ()
    assert selected.policy == ()
    assert selected.goal2_union == ()


def test_confirmation_baseline_k_formula() -> None:
    assert confirmation_baseline_k(corpus_user_count=50_000, estimated_member_count=0) == 5_000
    assert confirmation_baseline_k(corpus_user_count=50_000, estimated_member_count=10_000) == 15_000
    assert confirmation_baseline_k(corpus_user_count=10_000, estimated_member_count=20_000) == 10_000


def test_incomplete_fingerprint_refuses_reuse() -> None:
    complete = fingerprint()
    assert inspect_fingerprint(complete).complete is True
    incomplete = dict(complete)
    incomplete.pop("resource_manifest_sha256")
    decision = decide_artifact_reuse(incomplete, complete)
    assert decision.reusable is False
    assert decision.reason == "existing fingerprint incomplete"
    assert old_50k_reuse_decision() == {
        "decision": "pilot_only",
        "latency_reused": False,
        "ground_truth_reused": False,
        "reason": "historical fingerprint incomplete and new full-source snapshot differs",
    }


def test_environment_resource_and_fingerprint_are_sealable(tmp_path: Path) -> None:
    environment = build_environment_manifest(
        code_revision="deadbeef",
        postgres_version="17",
        pgvector_version="0.8.0",
        clickhouse_version="25.1",
        container_identity={"postgres": "pg-image", "clickhouse": "ch-image"},
    )
    resources = build_resource_manifest(
        postgres_settings={"work_mem": "4MB"},
        clickhouse_settings={"max_threads": "4"},
        container_limits={"postgres": {"memory_bytes": 1_000_000}},
        filesystem_path=tmp_path,
        logical_cpu_count=4,
        physical_memory_bytes=8_000_000,
    )
    assert environment["experiment_version"] == SCALE_EXPERIMENT_VERSION
    assert resources["logical_cpu_count"] == 4
    sealed = build_fingerprint(**fingerprint())
    assert sealed["fingerprint_sha256"] == inspect_fingerprint(sealed).fingerprint_sha256


class _NamedResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def named_results(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)


class _RecordingClickHouse:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict[str, object]]] = []

    def query(self, query: str, parameters=None):  # type: ignore[no-untyped-def]
        self.queries.append((query, dict(parameters or {})))
        return _NamedResult(self.rows)


class _ScalePostgres:
    def __init__(self, users: tuple[str, ...]) -> None:
        self.users = users
        self.sampled_count = 0

    def fetchall(self, query, params=()):  # type: ignore[no-untyped-def]
        return [{"user_id": value} for value in self.users]

    def fetchone(self, query, params=()):  # type: ignore[no-untyped-def]
        return {"sampled_count": self.sampled_count, "passed_count": self.sampled_count}


class _ScaleRepository:
    def __init__(self, postgres: _ScalePostgres) -> None:
        self.postgres = postgres

    def _replace_temp_user_ids(self, *, table_name, user_ids):  # type: ignore[no-untyped-def]
        assert table_name == "audience_hard_match_sample"
        self.postgres.sampled_count = len(set(user_ids))


def _live_scale_benchmark(
    *,
    clickhouse_rows: list[dict[str, object]],
) -> tuple[LiveAnnSearchBenchmark, _RecordingClickHouse]:
    postgres = _ScalePostgres(("inside-a", "inside-b"))
    clickhouse = _RecordingClickHouse(clickhouse_rows)
    benchmark = object.__new__(LiveAnnSearchBenchmark)
    benchmark._clickhouse = clickhouse
    benchmark._postgres = postgres
    benchmark._repository = _ScaleRepository(postgres)
    benchmark._scale_cohort_scope = ScaleCohortScope(
        scale_series_id="series-v2",
        cohort_size=2,
        reference_sample_seed="sample-seed",
    )
    benchmark._cohort_user_ids_by_generation = {}
    benchmark._cohort_hard_counts = {}
    return benchmark, clickhouse


def _scale_manifest_scenario_context():
    scenario = BenchmarkScenario(
        scenario_id="intent-tuning-a",
        candidate_type="intent_matched",
        query_vector=(0.0,) * 64,
        score_threshold=0.5,
        hard_predicate_keys=("hotel_product_interest",),
        predicate_parameters={},
        compiler_provenance={
            "manifest_hash": "manifest",
            "calibration_version": "v1",
            "calibration_hash": "calibration",
            "query_compiler_version": "v1",
            "query_compiler_hash": "compiler",
            "template_id": "template",
            "template_semantic_hash": "semantic",
        },
    )
    manifest = BenchmarkManifest(
        project_id="project",
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest",
        scenarios=(scenario,),
        experiment_version=SCALE_EXPERIMENT_VERSION,
    )
    context = SimpleNamespace(
        vector_generation_id="generation",
        source_revision_cutoff=datetime(2026, 7, 18, tzinfo=UTC),
        window_start=datetime(2013, 1, 1, tzinfo=UTC),
        source_cutoff=datetime(2015, 1, 1, tzinfo=UTC),
        exclusion_context=None,
    )
    return manifest, scenario, context


def test_live_filter_first_and_p_queries_reject_cohort_outsiders() -> None:
    manifest, scenario, context = _scale_manifest_scenario_context()
    benchmark, clickhouse = _live_scale_benchmark(
        clickhouse_rows=[{"user_id": "inside-a"}],
    )
    assert benchmark._hard_match_user_ids(
        manifest=manifest, scenario=scenario, context=context
    ) == ["inside-a"]
    query, parameters = clickhouse.queries[-1]
    assert COHORT_MEMBERSHIP_RELATION in query
    assert parameters["cohort_size"] == 2
    assert benchmark._cohort_score_pass_rate(
        manifest=manifest, scenario=scenario, context=context
    ) == 1.0

    outsider_benchmark, _ = _live_scale_benchmark(
        clickhouse_rows=[{"user_id": "outside"}],
    )
    with pytest.raises(ValueError, match="outside the cohort"):
        outsider_benchmark._hard_match_user_ids(
            manifest=manifest, scenario=scenario, context=context
        )
    with pytest.raises(ValueError, match="outside the cohort"):
        outsider_benchmark._cohort_score_pass_rate(
            manifest=manifest, scenario=scenario, context=context
        )


def test_scale_ann_applies_exact_score_filter_before_final_membership() -> None:
    manifest, scenario, context = _scale_manifest_scenario_context()
    benchmark = object.__new__(LiveAnnSearchBenchmark)
    benchmark._scale_cohort_scope = ScaleCohortScope(
        scale_series_id="series-v2",
        cohort_size=2,
        reference_sample_seed="sample-seed",
    )
    benchmark._postgres = SimpleNamespace(
        fetchall=lambda *_args, **_kwargs: [
            {"user_id": "score-pass", "behavior_fit_score": 0.9},
            {"user_id": "score-fail", "behavior_fit_score": 0.1},
        ]
    )
    benchmark._set_hnsw = lambda _settings: None
    captured: dict[str, object] = {}

    def exact_filter(
        self,  # type: ignore[no-untyped-def]
        *,
        manifest,
        scenario,
        context,
        user_ids,
    ):
        captured["user_ids"] = tuple(user_ids)
        return [
            SimpleNamespace(
                user_id="score-pass",
                behavior_fit_score=0.9,
                retrieval_rank=1,
            )
        ]

    benchmark._scale_exact_filter_user_ids = MethodType(exact_filter, benchmark)

    members = benchmark._ann_first(
        manifest=manifest,
        scenario=scenario,
        context=context,
        requested_k=2,
        hnsw=HnswSettings(100, "strict_order", 20_000),
    )

    assert captured["user_ids"] == ("score-pass", "score-fail")
    assert [member.user_id for member in members] == ["score-pass"]


def test_checkpoint_resume_never_duplicates_or_overwrites_rows(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    writer = AppendOnlyJsonl(path, identity_fields=("cell", "iteration"))
    first = {"cell": "a", "iteration": 1, "duration_ms": 10}
    result = writer.append((first,))
    assert result.appended_count == 1
    assert writer.append((first,)).skipped_identical_count == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ValueError, match="conflicting row"):
        writer.append(({**first, "duration_ms": 11},))
    immutable = tmp_path / "manifest.json"
    assert write_immutable_json(immutable, {"a": 1}) is True
    assert write_immutable_json(immutable, {"a": 1}) is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_immutable_json(immutable, {"a": 2})


def test_actual_only_report_uses_measured_points_and_writes_png(tmp_path: Path) -> None:
    points = (
        ActualScalePoint(50_000, "s", "b", "exact_all", 100.0, 10, True, HASH),
        ActualScalePoint(50_000, "s", "b", "ann_first", 80.0, 10, False, HASH),
        ActualScalePoint(100_000, "s", "b", "exact_all", 200.0, 10, True, HASH),
    )
    csv_path = tmp_path / "report.csv"
    md_path = tmp_path / "report.md"
    png_path = tmp_path / "report.png"
    write_actual_only_reports(
        points=points,
        expected_cohort_sizes=(50_000, 100_000),
        csv_path=csv_path,
        markdown_path=md_path,
        png_path=png_path,
    )
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 4
    assert "no ANN candidate" in md_path.read_text(encoding="utf-8")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="actual measurement"):
        ActualScalePoint(50_000, "s", "b", "exact_all", 1.0, 0, True, HASH)


def test_artifact_integrity_validator_checks_hashes_and_completeness(
    tmp_path: Path,
) -> None:
    root = tmp_path / SCALE_OUTPUT_ROOT
    raw = root / "cohorts/50000/raw.jsonl"
    result = root / "cohorts/50000/results.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("{}\n", encoding="utf-8")
    result.write_text("{}\n", encoding="utf-8")
    manifest = {
        **scale_manifest_template(),
        "status": "running",
        "completed_cohort_sizes": [50_000],
    }
    write_immutable_json(root / "manifest.json", manifest)
    write_immutable_json(root / "fingerprint.json", fingerprint())
    write_immutable_json(root / "old-50k-reuse-decision.json", old_50k_reuse_decision())
    checkpoint = checkpoint_payload(
        cohort_size=50_000,
        cohort_sha256=HASH,
        fingerprint_sha256=inspect_fingerprint(fingerprint()).fingerprint_sha256 or "",
        raw_path=raw,
        result_path=result,
        ground_truth={"s": {"row_count": 50_000, "sha256": HASH}},
        expected_scenario_count=1,
        expected_baseline_cell_count=4,
        observed_baseline_cell_count=4,
        scale_survivor_count=0,
        policy_survivor_count=0,
        rss_complete=True,
        spill_oom_complete=True,
    )
    write_immutable_json(root / "checkpoints/cohort-50000.json", checkpoint)
    validator = ScaleArtifactValidator(root)
    assert validator.validate().passed is True
    raw.write_text("changed\n", encoding="utf-8")
    report = validator.validate()
    assert report.passed is False
    assert any(issue.code == "artifact_hash" for issue in report.issues)


def test_checkpoint_hash_matches_written_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text('{"a":1}\n', encoding="utf-8")
    assert sha256_file(path) == "e346432021b04179518d9614f3560ccd71354a4ee101ddcb893d6959a9d6301c"
