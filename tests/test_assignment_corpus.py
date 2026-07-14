from __future__ import annotations

import json
import sys

import pytest


np = pytest.importorskip("numpy")

from offline_evaluation.assignment_corpus import (  # noqa: E402
    CORPUS_FORMAT_VERSION,
    LEGACY_CORPUS_FORMAT_VERSION,
    LEGACY_SOURCE_CUTOFF_ATTESTATION,
    LEGACY_UNATTESTED_PROVENANCE_MODE,
    PROVIDED_DISTRIBUTION,
    PROVIDED_PROVENANCE_MODE,
    PROVIDED_SOURCE_CUTOFF_ATTESTATION,
    SYNTHETIC_PROVENANCE_MODE,
    SYNTHETIC_SOURCE_CUTOFF_ATTESTATION,
    UNATTESTED_PROVENANCE_MODE,
    UNATTESTED_SOURCE_CUTOFF_ATTESTATION,
    compute_corpus_sha256,
    freeze_corpus,
    generate_synthetic_corpus,
    load_frozen_corpus,
    write_frozen_corpus,
)
from scripts.export_segment_assignment_corpus import main as export_main  # noqa: E402


def _vectors(count: int) -> object:
    values = np.zeros((count, 64), dtype=np.float32)
    for index in range(count):
        values[index, index % 64] = index + 1
    return values


def _freeze(user_ids: list[str], user_vectors: object):
    return freeze_corpus(
        user_ids=user_ids,
        user_vectors=user_vectors,
        segment_ids=["segment-z", "segment-a"],
        segment_vectors=_vectors(2),
        vector_version="v1",
        source_cutoff_at="2026-01-01T00:00:00Z",
        distribution="fixture",
        random_seed=7,
        git_commit="abc123",
        id_hash_salt="test-salt",
        matcher_config={"candidate_k": 2},
        provenance_mode=UNATTESTED_PROVENANCE_MODE,
        source_cutoff_attestation=UNATTESTED_SOURCE_CUTOFF_ATTESTATION,
    )


def test_freeze_is_canonical_normalized_and_pseudonymous(tmp_path) -> None:
    first = _freeze(["raw-user-b", "raw-user-a"], _vectors(2))
    second = _freeze(
        ["raw-user-a", "raw-user-b"],
        _vectors(2)[[1, 0]],
    )

    assert first.manifest.corpus_sha256 == second.manifest.corpus_sha256
    assert np.allclose(np.linalg.norm(first.user_vectors, axis=1), 1.0)
    assert all(value.startswith("user_sha256_") for value in first.user_ids)
    assert first.segment_ids == ("segment-a", "segment-z")

    destination = tmp_path / "corpus.jsonl"
    write_frozen_corpus(first, destination)
    text = destination.read_text(encoding="utf-8")
    assert "raw-user-a" not in text
    assert "raw-user-b" not in text
    assert "segment-a" in text
    loaded = load_frozen_corpus(destination)
    assert loaded.manifest.corpus_sha256 == first.manifest.corpus_sha256
    assert loaded.manifest.format_version == CORPUS_FORMAT_VERSION
    assert loaded.manifest.provenance_mode == UNATTESTED_PROVENANCE_MODE
    assert (
        loaded.manifest.source_cutoff_attestation
        == UNATTESTED_SOURCE_CUTOFF_ATTESTATION
    )
    with pytest.raises(FileExistsError):
        write_frozen_corpus(first, destination)


def test_freeze_rejects_invalid_dimension_and_zero_vector() -> None:
    with pytest.raises(ValueError, match="shape"):
        _freeze(["user"], np.ones((1, 63), dtype=np.float32))
    with pytest.raises(ValueError, match="zero"):
        _freeze(["user"], np.zeros((1, 64), dtype=np.float32))


def test_load_detects_vector_tampering(tmp_path) -> None:
    destination = tmp_path / "corpus.jsonl"
    write_frozen_corpus(_freeze(["user"], _vectors(1)), destination)
    rows = destination.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[1])
    payload["vector"][0] = 0.5
    rows[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="normalized|SHA-256"):
        load_frozen_corpus(destination)


def test_synthetic_corpus_records_explicit_generation_provenance() -> None:
    corpus = generate_synthetic_corpus(
        user_count=2,
        segment_count=2,
        distribution="random",
        random_seed=19,
        git_commit="abc123",
    )

    assert corpus.manifest.provenance_mode == SYNTHETIC_PROVENANCE_MODE
    assert (
        corpus.manifest.source_cutoff_attestation
        == SYNTHETIC_SOURCE_CUTOFF_ATTESTATION
    )
    assert corpus.manifest.random_seed == 19


def test_raw_export_records_provided_provenance_without_synthetic_seed(
    tmp_path,
    monkeypatch,
) -> None:
    input_path, salt_path = _write_raw_export_inputs(tmp_path)
    destination = tmp_path / "provided-corpus.jsonl"
    cutoff = "2026-07-01T00:00:00Z"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_segment_assignment_corpus.py",
            "--output",
            str(destination),
            "--input-jsonl",
            str(input_path),
            "--id-hash-salt-file",
            str(salt_path),
            "--source-cutoff",
            cutoff,
            "--attest-source-cutoff",
        ],
    )

    assert export_main() == 0
    corpus = load_frozen_corpus(destination)
    assert corpus.manifest.distribution == PROVIDED_DISTRIBUTION
    assert corpus.manifest.random_seed is None
    assert corpus.manifest.provenance_mode == PROVIDED_PROVENANCE_MODE
    assert (
        corpus.manifest.source_cutoff_attestation
        == PROVIDED_SOURCE_CUTOFF_ATTESTATION
    )
    assert corpus.manifest.source_cutoff_at == cutoff


@pytest.mark.parametrize(
    "extra_args",
    (
        ("--attest-source-cutoff",),
        ("--source-cutoff", "2026-07-01T00:00:00Z"),
        (
            "--source-cutoff",
            "2026-07-01T00:00:00Z",
            "--attest-source-cutoff",
            "--distribution",
            "random",
        ),
        (
            "--source-cutoff",
            "2026-07-01T00:00:00Z",
            "--attest-source-cutoff",
            "--seed",
            "7",
        ),
    ),
)
def test_raw_export_requires_attestation_and_rejects_synthetic_metadata(
    tmp_path,
    monkeypatch,
    extra_args,
) -> None:
    input_path, salt_path = _write_raw_export_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_segment_assignment_corpus.py",
            "--output",
            str(tmp_path / "rejected.jsonl"),
            "--input-jsonl",
            str(input_path),
            "--id-hash-salt-file",
            str(salt_path),
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        export_main()


def test_v2_corpus_loads_only_with_legacy_unattested_provenance(tmp_path) -> None:
    corpus = _freeze(["user"], _vectors(1))
    destination = tmp_path / "legacy-v2.jsonl"
    write_frozen_corpus(corpus, destination)
    rows = [
        json.loads(value)
        for value in destination.read_text(encoding="utf-8").splitlines()
    ]
    manifest = rows[0]["manifest"]
    manifest["format_version"] = LEGACY_CORPUS_FORMAT_VERSION
    manifest.pop("provenance_mode")
    manifest.pop("source_cutoff_attestation")
    manifest["corpus_sha256"] = compute_corpus_sha256(
        user_ids=corpus.user_ids,
        user_vectors=corpus.user_vectors,
        segment_ids=corpus.segment_ids,
        segment_vectors=corpus.segment_vectors,
        stable_metadata={
            "format_version": LEGACY_CORPUS_FORMAT_VERSION,
            "dimension": corpus.manifest.dimension,
            "vector_version": corpus.manifest.vector_version,
            "source_cutoff_at": corpus.manifest.source_cutoff_at,
            "distribution": corpus.manifest.distribution,
            "random_seed": corpus.manifest.random_seed,
        },
    )
    destination.write_text(
        "\n".join(json.dumps(value) for value in rows) + "\n",
        encoding="utf-8",
    )

    loaded = load_frozen_corpus(destination)
    assert loaded.manifest.format_version == LEGACY_CORPUS_FORMAT_VERSION
    assert loaded.manifest.provenance_mode == LEGACY_UNATTESTED_PROVENANCE_MODE
    assert (
        loaded.manifest.source_cutoff_attestation
        == LEGACY_SOURCE_CUTOFF_ATTESTATION
    )


def _write_raw_export_inputs(tmp_path):
    input_path = tmp_path / "raw.jsonl"
    input_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {"kind": "user", "id": "raw-user", "vector": _vectors(1)[0].tolist()}
                ),
                json.dumps(
                    {
                        "kind": "segment",
                        "id": "segment-a",
                        "vector": _vectors(1)[0].tolist(),
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    salt_path = tmp_path / "salt.txt"
    salt_path.write_text("test-salt\n", encoding="utf-8")
    return input_path, salt_path
