from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from offline_evaluation.external_final_test import (
    EXTERNAL_SEALED_FINAL_TEST_VERSION,
)
from offline_evaluation.external_manifest_registry import (
    ExternalManifestRegistry,
    ExternalManifestRegistryError,
)


def test_registry_loads_manifest_by_public_identity(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / f"{manifest['manifest_id']}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = ExternalManifestRegistry(tmp_path).load(manifest["manifest_id"])

    assert loaded.manifest_id == manifest["manifest_id"]
    assert loaded.dataset_id == "airbnb"


def test_registry_rejects_path_input_and_missing_manifest(tmp_path: Path) -> None:
    registry = ExternalManifestRegistry(tmp_path)

    with pytest.raises(ExternalManifestRegistryError, match="lowercase SHA-256"):
        registry.load("../manifest.json")
    with pytest.raises(ExternalManifestRegistryError, match="not published"):
        registry.load("a" * 64)


def test_registry_registers_without_overwriting(tmp_path: Path) -> None:
    manifest = _manifest()
    source = tmp_path / "source.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    registry = ExternalManifestRegistry(tmp_path / "registry")

    registered, destination = registry.register(source)

    assert destination.name == f"{registered.manifest_id}.json"
    assert registry.list() == (registered,)
    with pytest.raises(ExternalManifestRegistryError, match="do not overwrite"):
        registry.register(source)


def _manifest() -> dict[str, Any]:
    stable = {
        "version": EXTERNAL_SEALED_FINAL_TEST_VERSION,
        "dataset_id": "airbnb",
        "code_commit": "c" * 40,
        "code_tree": "d" * 40,
        "source": {},
        "model": {},
        "adapter_config": {},
        "backtest_config": {},
        "partition_contract": {},
        "outcome_contract": {},
        "acceptance_criteria": {},
    }
    manifest_id = _json_sha256(stable)
    payload = {
        **stable,
        "manifest_id": manifest_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    return {**payload, "integrity_sha256": _json_sha256(payload)}


def _json_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
