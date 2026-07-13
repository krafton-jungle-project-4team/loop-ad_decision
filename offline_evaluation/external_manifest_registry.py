from __future__ import annotations

from pathlib import Path

from offline_evaluation.external_final_test import (
    ExternalSealedFinalTestManifest,
    load_external_sealed_final_test_manifest,
)


class ExternalManifestRegistryError(ValueError):
    """Raised when a public sealed manifest cannot be resolved safely."""


class ExternalManifestRegistry:
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

    def path_for(self, manifest_id: str) -> Path:
        _validate_manifest_id(manifest_id)
        path = (self._root / f"{manifest_id}.json").resolve()
        if path.parent != self._root:
            raise ExternalManifestRegistryError(
                "manifest path escaped the registry"
            )
        return path

    def load(self, manifest_id: str) -> ExternalSealedFinalTestManifest:
        path = self.path_for(manifest_id)
        if not path.is_file():
            raise ExternalManifestRegistryError(
                "manifest is not published in the Git registry"
            )
        try:
            manifest = load_external_sealed_final_test_manifest(path)
        except (OSError, ValueError) as exc:
            raise ExternalManifestRegistryError(
                "registered manifest is invalid"
            ) from exc
        if manifest.manifest_id != manifest_id:
            raise ExternalManifestRegistryError(
                "registered manifest filename and identity differ"
            )
        return manifest

    def register(
        self,
        manifest_path: Path,
    ) -> tuple[ExternalSealedFinalTestManifest, Path]:
        try:
            manifest = load_external_sealed_final_test_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            raise ExternalManifestRegistryError(
                "manifest cannot be registered because it is invalid"
            ) from exc
        destination = self.path_for(manifest.manifest_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as output:
                output.write(manifest_path.read_bytes())
        except FileExistsError as exc:
            raise ExternalManifestRegistryError(
                "manifest is already registered; do not overwrite it"
            ) from exc
        return manifest, destination

    def list(self) -> tuple[ExternalSealedFinalTestManifest, ...]:
        if not self._root.is_dir():
            return ()
        manifests: list[ExternalSealedFinalTestManifest] = []
        for path in sorted(self._root.glob("*.json")):
            manifests.append(self.load(path.stem))
        return tuple(manifests)


def _validate_manifest_id(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ExternalManifestRegistryError(
            "manifest ID must be a lowercase SHA-256"
        )
