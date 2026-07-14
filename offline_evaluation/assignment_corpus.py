from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


CORPUS_FORMAT_VERSION = "loopad.assignment-corpus.v3"
LEGACY_CORPUS_FORMAT_VERSION = "loopad.assignment-corpus.v2"
VECTOR_DIM = 64
SUPPORTED_DISTRIBUTIONS = (
    "random",
    "clustered",
    "threshold_near",
    "low_margin",
)
PROVIDED_DISTRIBUTION = "provided"
SYNTHETIC_PROVENANCE_MODE = "synthetic_generator"
PROVIDED_PROVENANCE_MODE = "provided_input"
UNATTESTED_PROVENANCE_MODE = "programmatic_unattested"
LEGACY_UNATTESTED_PROVENANCE_MODE = "legacy_v2_unattested"
SYNTHETIC_SOURCE_CUTOFF_ATTESTATION = "synthetic_scenario_metadata"
PROVIDED_SOURCE_CUTOFF_ATTESTATION = (
    "caller_attested_input_filtered_at_or_before_cutoff"
)
UNATTESTED_SOURCE_CUTOFF_ATTESTATION = "not_attested"
LEGACY_SOURCE_CUTOFF_ATTESTATION = "not_recorded_in_v2"
THREAD_CONTROL_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    format_version: str
    corpus_sha256: str
    user_count: int
    segment_count: int
    dimension: int
    vector_version: str
    source_cutoff_at: str
    distribution: str
    random_seed: int | None
    provenance_mode: str
    source_cutoff_attestation: str
    git_commit: str
    numpy_version: str
    faiss_version: str | None
    cpu_model: str
    cpu_count: int | None
    thread_settings: Mapping[str, Any]
    matcher_config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrozenAssignmentCorpus:
    manifest: CorpusManifest
    user_ids: tuple[str, ...]
    user_vectors: Any
    segment_ids: tuple[str, ...]
    segment_vectors: Any


def freeze_corpus(
    *,
    user_ids: Sequence[str],
    user_vectors: Any,
    segment_ids: Sequence[str],
    segment_vectors: Any,
    vector_version: str,
    source_cutoff_at: str,
    distribution: str,
    random_seed: int | None,
    git_commit: str,
    id_hash_salt: str,
    matcher_config: Mapping[str, Any] | None = None,
    provenance_mode: str = UNATTESTED_PROVENANCE_MODE,
    source_cutoff_attestation: str = UNATTESTED_SOURCE_CUTOFF_ATTESTATION,
) -> FrozenAssignmentCorpus:
    np = _numpy()
    stable_random_seed = _normalize_random_seed(random_seed)
    _validate_manifest_inputs(
        vector_version=vector_version,
        source_cutoff_at=source_cutoff_at,
        distribution=distribution,
        random_seed=stable_random_seed,
        provenance_mode=provenance_mode,
        source_cutoff_attestation=source_cutoff_attestation,
        git_commit=git_commit,
        id_hash_salt=id_hash_salt,
    )
    normalized_users = normalize_float32_matrix(user_vectors, name="user_vectors")
    normalized_segments = normalize_float32_matrix(
        segment_vectors,
        name="segment_vectors",
    )
    if len(user_ids) != normalized_users.shape[0]:
        raise ValueError("user_ids count must match user_vectors rows")
    if len(segment_ids) != normalized_segments.shape[0]:
        raise ValueError("segment_ids count must match segment_vectors rows")
    if not user_ids:
        raise ValueError("corpus must contain at least one user")
    if not segment_ids:
        raise ValueError("corpus must contain at least one segment")

    hashed_users = tuple(
        hash_identifier(value, namespace="user", salt=id_hash_salt)
        for value in user_ids
    )
    # Segment IDs are durable, non-personal assignment keys. Preserve their
    # lexical order because it is the production tie-break contract. Hashing
    # them would order by digest and could select a different segment on ties.
    stable_segment_ids = tuple(str(value) for value in segment_ids)
    _validate_unique_ids(hashed_users, name="user")
    _validate_plain_ids(stable_segment_ids, name="segment")

    user_order = np.argsort(np.asarray(hashed_users), kind="stable")
    segment_order = np.argsort(np.asarray(stable_segment_ids), kind="stable")
    ordered_user_ids = tuple(hashed_users[int(index)] for index in user_order)
    ordered_segment_ids = tuple(
        stable_segment_ids[int(index)] for index in segment_order
    )
    ordered_users = np.ascontiguousarray(normalized_users[user_order], dtype=np.float32)
    ordered_segments = np.ascontiguousarray(
        normalized_segments[segment_order],
        dtype=np.float32,
    )

    stable_metadata = {
        "format_version": CORPUS_FORMAT_VERSION,
        "dimension": VECTOR_DIM,
        "vector_version": vector_version,
        "source_cutoff_at": source_cutoff_at,
        "distribution": distribution,
        "random_seed": stable_random_seed,
        "provenance_mode": provenance_mode,
        "source_cutoff_attestation": source_cutoff_attestation,
    }
    digest = compute_corpus_sha256(
        user_ids=ordered_user_ids,
        user_vectors=ordered_users,
        segment_ids=ordered_segment_ids,
        segment_vectors=ordered_segments,
        stable_metadata=stable_metadata,
    )
    manifest = CorpusManifest(
        format_version=CORPUS_FORMAT_VERSION,
        corpus_sha256=digest,
        user_count=len(ordered_user_ids),
        segment_count=len(ordered_segment_ids),
        dimension=VECTOR_DIM,
        vector_version=vector_version,
        source_cutoff_at=source_cutoff_at,
        distribution=distribution,
        random_seed=stable_random_seed,
        provenance_mode=provenance_mode,
        source_cutoff_attestation=source_cutoff_attestation,
        git_commit=git_commit,
        numpy_version=str(np.__version__),
        faiss_version=_optional_module_version("faiss"),
        cpu_model=cpu_model(),
        cpu_count=os.cpu_count(),
        thread_settings=thread_settings(),
        matcher_config=dict(matcher_config or {}),
    )
    return FrozenAssignmentCorpus(
        manifest=manifest,
        user_ids=ordered_user_ids,
        user_vectors=ordered_users,
        segment_ids=ordered_segment_ids,
        segment_vectors=ordered_segments,
    )


def generate_synthetic_corpus(
    *,
    user_count: int,
    segment_count: int,
    distribution: str,
    random_seed: int,
    vector_version: str = "synthetic-v1",
    source_cutoff_at: str = "2026-01-01T00:00:00Z",
    git_commit: str = "unknown",
    matcher_config: Mapping[str, Any] | None = None,
    threshold: float = 0.65,
) -> FrozenAssignmentCorpus:
    np = _numpy()
    if user_count <= 0 or segment_count <= 0:
        raise ValueError("synthetic corpus counts must be positive")
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"unsupported distribution: {distribution}")
    rng = np.random.default_rng(random_seed)
    segments = normalize_float32_matrix(
        rng.normal(size=(segment_count, VECTOR_DIM)),
        name="segment_vectors",
    )

    if distribution == "random":
        users = rng.normal(size=(user_count, VECTOR_DIM))
    elif distribution == "clustered":
        targets = rng.integers(0, segment_count, size=user_count)
        users = segments[targets] + rng.normal(
            scale=0.10,
            size=(user_count, VECTOR_DIM),
        )
    elif distribution == "threshold_near":
        targets = rng.integers(0, segment_count, size=user_count)
        target_vectors = segments[targets]
        noise = rng.normal(size=(user_count, VECTOR_DIM))
        noise -= np.sum(noise * target_vectors, axis=1, keepdims=True) * target_vectors
        noise = normalize_float32_matrix(noise, name="threshold_noise")
        cosines = np.clip(
            threshold + rng.uniform(-0.015, 0.015, size=user_count),
            -0.999,
            0.999,
        ).astype(np.float32)
        users = (
            cosines[:, None] * target_vectors
            + np.sqrt(1.0 - cosines * cosines)[:, None] * noise
        )
    else:
        pair_count = (segment_count + 1) // 2
        pair_bases = normalize_float32_matrix(
            rng.normal(size=(pair_count, VECTOR_DIM)),
            name="pair_bases",
        )
        paired_segments: list[Any] = []
        for index in range(segment_count):
            paired_segments.append(
                pair_bases[index // 2]
                + rng.normal(scale=0.012, size=VECTOR_DIM)
            )
        segments = normalize_float32_matrix(
            np.asarray(paired_segments),
            name="segment_vectors",
        )
        target_pairs = rng.integers(0, pair_count, size=user_count)
        users = pair_bases[target_pairs] + rng.normal(
            scale=0.025,
            size=(user_count, VECTOR_DIM),
        )

    return freeze_corpus(
        user_ids=[f"synthetic-user-{index:09d}" for index in range(user_count)],
        user_vectors=users,
        segment_ids=[
            f"synthetic-segment-{index:06d}" for index in range(segment_count)
        ],
        segment_vectors=segments,
        vector_version=vector_version,
        source_cutoff_at=source_cutoff_at,
        distribution=distribution,
        random_seed=random_seed,
        git_commit=git_commit,
        id_hash_salt="loopad-assignment-synthetic-v1",
        matcher_config=matcher_config,
        provenance_mode=SYNTHETIC_PROVENANCE_MODE,
        source_cutoff_attestation=SYNTHETIC_SOURCE_CUTOFF_ATTESTATION,
    )


def normalize_float32_matrix(values: Any, *, name: str) -> Any:
    np = _numpy()
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != VECTOR_DIM:
        raise ValueError(f"{name} must have shape (N, {VECTOR_DIM})")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if bool(np.any(norms == 0.0)):
        raise ValueError(f"{name} must not contain zero vectors")
    normalized = matrix / norms[:, None]
    return np.ascontiguousarray(normalized, dtype=np.float32)


def hash_identifier(value: str, *, namespace: str, salt: str) -> str:
    clean_value = str(value)
    if not clean_value:
        raise ValueError(f"{namespace} ID must not be empty")
    if not salt:
        raise ValueError("ID hash salt must not be empty")
    digest = hashlib.sha256(
        f"{namespace}\0{salt}\0{clean_value}".encode("utf-8")
    ).hexdigest()
    return f"{namespace}_sha256_{digest}"


def compute_corpus_sha256(
    *,
    user_ids: Sequence[str],
    user_vectors: Any,
    segment_ids: Sequence[str],
    segment_vectors: Any,
    stable_metadata: Mapping[str, Any],
) -> str:
    np = _numpy()
    users = np.ascontiguousarray(user_vectors, dtype="<f4")
    segments = np.ascontiguousarray(segment_vectors, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"loopad-assignment-corpus\0")
    digest.update(_canonical_json_bytes(dict(stable_metadata)))
    for label, identifiers, matrix in (
        (b"users", user_ids, users),
        (b"segments", segment_ids, segments),
    ):
        digest.update(label + b"\0")
        digest.update(len(identifiers).to_bytes(8, "big"))
        for identifier, row in zip(identifiers, matrix, strict=True):
            encoded = identifier.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(row.tobytes(order="C"))
    return digest.hexdigest()


def write_frozen_corpus(
    corpus: FrozenAssignmentCorpus,
    destination: Path,
    *,
    overwrite: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8") as output:
        output.write(
            _canonical_json(
                {"type": "manifest", "manifest": corpus.manifest.to_dict()}
            )
            + "\n"
        )
        for identifier, vector in zip(
            corpus.user_ids,
            corpus.user_vectors,
            strict=True,
        ):
            output.write(
                _canonical_json(
                    {
                        "type": "user",
                        "id": identifier,
                        "vector": [float(value) for value in vector],
                    }
                )
                + "\n"
            )
        for identifier, vector in zip(
            corpus.segment_ids,
            corpus.segment_vectors,
            strict=True,
        ):
            output.write(
                _canonical_json(
                    {
                        "type": "segment",
                        "id": identifier,
                        "vector": [float(value) for value in vector],
                    }
                )
                + "\n"
            )


def load_frozen_corpus(path: Path) -> FrozenAssignmentCorpus:
    np = _numpy()
    manifest_payload: Mapping[str, Any] | None = None
    user_ids: list[str] = []
    user_vectors: list[Any] = []
    segment_ids: list[str] = []
    segment_vectors: list[Any] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid corpus JSONL at line {line_number}") from exc
            row_type = row.get("type")
            if row_type == "manifest":
                if manifest_payload is not None or line_number != 1:
                    raise ValueError("corpus manifest must be the first and only manifest row")
                manifest_payload = row.get("manifest")
            elif row_type == "user":
                user_ids.append(str(row.get("id", "")))
                user_vectors.append(row.get("vector"))
            elif row_type == "segment":
                segment_ids.append(str(row.get("id", "")))
                segment_vectors.append(row.get("vector"))
            else:
                raise ValueError(f"unsupported corpus row type at line {line_number}")
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("corpus manifest is missing")
    normalized_manifest_payload = dict(manifest_payload)
    format_version = normalized_manifest_payload.get("format_version")
    if format_version == LEGACY_CORPUS_FORMAT_VERSION:
        # v2 did not distinguish generated corpora from caller-provided input.
        # Never infer trustworthy provenance from its distribution/seed fields.
        normalized_manifest_payload["provenance_mode"] = (
            LEGACY_UNATTESTED_PROVENANCE_MODE
        )
        normalized_manifest_payload["source_cutoff_attestation"] = (
            LEGACY_SOURCE_CUTOFF_ATTESTATION
        )
    elif format_version != CORPUS_FORMAT_VERSION:
        raise ValueError("unsupported assignment corpus format")
    manifest = CorpusManifest(**normalized_manifest_payload)
    _validate_loaded_manifest_metadata(manifest)
    users = _validate_stored_matrix(user_vectors, name="user_vectors")
    segments = _validate_stored_matrix(segment_vectors, name="segment_vectors")
    _validate_loaded_ids(user_ids, namespace="user")
    _validate_plain_ids(segment_ids, name="segment")
    if user_ids != sorted(user_ids) or segment_ids != sorted(segment_ids):
        raise ValueError("frozen corpus IDs must be in canonical lexical order")
    if manifest.dimension != VECTOR_DIM:
        raise ValueError("assignment corpus dimension must be 64")
    if manifest.user_count != len(user_ids) or manifest.segment_count != len(segment_ids):
        raise ValueError("assignment corpus manifest counts do not match rows")
    stable_metadata = _manifest_stable_metadata(manifest)
    actual_digest = compute_corpus_sha256(
        user_ids=user_ids,
        user_vectors=users,
        segment_ids=segment_ids,
        segment_vectors=segments,
        stable_metadata=stable_metadata,
    )
    if actual_digest != manifest.corpus_sha256:
        raise ValueError("assignment corpus SHA-256 verification failed")
    return FrozenAssignmentCorpus(
        manifest=manifest,
        user_ids=tuple(user_ids),
        user_vectors=np.ascontiguousarray(users, dtype=np.float32),
        segment_ids=tuple(segment_ids),
        segment_vectors=np.ascontiguousarray(segments, dtype=np.float32),
    )


def current_git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def thread_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {
        name: os.environ.get(name) for name in THREAD_CONTROL_ENV_NAMES
    }
    np = _numpy()
    try:
        numpy_config = np.show_config(mode="dicts")
    except (AttributeError, TypeError):
        # Some legacy-build NumPy 1.26 installations expose only the printing
        # form of show_config. Do not depend on private __config__ attributes.
        numpy_config = {}
    settings["numpy_config_dict_available"] = bool(numpy_config)
    build_dependencies = (
        numpy_config.get("Build Dependencies", {})
        if isinstance(numpy_config, Mapping)
        else {}
    )
    blas_build = (
        build_dependencies.get("blas")
        if isinstance(build_dependencies, Mapping)
        else None
    )
    settings["numpy_blas_build"] = (
        dict(blas_build) if isinstance(blas_build, Mapping) else None
    )
    try:
        faiss = __import__("faiss")
        settings["faiss_omp_threads"] = int(faiss.omp_get_max_threads())
    except (ImportError, AttributeError, TypeError, ValueError):
        settings["faiss_omp_threads"] = None
    try:
        runtime_threadpools = __import__(
            "threadpoolctl"
        ).threadpool_info()
        settings["runtime_threadpools"] = runtime_threadpools
        detected_blas_pools = [
            pool
            for pool in runtime_threadpools
            if pool.get("user_api") == "blas"
        ]
        if detected_blas_pools:
            blas_control_verification = "threadpoolctl_runtime_verified"
        elif (
            isinstance(blas_build, Mapping)
            and str(blas_build.get("name", "")).lower() == "accelerate"
        ):
            blas_control_verification = "environment_only_unverified"
        else:
            blas_control_verification = "no_supported_blas_pool_detected"
        settings["blas_control_verification"] = blas_control_verification
    except ImportError:
        settings["runtime_threadpools"] = None
        settings["blas_control_verification"] = "threadpoolctl_unavailable"
    return settings


def controlled_thread_environment(thread_count: int) -> dict[str, str]:
    if thread_count <= 0:
        raise ValueError("thread_count must be positive")
    return {
        name: str(thread_count) for name in THREAD_CONTROL_ENV_NAMES
    }


def _validate_stored_matrix(values: Any, *, name: str) -> Any:
    np = _numpy()
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != VECTOR_DIM or matrix.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, {VECTOR_DIM})")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not bool(np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6)):
        raise ValueError(f"{name} rows must already be normalized")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _validate_manifest_inputs(
    *,
    vector_version: str,
    source_cutoff_at: str,
    distribution: str,
    random_seed: int | None,
    provenance_mode: str,
    source_cutoff_attestation: str,
    git_commit: str,
    id_hash_salt: str,
) -> None:
    if not vector_version:
        raise ValueError("vector_version must not be empty")
    _validate_source_cutoff(source_cutoff_at)
    if not distribution:
        raise ValueError("distribution must not be empty")
    _validate_provenance_metadata(
        distribution=distribution,
        random_seed=random_seed,
        provenance_mode=provenance_mode,
        source_cutoff_attestation=source_cutoff_attestation,
    )
    if not git_commit:
        raise ValueError("git_commit must not be empty")
    if not id_hash_salt:
        raise ValueError("id_hash_salt must not be empty")


def _validate_loaded_manifest_metadata(manifest: CorpusManifest) -> None:
    if not manifest.vector_version:
        raise ValueError("vector_version must not be empty")
    _validate_source_cutoff(manifest.source_cutoff_at)
    if not manifest.distribution:
        raise ValueError("distribution must not be empty")
    if manifest.format_version == LEGACY_CORPUS_FORMAT_VERSION:
        if (
            manifest.provenance_mode != LEGACY_UNATTESTED_PROVENANCE_MODE
            or manifest.source_cutoff_attestation
            != LEGACY_SOURCE_CUTOFF_ATTESTATION
        ):
            raise ValueError("v2 corpus provenance must remain explicitly unattested")
        return
    _validate_provenance_metadata(
        distribution=manifest.distribution,
        random_seed=_normalize_random_seed(manifest.random_seed),
        provenance_mode=manifest.provenance_mode,
        source_cutoff_attestation=manifest.source_cutoff_attestation,
    )


def _validate_source_cutoff(source_cutoff_at: str) -> None:
    if not source_cutoff_at:
        raise ValueError("source_cutoff_at must not be empty")
    try:
        parsed_cutoff = datetime.fromisoformat(source_cutoff_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("source_cutoff_at must be ISO-8601") from exc
    if parsed_cutoff.tzinfo is None:
        raise ValueError("source_cutoff_at must include a timezone")


def _validate_provenance_metadata(
    *,
    distribution: str,
    random_seed: int | None,
    provenance_mode: str,
    source_cutoff_attestation: str,
) -> None:
    if provenance_mode == SYNTHETIC_PROVENANCE_MODE:
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError("synthetic corpus must use a supported synthetic distribution")
        if random_seed is None:
            raise ValueError("synthetic corpus must record its random seed")
        if source_cutoff_attestation != SYNTHETIC_SOURCE_CUTOFF_ATTESTATION:
            raise ValueError("synthetic corpus must use synthetic cutoff metadata")
        return
    if provenance_mode == PROVIDED_PROVENANCE_MODE:
        if distribution != PROVIDED_DISTRIBUTION:
            raise ValueError("provided corpus must use the provided distribution")
        if random_seed is not None:
            raise ValueError("provided corpus must not claim a synthetic random seed")
        if source_cutoff_attestation != PROVIDED_SOURCE_CUTOFF_ATTESTATION:
            raise ValueError("provided corpus requires explicit source cutoff attestation")
        return
    if provenance_mode == UNATTESTED_PROVENANCE_MODE:
        if distribution == PROVIDED_DISTRIBUTION:
            raise ValueError("provided distribution requires explicit provided provenance")
        if source_cutoff_attestation != UNATTESTED_SOURCE_CUTOFF_ATTESTATION:
            raise ValueError("unattested corpus must not claim source cutoff attestation")
        return
    raise ValueError(f"unsupported corpus provenance mode: {provenance_mode}")


def _normalize_random_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("random_seed must be an integer or null")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("random_seed must be an integer or null") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("random_seed must be an integer or null")
    return normalized


def _manifest_stable_metadata(manifest: CorpusManifest) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format_version": manifest.format_version,
        "dimension": manifest.dimension,
        "vector_version": manifest.vector_version,
        "source_cutoff_at": manifest.source_cutoff_at,
        "distribution": manifest.distribution,
        "random_seed": manifest.random_seed,
    }
    if manifest.format_version == CORPUS_FORMAT_VERSION:
        metadata["provenance_mode"] = manifest.provenance_mode
        metadata["source_cutoff_attestation"] = (
            manifest.source_cutoff_attestation
        )
    return metadata


def _validate_unique_ids(values: Sequence[str], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} IDs must be unique")


def _validate_plain_ids(values: Sequence[str], *, name: str) -> None:
    if any(not value for value in values):
        raise ValueError(f"{name} IDs must not be empty")
    _validate_unique_ids(values, name=name)


def _validate_loaded_ids(values: Sequence[str], *, namespace: str) -> None:
    _validate_unique_ids(values, name=namespace)
    prefix = f"{namespace}_sha256_"
    if any(
        not value.startswith(prefix)
        or len(value) != len(prefix) + 64
        or any(character not in "0123456789abcdef" for character in value[len(prefix) :])
        for value in values
    ):
        raise ValueError(f"frozen {namespace} IDs must be SHA-256 pseudonyms")


def cpu_model() -> str:
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                completed = subprocess.run(
                    ("sysctl", "-n", key),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            value = completed.stdout.strip()
            if value:
                return value
    uname = platform.uname()
    return uname.processor or platform.processor() or uname.machine or "unknown"


def _optional_module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _numpy() -> Any:
    try:
        return __import__("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "assignment benchmark requires the 'assignment-benchmark' optional extra"
        ) from exc
