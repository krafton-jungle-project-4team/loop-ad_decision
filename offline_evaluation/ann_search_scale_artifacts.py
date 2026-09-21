"""Checkpoint, integrity, and actual-only reports for ANN scale-series v2."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from offline_evaluation.ann_search_scale_series import (
    SCALE_COHORT_SIZES,
    SCALE_EXPERIMENT_VERSION,
    SCALE_OUTPUT_ROOT,
    inspect_fingerprint,
)


@dataclass(frozen=True, slots=True)
class AppendResult:
    appended_count: int
    skipped_identical_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class ActualScalePoint:
    cohort_size: int
    scenario_id: str
    bucket_id: str
    plan: str
    p95_ms: float
    measured_run_count: int
    quality_passed: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.cohort_size <= 0 or not self.scenario_id or not self.bucket_id:
            raise ValueError("scale point identity is invalid")
        if self.plan not in {"exact_all", "filter_first_exact", "ann_first"}:
            raise ValueError("scale point plan is invalid")
        if self.p95_ms < 0 or self.measured_run_count <= 0:
            raise ValueError("scale point must contain an actual measurement")
        if len(self.evidence_sha256) != 64:
            raise ValueError("scale point evidence hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    passed: bool
    checked_cohort_count: int
    issues: tuple[IntegrityIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "passed": self.passed,
            "checked_cohort_count": self.checked_cohort_count,
            "issues": [item.to_dict() for item in self.issues],
        }


class AppendOnlyJsonl:
    """Append records once; identical resume rows are skipped, conflicts fail."""

    def __init__(self, path: Path, *, identity_fields: Sequence[str]) -> None:
        if not identity_fields:
            raise ValueError("JSONL identity fields are required")
        self.path = path
        self.identity_fields = tuple(identity_fields)

    def append(self, records: Iterable[Mapping[str, Any]]) -> AppendResult:
        existing = self._load_existing()
        appended = 0
        skipped = 0
        pending: list[bytes] = []
        for raw in records:
            record = dict(raw)
            identity = self._identity(record)
            encoded = _canonical_json_bytes(record)
            previous = existing.get(identity)
            if previous is not None:
                if previous != encoded:
                    raise ValueError(
                        "checkpoint resume found conflicting row for identity "
                        + repr(identity)
                    )
                skipped += 1
                continue
            existing[identity] = encoded
            pending.append(encoded + b"\n")
            appended += 1
        if pending:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                for encoded in pending:
                    handle.write(encoded)
                handle.flush()
        return AppendResult(
            appended_count=appended,
            skipped_identical_count=skipped,
            total_count=len(existing),
        )

    def _load_existing(self) -> dict[tuple[Any, ...], bytes]:
        if not self.path.exists():
            return {}
        result: dict[tuple[Any, ...], bytes] = {}
        for line_number, line in enumerate(
            self.path.read_bytes().splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row {line_number} is not an object")
            identity = self._identity(payload)
            encoded = _canonical_json_bytes(payload)
            if identity in result:
                raise ValueError(f"duplicate JSONL identity at line {line_number}")
            result[identity] = encoded
        return result

    def _identity(self, record: Mapping[str, Any]) -> tuple[Any, ...]:
        missing = [field for field in self.identity_fields if field not in record]
        if missing:
            raise ValueError("JSONL identity fields missing: " + ", ".join(missing))
        return tuple(_hashable_identity(record[field]) for field in self.identity_fields)


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> bool:
    encoded = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() == encoded:
            return False
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_payload(
    *,
    cohort_size: int,
    cohort_sha256: str,
    fingerprint_sha256: str,
    raw_path: Path,
    result_path: Path,
    ground_truth: Mapping[str, Mapping[str, Any]],
    expected_scenario_count: int,
    expected_baseline_cell_count: int,
    observed_baseline_cell_count: int,
    scale_survivor_count: int,
    policy_survivor_count: int,
    rss_complete: bool,
    spill_oom_complete: bool,
) -> Mapping[str, Any]:
    if cohort_size <= 0:
        raise ValueError("checkpoint cohort size must be positive")
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": "complete",
        "cohort_size": cohort_size,
        "cohort_sha256": cohort_sha256,
        "prefix_validated": True,
        "fingerprint_sha256": fingerprint_sha256,
        "raw": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
        "result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "ground_truth": dict(ground_truth),
        "expected_scenario_count": expected_scenario_count,
        "ground_truth_complete": (
            len(ground_truth) == expected_scenario_count
            and all(
                int(item.get("row_count", -1)) == cohort_size
                and len(str(item.get("sha256", ""))) == 64
                for item in ground_truth.values()
            )
        ),
        "expected_baseline_cell_count": expected_baseline_cell_count,
        "observed_baseline_cell_count": observed_baseline_cell_count,
        "baseline_complete": (
            observed_baseline_cell_count == expected_baseline_cell_count
        ),
        "scale_survivor_count": scale_survivor_count,
        "policy_survivor_count": policy_survivor_count,
        "rss_complete": rss_complete,
        "spill_oom_complete": spill_oom_complete,
    }


class ScaleArtifactValidator:
    def __init__(self, root: Path = SCALE_OUTPUT_ROOT) -> None:
        self.root = root

    def validate(self) -> IntegrityReport:
        issues: list[IntegrityIssue] = []
        expected_root = SCALE_OUTPUT_ROOT.as_posix()
        if not self.root.as_posix().endswith(expected_root):
            issues.append(
                IntegrityIssue(
                    "wrong_output_root",
                    str(self.root),
                    f"v2 artifacts must remain under {expected_root}",
                )
            )
        manifest_path = self.root / "manifest.json"
        fingerprint_path = self.root / "fingerprint.json"
        reuse_path = self.root / "old-50k-reuse-decision.json"
        manifest = self._object(manifest_path, issues)
        fingerprint = self._object(fingerprint_path, issues)
        reuse = self._object(reuse_path, issues)
        if manifest and manifest.get("experiment_version") != SCALE_EXPERIMENT_VERSION:
            issues.append(
                IntegrityIssue(
                    "manifest_version",
                    str(manifest_path),
                    "manifest is not benchmark v2",
                )
            )
        if manifest and manifest.get("output_root") != str(SCALE_OUTPUT_ROOT):
            issues.append(
                IntegrityIssue(
                    "manifest_output_root",
                    str(manifest_path),
                    "manifest output root differs from the v2 contract",
                )
            )
        if fingerprint:
            check = inspect_fingerprint(fingerprint)
            if not check.complete:
                issues.append(
                    IntegrityIssue(
                        "fingerprint_incomplete",
                        str(fingerprint_path),
                        "missing=" + ",".join(check.missing_fields)
                        + ";invalid=" + ",".join(check.invalid_fields),
                    )
                )
        if reuse and (
            reuse.get("decision") != "pilot_only"
            or reuse.get("latency_reused") is not False
            or reuse.get("ground_truth_reused") is not False
        ):
            issues.append(
                IntegrityIssue(
                    "old_50k_reuse",
                    str(reuse_path),
                    "historical 50K must remain pilot-only",
                )
            )
        completed_sizes = self._completed_sizes(manifest)
        for size in completed_sizes:
            checkpoint_path = self.root / "checkpoints" / f"cohort-{size}.json"
            checkpoint = self._object(checkpoint_path, issues)
            if checkpoint:
                self._validate_checkpoint(checkpoint_path, checkpoint, size, issues)
        return IntegrityReport(
            passed=not issues,
            checked_cohort_count=len(completed_sizes),
            issues=tuple(issues),
        )

    def raise_for_errors(self) -> IntegrityReport:
        report = self.validate()
        if not report.passed:
            raise ValueError(
                "scale-series artifact validation failed: "
                + "; ".join(f"{item.code}:{item.message}" for item in report.issues)
            )
        return report

    def _object(
        self,
        path: Path,
        issues: list[IntegrityIssue],
    ) -> Mapping[str, Any] | None:
        if not path.exists():
            issues.append(IntegrityIssue("missing_file", str(path), "file is missing"))
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(IntegrityIssue("invalid_json", str(path), str(exc)))
            return None
        if not isinstance(payload, Mapping):
            issues.append(IntegrityIssue("invalid_object", str(path), "expected object"))
            return None
        return payload

    def _completed_sizes(self, manifest: Mapping[str, Any] | None) -> tuple[int, ...]:
        if not manifest:
            return ()
        raw = manifest.get("completed_cohort_sizes", ())
        if not isinstance(raw, list):
            return ()
        return tuple(sorted({int(value) for value in raw}))

    def _validate_checkpoint(
        self,
        path: Path,
        payload: Mapping[str, Any],
        expected_size: int,
        issues: list[IntegrityIssue],
    ) -> None:
        required_true = (
            "prefix_validated",
            "ground_truth_complete",
            "baseline_complete",
            "rss_complete",
            "spill_oom_complete",
        )
        if payload.get("experiment_version") != SCALE_EXPERIMENT_VERSION:
            issues.append(IntegrityIssue("checkpoint_version", str(path), "wrong version"))
        if payload.get("status") != "complete" or int(payload.get("cohort_size", -1)) != expected_size:
            issues.append(IntegrityIssue("checkpoint_status", str(path), "checkpoint is incomplete"))
        for field in required_true:
            if payload.get(field) is not True:
                issues.append(
                    IntegrityIssue("checkpoint_completeness", str(path), f"{field} is not true")
                )
        ground_truth = payload.get("ground_truth")
        expected_scenarios = int(payload.get("expected_scenario_count", -1))
        if not isinstance(ground_truth, Mapping) or len(ground_truth) != expected_scenarios:
            issues.append(
                IntegrityIssue("ground_truth_count", str(path), "ground-truth scenario count differs")
            )
        elif any(
            int(value.get("row_count", -1)) != expected_size
            for value in ground_truth.values()
            if isinstance(value, Mapping)
        ):
            issues.append(
                IntegrityIssue("ground_truth_rows", str(path), "ground-truth rank is incomplete")
            )
        for label in ("raw", "result"):
            artifact = payload.get(label)
            if not isinstance(artifact, Mapping):
                issues.append(IntegrityIssue("artifact_entry", str(path), f"{label} is missing"))
                continue
            artifact_path = Path(str(artifact.get("path", "")))
            try:
                artifact_path.resolve().relative_to(self.root.resolve())
            except (OSError, ValueError):
                issues.append(
                    IntegrityIssue("artifact_outside_root", str(artifact_path), label)
                )
                continue
            if not artifact_path.exists():
                issues.append(IntegrityIssue("missing_artifact", str(artifact_path), label))
            elif sha256_file(artifact_path) != artifact.get("sha256"):
                issues.append(IntegrityIssue("artifact_hash", str(artifact_path), label))


def write_actual_only_reports(
    *,
    points: Sequence[ActualScalePoint],
    expected_cohort_sizes: Sequence[int],
    csv_path: Path,
    markdown_path: Path,
    png_path: Path,
) -> None:
    if not points:
        raise ValueError("actual scale report requires measured points")
    sizes = tuple(sorted(set(int(value) for value in expected_cohort_sizes)))
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("expected cohort sizes are invalid")
    if any(point.cohort_size not in sizes for point in points):
        raise ValueError("graph point uses an unregistered cohort size")
    identities = [
        (point.cohort_size, point.scenario_id, point.bucket_id, point.plan)
        for point in points
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("actual scale report contains duplicate points")
    ordered = sorted(points, key=lambda item: (item.bucket_id, item.scenario_id, item.plan, item.cohort_size))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(ordered[0].to_dict()))
        writer.writeheader()
        writer.writerows(item.to_dict() for item in ordered)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown_report(ordered, sizes), encoding="utf-8")
    _render_png(ordered, sizes, png_path)


def _markdown_report(
    points: Sequence[ActualScalePoint],
    sizes: Sequence[int],
) -> str:
    by_cell = {
        (point.cohort_size, point.scenario_id, point.bucket_id, point.plan): point
        for point in points
    }
    scenarios = sorted({(point.scenario_id, point.bucket_id) for point in points})
    lines = [
        "# ANN scale-series v2 preliminary actual-only report",
        "",
        "그래프와 표는 실제 measured row가 있는 point만 사용한다. 품질 gate를 실패한 ANN point는 선에서 제외하고 `no ANN candidate`로 표시한다.",
        "",
        "| scenario/bucket | cohort | exact_all p95 | filter_first p95 | ANN p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario_id, bucket_id in scenarios:
        for size in sizes:
            exact = by_cell.get((size, scenario_id, bucket_id, "exact_all"))
            filtered = by_cell.get((size, scenario_id, bucket_id, "filter_first_exact"))
            ann = by_cell.get((size, scenario_id, bucket_id, "ann_first"))
            ann_value = (
                f"{ann.p95_ms:.3f}"
                if ann is not None and ann.quality_passed
                else "no ANN candidate"
            )
            if exact is None and filtered is None and ann is None:
                continue
            lines.append(
                f"| {scenario_id}/{bucket_id} | {size} | "
                f"{_point_value(exact)} | {_point_value(filtered)} | {ann_value} |"
            )
    lines.append("")
    return "\n".join(lines)


def _render_png(
    points: Sequence[ActualScalePoint],
    sizes: Sequence[int],
    path: Path,
) -> None:
    width, height = 960, 540
    pixels = bytearray([255] * width * height * 3)
    left, right, top, bottom = 70, width - 30, 30, height - 55
    _line(pixels, width, height, left, top, left, bottom, (30, 30, 30))
    _line(pixels, width, height, left, bottom, right, bottom, (30, 30, 30))
    visible = [
        point
        for point in points
        if point.plan != "ann_first" or point.quality_passed
    ]
    max_y = max(point.p95_ms for point in visible) if visible else 1.0
    max_y = max(max_y, 1.0)
    colors = {
        "exact_all": (30, 80, 180),
        "filter_first_exact": (0, 145, 110),
        "ann_first": (210, 80, 50),
    }
    groups: dict[tuple[str, str, str], list[ActualScalePoint]] = {}
    for point in visible:
        groups.setdefault((point.scenario_id, point.bucket_id, point.plan), []).append(point)
    x_positions = {
        size: int(left + index * (right - left) / max(1, len(sizes) - 1))
        for index, size in enumerate(sizes)
    }
    for (_, _, plan), group in sorted(groups.items()):
        coords: list[tuple[int, int]] = []
        for point in sorted(group, key=lambda item: item.cohort_size):
            x = x_positions[point.cohort_size]
            y = int(bottom - (point.p95_ms / max_y) * (bottom - top))
            coords.append((x, y))
            _disc(pixels, width, height, x, y, 3, colors[plan])
        for first, second in zip(coords, coords[1:]):
            _line(pixels, width, height, *first, *second, colors[plan])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(width, height, pixels))


def _png_bytes(width: int, height: int, pixels: bytes) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(
        b"\x00" + pixels[offset : offset + width * 3]
        for offset in range(0, len(pixels), width * 3)
    )
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(rows, 9)) + _png_chunk(b"IEND", b"")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _pixel(pixels, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _disc(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                _pixel(pixels, width, height, x + dx, y + dy, color)


def _pixel(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if not (0 <= x < width and 0 <= y < height):
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def _point_value(point: ActualScalePoint | None) -> str:
    return f"{point.p95_ms:.3f}" if point is not None else "-"


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hashable_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable_identity(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_hashable_identity(item) for item in value)
    return value
