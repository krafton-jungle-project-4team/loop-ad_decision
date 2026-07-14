#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from offline_evaluation.assignment_corpus import (  # noqa: E402
    PROVIDED_DISTRIBUTION,
    PROVIDED_PROVENANCE_MODE,
    PROVIDED_SOURCE_CUTOFF_ATTESTATION,
    SUPPORTED_DISTRIBUTIONS,
    current_git_commit,
    freeze_corpus,
    generate_synthetic_corpus,
    write_frozen_corpus,
)
from offline_evaluation.assignment_matchers import MatcherConfig  # noqa: E402


DEFAULT_SYNTHETIC_SEED = 20260714
DEFAULT_SYNTHETIC_SOURCE_CUTOFF = "2026-01-01T00:00:00Z"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a pseudonymous D64 assignment benchmark corpus.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument(
        "--id-hash-salt-file",
        type=Path,
        help="Required for raw input; the salt is never stored in the corpus.",
    )
    parser.add_argument("--users", type=int, default=256)
    parser.add_argument("--segments", type=int, default=256)
    parser.add_argument(
        "--distribution",
        choices=SUPPORTED_DISTRIBUTIONS,
        help="Synthetic generation only; raw input is always labeled provided.",
    )
    parser.add_argument("--seed", type=int, help="Synthetic generation only.")
    parser.add_argument("--vector-version", default="assignment-benchmark-v1")
    parser.add_argument("--source-cutoff")
    parser.add_argument(
        "--attest-source-cutoff",
        action="store_true",
        help=(
            "With --input-jsonl, attest that the input was materialized only from "
            "source rows at or before --source-cutoff. The exporter cannot verify "
            "this because input rows contain no timestamps."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = MatcherConfig()
    commit = current_git_commit(REPOSITORY_ROOT)
    if args.input_jsonl is None:
        if args.attest_source_cutoff:
            parser.error("--attest-source-cutoff is only valid with --input-jsonl")
        distribution = args.distribution or "random"
        seed = args.seed if args.seed is not None else DEFAULT_SYNTHETIC_SEED
        source_cutoff = args.source_cutoff or DEFAULT_SYNTHETIC_SOURCE_CUTOFF
        corpus = generate_synthetic_corpus(
            user_count=args.users,
            segment_count=args.segments,
            distribution=distribution,
            random_seed=seed,
            vector_version=args.vector_version,
            source_cutoff_at=source_cutoff,
            git_commit=commit,
            matcher_config=config.to_dict(),
            threshold=config.threshold,
        )
    else:
        if args.id_hash_salt_file is None:
            parser.error("--id-hash-salt-file is required with --input-jsonl")
        if args.distribution is not None:
            parser.error("--distribution cannot be used with --input-jsonl")
        if args.seed is not None:
            parser.error("--seed cannot be used with --input-jsonl")
        if args.source_cutoff is None:
            parser.error("--source-cutoff is required with --input-jsonl")
        if not args.attest_source_cutoff:
            parser.error("--attest-source-cutoff is required with --input-jsonl")
        salt = args.id_hash_salt_file.read_text(encoding="utf-8").strip()
        users, user_vectors, segments, segment_vectors = _read_input_jsonl(
            args.input_jsonl
        )
        corpus = freeze_corpus(
            user_ids=users,
            user_vectors=user_vectors,
            segment_ids=segments,
            segment_vectors=segment_vectors,
            vector_version=args.vector_version,
            source_cutoff_at=args.source_cutoff,
            distribution=PROVIDED_DISTRIBUTION,
            random_seed=None,
            git_commit=commit,
            id_hash_salt=salt,
            matcher_config=config.to_dict(),
            provenance_mode=PROVIDED_PROVENANCE_MODE,
            source_cutoff_attestation=PROVIDED_SOURCE_CUTOFF_ATTESTATION,
        )
    write_frozen_corpus(corpus, args.output, overwrite=args.force)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "corpus_sha256": corpus.manifest.corpus_sha256,
                "user_count": corpus.manifest.user_count,
                "segment_count": corpus.manifest.segment_count,
                "dimension": corpus.manifest.dimension,
            },
            sort_keys=True,
        )
    )
    return 0


def _read_input_jsonl(
    path: Path,
) -> tuple[list[str], list[Any], list[str], list[Any]]:
    user_ids: list[str] = []
    user_vectors: list[Any] = []
    segment_ids: list[str] = []
    segment_vectors: list[Any] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid input JSONL at line {line_number}") from exc
            kind = row.get("kind", row.get("type"))
            if kind == "user":
                user_ids.append(str(row.get("id", "")))
                user_vectors.append(row.get("vector"))
            elif kind == "segment":
                segment_ids.append(str(row.get("id", "")))
                segment_vectors.append(row.get("vector"))
            else:
                raise ValueError(f"unsupported input row kind at line {line_number}")
    return user_ids, user_vectors, segment_ids, segment_vectors


if __name__ == "__main__":
    raise SystemExit(main())
