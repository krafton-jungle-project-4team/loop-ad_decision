from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True, slots=True)
class CleanGitIdentity:
    commit: str
    tree: str
    branch: str


def inspect_clean_git_identity(
    repository_root: Path,
    *,
    required_branch: str | None = None,
) -> CleanGitIdentity:
    branch = _git_output(repository_root, "rev-parse", "--abbrev-ref", "HEAD")
    if required_branch is not None and branch != required_branch:
        raise ValueError(
            "sealed final test manifest must be created from "
            f"the {required_branch!r} branch"
        )

    tracked_status = _git_output(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise ValueError(
            "sealed final test requires a clean tracked working tree"
        )

    return CleanGitIdentity(
        commit=_git_output(repository_root, "rev-parse", "HEAD"),
        tree=_git_output(repository_root, "rev-parse", "HEAD^{tree}"),
        branch=branch,
    )


def _git_output(repository_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"failed to inspect frozen git state: {' '.join(args)}"
        ) from exc
    return completed.stdout.strip()
