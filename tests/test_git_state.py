from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from offline_evaluation.git_state import inspect_clean_git_identity


def test_manifest_sealing_requires_clean_dev_branch(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path)

    identity = inspect_clean_git_identity(repository, required_branch="dev")

    assert identity.branch == "dev"
    assert identity.commit
    assert identity.tree


def test_execution_allows_clean_detached_sealed_commit(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path)
    commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--detach", commit)

    identity = inspect_clean_git_identity(repository)

    assert identity.branch == "HEAD"
    assert identity.commit == commit


def test_manifest_sealing_rejects_detached_head(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path)
    _git(repository, "checkout", "--detach", "HEAD")

    with pytest.raises(ValueError, match="must be created from the 'dev' branch"):
        inspect_clean_git_identity(repository, required_branch="dev")


def test_execution_rejects_tracked_working_tree_changes(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path)
    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean tracked working tree"):
        inspect_clean_git_identity(repository)


def _git_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=dev")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "test: initial")
    return repository


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
