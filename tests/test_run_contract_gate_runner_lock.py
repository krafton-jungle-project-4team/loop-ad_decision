from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def test_runner_lock_covers_project_and_dev_requirements() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    required = [
        Requirement(value)
        for value in project["dependencies"]
        + project["optional-dependencies"]["dev"]
    ]

    locked: dict[str, str] = {}
    for line in (ROOT / "tools/run_contract_gate/requirements.lock").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1 and specifiers[0].operator == "==", line
        name = canonicalize_name(requirement.name)
        assert name not in locked, f"duplicate lock entry: {name}"
        locked[name] = specifiers[0].version

    for requirement in required:
        name = canonicalize_name(requirement.name)
        assert name in locked, f"runner lock is missing {requirement}"
        assert locked[name] in requirement.specifier, (
            f"runner lock has {name}=={locked[name]}, which does not satisfy {requirement}"
        )
