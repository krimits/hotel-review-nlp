"""CI installs from requirements/ci.lock, and the lock agrees with what it was compiled from."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement

tomllib = pytest.importorskip("tomllib")  # Python 3.11+; CI runs 3.11

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements" / "ci.lock"
PINS = ROOT / "requirements" / "ci.in"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _entries(path: Path) -> list[str]:
    return [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()]


def _locked() -> dict[str, str]:
    locked = {}
    for entry in _entries(LOCK):
        name, version = entry.split("==")
        locked[re.sub(r"[-_.]+", "-", name).lower()] = version
    return locked


def test_every_locked_line_is_one_exact_pin():
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+==[^\s=]+", entry) for entry in _entries(LOCK))


def test_the_lock_keeps_the_pins_it_was_compiled_from():
    locked = _locked()
    for entry in _entries(PINS):
        requirement = Requirement(entry)
        (spec,) = requirement.specifier
        assert spec.operator == "==" and locked[requirement.name.lower()] == spec.version, entry


def test_the_lock_covers_every_dependency_ci_installs():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = _locked()
    wanted = [*project["dependencies"], *project["optional-dependencies"]["dev"],
              *project["optional-dependencies"]["serving"]]
    for text in wanted:
        requirement = Requirement(text)
        version = locked[re.sub(r"[-_.]+", "-", requirement.name).lower()]
        assert requirement.specifier.contains(version.split("+")[0], prereleases=True), (text, version)


def test_ci_installs_with_the_lock():
    workflow = CI.read_text(encoding="utf-8")
    installs = [line.strip() for line in workflow.splitlines() if "pip install" in line]
    assert installs and all("-c requirements/ci.lock" in line for line in installs), installs
    assert "-r requirements/ci.in" in workflow
    assert 'requires-python = ">=3.10"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
