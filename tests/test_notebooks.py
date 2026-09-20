"""Health checks for the committed notebooks — offline, no execution.

The notebooks are linked from the README as evidence, but nothing verified
they still matched the package. Executing them needs a GPU, the 515k Kaggle
CSV and trained checkpoints, so CI cannot run them; it can still prove they
would not fail on trivia. These checks catch the drift that actually happens:
a helper that got renamed, a config that was never committed, a cell that
stopped being valid Python.

Deliberately NOT asserted: that outputs are saved. Whether a notebook has
been executed is a fact about the repository recorded in the README, not a
gate on every pull request.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))

# Paths under these roots are produced by a run or downloaded by the user, so
# their absence is expected in a clean checkout.
GENERATED_PREFIXES = ("data/processed", "data/raw/booking", "runs/", "models/")

# Referencing one of these that does not exist is a genuine breakage: they are
# meant to be committed.
TRACKED_PREFIXES = ("configs/", "scripts/", "src/")


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def _parseable(source: str) -> str:
    """Neutralise IPython magics and shell escapes so ast can read the rest."""
    return "\n".join(
        f"pass  # {line}" if line.lstrip().startswith(("!", "%")) else line
        for line in source.splitlines()
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_at_least_one_notebook_is_discovered():
    """Guards against the glob silently matching nothing."""
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_well_formed(path: Path):
    notebook = _load(path)
    assert notebook.get("nbformat", 0) >= 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"code", "markdown", "raw"}, f"cell {index}"
        assert isinstance(cell["source"], (str, list)), f"cell {index}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_are_valid_python(path: Path):
    try:
        ast.parse(_parseable(_code(_load(path))))
    except SyntaxError as exc:  # pragma: no cover - message is the point
        pytest.fail(f"{path.name} line {exc.lineno}: {exc.msg}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_reviewnlp_imports_still_exist(path: Path):
    """Catches helpers renamed or removed out from under a notebook."""
    tree = ast.parse(_parseable(_code(_load(path))))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not (node.module or "").startswith("reviewnlp"):
            continue
        try:
            module = importlib.import_module(node.module)
        except ModuleNotFoundError as exc:
            if (exc.name or "").startswith("reviewnlp"):
                missing.append(f"module {node.module} does not exist")
            # An absent optional extra (trl, bitsandbytes) is not our drift.
            continue
        for alias in node.names:
            if not hasattr(module, alias.name):
                missing.append(f"{node.module}.{alias.name}")
    assert not missing, f"{path.name} imports names that no longer exist: {missing}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_referenced_tracked_files_are_committed(path: Path):
    """A notebook pointing at a config that was never committed is broken.

    Generated artefacts (processed parquet, runs/, the Kaggle CSV) are
    exempt: they are produced or downloaded, not tracked.
    """
    source = _code(_load(path))
    referenced = set()
    for node in ast.walk(ast.parse(_parseable(source))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.removeprefix("./")
            if value.startswith(TRACKED_PREFIXES) and not value.startswith(GENERATED_PREFIXES):
                referenced.add(value)

    absent = sorted(ref for ref in referenced if not (ROOT / ref).exists())
    assert not absent, (
        f"{path.name} references tracked file(s) that are not in the repository: {absent}"
    )
