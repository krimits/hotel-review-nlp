"""Notebook 29 trains the triage diagnosis student once (PROTOCOL.md version 6); checked offline."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "29_triage_student_colab.ipynb"


@pytest.fixture(scope="module")
def cells() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def test_the_notebook_pins_a_commit_that_holds_the_script_the_protocol_and_the_labels(cells):
    commit = re.search(r'SOURCE_COMMIT = "([0-9a-f]{40})"', cells["parameters"])[1]
    for path in ("scripts/train_triage_student.py", "docs/experiments/triage_diagnosis/PROTOCOL.md",
                 "docs/experiments/triage_diagnosis/labels/teacher_haiku_teacher.csv",
                 "docs/experiments/triage_diagnosis/samples/teacher_key.csv"):
        assert subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], cwd=ROOT).returncode == 0, path
    protocol = subprocess.run(["git", "show", f"{commit}:docs/experiments/triage_diagnosis/PROTOCOL.md"], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8").stdout
    assert protocol.startswith("# Triage diagnosis and routing: protocol — version 6, locked")
    assert '"--detach", SOURCE_COMMIT' in cells["checkout"] and "GIT_COMMIT != SOURCE_COMMIT" in cells["checkout"]


def test_the_data_is_the_checked_csv_on_a_gpu(cells):
    assert 'run("python scripts/fetch_booking_515k.py")' in cells["data"]
    assert "torch.cuda.is_available()" in cells["data"]


def test_the_run_trains_once_and_bundles_logits_without_weights(cells):
    assert cells["run"] == 'run("python scripts/train_triage_student.py --csv data/raw/booking_reviews_515k.csv")'
    assert 'OUT = Path("docs/experiments/triage_diagnosis/student")' in cells["parameters"]
    assert '".safetensors", ".bin", ".pt"' in cells["collect"]
    assert '{"run.json", "logits_development.csv", "logits_test.csv", "logits_check.csv"}' in cells["collect"]
    assert "artifact_manifest.json" in cells["collect"] and "triage_student_results.zip" in cells["download"]


def test_the_notebook_needs_no_secret(cells):
    joined = "\n".join(cells.values())
    assert "OPENROUTER" not in joined and "HF_TOKEN" not in joined and "userdata" not in joined
