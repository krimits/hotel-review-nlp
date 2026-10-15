"""Notebook 28 runs the INT8 and batching protocol; checked offline, nothing downloaded."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "28_int8_and_batching_colab.ipynb"


@pytest.fixture(scope="module")
def cells() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def test_the_notebook_pins_a_commit_that_holds_the_tooling(cells):
    commit = re.search(r'SOURCE_COMMIT = "([0-9a-f]{40})"', cells["parameters"])[1]
    for path in ("scripts/benchmark_int8.py", "docs/experiments/int8_and_batching/PROTOCOL.md"):
        assert subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], cwd=ROOT).returncode == 0, path
    assert '"--detach", SOURCE_COMMIT' in cells["checkout"] and "GIT_COMMIT != SOURCE_COMMIT" in cells["checkout"]


def test_the_data_is_the_random_split_the_model_was_scored_on(cells):
    assert 'DATA_CONFIG = "configs/baselines.yaml"' in cells["data"]
    assert 'fingerprint_splits(processed) != views["random"]["splits"]' in cells["data"]


def test_the_model_is_the_pinned_checkpoint_and_stays_outside_the_bundle(cells):
    assert "revision=int8.MODEL_REVISION" in cells["model"]
    assert "int8.weights_sha256(MODEL_DIR) != int8.WEIGHTS_SHA256" in cells["model"]
    model_dir = re.search(r'MODEL_DIR = Path\("([^"]+)"\)', cells["parameters"])[1]
    assert not model_dir.startswith("/content/hotel-review-nlp")
    assert '".safetensors", ".bin", ".pt"' in cells["collect"]


def test_the_run_writes_where_the_results_are_read(cells):
    assert "scripts/benchmark_int8.py --model {MODEL_DIR}" in cells["run"]
    assert 'OUT = Path("docs/experiments/results/int8_v1")' in cells["run"]
    assert "artifact_manifest.json" in cells["collect"] and "int8_and_batching_results.zip" in cells["download"]
