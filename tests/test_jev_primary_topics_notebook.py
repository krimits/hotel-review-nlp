"""Check the actual Colab execution cells without any API call or credential."""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/24_jev_primary_topics_dev_comparison_colab.ipynb"


def code_cells():
    return [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_bytes())["cells"]
        if c["cell_type"] == "code"
    ]


def test_notebook_is_pinned_cpu_only_clean_and_does_not_modify_the_original_experiment():
    notebook = json.loads(NOTEBOOK.read_bytes())
    code = "\n".join(code_cells())
    ast.parse(code)
    pin = re.search(r'SOURCE_COMMIT\s*=\s*"([0-9a-f]{40})"', code).group(1)
    for path in [
        "scripts/benchmark_jev_primary_topics.py",
        "scripts/unpack_decision_topic_inputs.py",
        "configs/decision_topic_sample_requirements.txt",
        "docs/experiments/decision_topic_benchmark/JEV_EXTENSION.md",
        "docs/experiments/decision_topic_benchmark/annotation_record.json",
    ]:
        assert (
            subprocess.check_output(["git", "show", f"{pin}:{path}"], cwd=ROOT)
            == (ROOT / path).read_bytes()
        )
    assert "ALLOW_JEV = False" in code and "MAX_COST_USD = None" in code
    assert "system_site_packages=False" in code and '"--isolated", "install", "--no-user"' in code
    assert "module.__file__" in code and "module.__version__" in code
    assert "torch" not in code and "cuda" not in code and "snapshot_download" not in code
    assert '"--partition", "test"' not in code and '"--arms"' not in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_execution_cell_reads_secret_only_after_opt_in_and_never_logs_it(
    tmp_path, monkeypatch, capsys, enabled
):
    code = next(c for c in code_cells() if c.startswith("run_env = dict(clean_env)"))
    reads, calls = [], []
    colab = types.ModuleType("google.colab")

    class Userdata:
        def get(self, name):
            reads.append(name)
            assert enabled and name == "OPENROUTER_API_KEY"
            return "synthetic-secret-canary"

    colab.userdata = Userdata()
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def run(command, *, cwd, env, stdout, stderr):
        calls.append((command, dict(env)))
        stdout.write("fake invocation; not a real model result\n")
        return types.SimpleNamespace(returncode=1)

    namespace = {
        "ALLOW_JEV": enabled,
        "MAX_COST_USD": 0.1,
        "clean_env": {},
        "math": math,
        "BASE": tmp_path,
        "REPO": ROOT,
        "PYTHON": Path("/fake/python"),
        "INPUTS": tmp_path / "inputs",
        "OUT": tmp_path / "out",
        "subprocess": types.SimpleNamespace(run=run, STDOUT=subprocess.STDOUT),
    }
    exec(compile(code, "actual-execution-cell", "exec"), namespace)
    assert reads == (["OPENROUTER_API_KEY"] if enabled else [])
    assert len(calls) == int(enabled)
    if enabled:
        command, environment = calls[0]
        assert "--allow-external-api" in command and "synthetic-secret-canary" not in " ".join(
            command
        )
        assert environment["OPENROUTER_API_KEY"] == "synthetic-secret-canary"
        assert namespace["RUN_EXIT_CODE"] == 1  # a partial run still reaches the export cell
        assert "synthetic-secret-canary" not in (tmp_path / "execution.txt").read_text()
    assert "OPENROUTER_API_KEY" not in namespace["run_env"]
    assert "synthetic-secret-canary" not in capsys.readouterr().out


@pytest.mark.parametrize("budget", [None, 0, -1, math.nan, math.inf, True])
def test_budget_is_required_before_secret_lookup(budget, tmp_path):
    code = next(c for c in code_cells() if c.startswith("run_env = dict(clean_env)"))
    with pytest.raises(ValueError, match="positive USD"):
        exec(
            compile(code, "actual-execution-cell", "exec"),
            {
                "ALLOW_JEV": True,
                "MAX_COST_USD": budget,
                "clean_env": {},
                "math": math,
                "OUT": tmp_path / "new_capture",
            },
        )


def test_actual_export_allowlist_excludes_inputs_and_unexpected_output_files(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "completed_A.csv").write_text("private human labels and reviews")
    (out / "jev_predictions.jsonl").write_text('{"arm":"jev","status":"error"}\n')
    (out / "run.json").write_text('{"execution_complete":false}\n')
    (out / "unexpected_secrets.txt").write_text("private canary")
    (tmp_path / "execution.txt").write_text("partial execution, no secret")
    downloads = []
    fake_files = types.SimpleNamespace(download=lambda p: downloads.append(p))
    colab = types.ModuleType("google.colab")
    colab.files = fake_files
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    code = next(c for c in code_cells() if c.startswith("import hashlib"))
    exec(
        compile(code, "actual-export-cell", "exec"),
        {
            "BASE": tmp_path,
            "OUT": out,
            "RUN_EXIT_CODE": 1,
            "files": fake_files,
            "CAPTURE_IS_CURRENT": True,
        },
    )
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        assert set(archive.namelist()) == {
            "comparison/jev_predictions.jsonl",
            "comparison/run.json",
            "execution.txt",
        }
        assert all(b"private canary" not in archive.read(name) for name in archive.namelist())


def test_old_capture_is_rejected_before_secret_lookup_and_cannot_be_exported_as_new(
    tmp_path, monkeypatch
):
    out = tmp_path / "old_capture"
    out.mkdir()
    (out / "run.json").write_text('{"execution_complete":true,"old":true}\n')
    code = next(c for c in code_cells() if c.startswith("run_env = dict(clean_env)"))
    namespace = {
        "ALLOW_JEV": True,
        "MAX_COST_USD": 0.1,
        "clean_env": {},
        "math": math,
        "OUT": out,
        "BASE": tmp_path,
    }
    with pytest.raises(ValueError, match="Existing capture"):
        exec(compile(code, "actual-execution-cell", "exec"), namespace)
    assert namespace["CAPTURE_IS_CURRENT"] is False
    downloads = []
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=lambda p: downloads.append(p))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    export = next(c for c in code_cells() if c.startswith("import hashlib"))
    exec(compile(export, "actual-export-cell", "exec"), namespace)
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "comparison/run.json" not in archive.namelist()
