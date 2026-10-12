"""Colab's provider opt-in must be real, and the test partition must remain untouched."""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/23_decision_topics_dev_comparison_colab.ipynb"


def cells():
    notebook = json.loads(NOTEBOOK.read_bytes())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_comparison_notebook_is_pinned_clean_gpu_dev_only_and_defaults_to_no_provider():
    notebook = json.loads(NOTEBOOK.read_bytes())
    code = "\n".join(cells())
    ast.parse(code)
    pin = re.search(r"SOURCE_COMMIT\s*=\s*['\"]([0-9a-f]{40})['\"]", code).group(1)
    for path in (
        "scripts/benchmark_decision_topics.py",
        "scripts/unpack_decision_topic_inputs.py",
        "src/reviewnlp/experiments/topic_models.py",
        "configs/decision_topic_comparison_requirements.txt",
        "docs/experiments/decision_topic_benchmark/annotation_record.json",
    ):
        assert (
            subprocess.check_output(["git", "show", f"{pin}:{path}"], cwd=ROOT)
            == (ROOT / path).read_bytes()
        )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
    assert "ALLOW_MICROSOFT = False" in code
    assert '"--partition", "dev"' in code and '"--partition", "test"' not in code
    assert "system_site_packages=False" in code and '"--isolated", "install", "--no-user"' in code
    assert "module.__file__" in code and "module.__version__" in code
    assert "torch.cuda.is_available()" in code
    assert "files.upload()" in code and "scripts/unpack_decision_topic_inputs.py" in code
    assert (
        "source_record.json" not in code
    )  # private provenance stays in validated inputs, not result ZIP


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_execution_cell_only_reads_and_passes_a_secret_after_opt_in(
    tmp_path, monkeypatch, capsys, enabled
):
    execution = next(cell for cell in cells() if cell.startswith("run_env=dict(clean_env)"))
    reads, invoked = [], []
    colab = types.ModuleType("google.colab")

    class Userdata:
        def get(self, name):
            reads.append(name)
            assert enabled
            return "synthetic-secret-not-a-real-key"

    colab.userdata = Userdata()
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def run(command, *, cwd, env, stdout, stderr):
        invoked.append({"command": command, "env": dict(env)})
        stdout.write("offline invocation fixture; no model or API call\n")
        return types.SimpleNamespace(returncode=0)

    fake_subprocess = types.SimpleNamespace(run=run, STDOUT=subprocess.STDOUT)
    namespace = {
        "clean_env": {},
        "ALLOW_MICROSOFT": enabled,
        "MAX_COST_USD": 0.1,
        "math": math,
        "BASE": tmp_path,
        "REPO": ROOT,
        "OUT": tmp_path / "out",
        "PYTHON": Path("/fake/python"),
        "INPUTS": tmp_path / "inputs",
        "ARMS": ["logistic", "qwen"] + (["decision"] if enabled else []),
        "subprocess": fake_subprocess,
    }
    exec(compile(execution, "execution-cell", "exec"), namespace)
    assert reads == (["OPENROUTER_API_KEY"] if enabled else [])
    assert len(invoked) == 1
    assert ("--allow-external-api" in invoked[0]["command"]) is enabled
    assert ("OPENROUTER_API_KEY" in invoked[0]["env"]) is enabled
    assert "synthetic-secret-not-a-real-key" not in " ".join(invoked[0]["command"])
    assert "OPENROUTER_API_KEY" not in namespace["run_env"]
    assert "synthetic-secret-not-a-real-key" not in capsys.readouterr().out
    assert "synthetic-secret-not-a-real-key" not in (tmp_path / "execution.txt").read_text()


def test_invalid_api_budget_fails_before_secret_lookup_or_execution(tmp_path):
    execution = next(cell for cell in cells() if cell.startswith("run_env=dict(clean_env)"))
    with pytest.raises(ValueError, match="positive USD budget"):
        exec(
            compile(execution, "execution-cell", "exec"),
            {"clean_env": {}, "ALLOW_MICROSOFT": True, "MAX_COST_USD": 0, "math": math},
        )
