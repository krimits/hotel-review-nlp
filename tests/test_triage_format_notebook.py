"""Run the notebook handoff cells offline; verify pins, isolation and failure-preserving exports."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "1215e4008225c7c1b766af37d609dbd63535ef2c"


def cells():
    notebook = json.loads((ROOT / "notebooks/21_triage_format_comparison_colab.ipynb").read_bytes())
    return notebook, {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}


def parameters():
    namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(cells()[1]["parameters"], namespace)
    return namespace


def test_notebook_pins_actual_source_and_all_cases_without_any_secret_or_reserved_work():
    notebook, code = cells()
    namespace = parameters()
    assert namespace["SOURCE_COMMIT"] == SOURCE
    assert namespace["CASE_SCOPE"] == "all" and namespace["PRECISION"] == "native"
    assert "lm-format-enforcer==0.11.3" in namespace["DEPENDENCIES"]
    for name, expected in namespace["SOURCE_HASHES"].items():
        content = subprocess.check_output(["git", "show", SOURCE + ":" + name], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    assert {"scripts/compare_triage_formats.py", "scripts/score_triage_review.py",
            "src/reviewnlp/triage/structured_span_generator.py",
            "docs/experiments/triage_generator/dev.json"} <= namespace["SOURCE_HASHES"].keys()
    source = "\n".join(code.values())
    for forbidden in ("userdata", "notebook_login", "holdout.json", "upload_folder", "HfApi", "publish_triage_space.py"):
        assert forbidden not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("cuda,head", [(True, SOURCE), (False, SOURCE), (True, "unrelated-checkout")])
def test_setup_installs_only_in_the_child_venv_and_requires_gpu_and_the_pinned_checkout(tmp_path, cuda, head):
    namespace = parameters()
    namespace.update(REPO_DIR=ROOT, ENV_DIR=tmp_path / "venv", RUN_DIR=tmp_path / "run")
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        if "rev-parse" in arguments:
            output = head
        elif "-c" in arguments:
            output = json.dumps({"cuda": cuda, "torch": "test", "gpu": "fake"})
        else:
            output = ""
        return SimpleNamespace(stdout=output, returncode=0)

    with patch.dict("os.environ", {"HF_TOKEN": "FAKE-HF-SECRET", "OPENROUTER_API_KEY": "FAKE-JEV-SECRET"}), \
            patch("subprocess.run", side_effect=run):
        if cuda and head == SOURCE:
            exec(cells()[1]["setup"], namespace)
            assert namespace["SETUP_READY"] and (tmp_path / "run/versions.txt").is_file()
        else:
            with pytest.raises(RuntimeError):
                exec(cells()[1]["setup"], namespace)
            assert namespace["SETUP_READY"] is False
    for arguments, kwargs in calls:
        assert "HF_TOKEN" not in kwargs["env"] and "OPENROUTER_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert kwargs["env"]["GRADIO_ANALYTICS_ENABLED"] == "False"
        if "pip" in arguments:
            assert arguments[0] == str(tmp_path / "venv/bin/python")
    installs = [args for args, _ in calls if "install" in args]
    assert len(installs) == (0 if head != SOURCE else 1)
    if installs:
        assert installs[0][-len(namespace["DEPENDENCIES"]):] == namespace["DEPENDENCIES"]


@pytest.mark.parametrize("exit_code,complete", [(0, True), (0, False), (1, False)])
def test_compare_preserves_exit_status_and_incomplete_reports_without_claiming_success(tmp_path, exit_code, complete):
    calls = []
    output = tmp_path / "comparison"

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        output.mkdir()
        (output / "run.json").write_text(json.dumps({"execution_complete": complete, "summary": {}}))
        kwargs["stdout"].write("real execution diagnostics\n")
        return SimpleNamespace(returncode=exit_code)

    namespace = {"SETUP_READY": True, "SOURCE_COMMIT": SOURCE, "CASE_SCOPE": "all", "PRECISION": "native",
        "PYTHON": "isolated-python", "REPO_DIR": ROOT, "RUN_DIR": tmp_path,
        "OUTPUT_DIR": output, "CACHE_DIR": tmp_path / "cache", "base_env": {}, "json": json,
        "subprocess": SimpleNamespace(run=run, STDOUT=-2)}
    exec(cells()[1]["compare"], namespace)
    receipt = json.loads((tmp_path / "launcher.json").read_bytes())
    assert receipt["execution_complete"] is (complete and exit_code == 0)
    assert receipt["exit_code"] == exit_code
    arguments, kwargs = calls[0]
    assert arguments[:2] == ["isolated-python", "scripts/compare_triage_formats.py"]
    assert arguments[arguments.index("--cases") + 1] == "all"
    assert arguments[arguments.index("--device") + 1] == "cuda"
    assert arguments[arguments.index("--precision") + 1] == "native"
    assert kwargs["env"] == {} and not receipt["hub_writes"] and not receipt["jev_requested"]
    assert (tmp_path / "execution.txt").read_text() == "real execution diagnostics\n"


@pytest.mark.parametrize("setup_ready", [False, True])
def test_launch_failure_and_incomplete_setup_produce_sanitized_receipts_and_no_fake_complete_capture(tmp_path, setup_ready):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("untrusted exception with FAKE-SECRET")

    namespace = {"SETUP_READY": setup_ready, "SOURCE_COMMIT": SOURCE, "CASE_SCOPE": "all", "PRECISION": "native",
        "PYTHON": "isolated-python", "REPO_DIR": ROOT, "RUN_DIR": tmp_path,
        "OUTPUT_DIR": tmp_path / "comparison", "CACHE_DIR": tmp_path / "cache", "base_env": {}, "json": json,
        "subprocess": SimpleNamespace(run=run, STDOUT=-2)}
    captured = io.StringIO()
    with redirect_stdout(captured):
        exec(cells()[1]["compare"], namespace)
    receipt = json.loads((tmp_path / "launcher.json").read_bytes())
    assert receipt["execution_complete"] is False and receipt["exit_code"] is None
    assert receipt["error"] == ("RuntimeError" if setup_ready else "setup_incomplete")
    assert len(calls) == int(setup_ready)
    assert "FAKE-SECRET" not in captured.getvalue() + (tmp_path / "launcher.json").read_text()


def test_zip_exports_only_the_declared_reports_even_when_model_run_failed(tmp_path):
    output = tmp_path / "comparison"
    output.mkdir()
    for name in ("run.json", "results.jsonl", "coverage_A.csv", "coverage_B.csv", "upstream.json"):
        (output / name).write_text("failed-run evidence")
    for name in ("model.safetensors", "HF_TOKEN", "completed_A.csv", "unexpected.txt"):
        (output / name).write_text("must not be exported")
    for name in ("execution.txt", "launcher.json", "versions.txt"):
        (tmp_path / name).write_text("diagnostics")
    (tmp_path / "model.safetensors").write_text("weights")
    downloads = []
    namespace = {"RUN_DIR": tmp_path, "OUTPUT_DIR": output, "COMPARE_COMPLETE": False}
    colab = SimpleNamespace(files=SimpleNamespace(download=downloads.append))
    with patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}):
        exec(cells()[1]["export"], namespace)
    with zipfile.ZipFile(downloads[0]) as bundle:
        assert set(bundle.namelist()) == {"execution.txt", "launcher.json", "versions.txt",
            "comparison/run.json", "comparison/results.jsonl", "comparison/coverage_A.csv",
            "comparison/coverage_B.csv", "comparison/upstream.json"}
