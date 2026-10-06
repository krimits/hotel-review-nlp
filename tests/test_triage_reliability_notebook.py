"""Real notebook cells must guard remote calls and preserve interrupted review evidence."""
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
SOURCE = "321c20689418c45267122f537ea6451f3ef30417"


def cells():
    notebook = json.loads((ROOT / "notebooks/20_triage_reliability_colab.ipynb").read_bytes())
    code = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    return notebook, code


def test_notebook_binds_package_capture_and_scorer_without_reserved_cases_or_saved_outputs():
    notebook, code = cells()
    namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(code["parameters"], namespace)
    assert namespace["SOURCE_COMMIT"] == SOURCE
    assert namespace["RESUME"] and namespace["CAPTURE_REVIEW"]
    assert not namespace["REVIEW_JEV"] and not namespace["DRY_RUN"]
    for path, expected in namespace["SOURCE_HASHES"].items():
        content = subprocess.check_output(["git", "show", SOURCE + ":" + path], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    assert {"scripts/capture_triage_review.py", "scripts/score_triage_review.py",
            "docs/experiments/triage_generator/dev.json"} <= namespace["SOURCE_HASHES"].keys()
    assert "holdout.json" not in "\n".join(code.values())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("live_ready,dry_run,capture_review", [(False, False, True), (True, True, True), (True, False, False)])
def test_review_cell_makes_no_remote_calls_without_verified_publication(live_ready, dry_run, capture_review):
    _, code = cells()
    namespace = {"LIVE_READY": live_ready, "DRY_RUN": dry_run, "CAPTURE_REVIEW": capture_review}
    exec(code["review"], namespace)
    assert namespace["review_code"] is None


@pytest.mark.parametrize("jev,complete", [(False, True), (True, False)])
def test_review_cell_uses_secret_only_in_child_environment_and_preserves_incomplete_capture(tmp_path, jev, complete):
    _, code = cells()
    folder = tmp_path / "review"
    secret, calls = "FAKE-REVIEW-SECRET", []

    def run(arguments, **kwargs):
        calls.append((arguments, dict(kwargs["env"])))
        folder.mkdir()
        (folder / "run.json").write_text(json.dumps({"execution_complete": complete,
            "planned_cases": 24, "executed_cases": 24 if complete else 1,
            "failures": 0 if complete else 1, "stop_reason": None if complete else {"error": "gpu_quota_exceeded"},
            "jev_requested": jev}))
        return SimpleNamespace(returncode=0 if complete else 1)

    namespace = {"LIVE_READY": True, "DRY_RUN": False, "CAPTURE_REVIEW": True, "REVIEW_JEV": jev,
        "REVIEW_READY": False, "SECRET_NAME": "HF_TOKEN", "SOURCE_COMMIT": SOURCE,
        "base_env": {"REVIEWNLP_JEV_ENABLED": "0"}, "PYTHON": "python", "REPO_DIR": ROOT,
        "RUN_DIR": tmp_path, "REVIEW_DIR": folder, "json": json,
        "subprocess": SimpleNamespace(run=run, STDOUT=-2)}
    colab = SimpleNamespace(userdata=SimpleNamespace(get=lambda name: secret))
    output = io.StringIO()
    with patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}), redirect_stdout(output):
        exec(code["review"], namespace)
    assert namespace["REVIEW_READY"] is complete
    assert len(calls) == 1 and ("--jev" in calls[0][0]) is jev
    assert calls[0][1]["HF_TOKEN"] == secret and "HF_TOKEN" not in namespace["review_env"]
    assert secret not in output.getvalue() and secret not in str(calls[0][0])


def test_export_preserves_failure_receipt_and_review_files_without_a_package(tmp_path):
    _, code = cells()
    folder = tmp_path / "review"
    folder.mkdir()
    (folder / "run.json").write_text('{"execution_complete": false}')
    (tmp_path / "published.json").write_text('{"runtime_verified": false}')
    downloaded = []
    namespace = {"RUN_DIR": tmp_path, "REVIEW_DIR": folder, "PACKAGE_DIR": tmp_path / "package",
                 "LIVE_READY": False, "REVIEW_READY": False}
    colab = SimpleNamespace(files=SimpleNamespace(download=downloaded.append))
    with patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}):
        exec(code["export"], namespace)
    with zipfile.ZipFile(downloaded[0]) as bundle:
        assert set(bundle.namelist()) == {"published.json", "review/run.json"}


def test_setup_disables_telemetry_and_corrects_the_colab_backend():
    _, code = cells()
    namespace = {"SECRET_NAME": "HF_TOKEN"}
    exec(code["setup"].split("if not REPO_DIR.exists():", 1)[0], namespace)
    assert namespace["base_env"]["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert namespace["base_env"]["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert namespace["base_env"]["MPLBACKEND"] == "Agg"
