"""The Space update notebook pins the fix and cannot upload after a failed regression."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cells():
    nb = json.loads((ROOT / "notebooks/19_update_triage_evidence_space_colab.ipynb").read_text())
    return nb, {cell["id"]: "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"}


def test_update_pins_all_sources_and_existing_space_without_saved_outputs_or_reserved_access():
    nb, code = cells()
    namespace = {}
    exec(code["parameters"], namespace)
    assert namespace["RESUME"] is True and namespace["DRY_RUN"] is False
    assert namespace["SPACE_ID"] == "krimits/hotel-triage-demo"
    assert namespace["SOURCE_COMMIT"] == "5fa1f5635f8df4d537a6d7adbdcb4b22e8ad3bac"
    for name, expected in namespace["SOURCE_HASHES"].items():
        content = subprocess.check_output(["git", "show", namespace["SOURCE_COMMIT"] + ":" + name], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    assert "src/reviewnlp/triage/span_evidence_generator.py" in namespace["SOURCE_HASHES"]
    all_code = "\n".join(code.values())
    assert "holdout.json" not in all_code and "files.upload" not in all_code
    assert "userdata.get" not in "\n".join(value for key, value in code.items() if key != "publish")
    assert '"-m", "venv"' in code["setup"] and "--system-site-packages" not in all_code
    assert "TYPESAFE_API_KEY" in code["setup"] and "triage_space_update.zip" in code["export"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("regressions_pass", [True, False])
def test_smoke_cell_blocks_upload_even_if_process_says_success_but_regressions_fail(tmp_path, regressions_pass):
    _, code = cells()
    report = tmp_path / "smoke.json"
    report.write_text(json.dumps({"passed": True, "regression_checks_passed": regressions_pass,
                                 "records": [], "problems": [], "runtime": {}}))
    namespace = {"PACKAGE_READY": True, "RUN_DIR": tmp_path, "SMOKE_REPORT": report,
        "PYTHON": "python", "PACKAGE_DIR": tmp_path / "package", "REPO_DIR": ROOT,
        "base_env": {}, "json": json,
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(returncode=0), STDOUT=-2)}
    exec(code["smoke"], namespace)
    assert namespace["SMOKE_READY"] is regressions_pass


def test_update_cell_requests_resume_and_keeps_token_out_of_output_and_arguments(tmp_path):
    _, code = cells()
    calls = []
    receipt = tmp_path / "published.json"
    secret = "FAKE-UPDATE-TOKEN"

    def run(arguments, **kwargs):
        calls.append((arguments, dict(kwargs["env"])))
        receipt.write_text(json.dumps({"url": "https://huggingface.co/spaces/krimits/hotel-triage-demo",
            "space_commit": "a" * 40, "upload_verified": True, "runtime_verified": True}))
        return SimpleNamespace(returncode=0)

    namespace = {"SMOKE_READY": True, "DRY_RUN": False, "RESUME": True, "SECRET_NAME": "HF_TOKEN",
        "base_env": {"REVIEWNLP_JEV_ENABLED": "0"}, "PYTHON": "python", "REPO_DIR": ROOT,
        "PACKAGE_DIR": tmp_path / "package", "SMOKE_REPORT": tmp_path / "smoke.json",
        "PUBLICATION_REPORT": receipt, "RUN_DIR": tmp_path,
        "subprocess": SimpleNamespace(run=run, STDOUT=-2), "json": json}
    output = io.StringIO()
    colab = SimpleNamespace(userdata=SimpleNamespace(get=lambda name: secret))
    with patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}), redirect_stdout(output):
        exec(code["publish"], namespace)
    assert "--resume" in calls[0][0] and calls[0][1]["HF_TOKEN"] == secret
    assert secret not in str(calls[0][0]) and secret not in output.getvalue()
    assert "HF_TOKEN" not in namespace["publication_env"] and namespace["LIVE_READY"]


def test_archived_qwen_probe_is_bound_to_the_fix_and_does_not_claim_deployment_or_quality_evaluation():
    folder = ROOT / "docs/experiments/triage_space_publication/source-spans-v6"
    report = json.loads((folder / "qwen_regression_probe.json").read_text())
    assert report["deployed"] is report["quality_evaluated"] is report["reserved_evaluation_used"] is False
    assert report["real_weights_loaded"] and len(report["records"]) == 7
    for name, expected in report["source_sha256"].items():
        content = subprocess.check_output(["git", "show", report["source_commit"] + ":" + name], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    by_id = {row["id"]: row for row in report["records"]}
    assert all(row["failure"] is None for row in by_id.values())
    for name in ("lamp-mixed", "lamp-short", "lamp-negative", "not-resolved"):
        row = by_id[name]
        assert row["actions"] and all(action["department"] == "maintenance" for action in row["actions"])
        assert all(issue["excerpt"] in row["review"] for issue in row["issues"])
    assert by_id["hypothetical"]["issues"][0]["status"] == "UNCERTAIN"
    assert not any(by_id[name]["actions"] for name in ("resolved", "hypothetical", "praise"))
    baseline = json.loads((folder / "historical_failures.json").read_text())
    assert [row["failure"] for row in baseline["records"]] == ["issue_quote_missing", "evidence_quote_missing_or_invalid"]


def test_setup_overrides_colab_inline_backend_for_child_processes(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    _, code = cells()
    namespace = {"SECRET_NAME": "HF_TOKEN"}
    # Execute the actual environment setup without installing packages or cloning.
    exec(code["setup"].split("if not REPO_DIR.exists():", 1)[0], namespace)
    assert namespace["base_env"]["MPLBACKEND"] == "Agg"
    with patch("subprocess.run", return_value=SimpleNamespace(stdout="ok")) as run:
        namespace["run_command"](["python", "-c", "pass"], capture=True)
    assert run.call_args.kwargs["env"]["MPLBACKEND"] == "Agg"
