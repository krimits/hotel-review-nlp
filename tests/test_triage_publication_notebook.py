"""Notebook 18 pins, credentials and publication result handling; no Colab/Hub writes."""

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
    notebook = json.loads((ROOT / "notebooks/18_publish_triage_space_colab.ipynb").read_text())
    return notebook, {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}


def test_original_publication_notebook_pins_its_historical_sources_without_outputs_or_reserved_access():
    nb, code = cells()
    namespace = {}
    exec(code["parameters"], namespace)
    assert namespace["SPACE_ID"] == "krimits/hotel-triage-demo"
    assert namespace["DRY_RUN"] is False and namespace["RESUME"] is False
    for name, expected in namespace["SOURCE_HASHES"].items():
        content = subprocess.check_output(["git", "show", namespace["SOURCE_COMMIT"] + ":" + name], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    combined = "\n".join(code.values())
    assert "holdout.json" not in combined and "files.upload" not in combined
    assert "userdata.get" not in "\n".join(value for key, value in code.items() if key != "publish")
    assert '"--no-deps"' in code["setup"] and '"-m", "venv"' in code["setup"]
    assert "--system-site-packages" not in combined
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("dry_run,verified", [(False, True), (False, False), (True, False)])
def test_publication_cell_keeps_token_only_in_child_environment_and_requires_live_receipt(tmp_path, dry_run, verified):
    _, code = cells()
    report_path, calls, secrets = tmp_path / "published.json", [], []
    secret = "FAKE-HF-TOKEN-DO-NOT-PRINT"

    def get(name):
        secrets.append(name)
        return secret

    def run(arguments, **kwargs):
        calls.append((arguments, dict(kwargs["env"])))
        if not dry_run:
            report_path.write_text(json.dumps({"url": "https://huggingface.co/spaces/krimits/hotel-triage-demo",
                "space_commit": "a" * 40, "upload_verified": True, "runtime_verified": verified}))
        return SimpleNamespace(returncode=0)

    namespace = {"SMOKE_READY": True, "DRY_RUN": dry_run, "RESUME": False, "SECRET_NAME": "HF_TOKEN",
        "base_env": {"REVIEWNLP_JEV_ENABLED": "0"}, "PYTHON": "python", "REPO_DIR": ROOT,
        "PACKAGE_DIR": tmp_path / "package", "SMOKE_REPORT": tmp_path / "smoke.json",
        "PUBLICATION_REPORT": report_path, "RUN_DIR": tmp_path,
        "subprocess": SimpleNamespace(run=run, STDOUT=-2), "json": json}
    stdout = io.StringIO()
    colab = SimpleNamespace(userdata=SimpleNamespace(get=get))
    with patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}), redirect_stdout(stdout):
        exec(code["publish"], namespace)
    assert secret not in stdout.getvalue() and secret not in str(calls[0][0])
    assert "HF_TOKEN" not in namespace["publication_env"]
    assert namespace["LIVE_READY"] is (verified and not dry_run)
    if dry_run:
        assert secrets == [] and "HF_TOKEN" not in calls[0][1]
        assert "--dry-run" in calls[0][0]
    else:
        assert secrets == ["HF_TOKEN"] and calls[0][1]["HF_TOKEN"] == secret


def test_recorded_real_weight_smoke_matches_the_package_snapshot_without_a_quality_claim():
    folder = ROOT / "docs/experiments/triage_space_publication/f2da186"
    manifest_bytes = (folder / "source_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    report = json.loads((folder / "cpu_smoke.json").read_text())
    assert report["passed"] and report["real_weights_loaded"] and report["problems"] == []
    assert report["source_commit"] == manifest["source_commit"]
    assert report["source_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert report["package_files"] == manifest["files"]
    assert len(report["records"]) == 14
    assert report["quality_evaluated"] is False and report["reserved_evaluation_used"] is False
    assert report["runtime"]["qwen_device"] == "cpu" and not report["runtime"]["zero_gpu"]
