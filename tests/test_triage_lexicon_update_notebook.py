"""Run the real notebook cells with offline subprocess/Colab fakes, never Hub writes."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.prepare_triage_space import build_package

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/25_update_triage_lexicon_space_colab.ipynb"


def cells():
    nb = json.loads(NOTEBOOK.read_text())
    return nb, {cell["id"]: "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"}


def test_notebook_pins_the_complete_local_primary_source_and_preserves_historical_templates():
    nb, code = cells()
    parameters = {}
    exec(code["parameters"], parameters)
    assert parameters["SOURCE_COMMIT"] == "02d6e8c7b7297bd87b4060b8f9f504421e60244c"
    assert parameters["RESUME"] is True and parameters["DRY_RUN"] is False
    assert parameters["SPACE_ID"] == "krimits/hotel-triage-demo"
    assert "src/reviewnlp/triage/lexicon.py" in parameters["SOURCE_HASHES"]
    for name, expected in parameters["SOURCE_HASHES"].items():
        content = subprocess.check_output(["git", "show", parameters["SOURCE_COMMIT"] + ":" + name], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    assert "spaces==0.51.3" in code["setup"] and "--system-site-packages" not in code["setup"]
    assert "userdata.get" not in "\n".join(source for name, source in code.items() if name != "publish")
    assert "holdout.json" not in "\n".join(code.values()) and "files.upload" not in "\n".join(code.values())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("policy", ["valid", "unverified", "jev_enabled", "wrong_lexicon"])
def test_smoke_gate_cannot_accept_wrong_primary_even_with_successful_model_execution(tmp_path, policy):
    _, code = cells()
    report = {"passed": True, "regression_checks_passed": True, "records": [], "problems": [], "runtime": {},
              "complaint_policy": {"primary": "lexicon", "lexicon_sha256": "expected", "jev_enabled": False,
                                   "verified": True}}
    if policy == "unverified":
        report["complaint_policy"]["verified"] = False
    elif policy == "jev_enabled":
        report["complaint_policy"]["jev_enabled"] = True
    elif policy == "wrong_lexicon":
        report["complaint_policy"]["lexicon_sha256"] = "old"
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report))
    namespace = {"PACKAGE_READY": True, "RUN_DIR": tmp_path, "SMOKE_REPORT": path,
        "PYTHON": "python", "PACKAGE_DIR": tmp_path / "package", "REPO_DIR": ROOT, "base_env": {},
        "json": json, "EXPECTED_LEXICON_SHA256": "expected", "LIVE_READY": True, "SMOKE_READY": True,
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(returncode=0), STDOUT=-2)}
    exec(code["smoke"], namespace)
    assert namespace["SMOKE_READY"] is (policy == "valid")
    assert namespace["LIVE_READY"] is False


@pytest.mark.parametrize("smoke_pass,dry_run,published_policy", [(True, False, True), (False, False, False),
                                                             (True, True, False), (True, False, False)])
def test_top_to_bottom_real_cells_keep_secret_safe_skip_bad_smoke_and_require_live_primary_receipt(
        tmp_path, smoke_pass, dry_run, published_policy):
    _, code = cells()
    namespace = {}
    exec(code["parameters"], namespace)
    checkout = tmp_path / "checkout"
    for name in namespace["SOURCE_HASHES"]:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", namespace["SOURCE_COMMIT"] + ":" + name], cwd=ROOT))
    run_dir = checkout / "runs" / "offline"
    namespace.update(REPO_DIR=checkout, ENV_DIR=tmp_path / "venv", RUN_DIR=run_dir,
        PACKAGE_DIR=run_dir / "package", SMOKE_REPORT=run_dir / "smoke.json",
        PUBLICATION_REPORT=run_dir / "published.json", SETUP_LOG=tmp_path / "setup.txt", DRY_RUN=dry_run)
    calls, reads, downloads = [], [], []
    secret = "FAKE-LEXICON-HF-SECRET"

    def userdata_get(name):
        reads.append(name)
        return secret

    def run(arguments, **kwargs):
        calls.append((list(arguments), dict(kwargs["env"])))
        stdout = ""
        if arguments[:3] == ["git", "rev-parse", "HEAD"]:
            stdout = namespace["SOURCE_COMMIT"]
        elif "-c" in arguments:
            stdout = '{"cuda":true,"torch":"fake"}'
        elif "scripts/prepare_triage_space.py" in arguments:
            build_package(ROOT, namespace["PACKAGE_DIR"], source_commit=namespace["SOURCE_COMMIT"])
        elif "scripts/check_triage_space.py" in arguments:
            report = {"passed": smoke_pass, "regression_checks_passed": smoke_pass,
                "problems": [], "runtime": {"ready": True}, "records": [],
                "complaint_policy": {"primary": "lexicon", "verified": smoke_pass, "jev_enabled": False,
                                     "lexicon_sha256": namespace["EXPECTED_LEXICON_SHA256"]}}
            namespace["SMOKE_REPORT"].write_text(json.dumps(report))
            return SimpleNamespace(returncode=0 if smoke_pass else 1)
        elif "scripts/publish_triage_space.py" in arguments:
            assert "--local-primary" in arguments and "--resume" in arguments
            if dry_run:
                assert "--dry-run" in arguments and "HF_TOKEN" not in kwargs["env"]
            else:
                assert kwargs["env"]["HF_TOKEN"] == secret
                namespace["PUBLICATION_REPORT"].write_text(json.dumps({"runtime_verified": True,
                    "local_primary_verified": published_policy, "upload_verified": True,
                    "url": "https://huggingface.co/spaces/krimits/hotel-triage-demo", "space_commit": "a" * 40}))
                kwargs["stdout"].write(secret + "\n")  # Export must redact even an unexpected echo.
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    colab = SimpleNamespace(userdata=SimpleNamespace(get=userdata_get), files=SimpleNamespace(download=downloads.append))
    output = io.StringIO()
    with patch("subprocess.run", side_effect=run), patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}), \
            patch.dict(os.environ, {"HF_TOKEN": "AMBIENT-SECRET", "OPENROUTER_API_KEY": "AMBIENT-PROVIDER", "REVIEWNLP_JEV_ENABLED": "1"}), \
            redirect_stdout(output):
        for stage in ("setup", "package", "smoke", "publish", "export"):
            exec(code[stage], namespace)
    assert namespace["LIVE_READY"] is (smoke_pass and not dry_run and published_policy)
    assert reads == (["HF_TOKEN"] if smoke_pass and not dry_run else [])
    if not smoke_pass:
        assert not any("scripts/publish_triage_space.py" in args for args, _ in calls)
    assert namespace["base_env"]["REVIEWNLP_TRIAGE_LEXICON_FALLBACK"] == "1"
    assert namespace["base_env"]["REVIEWNLP_JEV_ENABLED"] == "0"
    assert all("OPENROUTER_API_KEY" not in env for _, env in calls)
    assert secret not in output.getvalue() and "AMBIENT-SECRET" not in output.getvalue()
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "source_manifest.json" in archive.namelist() and "setup_execution.txt" in archive.namelist()
        assert all(secret.encode() not in archive.read(name) for name in archive.namelist())
        assert not any(name.endswith(".csv") for name in archive.namelist())


def test_export_after_setup_failure_does_not_require_a_package_or_upload(tmp_path):
    _, code = cells()
    setup = tmp_path / "setup.txt"
    setup.write_text("pinned dependency install failed\n")
    downloaded = []
    namespace = {"RUN_DIR": tmp_path / "run", "PACKAGE_DIR": tmp_path / "missing", "SETUP_LOG": setup,
                 "LIVE_READY": False}
    colab = SimpleNamespace(files=SimpleNamespace(download=downloaded.append))
    with patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}):
        exec(code["export"], namespace)
    with zipfile.ZipFile(downloaded[0]) as archive:
        assert archive.namelist() == ["setup_execution.txt"]
