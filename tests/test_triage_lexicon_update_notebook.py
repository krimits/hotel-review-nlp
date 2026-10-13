"""Run the real notebook cells with offline subprocess/Colab fakes, never Hub writes."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from importlib.metadata import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement

from scripts.prepare_triage_space import build_package

ROOT = Path(__file__).resolve().parents[1]
# Notebook 26 republishes notebook 25's local-primary Space with cached examples; both stay runnable.
NOTEBOOKS = {
    "notebooks/25_update_triage_lexicon_space_colab.ipynb": "02d6e8c7b7297bd87b4060b8f9f504421e60244c",
    "notebooks/26_update_triage_examples_space_colab.ipynb": "c65e638c0eccb7979c3d2a1fa48d0ce5e06735fc",
}


@pytest.fixture(params=sorted(NOTEBOOKS))
def notebook(request):
    return ROOT / request.param


def cells(notebook):
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    return nb, {cell["id"]: "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"}


def test_notebook_pins_the_complete_local_primary_source_and_preserves_historical_templates(notebook):
    nb, code = cells(notebook)
    parameters = {}
    exec(code["parameters"], parameters)
    assert parameters["SOURCE_COMMIT"] == NOTEBOOKS[notebook.relative_to(ROOT).as_posix()]
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
def test_smoke_gate_cannot_accept_wrong_primary_even_with_successful_model_execution(tmp_path, policy, notebook):
    _, code = cells(notebook)
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
        tmp_path, smoke_pass, dry_run, published_policy, notebook):
    _, code = cells(notebook)
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


def test_export_after_setup_failure_does_not_require_a_package_or_upload(tmp_path, notebook):
    _, code = cells(notebook)
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


def metadata_wheel(directory, name, version, dependencies=()):
    """Small local distributions exercise pip resolution, not model execution."""
    stem = name.replace("-", "_") + "-" + version
    info = stem + ".dist-info/"
    contents = {
        info + "METADATA": "Metadata-Version: 2.1\nName: " + name + "\nVersion: " + version + "\n"
            + "".join("Requires-Dist: " + dependency + "\n" for dependency in dependencies),
        info + "WHEEL": "Wheel-Version: 1.0\nGenerator: offline-setup-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = io.StringIO()
    writer = csv.writer(record)
    for filename in (*contents, info + "RECORD"):
        writer.writerow([filename, "", ""])
    contents[info + "RECORD"] = record.getvalue()
    wheel = directory / (stem + "-py3-none-any.whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        for filename, content in contents.items():
            archive.writestr(filename, content)
    return wheel


@pytest.mark.parametrize("partial_environment", [False, True])
def test_actual_notebook_installation_resolves_core_dependencies_in_a_pipless_isolated_environment(
        tmp_path, partial_environment, notebook):
    _, code = cells(notebook)
    function = next(node for node in ast.parse(code["setup"]).body
                    if isinstance(node, ast.FunctionDef) and node.name == "install_environment")
    project = metadata("reviewnlp")
    dependencies = [text for text in project.get_all("Requires-Dist", [])
                    if Requirement(text).marker is None or Requirement(text).marker.evaluate({"extra": ""})]
    requirements = (ROOT / "spaces/hotel-triage-demo/requirements.txt").read_text().splitlines()
    versions = {}
    for text in [*dependencies, *requirements, "spaces==0.51.3"]:
        requirement = Requirement(text)
        pins = [spec.version for spec in requirement.specifier if spec.operator == "=="]
        minima = [spec.version for spec in requirement.specifier if spec.operator == ">="]
        if pins or (minima and requirement.name not in versions):
            versions[requirement.name] = (pins or minima)[0]
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name, version in versions.items():
        metadata_wheel(wheels, name, version)
    project_wheel = metadata_wheel(wheels, project["Name"], project["Version"], dependencies)
    target = tmp_path / "environment"
    if partial_environment:
        target.mkdir()  # A failed ensurepip may leave an existing directory behind.
    calls = []
    environment = {**os.environ, "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(wheels),
                   "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    environment.pop("PYTHONPATH", None)

    def run(arguments):
        arguments = list(arguments)
        calls.append(arguments[:])
        if "-e" in arguments:
            index = arguments.index("-e")
            # Real project dependency metadata, in a wheel, avoids an online build-backend download.
            arguments[index:index + 2] = [str(project_wheel)]
        if "venv" in arguments:
            assert "--without-pip" in arguments
        subprocess.run(arguments, env=environment, capture_output=True, text=True, check=True)

    namespace = {"sys": sys, "run_command": run}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)
    python = namespace["install_environment"](ROOT, target)
    info = json.loads(subprocess.check_output([python, "-c",
        "import sys,json,importlib.util,importlib.metadata as m; "
        "print(json.dumps({'isolated':sys.prefix!=sys.base_prefix,'pip':importlib.util.find_spec('pip') is not None,"
        "'versions':{n:m.version(n) for n in " + repr(list(versions)) + "}}))"], env=environment, text=True))
    assert info["isolated"] and info["pip"] is False
    assert info["versions"] == versions
    install = next(arguments for arguments in calls if "install" in arguments)
    assert "--no-deps" not in install
    assert "-r" in install and "-e" in install
    assert calls[-1][-1] == "check"
    assert "include-system-site-packages = false" in (target / "pyvenv.cfg").read_text()
