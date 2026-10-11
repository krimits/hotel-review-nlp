"""Block the submitted dependency/interpreter failure before model download, including import shadowing."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reviewnlp.triage import format_runtime as runtime
from scripts import check_triage_format_runtime as checker
from scripts import compare_triage_formats as experiment


def fake_runtime(monkeypatch, tmp_path):
    prefix = tmp_path / "venv"
    prefix.mkdir()
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    versions = dict(runtime.PACKAGE_PINS)
    versions["torch"] += "+cu130"
    monkeypatch.setattr(runtime.importlib.metadata, "version", versions.__getitem__)
    modules = {runtime.MODULES[name]: SimpleNamespace(
        __file__=str(prefix / "lib" / runtime.MODULES[name] / "__init__.py"), __version__=version)
        for name, version in versions.items()}
    modules["torch"].cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fake T4")
    modules["lmformatenforcer"].JsonSchemaParser = lambda _: None
    modules["lmformatenforcer.integrations.transformers"] = SimpleNamespace(
        build_token_enforcer_tokenizer_data=lambda _: None,
        build_transformers_prefix_allowed_tokens_fn=lambda *_: None)
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    return prefix, versions, modules


def test_real_integration_imports_with_the_installed_ci_libraries_without_models_or_provider_requests():
    # CI's CPU torch/pydantic may differ; this test checks real integration imports, not production pins.
    installed = {name: importlib.metadata.version(name).split("+", 1)[0] for name in runtime.PACKAGE_PINS}
    result = runtime.check_runtime(required_versions=installed)
    assert result["runtime_ready"] and result["format_integration_import_ok"]
    assert result["model_loaded"] is result["hub_requests"] is False


def test_compatible_pins_and_module_origins_accept_gpu_build_suffix_without_loading_any_model(monkeypatch, tmp_path):
    prefix, _, _ = fake_runtime(monkeypatch, tmp_path)
    result = runtime.check_runtime(expected_prefix=prefix, require_cuda=True)
    assert result["runtime_ready"] and result["isolated"] and result["cuda"]
    assert result["packages"]["torch"] == "2.11.0+cu130"


@pytest.mark.parametrize("failure,expected", [
    ("wrong_interpreter", "isolated_interpreter_required"),
    ("system_interpreter", "isolated_interpreter_required"),
    ("system_site", "system_site_packages_disallowed"),
    ("submitted_version", "pinned_dependency_mismatch"),
    ("missing_distribution", "pinned_dependency_mismatch"),
    ("shadowed_version", "imported_dependency_mismatch"),
    ("global_package", "dependency_outside_isolated_environment"),
    ("integration", "format_integration_import_failed"),
    ("no_cuda", "cuda_required"),
])
def test_actual_import_and_environment_failures_are_rejected_with_bounded_diagnostics(monkeypatch, tmp_path, failure, expected):
    prefix, versions, modules = fake_runtime(monkeypatch, tmp_path)
    if failure == "wrong_interpreter":
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "different"))
    elif failure == "system_interpreter":
        monkeypatch.setattr(sys, "base_prefix", str(prefix))
    elif failure == "system_site":
        (prefix / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    elif failure == "submitted_version":
        versions["transformers"] = "5.18.0"
    elif failure == "missing_distribution":
        def missing(name):
            if name == "transformers":
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        monkeypatch.setattr(runtime.importlib.metadata, "version", missing)
    elif failure == "shadowed_version":
        modules["transformers"].__version__ = "5.18.0"
    elif failure == "global_package":
        modules["transformers"].__file__ = str(tmp_path / "global/transformers.py")
    elif failure == "integration":
        def broken(name):
            if name.endswith("integrations.transformers"):
                raise ImportError("untrusted exception FAKE-SECRET")
            return modules[name]
        monkeypatch.setattr(runtime.importlib, "import_module", broken)
    else:
        modules["torch"].cuda.is_available = lambda: False
    with pytest.raises(runtime.RuntimePreflightError) as raised:
        runtime.check_runtime(expected_prefix=prefix, require_cuda=True)
    assert raised.value.code == expected
    assert "FAKE-SECRET" not in json.dumps(raised.value.details)


def test_comparison_does_not_import_model_loader_or_create_output_when_preflight_fails(monkeypatch, tmp_path):
    seen = []

    def blocked(**kwargs):
        seen.append(kwargs)
        raise runtime.RuntimePreflightError("pinned_dependency_mismatch", {"packages": {"transformers": "5.18.0"}})

    monkeypatch.setattr(experiment, "check_runtime", blocked)
    monkeypatch.setattr(sys, "argv", ["compare_triage_formats.py", "--cases", "all", "--device", "cpu",
                                      "--output", str(tmp_path / "comparison")])
    with pytest.raises(runtime.RuntimePreflightError):
        experiment.main()
    assert len(seen) == 1 and not (tmp_path / "comparison").exists()


def test_checker_preserves_failure_receipt_and_exit_one_before_model_loading(monkeypatch, tmp_path):
    def blocked(**_):
        raise runtime.RuntimePreflightError("pinned_dependency_mismatch", {
            "packages": {"transformers": {"expected": "4.56.2", "actual": "5.18.0"}}})

    monkeypatch.setattr(checker, "check_runtime", blocked)
    output = tmp_path / "runtime_preflight.json"
    monkeypatch.setattr(sys, "argv", ["check_triage_format_runtime.py", "--expected-prefix", str(tmp_path / "env"),
                                      "--output", str(output), "--require-cuda"])
    with pytest.raises(SystemExit) as raised:
        checker.main()
    assert raised.value.code == 1
    receipt = json.loads(output.read_bytes())
    assert receipt["error"] == "pinned_dependency_mismatch"
    assert receipt["runtime_ready"] is receipt["model_loaded"] is receipt["hub_requests"] is False
    with pytest.raises(ValueError, match="new_preflight_receipt"):
        checker.main()


def test_install_requirements_and_runtime_pins_are_the_same_contract():
    path = Path(__file__).resolve().parents[1] / "configs/triage_format_requirements.txt"
    lines = path.read_text().splitlines()
    assert lines == [name + "==" + version for name, version in runtime.PACKAGE_PINS.items()]
