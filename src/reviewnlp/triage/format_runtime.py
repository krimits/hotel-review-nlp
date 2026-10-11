"""Check the format experiment's actual imports before any model download or generation."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path

PACKAGE_PINS = {
    "torch": "2.11.0",
    "transformers": "4.56.2",
    "huggingface-hub": "0.36.2",
    "accelerate": "1.10.1",
    "pydantic": "2.11.7",
    "lm-format-enforcer": "0.11.3",
}
MODULES = {
    "torch": "torch", "transformers": "transformers", "huggingface-hub": "huggingface_hub",
    "accelerate": "accelerate", "pydantic": "pydantic", "lm-format-enforcer": "lmformatenforcer",
}


class RuntimePreflightError(ValueError):
    """Closed error code and deliberately constructed diagnostics; no exception text or environment dump."""

    def __init__(self, code: str, details: dict):
        super().__init__(code)
        self.code, self.details = code, details


def check_runtime(*, expected_prefix: Path | None = None, require_cuda: bool = False,
                  required_versions: dict | None = None) -> dict:
    """A version check, actual module-origin check and real integration import; no model loading."""
    required = dict(PACKAGE_PINS if required_versions is None else required_versions)
    prefix, base = Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()
    if expected_prefix is not None and (prefix != expected_prefix.resolve() or prefix == base):
        raise RuntimePreflightError("isolated_interpreter_required", {
            "expected_prefix": str(expected_prefix.resolve()), "actual_prefix": str(prefix),
            "base_prefix": str(base)})
    if expected_prefix is not None:
        config = prefix / "pyvenv.cfg"
        settings = {key.strip(): value.strip() for line in config.read_text().lower().splitlines()
                    if "=" in line for key, value in [line.split("=", 1)]} if config.is_file() else {}
        if settings.get("include-system-site-packages") != "false":
            raise RuntimePreflightError("system_site_packages_disallowed", {"prefix": str(prefix)})
    installed = {}
    for name in required:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    mismatches = {name: {"expected": version, "actual": installed[name]}
                  for name, version in required.items()
                  if (installed[name] or "").split("+", 1)[0] != version}
    if mismatches:
        raise RuntimePreflightError("pinned_dependency_mismatch", {"packages": mismatches})
    origins, loaded = {}, {}
    for name in required:
        try:
            module = importlib.import_module(MODULES[name])
        except Exception as error:
            raise RuntimePreflightError("dependency_import_failed", {
                "package": name, "error_type": type(error).__name__}) from None
        origin = Path(module.__file__).resolve()
        origins[name], loaded[name] = str(origin), module
        if expected_prefix is not None and not origin.is_relative_to(prefix):
            raise RuntimePreflightError("dependency_outside_isolated_environment", {
                "package": name, "module_file": str(origin), "expected_prefix": str(prefix)})
        imported_version = getattr(module, "__version__", None)
        if imported_version is not None and imported_version.split("+", 1)[0] != required[name]:
            raise RuntimePreflightError("imported_dependency_mismatch", {
                "package": name, "expected": required[name], "actual": imported_version})
    try:
        integration = importlib.import_module("lmformatenforcer.integrations.transformers")
        if not all(callable(value) for value in (integration.build_token_enforcer_tokenizer_data,
                integration.build_transformers_prefix_allowed_tokens_fn,
                loaded["lm-format-enforcer"].JsonSchemaParser)):
            raise TypeError("format_integration_contract")
    except Exception as error:
        raise RuntimePreflightError("format_integration_import_failed", {
            "error_type": type(error).__name__}) from None
    cuda = bool(loaded["torch"].cuda.is_available())
    if require_cuda and not cuda:
        raise RuntimePreflightError("cuda_required", {})
    return {"runtime_ready": True, "python": platform.python_version(),
            "python_executable": sys.executable, "prefix": str(prefix), "base_prefix": str(base),
            "isolated": prefix != base, "packages": installed, "module_files": origins,
            "format_integration_import_ok": True, "cuda": cuda,
            "gpu": loaded["torch"].cuda.get_device_name(0) if cuda else None,
            "model_loaded": False, "hub_requests": False}
