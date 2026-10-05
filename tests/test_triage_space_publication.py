"""Publication guards without network, model downloads, account access or Hub writes."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_triage_space import (  # noqa: E402
    LAMP_REGRESSIONS,
    REQUIRED_REGRESSIONS,
    regression_problems,
)
from scripts.prepare_triage_space import build_package, verify_package  # noqa: E402
from scripts.publish_triage_space import (  # noqa: E402
    HARDWARE,
    REPO_ID,
    check_live,
    deploy,
    validate_smoke,
)


@pytest.fixture
def checked(tmp_path):
    package = tmp_path / "package"
    manifest = build_package(ROOT, package, source_commit="a" * 40)
    report = {"kind": "real_weight_functional_smoke", "passed": True, "real_weights_loaded": True,
              "problems": [], "quality_evaluated": False, "reserved_evaluation_used": False,
              "source_commit": manifest["source_commit"], "package_files": manifest["files"],
              "required_regression_ids": sorted(REQUIRED_REGRESSIONS), "regression_checks_passed": True,
              "records": [{"id": name, "regression_passed": True, "status": "complete"} for name in REQUIRED_REGRESSIONS],
              "source_manifest_sha256": hashlib.sha256((package / "source_manifest.json").read_bytes()).hexdigest()}
    return package, manifest, report


class FakeApi:
    token = None

    def __init__(self, package, name="krimits"):
        self.package, self.name, self.calls = package, name, []
        self.head = "b" * 40

    def whoami(self):
        return {"name": self.name}

    def create_repo(self, **kwargs):
        self.calls.append(("create", kwargs))

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.head)

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs))
        self.head = "c" * 40
        return SimpleNamespace(oid=self.head)

    def list_repo_files(self, **kwargs):
        assert kwargs["revision"] == "c" * 40
        return [".gitattributes", *[p.relative_to(self.package).as_posix()
                                  for p in self.package.rglob("*") if p.is_file()]]

    def get_space_runtime(self, repo_id):
        assert repo_id == REPO_ID
        return SimpleNamespace(hardware=HARDWARE, stage="RUNNING")

    def download(self, **kwargs):
        assert kwargs["repo_id"] == REPO_ID and kwargs["repo_type"] == "space"
        assert kwargs["revision"] in {"b" * 40, "c" * 40}
        return self.package / kwargs["filename"]


def test_upload_checks_owner_uses_only_zero_gpu_and_guards_the_parent_commit(checked):
    package, manifest, report = checked
    api = FakeApi(package)
    receipt = deploy(api, package, report, download=api.download)
    create, upload = api.calls
    assert create[1]["repo_id"] == REPO_ID and create[1]["space_hardware"] == HARDWARE
    assert create[1]["private"] is False and create[1]["exist_ok"] is False
    assert upload[1]["parent_commit"] == "b" * 40 and upload[1]["delete_patterns"] == ["*"]
    assert set(upload[1]["allow_patterns"]) == {*manifest["files"], "source_manifest.json"}
    assert receipt["upload_verified"] and not receipt["runtime_verified"]
    wrong = FakeApi(package, name="wrong-owner")
    with pytest.raises(ValueError, match="expected_hf_account"):
        deploy(wrong, package, report, download=wrong.download)
    assert wrong.calls == []


@pytest.mark.parametrize("change", ["extra", "modified", "symlink"])
def test_packages_reject_extra_sensitive_files_changes_and_symlinks_before_any_upload(checked, change):
    package, _, report = checked
    if change == "extra":
        (package / "annotation_key.json").write_text("{}")
    elif change == "modified":
        (package / "app.py").write_text("changed")
    else:
        original = package / "app.py"
        content = original.read_bytes()
        outside = package.parent / "outside.py"
        outside.write_bytes(content)
        original.unlink()
        original.symlink_to(outside)
    api = FakeApi(package)
    with pytest.raises(ValueError):
        deploy(api, package, report, download=api.download)
    assert api.calls == []


def test_real_weight_receipt_must_match_the_entire_package_and_manifest(checked):
    package, _, report = checked
    for key, value in (("passed", False), ("real_weights_loaded", False), ("source_commit", "b" * 40),
                       ("source_manifest_sha256", "bad"), ("package_files", {}),
                       ("regression_checks_passed", False), ("required_regression_ids", []), ("records", []),
                       ("reserved_evaluation_used", True), ("quality_evaluated", True)):
        with pytest.raises(ValueError, match="real_weight_smoke"):
            validate_smoke(package, {**report, key: value})


def test_exact_uploaded_commit_is_checked_and_a_different_download_is_rejected(checked, tmp_path):
    package, _, report = checked
    api = FakeApi(package)
    bad = tmp_path / "bad"
    bad.write_text("changed")
    with pytest.raises(ValueError, match="uploaded_file_content_mismatch"):
        deploy(api, package, report, download=lambda **kwargs: bad)


def test_live_check_requires_loaded_snapshot_and_exercises_both_inputs_without_jev(checked):
    package, manifest, _ = checked
    predictions = []

    class Client:
        def predict(self, *args, api_name):
            predictions.append((args, api_name))
            if api_name == "/model_info":
                return {"source_snapshot": manifest, "runtime": {"ready": True, "zero_gpu": True}}
            lamp = args[0].startswith("The reading lamp flickered")
            return "completed", [], [], {"stored": False, "api_cost": {"attempts": 0},
                "stage_reports": [{"stage": "issues", "status": "ok", "error": None}],
                "issue_assessments": [{"status": "REAL_PENDING", "department": "maintenance", "excerpt": args[0]}] if lamp else [],
                "actions": {"actions": [{"department": "maintenance", "excerpt": args[0]}] if lamp else []},
                "stage_failure": None, "status": "complete", "timings": {"total_ms": 100}}

    receipt = check_live(FakeApi(package), {}, manifest, client_factory=lambda *args, **kwargs: Client())
    assert receipt["runtime_verified"] and not receipt["quality_evaluated"]
    assert len(predictions) == len(REQUIRED_REGRESSIONS) + 1
    assert all(args[-1] is False for args, name in predictions if name == "/analyze")
    assert verify_package(package) == manifest


@pytest.mark.parametrize("case", sorted(REQUIRED_REGRESSIONS))
def test_fix_regressions_cannot_pass_with_an_extraction_failure_or_missing_lamp_actions(case):
    result = {"stage_failure": {"stage": "issues", "error": "invalid_evidence_span_id"}, "status": "partial"}
    assert regression_problems(case, "The lamp flickered.", result)
    result = {"stage_failure": None, "status": "complete", "issue_assessments": [], "actions": {"actions": []}}
    assert bool(regression_problems(case, "The lamp flickered.", result)) == (case in LAMP_REGRESSIONS)


def test_existing_space_update_waits_for_new_snapshot_and_preserves_configuration(checked, monkeypatch):
    package, manifest, report = checked
    api = FakeApi(package)
    receipt = deploy(api, package, report, resume=True, download=api.download)
    assert [name for name, _ in api.calls] == ["upload"]
    assert api.calls[0][1]["parent_commit"] == "b" * 40
    reads = []

    class Client:
        def predict(self, *args, api_name):
            if api_name == "/model_info":
                reads.append(1)
                return {"source_snapshot": {} if len(reads) == 1 else manifest,
                        "runtime": {"ready": True, "zero_gpu": True}}
            lamp = args[0].startswith("The reading lamp flickered")
            return "completed", [], [], {"stored": False, "api_cost": {"attempts": 0},
                "stage_reports": [{"stage": "issues", "status": "ok", "error": None}],
                "issue_assessments": [{"status": "REAL_PENDING", "department": "maintenance", "excerpt": args[0]}] if lamp else [],
                "actions": {"actions": [{"department": "maintenance", "excerpt": args[0]}] if lamp else []},
                "stage_failure": None, "status": "complete", "timings": {"total_ms": 100}}

    monkeypatch.setattr("scripts.publish_triage_space.time.sleep", lambda seconds: None)
    assert check_live(api, receipt, manifest, client_factory=lambda *args, **kwargs: Client())["runtime_verified"]
    assert len(reads) == 2
