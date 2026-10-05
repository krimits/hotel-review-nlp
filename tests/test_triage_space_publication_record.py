"""Offline integrity of the archived Colab receipt and independent live verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/experiments/triage_space_publication/a79717c"


def read(name):
    return json.loads((RECORD / name).read_bytes())


def test_recorded_publication_binds_the_original_colab_evidence_to_the_verified_live_snapshot():
    archive = read("record_manifest.json")
    assert set(archive["files"]) == {
        "source_manifest.json", "colab_smoke.json", "published.json", "verification.json",
        "smoke_execution.txt", "publish_execution.txt",
    }
    for name, expected in archive["files"].items():
        assert hashlib.sha256((RECORD / name).read_bytes()).hexdigest() == expected
    manifest, smoke, receipt, verification = (
        read("source_manifest.json"), read("colab_smoke.json"), read("published.json"), read("verification.json")
    )
    digest = hashlib.sha256((RECORD / "source_manifest.json").read_bytes()).hexdigest()
    for report in (smoke, receipt, verification):
        assert report["source_commit"] == manifest["source_commit"] == archive["source_commit"]
        assert report["source_manifest_sha256"] == digest
        assert report["quality_evaluated"] is False
    assert receipt["space_commit"] == verification["space_commit"] == archive["space_commit"]
    assert receipt["upload_verified"] and receipt["runtime_verified"]
    assert receipt["live_model_info"]["source_snapshot"] == manifest
    assert receipt["selection_status"] == verification["selection_status"] == "experimental_not_selected_not_promoted"
    assert smoke["passed"] and smoke["real_weights_loaded"] and smoke["problems"] == []
    assert smoke["package_files"] == manifest["files"]
    assert smoke["reserved_evaluation_used"] is verification["reserved_evaluation_used"] is False
    assert len(smoke["records"]) == 14
    assert verification["passed"] and verification["loaded_snapshot_matches"]
    assert verification["public"] and verification["runtime_stage"] == "RUNNING"
    assert verification["hardware"] == receipt["hardware_requested"] == "zero-a10g"
    assert verification["runtime_commit"] == receipt["space_commit"]
    assert verification["verified_files_sha256"] == {**manifest["files"], "source_manifest.json": digest}
    assert set(verification["hub_files"]) - {".gitattributes"} == set(verification["verified_files_sha256"])
    assert verification["live_runtime"]["ready"] and verification["live_runtime"]["zero_gpu"]
    assert {item["case"] for item in verification["functional_checks"]} == {"praise", "complaint", "known-dev-07"}
    for case in verification["functional_checks"]:
        assert case["stored"] is False and case["jev_attempts"] == 0
        assert any(item["stage"] == "issues" for item in case["stage_reports"])
        assert case["stage_failure"] is None or case["stage_failure_visible"]
        assert not any(item["error"] in {"generation_failed", "gpu_unavailable_or_timeout"} for item in case["stage_reports"])
    for pending in ("mobile_layout_checked", "cold_start_after_sleep_checked", "jev_opt_in_checked",
                    "gpu_quota_timeout_checked", "independent_human_pilot_completed"):
        assert verification[pending] is False
