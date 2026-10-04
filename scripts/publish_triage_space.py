"""Publish the checked experimental package to krimits/hotel-triage-demo on ZeroGPU.

Authentication uses HF_TOKEN or an existing local HF login. No token is accepted as a CLI argument.
No paid GPU fallback, model upload, database, reserved evaluation or production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_triage_space import verify_package  # noqa: E402

REPO_ID = "krimits/hotel-triage-demo"
HARDWARE = "zero-a10g"  # HF API name for ZeroGPU; never a dedicated paid A10G.


def validate_smoke(package: Path, report: dict) -> dict:
    manifest = verify_package(package)
    if (report.get("kind") != "real_weight_functional_smoke" or not report.get("passed")
            or not report.get("real_weights_loaded") or report.get("problems")
            or report.get("quality_evaluated") is not False or report.get("reserved_evaluation_used") is not False
            or report.get("source_commit") != manifest["source_commit"]
            or report.get("package_files") != manifest["files"]
            or report.get("source_manifest_sha256") != hashlib.sha256((package / "source_manifest.json").read_bytes()).hexdigest()):
        raise ValueError("matching_successful_real_weight_smoke_required")
    return manifest


def deploy(api, package: Path, report: dict, *, resume=False, download=hf_hub_download) -> dict:
    manifest = validate_smoke(package, report)
    if api.whoami()["name"] != "krimits":
        raise ValueError("expected_hf_account_krimits")
    parent = None
    if resume:
        info = api.repo_info(repo_id=REPO_ID, repo_type="space")
        parent = info.sha
        old_path = download(repo_id=REPO_ID, repo_type="space", revision=parent,
                            filename="source_manifest.json", token=api.token)
        previous = json.loads(Path(old_path).read_text(encoding="utf-8"))
        if previous.get("selection_status") != "experimental_not_selected_not_promoted":
            raise ValueError("refusing_to_replace_an_unrelated_space")
        if api.get_space_runtime(REPO_ID).hardware != HARDWARE:
            raise ValueError("existing_space_must_already_use_zero_gpu")
    else:
        api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio",
                        private=False, space_hardware=HARDWARE, exist_ok=False)
        parent = api.repo_info(repo_id=REPO_ID, repo_type="space").sha
    expected = {*manifest["files"], "source_manifest.json"}
    uploaded = api.upload_folder(repo_id=REPO_ID, repo_type="space", folder_path=str(package),
                                 allow_patterns=sorted(expected), delete_patterns=["*"], parent_commit=parent,
                                 commit_message="Publish experimental G snapshot; no production promotion")
    commit = uploaded.oid
    actual = set(api.list_repo_files(repo_id=REPO_ID, repo_type="space", revision=commit))
    if actual - {".gitattributes"} != expected:
        raise ValueError("uploaded_file_allowlist_mismatch")
    for name in sorted(expected):
        path = download(repo_id=REPO_ID, repo_type="space", revision=commit, filename=name, token=api.token)
        if Path(path).read_bytes() != (package / name).read_bytes():
            raise ValueError("uploaded_file_content_mismatch: " + name)
    if api.repo_info(repo_id=REPO_ID, repo_type="space").sha != commit:
        raise ValueError("space_head_changed_after_upload")
    return {"space_id": REPO_ID, "url": "https://huggingface.co/spaces/" + REPO_ID,
            "space_commit": commit, "source_commit": manifest["source_commit"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "selection_status": manifest["selection_status"], "hardware_requested": HARDWARE,
            "upload_verified": True, "runtime_verified": False}


def check_live(api, receipt: dict, manifest: dict, *, timeout=1500, client_factory=None) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runtime = api.get_space_runtime(REPO_ID)
        stage = getattr(runtime.stage, "value", runtime.stage)
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED"}:
            raise RuntimeError("space_not_running: " + stage)
        if stage == "RUNNING":
            break
        time.sleep(5)
    else:
        raise TimeoutError("space_startup_timeout")
    if client_factory is None:
        from gradio_client import Client
        client_factory = Client
    client = client_factory(REPO_ID, hf_token=api.token)
    info = client.predict(api_name="/model_info")
    if (info["source_snapshot"] != manifest or not info["runtime"]["ready"]
            or not info["runtime"]["zero_gpu"] or runtime.hardware != HARDWARE):
        raise ValueError("live_runtime_snapshot_or_hardware_mismatch")
    checked = []
    for label, text in (("praise", "The lobby tea was delicious and our suite was comfortable."),
                        ("complaint", "The reading lamp flickered all evening and nobody fixed it.")):
        summary, _, _, result = client.predict(text, False, api_name="/analyze")
        if (result["stored"] or result["api_cost"]["attempts"]
                or not any(item["stage"] == "issues" for item in result["stage_reports"])
                or any(item["error"] in {"generation_failed", "gpu_unavailable_or_timeout"} for item in result["stage_reports"])
                or (result["stage_failure"] and "Αποτυχία σταδίου" not in summary)):
            raise ValueError("live_functional_check_failed: " + label)
        checked.append({"case": label, "status": result["status"], "stage_failure": result["stage_failure"],
                        "total_ms": result["timings"]["total_ms"], "stored": False})
    return {**receipt, "runtime_verified": True, "live_model_info": info, "functional_checks": checked,
            "quality_evaluated": False, "manual_mobile_and_pilot_checks": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    manifest = validate_smoke(args.package, report)
    if args.dry_run:
        print("Package and real-weight check verified. No Hub writes.")
        return
    receipt = deploy(HfApi(), args.package, report, resume=args.resume)
    # Save the verified upload even if startup subsequently fails.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Upload verified:", receipt["url"], receipt["space_commit"], flush=True)
    receipt = check_live(HfApi(), receipt, manifest)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Live functional check passed:", receipt["url"], "— experimental, quality unvalidated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Exception messages may contain provider URLs, request content or credentials.
        print("Publication/check stopped:", type(error).__name__, "— review the last completed step and HF build status.")
        raise SystemExit(1) from None
