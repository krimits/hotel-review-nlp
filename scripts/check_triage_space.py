"""Run an allowlisted Space package with real pinned weights. No Hub writes or Jev calls.

This is a functional check, not a quality evaluation or a reserved evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_triage_space import verify_package  # noqa: E402

EDGE_CASES = [
    {"id": "smoke-positive", "text": "The lobby tea was delicious and our suite was comfortable."},
    {"id": "smoke-negative", "text": "The reading lamp flickered all evening and nobody fixed it."},
    {"id": "smoke-hypothetical", "text": "The lift worked every time. It would be awkward if it broke."},
    {"id": "smoke-resolved", "text": "The reading lamp failed. Staff replaced it immediately and it worked perfectly."},
    {"id": "smoke-praise-inversion", "text": "We loved the breakfast and the welcome drink. The clean room was comfortable."},
    {"id": "smoke-suggestion", "text": "The desserts were lovely. A vegan dessert would be a welcome addition."},
]
DEV_FAILURE_IDS = {"dev-07", "dev-09", "dev-12", "dev-13", "dev-14", "dev-21", "dev-22", "dev-23"}


def check(package: Path, *, include_dev_failures=True) -> dict:
    manifest = verify_package(package)
    # Bytecode is not part of the upload allowlist, and must not pollute the package during its check.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package.resolve()))
    spec = importlib.util.spec_from_file_location("triage_deployment_app", package / "app.py")
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
    runtime = app.initialize()
    information = app.model_info()
    if not runtime.information()["ready"]:
        raise RuntimeError("real_model_startup_failed: " + runtime.startup_failure["stage"])
    rows = list(EDGE_CASES)
    if include_dev_failures:
        development = json.loads((ROOT / "docs/experiments/triage_generator/dev.json").read_text(encoding="utf-8"))
        selected = [item for item in development["reviews"] if item["id"] in DEV_FAILURE_IDS]
        if {item["id"] for item in selected} != DEV_FAILURE_IDS:
            raise ValueError("development_smoke_cases_missing")
        rows += selected
    records, problems = [], []
    started = time.perf_counter()
    try:
        for row in rows:
            summary, _, _, result = app.analyze(row["text"], False)
            stage_reports = result["stage_reports"]
            if result["stored"] or result["api_cost"]["attempts"] or result["complaints"]["status"] != "disabled":
                problems.append(row["id"] + ": opt_out_or_storage_contract")
            if not any(item["stage"] == "issues" for item in stage_reports):
                problems.append(row["id"] + ": issue_generation_not_exercised")
            if any(item["error"] in {"generation_failed", "gpu_unavailable_or_timeout"} for item in stage_reports):
                problems.append(row["id"] + ": model_execution_failed")
            if result["stage_failure"] and "Αποτυχία σταδίου" not in summary:
                problems.append(row["id"] + ": invisible_stage_failure")
            if any(item["excerpt"] not in row["text"] for item in result["issue_assessments"]):
                problems.append(row["id"] + ": literal_quote_contract")
            records.append({"id": row["id"], "status": result["status"], "stage_reports": stage_reports,
                            "issues": len(result["issue_assessments"]), "actions": len(result["actions"]["actions"]),
                            "total_ms": result["timings"]["total_ms"], "needs_review": result["routing"]["needs_review"]})
            print(row["id"], result["status"], result["stage_failure"] or "stages completed", flush=True)
    finally:
        app.demo.close()
    verify_package(package)
    return {"kind": "real_weight_functional_smoke", "quality_evaluated": False,
            "reserved_evaluation_used": False, "real_weights_loaded": True,
            "source_commit": manifest["source_commit"], "package_files": manifest["files"],
            "source_manifest_sha256": hashlib.sha256((package / "source_manifest.json").read_bytes()).hexdigest(),
            "runtime": information["runtime"], "records": records, "problems": problems,
            "passed": not problems, "seconds": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Functional check:", "passed" if report["passed"] else "failed", "— no quality claim.")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
