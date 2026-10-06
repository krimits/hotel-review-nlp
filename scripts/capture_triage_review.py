"""Capture the deployed source-span workflow on authored development cases for two human reviewers.

No model download, Hub write, reserved cases, provider call by default, or automatic promotion.
HF_TOKEN is read from the environment by the CLI, never accepted as a command-line argument.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp.triage.generator_experiment import RATINGS  # noqa: E402
from reviewnlp.triage.workflow_review import export_coverage_review  # noqa: E402

SPACE_ID = "krimits/hotel-triage-demo"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture(client, rows: list[dict], output: Path, *, source_commit: str, use_jev=False,
            timeout_s=180, seed=20261006) -> dict:
    """One request per authored case; runtime failure stops the run and preserves unexecuted cases."""
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("full_source_commit_required")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unique_development_cases_required")
    if timeout_s <= 0 or timeout_s > 600:
        raise ValueError("timeout_requires_1_to_600_seconds")
    # No empty folder is created until the active snapshot has been checked.
    info = client.predict(api_name="/model_info")
    manifest = info.get("source_snapshot") or {}
    if (manifest.get("source_commit") != source_commit or info.get("prompt_version") != "actions-v6-source-spans"
            or manifest.get("selection_status") != "experimental_not_selected_not_promoted"
            or not info.get("runtime", {}).get("ready")):
        raise ValueError("matching_ready_source_span_snapshot_required")
    if use_jev and not info.get("jev_enabled"):
        raise ValueError("jev_not_enabled_on_server")
    output.mkdir(parents=True, exist_ok=False)
    candidate = "G-source-spans" + ("+Jev" if use_jev else "")
    records, stopped = [], None
    for row in rows:
        record = {"candidate": candidate, "id": row["id"], "actions": [], "extracted_issues": [],
                  "hit_token_budget": False, "status": "not_executed", "error": None,
                  "workflow_error": None, "api_cost": None, "total_ms": None}
        if stopped:
            record["error"] = "not_executed_after_failure"
        else:
            job = None
            try:
                job = client.submit(row["text"], bool(use_jev), api_name="/analyze")
                _, _, _, result = job.result(timeout=timeout_s)
                if result.get("deployment", {}).get("source_snapshot") != manifest:
                    raise ValueError("snapshot_changed")
                if result.get("stored") is not False:
                    raise ValueError("unexpected_review_storage")
                if not use_jev and (result["api_cost"]["attempts"] or result["complaints"]["status"] != "disabled"):
                    raise ValueError("jev_opt_out_contract_failed")
                if use_jev and (result["complaints"]["status"] == "disabled" or not result["api_cost"]["attempts"]):
                    raise ValueError("jev_opt_in_not_exercised")
                if any(issue["excerpt"] not in row["text"] for issue in result["issue_assessments"]):
                    raise ValueError("literal_evidence_contract_failed")
                stages = result["stage_reports"]
                failure = result["stage_failure"]
                record.update({"status": result["status"], "actions": result["actions"]["actions"],
                    "extracted_issues": result["issue_assessments"], "stage_reports": stages,
                    "workflow_error": (failure["stage"] + ":" + failure["error"]) if failure else None,
                    "hit_token_budget": any(stage.get("error") == "hit_token_budget" for stage in stages),
                    "api_cost": result["api_cost"], "total_ms": result["timings"]["total_ms"],
                    "sentiment": result["sentiment"], "complaints": result["complaints"]})
                if failure and failure["stage"] in {"qwen_runtime", "jev"}:
                    stopped = {"case": row["id"], "error": record["workflow_error"]}
            except Exception as error:
                if job is not None:
                    try:
                        job.cancel()
                    except Exception:
                        pass
                allowed = {"snapshot_changed", "unexpected_review_storage", "jev_opt_out_contract_failed",
                           "literal_evidence_contract_failed", "jev_opt_in_not_exercised"}
                code = (str(error) if isinstance(error, ValueError) and str(error) in allowed else
                        "client_timeout" if isinstance(error, TimeoutError) else "client_transport_or_contract_failed")
                record["status"], record["error"] = "execution_failed", code
                stopped = {"case": row["id"], "error": code}
        records.append(record)
        print(row["id"], record["status"], record["workflow_error"] or record["error"] or "completed", flush=True)
    # Re-read configuration after requests; a changed snapshot invalidates the run.
    if not stopped:
        try:
            if client.predict(api_name="/model_info").get("source_snapshot") != manifest:
                stopped = {"case": "postflight", "error": "snapshot_changed"}
        except Exception:
            stopped = {"case": "postflight", "error": "client_transport_failed"}
    order = list(rows)
    random.Random(seed).shuffle(order)
    key = [{"blind_id": f"{index:04d}", "candidate": candidate, "review_id": row["id"]}
           for index, row in enumerate(order, 1)]
    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write("source_manifest.json", manifest)
    write("model_info.json", info)
    write("annotation_key.json", key)
    (output / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    export_coverage_review(output, rows, records)
    with (output / "coverage_review.csv").open(encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    with (output / "coverage_review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [name for name in original[0] if name != "notes"] + list(RATINGS) + ["notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, **dict.fromkeys(RATINGS, "")} for row in original)
    for name in ("coverage_A.csv", "coverage_B.csv"):
        (output / name).write_bytes((output / "coverage_review.csv").read_bytes())
    report = {"split": "dev", "population": "synthetic-authored development; not an independent real-review pilot",
        "source_commit": source_commit, "shuffle_seed": seed,
        "capture_script_sha256": digest(Path(__file__).read_bytes()),
        "development_cases_sha256": digest(json.dumps(rows, sort_keys=True).encode()),
        "source_manifest_sha256": digest((output / "source_manifest.json").read_bytes()),
        "candidates": [candidate], "jev_requested": bool(use_jev), "selection_frozen": False,
        "quality_evaluated": False, "reserved_evaluation_used": False, "execution_complete": stopped is None,
        "stop_reason": stopped, "planned_cases": len(rows),
        "executed_cases": sum(row["status"] != "not_executed" for row in records),
        "failures": sum(bool(row["workflow_error"] or row["error"]) for row in records),
        "files": {path.name: digest(path.read_bytes()) for path in sorted(output.iterdir())}}
    write("run.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jev", action="store_true", help="Opt in to provider calls using the Space's server-side key")
    args = parser.parse_args()
    from gradio_client import Client
    client = Client(SPACE_ID, hf_token=os.environ.get("HF_TOKEN") or False,
                    httpx_kwargs={"timeout": 45.0}, verbose=False)
    data = json.loads((ROOT / "docs/experiments/triage_generator/dev.json").read_bytes())
    if data.get("split") != "dev":
        raise ValueError("authored_development_only")
    report = capture(client, data["reviews"], args.output, source_commit=args.source_commit, use_jev=args.jev)
    print("Capture complete:", report["execution_complete"], "· human ratings still required · no promotion.")
    raise SystemExit(0 if report["execution_complete"] else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Capture stopped:", type(error).__name__, "· existing bundles are never overwritten.")
        raise SystemExit(1) from None
