"""Replay frozen development hints with plain versus schema-constrained issue generation.

No Space update, provider request, sentiment rerun, label reuse or reserved evaluation.
Raw model text is retained only for the verified synthetic development corpus.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp.triage.demo_service import (  # noqa: E402
    QWEN_REPO,
    QWEN_REVISION,
    public_stage_reports,
)
from reviewnlp.triage.evidence_generator import messages_evidence_measures  # noqa: E402
from reviewnlp.triage.generator_experiment import RATINGS  # noqa: E402
from reviewnlp.triage.qwen_generator import (  # noqa: E402
    MAX_NEW_TOKENS,
    ActionSignals,
    QwenActionGenerator,
    _dtype_for,
)
from reviewnlp.triage.span_evidence_generator import (  # noqa: E402
    CANDIDATE,
    VERSION,
    SourceSpanGenerator,
    messages_span_evidence,
    prompt_fingerprint,
)
from reviewnlp.triage.structured_span_generator import (  # noqa: E402
    CANDIDATE as FORMAT_CANDIDATE,
)
from reviewnlp.triage.structured_span_generator import (  # noqa: E402
    FORMAT_ENFORCER_VERSION,
    StructuredSpanExtractor,
    schema_fingerprint,
)
from reviewnlp.triage.structured_span_generator import VERSION as FORMAT_VERSION  # noqa: E402
from reviewnlp.triage.workflow_review import export_coverage_review  # noqa: E402

BASELINE = ROOT / "docs/experiments/triage_space_publication/e1e37e5/review"
DEVELOPMENT = ROOT / "docs/experiments/triage_generator/dev.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_inputs(*, baseline=BASELINE, development=DEVELOPMENT, root=ROOT, scope="failures"):
    """Reject changed corpora, captured outputs, hints or deployed source before model loading."""
    info = json.loads((baseline / "run.json").read_bytes())
    if (info.get("split") != "dev" or info.get("execution_complete") is not True
            or info.get("jev_requested") is not False):
        raise ValueError("completed_jev_off_development_capture_required")
    for name, expected in info["files"].items():
        if Path(name).name != name or digest(baseline / name) != expected:
            raise ValueError("captured_artifact_changed")
    corpus = json.loads(development.read_bytes())
    rows = corpus["reviews"]
    if (corpus.get("split") != "dev" or corpus.get("provenance") != "synthetic-authored"
            or hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            != info["development_cases_sha256"]):
        raise ValueError("original_authored_development_corpus_required")
    manifest = json.loads((baseline / "source_manifest.json").read_bytes())
    if manifest["runtime_snapshot"]["prompt_sha256"] != prompt_fingerprint():
        raise ValueError("frozen_v6_prompt_required")
    for name, expected in manifest["files"].items():
        if name.startswith("reviewnlp/"):
            if digest(root / "src" / name) != expected:
                raise ValueError("frozen_v6_backend_required")
    captured = [json.loads(line) for line in (baseline / "results.jsonl").read_text().splitlines()]
    if len(captured) != len(rows) or [row["id"] for row in captured] != [row["id"] for row in rows]:
        raise ValueError("complete_ordered_capture_required")
    selected = []
    for row, record in zip(rows, captured, strict=True):
        if record["api_cost"]["attempts"] or record["complaints"]["status"] != "disabled":
            raise ValueError("jev_off_hints_required")
        if scope == "all" or record["workflow_error"]:
            sentiment = record["sentiment"]
            selected.append({"id": row["id"], "text": row["text"],
                "signals": ActionSignals(sentiment["label"], sentiment["confidence"], []),
                "hosted_error": record["workflow_error"]})
    if scope not in {"failures", "all"} or not selected:
        raise ValueError("failures_or_all_scope_required")
    return selected, {"source_run_sha256": digest(baseline / "run.json"),
        "source_manifest_sha256": digest(baseline / "source_manifest.json"),
        "deployed_source_commit": info["source_commit"],
        "development_cases_sha256": info["development_cases_sha256"], "case_scope": scope}


def generators(bundle) -> dict:
    def actioner(pending):
        return QwenActionGenerator(QWEN_REPO, revision=QWEN_REVISION, loader=lambda: bundle,
            message_builder=lambda review, signals: messages_evidence_measures(review, pending),
            prompt_version=VERSION + ":measures")

    plain = QwenActionGenerator(QWEN_REPO, revision=QWEN_REVISION, loader=lambda: bundle,
        message_builder=messages_span_evidence, prompt_version=VERSION + ":issues")
    constrained = StructuredSpanExtractor(QWEN_REPO, revision=QWEN_REVISION, loader=lambda: bundle)
    result = {CANDIDATE: SourceSpanGenerator(plain, actioner),
              FORMAT_CANDIDATE: SourceSpanGenerator(constrained, actioner)}
    result[FORMAT_CANDIDATE].prompt_version = FORMAT_VERSION
    return result


def compare(rows: list[dict], candidates: dict, output: Path, *, provenance: dict, runtime: dict) -> dict:
    """Capture up to two calls per review/arm, preserve failures and export fresh blank judgments."""
    output.mkdir(parents=True, exist_ok=False)
    records, stopped = [], None
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for name, generator in candidates.items():
                record = {"candidate": name, "id": row["id"], "error": None, "workflow_error": None,
                    "status": "not_executed", "hit_token_budget": False, "actions": [],
                    "extracted_issues": [], "stages": [], "hosted_error": row["hosted_error"]}
                if stopped:
                    record["error"] = "not_executed_after_failure"
                else:
                    started = time.perf_counter()
                    try:
                        generated = generator.generate(row["text"], row["signals"])
                        record.update(workflow_error=generator.workflow_error,
                            extracted_issues=copy.deepcopy(generator.issues),
                            stages=copy.deepcopy(generator.last_stages),
                            stage_reports=public_stage_reports(generator))
                        record["hit_token_budget"] = any(stage["hit_token_budget"] for stage in record["stages"])
                        if not record["workflow_error"]:
                            # This is the software-assembled result of the unchanged strict validator.
                            record["actions"] = json.loads(generated.raw)["actions"]
                        record["status"] = "partial" if record["workflow_error"] else "complete"
                    except Exception as error:
                        record.update(status="execution_failed", error=type(error).__name__,
                                      stages=copy.deepcopy(generator.last_stages))
                        stopped = {"candidate": name, "id": row["id"], "error": type(error).__name__}
                    record["seconds"] = round(time.perf_counter() - started, 6)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(row["id"], name, record["status"], record["workflow_error"] or record["error"] or "completed", flush=True)
    shuffled = [(name, row["id"]) for row in rows for name in candidates]
    random.Random(20261006).shuffle(shuffled)
    key = [{"blind_id": f"{i:04d}", "candidate": name, "review_id": identity}
           for i, (name, identity) in enumerate(shuffled, 1)]
    (output / "annotation_key.json").write_text(json.dumps(key, indent=2) + "\n")
    (output / "upstream.json").write_text(json.dumps([
        {"id": row["id"], "signals": row["signals"].__dict__, "hosted_error": row["hosted_error"]}
        for row in rows], ensure_ascii=False, indent=2) + "\n")
    export_coverage_review(output, rows, records)
    with (output / "coverage_review.csv").open(encoding="utf-8", newline="") as handle:
        blank = list(csv.DictReader(handle))
    with (output / "coverage_review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [name for name in blank[0] if name != "notes"] + list(RATINGS) + ["notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, **dict.fromkeys(RATINGS, "")} for row in blank)
    for name in ("coverage_A.csv", "coverage_B.csv"):
        (output / name).write_bytes((output / "coverage_review.csv").read_bytes())
    summaries = {}
    for name in candidates:
        arm = [row for row in records if row["candidate"] == name]
        summaries[name] = {"planned_cases": len(rows),
            "executed_cases": sum(row["status"] != "not_executed" for row in arm),
            "workflow_errors": sum(bool(row["workflow_error"]) for row in arm),
            "execution_errors": sum(bool(row["error"]) for row in arm),
            "accepted_actions": sum(len(row["actions"]) for row in arm),
            "uncertain_issues": sum(issue["status"] == "UNCERTAIN" for row in arm for issue in row["extracted_issues"])}
    report = {"split": "dev", "population": "paired synthetic-authored development; not production accuracy",
        **provenance, "runtime": runtime, "prompt_sha256": prompt_fingerprint(),
        "schema_sha256_by_review": {row["id"]: schema_fingerprint(row["text"]) for row in rows},
        "candidates": list(candidates), "summary": summaries,
        "signals_policy": "replayed captured DistilBERT hints; sentiment not rerun; Jev not called",
        "changed_variable": "issue-stage schema-constrained decoding only; prompt, parser and measures unchanged",
        "max_calls_per_review_per_arm": 2, "max_new_tokens_per_call": MAX_NEW_TOKENS, "do_sample": False,
        "execution_complete": stopped is None, "stop_reason": stopped,
        "failures": sum(bool(row["error"] or row["workflow_error"]) for row in records),
        "jev_requested": False, "hub_writes": False, "quality_evaluated": False,
        "reserved_evaluation_used": False, "selection_frozen": False, "promoted": False,
        "script_sha256": digest(Path(__file__)),
        "format_source_sha256": digest(ROOT / "src/reviewnlp/triage/structured_span_generator.py"),
        "files": {path.name: digest(path) for path in output.iterdir()}}
    (output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", choices=("failures", "all"), default="failures")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("native", "fp32"), default="native")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    rows, provenance = frozen_inputs(scope=args.cases)
    if args.output.exists():
        raise ValueError("new_output_directory_required")
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if importlib.metadata.version("lm-format-enforcer") != FORMAT_ENFORCER_VERSION:
        raise ValueError("pinned_format_enforcer_required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_required_or_explicit_cpu")
    torch.manual_seed(42)
    if args.device == "cpu":
        torch.set_num_threads(4)
    path = snapshot_download(QWEN_REPO, revision=QWEN_REVISION, token=False, cache_dir=args.cache_dir,
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.precision == "fp32" else _dtype_for(torch, args.device)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, local_files_only=True).eval().to(args.device)
    provenance["experiment_source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime = {"real_weights_loaded": True, "model": QWEN_REPO, "model_revision": QWEN_REVISION,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "lm_format_enforcer": FORMAT_ENFORCER_VERSION, "python": platform.python_version(),
        "device": str(model.device), "dtype": str(model.dtype), "precision_policy": args.precision,
        "gpu_name": torch.cuda.get_device_name(model.device) if args.device == "cuda" else None,
        "torch_threads": torch.get_num_threads(), "attention_implementation": model.config._attn_implementation,
        "precision_matches_hosted": dtype == torch.bfloat16, "hosted_runtime_reproduced": False,
        "gpu_billing": "unknown"}
    report = compare(rows, generators((tokenizer, model)), args.output, provenance=provenance, runtime=runtime)
    print("Execution complete:", report["execution_complete"], "· new human ratings required · no promotion.")
    raise SystemExit(0 if report["execution_complete"] else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Comparison stopped:", type(error).__name__, "· existing evidence is never overwritten.")
        raise SystemExit(1) from None
