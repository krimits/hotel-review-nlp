"""Prepare human labels, then compare TF-IDF logistic regression, Qwen and a decision model.

Default action is offline sample preparation. External inference needs an explicit opt-in
and budget; no model is automatically promoted to the hotel operations Space.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.experiments import topic_benchmark as task  # noqa: E402
from reviewnlp.experiments import topic_models as adapters  # noqa: E402

PROTOCOL = ROOT / "docs/experiments/decision_topic_benchmark/PROTOCOL.md"
SOURCE = ROOT / "data/raw/hotel_reviews.csv"
ARMS = ("logistic", "qwen", "decision")


def configuration() -> dict:
    return {
        "task": task.VERSION,
        "labels": list(task.LABELS),
        "question": task.QUESTION,
        "logistic": task.LR_SETTINGS,
        "qwen_model": adapters.QWEN_MODEL,
        "qwen_revision": adapters.QWEN_REVISION,
        "qwen_scoring": "next-token A-E; probabilities conditional on these five options; no generation",
        "decision_model": adapters.DECISION_MODEL,
        "decision_endpoint": adapters.DECISIONS_URL,
        "threshold": 0.7,
        "ece_bins": 10,
        "temperature_fit": "dev-only log-temperature bounded [-3,3]",
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): task.digest(p)
            for p in (
                Path(__file__),
                ROOT / "src/reviewnlp/experiments/topic_benchmark.py",
                ROOT / "src/reviewnlp/experiments/topic_models.py",
                PROTOCOL,
            )
        },
    }


def runtime(device: str) -> dict:
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "device_requested": device,
        "local_compute_cost_usd": None,
        "gpu_billing": "unknown",
    }
    for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        result["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["code_commit"] = "unknown"
    return result


def validate_gold(sample: Path, labels_path: Path) -> tuple[list[dict], dict]:
    rows, _ = task.load_sample(sample, protocol=PROTOCOL)
    record = json.loads(labels_path.read_bytes())
    labels = record.get("labels", {})
    if (
        record.get("version") != task.VERSION
        or record.get("sample_manifest_sha256") != task.digest(sample / "sample_manifest.json")
        or set(labels) != {r["review_id"] for r in rows}
        or any(v.get("label") not in task.LABELS for v in labels.values())
    ):
        raise ValueError("verified_human_labels_required")
    return rows, record


def runtime_signature(environment: dict) -> dict:
    return {
        key: environment.get(key)
        for key in (
            "numpy",
            "scipy",
            "scikit-learn",
            "torch",
            "transformers",
            "huggingface-hub",
            "qwen_device",
            "qwen_dtype",
            "gpu",
        )
    }


def check_freeze(
    path: Path, *, sample: Path, labels_path: Path, arms: list[str], device: str
) -> dict:
    freeze = json.loads(path.read_bytes())
    if (
        freeze.get("config_sha256") != task.fingerprint(configuration())
        or freeze.get("labels_sha256") != task.digest(labels_path)
        or freeze.get("sample_sha256") != task.digest(sample / "sample_manifest.json")
        or freeze.get("arms") != arms
        or freeze.get("device") != device
    ):
        raise ValueError("freeze_inputs_or_configuration_changed")
    for name, sha in freeze["development_files"].items():
        if Path(name).name != name or task.digest(path.parent / name) != sha:
            raise ValueError("development_capture_changed")
    if set(freeze["temperatures"]) != set(arms):
        raise ValueError("complete_dev_calibration_required")
    for temperature in freeze["temperatures"].values():
        if (
            not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("invalid_frozen_temperature")
    current = runtime_signature(runtime(device))
    for key in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "huggingface-hub"):
        if current[key] != freeze["runtime_signature"][key]:
            raise ValueError("frozen_dependency_versions_changed")
    return freeze


def capture(
    rows: list[dict], predictors: dict, out: Path, *, max_cost_usd: float | None = None
) -> list[dict]:
    """Record every planned pair, including failures and budget-stopped pairs. No retries."""
    records, spent, stopped = [], 0.0, None
    order = random.Random(task.SEED)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for arm in order.sample(list(predictors), len(predictors)):
                record = {
                    "review_id": row["review_id"],
                    "review_sha256": row["review_sha256"],
                    "arm": arm,
                    "status": "not_executed",
                    "error": None,
                    "probabilities": None,
                    "milliseconds": None,
                    "cost_usd": None,
                }
                if arm == "decision" and stopped:
                    record["error"] = stopped
                else:
                    started = time.perf_counter()
                    try:
                        result = predictors[arm](row["review_text"])
                        task.probability_vector(result["probabilities"])
                        record.update(result, status="ok")
                    except Exception as error:
                        # Exception text may contain review text or credentials: retain type only.
                        record.update(status="error", error=type(error).__name__)
                    record["milliseconds"] = 1000 * (time.perf_counter() - started)
                    if arm == "decision":
                        cost = record.get("cost_usd")
                        if cost is None:
                            stopped = "stopped_after_unknown_cost_or_failed_request"
                        else:
                            spent += cost
                            if max_cost_usd is not None and spent >= max_cost_usd:
                                stopped = "reported_cost_budget_reached"
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(row["review_id"], arm, record["status"], flush=True)
    return records


def summarize(records: list[dict], gold: dict, arms: list[str], temperatures: dict) -> dict:
    report = {}
    for arm in arms:
        all_rows = [row for row in records if row["arm"] == arm]
        ok = [row for row in all_rows if row["status"] == "ok"]
        attempted = [row for row in all_rows if row["status"] != "not_executed"]
        complete = len(ok) == len(all_rows) and bool(all_rows)
        observed_costs = [row["cost_usd"] for row in attempted if row.get("cost_usd") is not None]
        full_cost = bool(attempted) and len(observed_costs) == len(attempted)
        latencies = [row["milliseconds"] for row in attempted]
        result = {
            "planned": len(all_rows),
            "successful": len(ok),
            "complete": complete,
            "errors": len(attempted) - len(ok),
            "not_executed": len(all_rows) - len(attempted),
            "human_review_due_to_failure": len(all_rows) - len(ok),
            "p95_request_ms": float(np_percentile(latencies, 95)) if latencies else None,
            "latency_scope": "batch 1; text to validated probabilities; attempts including errors; excludes load/warmup; API includes network",
            "cost_observations": len(observed_costs),
            "attempts": len(attempted),
            "known_cost_usd_lower_bound": sum(observed_costs) if observed_costs else None,
            "cost_usd_per_1000_attempts": 1000 * sum(observed_costs) / len(attempted)
            if full_cost
            else None,
            "cost_scope": "provider-reported OpenRouter USD; local compute/GPU billing unknown",
            "raw": None,
            "calibrated": None,
        }
        if complete:
            p = [task.probability_vector(row["probabilities"]) for row in all_rows]
            y = [gold[row["review_id"]]["label"] for row in all_rows]
            result["raw"] = task.classification_metrics(p, y)
            if arm in temperatures:
                result["calibrated"] = task.classification_metrics(
                    task.temperature_probabilities(p, temperatures[arm]), y
                )
        report[arm] = result
    return report


def np_percentile(values, percentile):
    import numpy as np

    return np.percentile(values, percentile)


def benchmark(
    sample: Path,
    labels_path: Path,
    out: Path,
    *,
    partition: str,
    arms: list[str],
    device: str,
    freeze_path: Path | None = None,
    allow_external: bool = False,
    max_cost_usd: float | None = None,
    key: str = "",
) -> dict:
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("unique_known_arms_required")
    if partition not in {"dev", "test"}:
        raise ValueError("dev_or_test_required")
    if "decision" in arms and (
        not allow_external
        or not key
        or max_cost_usd is None
        or not math.isfinite(max_cost_usd)
        or max_cost_usd <= 0
    ):
        raise ValueError("decision_requires_external_opt_in_key_and_positive_usd_budget")
    rows, annotation = validate_gold(sample, labels_path)
    freeze = None
    if partition == "test":
        if freeze_path is None:
            raise ValueError("test_requires_development_freeze")
        freeze = check_freeze(
            freeze_path, sample=sample, labels_path=labels_path, arms=arms, device=device
        )
    elif freeze_path is not None:
        raise ValueError("freeze_is_only_an_input_for_test")
    if out.exists():
        raise ValueError("new_capture_directory_required; do not rerun an inspected test")
    selected = [r for r in rows if r["partition"] == partition]
    if not selected:
        raise ValueError("empty_partition")
    gold = annotation["labels"]
    predictors, loads, setup_errors = {}, {}, {}
    # A training-data defect blocks all arms before any paid or model-loading work.
    logistic = task.fit_logistic(rows, gold)
    out.mkdir(parents=True, exist_ok=False)
    environment = runtime(device)
    for arm in arms:
        started = time.perf_counter()
        if arm == "logistic":
            predictors[arm] = lambda text: task.logistic_predict(logistic, text)
        elif arm == "decision":
            predictors[arm] = lambda text: adapters.decision_predict(text, key)
        else:
            try:
                qwen = adapters.QwenTopicScorer.load(device=device)
                # Warm-up excluded from request latency, and never uses a held-out review.
                qwen.predict("The hotel staff were helpful.")
                predictors[arm] = qwen.predict
                environment["qwen_device"] = str(qwen.model.device)
                environment["qwen_dtype"] = str(qwen.model.dtype)
                import torch
                environment["torch_threads"] = torch.get_num_threads()
                if device == "cuda":
                    environment["gpu"] = torch.cuda.get_device_name(0)
            except Exception as error:
                setup_errors[arm] = type(error).__name__

                def unavailable(text):
                    raise RuntimeError("qwen_setup_failed")

                predictors[arm] = unavailable
        loads[arm] = time.perf_counter() - started
    if freeze and runtime_signature(environment) != freeze["runtime_signature"]:
        raise ValueError("frozen_hardware_or_precision_changed")
    records = capture(selected, predictors, out, max_cost_usd=max_cost_usd)
    temperatures = freeze["temperatures"] if freeze else {}
    if partition == "dev":
        for arm in arms:
            candidate = [r for r in records if r["arm"] == arm]
            if all(r["status"] == "ok" for r in candidate):
                temperatures[arm] = task.fit_temperature(
                    [task.probability_vector(r["probabilities"]) for r in candidate],
                    [gold[r["review_id"]]["label"] for r in candidate],
                )
    summaries = summarize(records, gold, arms, temperatures)
    report = {
        "partition": partition,
        "sample_sha256": task.digest(sample / "sample_manifest.json"),
        "labels_sha256": task.digest(labels_path),
        "annotation_independence": annotation["independence"],
        "configuration": configuration(),
        "runtime": environment,
        "arms": arms,
        "setup_seconds": loads,
        "setup_errors": setup_errors,
        "temperatures": temperatures,
        "execution_complete": all(s["complete"] for s in summaries.values()),
        "summary": summaries,
        "freeze_sha256": task.digest(freeze_path) if freeze_path else None,
        "population": "random reviews from two NYC hotels, 2008-2009; not unseen hotels or current traffic",
        "test_policy": "one frozen evaluation; no prompt/threshold/model selection on test",
        "threshold_policy": "0.7 illustrative preset on max class probability; not a validated business policy",
        "development_calibrated_metrics": "in-sample fit diagnostics, not held-out estimates"
        if partition == "dev"
        else None,
        "budget_policy": "no retry; stop before next call once reported cost reaches cap or cost is unknown; one request can overshoot",
        "max_reported_cost_usd": max_cost_usd,
        "promoted": False,
        "space_updated": False,
        "files": {"predictions.jsonl": task.digest(out / "predictions.jsonl")},
    }
    task.write_json(out / "run.json", report)
    if partition == "dev" and report["execution_complete"]:
        task.write_json(
            out / "freeze.json",
            {
                "config_sha256": task.fingerprint(configuration()),
                "sample_sha256": report["sample_sha256"],
                "labels_sha256": report["labels_sha256"],
                "arms": arms,
                "device": device,
                "temperatures": temperatures,
                "runtime_signature": runtime_signature(environment),
                "development_files": {
                    name: task.digest(out / name) for name in ("predictions.jsonl", "run.json")
                },
            },
        )
    # A queue is a file for this experiment, not a database or automated hotel action.
    with (out / "human_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(
            handle,
            fieldnames=["review_id", "arm", "reason", "predicted", "confidence"],
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            if record["status"] != "ok":
                writer.writerow(
                    {
                        "review_id": record["review_id"],
                        "arm": record["arm"],
                        "reason": record["status"],
                    }
                )
                continue
            p = task.probability_vector(record["probabilities"])
            if record["arm"] in temperatures:
                p = task.temperature_probabilities([p], temperatures[record["arm"]])[0]
            if p.max() < 0.7:
                writer.writerow(
                    {
                        "review_id": record["review_id"],
                        "arm": record["arm"],
                        "reason": "max_probability_below_0.7",
                        "predicted": task.LABELS[int(p.argmax())],
                        "confidence": float(p.max()),
                    }
                )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--output", type=Path, default=ROOT / "runs/decision_topics/sample")
    finish = sub.add_parser("labels")
    finish.add_argument("--sample", type=Path, required=True)
    finish.add_argument("--a", type=Path, required=True)
    finish.add_argument("--b", type=Path, required=True)
    finish.add_argument("--final", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    finish.add_argument("--independent-reviewers", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--sample", type=Path, required=True)
    run.add_argument("--labels", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--partition", choices=("dev", "test"), default="dev")
    run.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--freeze", type=Path)
    run.add_argument("--allow-external-api", action="store_true")
    run.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args()
    if args.command == "sample":
        report = task.prepare_sample(SOURCE, args.output, protocol=PROTOCOL)
    elif args.command == "labels":
        report = task.finish_labels(
            args.sample,
            args.a,
            args.b,
            args.final,
            args.output,
            protocol=PROTOCOL,
            independent_reviewers=args.independent_reviewers,
        )
    else:
        report = benchmark(
            args.sample,
            args.labels,
            args.output,
            partition=args.partition,
            arms=args.arms,
            device=args.device,
            freeze_path=args.freeze,
            allow_external=args.allow_external_api,
            max_cost_usd=args.max_cost_usd,
            key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("sample_size", "partition", "execution_complete", "summary")
                if k in report
            },
            indent=2,
        )
    )
    return 0 if report.get("execution_complete", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
