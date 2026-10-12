"""Dev-only Jev extension against the immutable Microsoft primary-topic capture.

No model downloads, GPU work, test predictions, provider retries or automatic promotion.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from reviewnlp.experiments import topic_benchmark as task  # noqa: E402
from reviewnlp.experiments import topic_models as adapters  # noqa: E402
from scripts import benchmark_decision_topics as original  # noqa: E402

JEV_MODEL = "typesafe/jev-1.13-20260917"
REFERENCE = ROOT / "docs/experiments/decision_topic_benchmark/results/comparison_dev_20261012"
LOCK = ROOT / "docs/experiments/decision_topic_benchmark/annotation_record.json"
PROTOCOL = ROOT / "docs/experiments/decision_topic_benchmark/JEV_EXTENSION.md"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261012


def close(left, right):
    """Compare a recomputed report with a recorded report, allowing numeric round-off only."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(v, right[k]) for k, v in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and not isinstance(left, bool):
        return isinstance(right, (float, int)) and math.isclose(left, right, abs_tol=1e-10)
    return left == right


def checked_reference(dev, gold, sample_sha, labels_sha, *, reference=REFERENCE):
    report = json.loads((reference / "run.json").read_bytes())
    freeze = json.loads((reference / "freeze.json").read_bytes())
    verification = json.loads((reference / "verification.json").read_bytes())
    for name, sha in verification["archived_sha256"].items():
        if Path(name).name != name or task.digest(reference / name) != sha:
            raise ValueError("reference_artifact_changed")
    for name, sha in freeze["development_files"].items():
        if Path(name).name != name or task.digest(reference / name) != sha:
            raise ValueError("reference_freeze_changed")
    if (
        report["partition"] != "dev"
        or not report["execution_complete"]
        or report["sample_sha256"] != sample_sha
        or report["labels_sha256"] != labels_sha
        or report["configuration"]["question"] != task.QUESTION
        or report["configuration"]["decision_model"] != adapters.DECISION_MODEL
        or report["configuration"]["threshold"] != 0.7
    ):
        raise ValueError("reference_task_or_inputs_differ")
    records = [
        json.loads(line) for line in (reference / "predictions.jsonl").read_text().splitlines()
    ]
    records = [row for row in records if row["arm"] == "decision"]
    expected = {row["review_id"]: row["review_sha256"] for row in dev}
    if (
        len(records) != len(dev)
        or len({r["review_id"] for r in records}) != len(records)
        or {r["review_id"]: r["review_sha256"] for r in records} != expected
        or any(r["status"] != "ok" or r["model"] != adapters.DECISION_MODEL for r in records)
    ):
        raise ValueError("complete_paired_microsoft_reference_required")
    temperature = report["temperatures"]["decision"]
    summary = original.summarize(records, gold, ["decision"], {"decision": temperature})["decision"]
    if not close(summary, report["summary"]["decision"]):
        raise ValueError("microsoft_reference_metrics_differ")
    return (
        records,
        temperature,
        {
            "code_commit": report["runtime"]["code_commit"],
            "run_sha256": task.digest(reference / "run.json"),
            "predictions_sha256": task.digest(reference / "predictions.jsonl"),
            "freeze_sha256": task.digest(reference / "freeze.json"),
            "request_time_policy": "historical API path; not simultaneous with this Jev run",
        },
    )


def checked_inputs(inputs: Path):
    sample, labels_path = inputs / "sample", inputs / "labels_rechecked.json"
    lock = json.loads(LOCK.read_bytes())
    if (
        task.digest(labels_path) != lock["labels_sha256"]
        or task.digest(sample / "sample_manifest.json") != lock["sample_manifest_sha256"]
    ):
        raise ValueError("locked_human_inputs_required")
    rows, annotation = original.validate_gold(sample, labels_path)
    dev = [row for row in rows if row["partition"] == "dev"]
    if len(dev) != 100:
        raise ValueError("exact_100_dev_reviews_required")
    gold = annotation["labels"]
    records, temperature, provenance = checked_reference(
        dev, gold, lock["sample_manifest_sha256"], lock["labels_sha256"]
    )
    return dev, gold, records, temperature, provenance, annotation


def jev_predict(text: str, key: str, *, transport=None):
    # One five-way choice, with exactly the original rubric and no independent binary scores.
    return adapters.decision_predict(text, key, model=JEV_MODEL, transport=transport)


def capture_jev(dev, output: Path, predictor, *, max_cost_usd: float):
    records, spent, stopped = [], 0.0, None
    with (output / "jev_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in dev:
            record = {
                "review_id": row["review_id"],
                "review_sha256": row["review_sha256"],
                "arm": "jev",
                "status": "not_executed",
                "error": stopped,
                "error_reason": None,
                "probabilities": None,
                "milliseconds": None,
                "cost_usd": None,
            }
            if stopped is None:
                started = time.perf_counter()
                try:
                    result = predictor(row["review_text"])
                    task.probability_vector(result["probabilities"])
                    if result["model"] != JEV_MODEL:
                        raise ValueError("jev_model_revision_changed")
                    cost = result.get("cost_usd")
                    if cost is not None and (
                        isinstance(cost, bool)
                        or not isinstance(cost, (int, float))
                        or not math.isfinite(cost)
                        or cost < 0
                    ):
                        raise ValueError("invalid_provider_cost")
                    record.update(result, status="ok", error=None)
                except Exception as error:
                    # No raw exception, response body, review or secret in diagnostics.
                    record.update(status="error", error=type(error).__name__)
                    allowed = {
                        "decision_object_required",
                        "decision_model_revision_changed",
                        "choice_answer_required",
                        "choice_probability_mismatch",
                        "full_five_class_distribution_required",
                        "invalid_probability",
                        "invalid_probability_distribution",
                        "invalid_provider_cost",
                        "usage_object_required",
                        "invalid_provider_token_usage",
                        "invalid_provider_confidence",
                        "jev_model_revision_changed",
                    }
                    if isinstance(error, ValueError):
                        reason = str(error)
                        if reason in allowed or re.fullmatch(r"decision_http_[0-9]{3}", reason):
                            record["error_reason"] = reason
                record["milliseconds"] = 1000 * (time.perf_counter() - started)
                cost = record.get("cost_usd")
                if cost is None:
                    stopped = "stopped_after_unknown_cost_or_failed_request"
                else:
                    spent += cost
                    if spent >= max_cost_usd:
                        stopped = "reported_cost_budget_reached"
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(row["review_id"], "jev", record["status"], flush=True)
    return records


def macro_f1_from_indices(gold, predicted):
    confusion = np.bincount(5 * gold + predicted, minlength=25).reshape(5, 5)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    scores = np.divide(
        2 * confusion.diagonal(), denominator, out=np.zeros(5), where=denominator != 0
    )
    return float(scores.mean())


def paired_comparison(jev, microsoft, gold):
    if not all(row["status"] == "ok" for row in jev):
        return None
    by_id = {row["review_id"]: row for row in microsoft}
    y = np.array([task.LABELS.index(gold[row["review_id"]]["label"]) for row in jev])
    a = np.array([int(task.probability_vector(row["probabilities"]).argmax()) for row in jev])
    b = np.array(
        [
            int(task.probability_vector(by_id[row["review_id"]]["probabilities"]).argmax())
            for row in jev
        ]
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = []
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = rng.integers(0, len(y), size=len(y))
        differences.append(
            macro_f1_from_indices(y[selected], a[selected])
            - macro_f1_from_indices(y[selected], b[selected])
        )
    return {
        "n": len(y),
        "delta_macro_f1_jev_minus_microsoft": macro_f1_from_indices(y, a)
        - macro_f1_from_indices(y, b),
        "delta_macro_f1_percentile_ci_95": np.percentile(differences, [2.5, 97.5]).tolist(),
        "jev_only_correct": int(((a == y) & (b != y)).sum()),
        "microsoft_only_correct": int(((a != y) & (b == y)).sum()),
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired whole dev review; five classes; absent-class F1=0",
            "scope": "dev diagnostic; sparse classes and unverified reviewer independence limit inference",
        },
        "automatic_winner": None,
    }


def write_queue(output, records, temperatures):
    with (output / "human_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_id", "arm", "reason", "predicted", "confidence"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in records:
            if row["status"] != "ok":
                writer.writerow(
                    {"review_id": row["review_id"], "arm": row["arm"], "reason": row["status"]}
                )
                continue
            p = task.probability_vector(row["probabilities"])
            if row["arm"] in temperatures:
                p = task.temperature_probabilities([p], temperatures[row["arm"]])[0]
            if p.max() < 0.7:
                writer.writerow(
                    {
                        "review_id": row["review_id"],
                        "arm": row["arm"],
                        "reason": "max_probability_below_0.7",
                        "predicted": task.LABELS[int(p.argmax())],
                        "confidence": float(p.max()),
                    }
                )


def benchmark(inputs, output, *, allow_external=False, key="", max_cost_usd=None):
    if (
        not allow_external
        or not key
        or isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, (float, int))
        or not math.isfinite(max_cost_usd)
        or max_cost_usd <= 0
    ):
        raise ValueError("jev_requires_external_opt_in_key_and_positive_usd_budget")
    if output.exists():
        raise ValueError("new_capture_directory_required")
    dev, gold, microsoft, ms_temperature, reference, annotation = checked_inputs(inputs)
    output.mkdir(parents=True, exist_ok=False)
    jev = capture_jev(dev, output, lambda text: jev_predict(text, key), max_cost_usd=max_cost_usd)
    temperatures = {"decision": ms_temperature}
    if all(row["status"] == "ok" for row in jev):
        temperatures["jev"] = task.fit_temperature(
            [task.probability_vector(row["probabilities"]) for row in jev],
            [gold[row["review_id"]]["label"] for row in jev],
        )
    task.write_json(output / "microsoft_reference_predictions.json", microsoft)
    summary = original.summarize(jev + microsoft, gold, ["jev", "decision"], temperatures)
    source_paths = [
        Path(__file__),
        PROTOCOL,
        ROOT / "src/reviewnlp/experiments/topic_models.py",
        ROOT / "src/reviewnlp/experiments/topic_benchmark.py",
        ROOT / "scripts/benchmark_decision_topics.py",
    ]
    report = {
        "partition": "dev",
        "planned_jev_requests": len(dev),
        "execution_complete": summary["jev"]["complete"],
        "sample_sha256": annotation["sample_manifest_sha256"],
        "labels_sha256": task.digest(inputs / "labels_rechecked.json"),
        "annotation_independence": annotation["independence"],
        "configuration": {
            "jev_model": JEV_MODEL,
            "endpoint": adapters.DECISIONS_URL,
            "question": task.QUESTION,
            "threshold": 0.7,
            "source_sha256": {p.relative_to(ROOT).as_posix(): task.digest(p) for p in source_paths},
        },
        "runtime": {
            "python": sys.version,
            "code_commit": original.runtime("cpu")["code_commit"],
            "executable": sys.executable,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "scikit-learn"]
            },
            "local_compute_cost_usd": None,
        },
        "reference": reference,
        "temperatures": temperatures,
        "summary": summary,
        "paired_comparison": paired_comparison(jev, microsoft, gold),
        "calibration_scope": "each temperature fitted and measured on the same dev set; not held-out calibration evidence",
        "latency_scope": "serial batch 1 API path including network; historical Microsoft and new Jev timings are not simultaneous",
        "budget_policy": "no retry; stop after unknown cost/failure or observed cap; one request can overshoot",
        "max_reported_cost_usd": max_cost_usd,
        "test_predictions_made": False,
        "promoted": False,
        "space_updated": False,
    }
    write_queue(output, jev + microsoft, temperatures)
    report["files"] = {p.name: task.digest(p) for p in sorted(output.iterdir())}
    task.write_json(output / "run.json", report)
    if report["execution_complete"]:
        task.write_json(
            output / "freeze.json",
            {
                "scope": "Jev dev extension only; not a final four-arm test freeze",
                "configuration_sha256": task.fingerprint(report["configuration"]),
                "sample_sha256": report["sample_sha256"],
                "labels_sha256": report["labels_sha256"],
                "temperatures": temperatures,
                "reference": reference,
                "development_files": {p.name: task.digest(p) for p in sorted(output.iterdir())},
            },
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "run"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-external-api", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args()
    if args.command == "check":
        dev, _, _, _, reference, _ = checked_inputs(args.inputs)
        print(
            json.dumps(
                {
                    "ready_for_opt_in": True,
                    "dev_reviews": len(dev),
                    "jev_model": JEV_MODEL,
                    "reference": reference,
                    "provider_requests_made": 0,
                },
                indent=2,
            )
        )
        return 0
    if args.output is None:
        parser.error("run requires --output")
    report = benchmark(
        args.inputs,
        args.output,
        allow_external=args.allow_external_api,
        key=os.environ.get("OPENROUTER_API_KEY", ""),
        max_cost_usd=args.max_cost_usd,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["partition", "execution_complete", "summary", "paired_comparison"]
            },
            indent=2,
        )
    )
    return 0 if report["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
