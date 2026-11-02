"""Does the model the API serves reproduce the INT8 run the decision note recorded?

The note measured dynamic INT8 on the full random-split test set (macro-F1 0.9637, 13,182 of
13,263 labels agreeing with FP32), but the served path was FP32. This runs the quantized
model through the same helper the API uses and checks it against the committed logits:
agreement with the recorded INT8 labels, the one-point margin against FP32, and the
single-review latency ratio.

    python scripts/verify_int8_serving.py --model models/distilbert

Writes docs/experiments/results/int8_serving/receipt.json. Exits non-zero when the rule
fails, so it can gate a deployment.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.serving.model_wrapper import quantize_encoder  # noqa: E402

RECORDED = ROOT / "docs" / "experiments" / "results" / "int8_v1"
MODEL_REVISION = "7306aebcaaebc00d579f5d0a91001ae376f18158"
LABEL2ID = {"negative": 0, "positive": 1}
RECORDED_INT8_MACRO_F1 = 0.9637
RECORDED_FP32_MACRO_F1 = 0.9642
FP32_MARGIN = 0.01          # the protocol's "within one point"
AGREEMENT_FLOOR = 0.999     # this build must reproduce the recorded labels
MAX_MACRO_F1_DRIFT = 0.001  # and the recorded score


def _relative(path: Path) -> str:
    """Repo-relative when the path is inside the checkout, so the receipt carries no home directory."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def macro_f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for value in (0, 1):
        true_positive = int(np.sum((predicted == value) & (labels == value)))
        false_positive = int(np.sum((predicted == value) & (labels != value)))
        false_negative = int(np.sum((predicted != value) & (labels == value)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def predict(model, tokenizer, texts: list[str], batch_size: int) -> np.ndarray:
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            predictions += model(**encoded).logits.argmax(-1).tolist()
    return np.asarray(predictions)


def timed(model, tokenizer, texts: list[str]) -> tuple[float, float]:
    """Median ms per review, one review at a time, matching the recorded measurement."""
    times = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer([text], truncation=True, max_length=256, padding=True, return_tensors="pt")
            start = time.perf_counter()
            model(**encoded)
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ROOT / "models" / "distilbert"))
    parser.add_argument("--split", default=str(ROOT / "data" / "processed_v2"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "results" / "int8_serving" / "receipt.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-n", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows (0 = the full test set)")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads; the recorded run used one, and the default count can be slower "
        "for dynamic-quantized layers (0 leaves torch's default)",
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    test = pd.read_parquet(Path(args.split) / "test.parquet")
    if args.limit:
        test = test.head(args.limit)
    labels = test["label"].map(LABEL2ID).to_numpy()
    texts = test["text"].tolist()

    recorded_labels = np.load(RECORDED / "test_labels.npy")
    recorded_int8 = np.load(RECORDED / "int8_test_logits.npy").argmax(axis=1)
    if len(recorded_labels) < len(labels) or not np.array_equal(recorded_labels[: len(labels)], labels):
        raise SystemExit("the rebuilt test rows do not line up with the recorded INT8 logits")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp32 = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    int8 = quantize_encoder(AutoModelForSequenceClassification.from_pretrained(args.model).eval())

    int8_predicted = predict(int8, tokenizer, texts, args.batch_size)
    fp32_predicted = predict(fp32, tokenizer, texts, args.batch_size)

    agreement_recorded = float(np.mean(int8_predicted == recorded_int8[: len(labels)]))
    agreement_fp32 = float(np.mean(int8_predicted == fp32_predicted))
    int8_f1 = macro_f1(labels, int8_predicted)
    fp32_f1 = macro_f1(labels, fp32_predicted)
    fp32_latency = timed(fp32, tokenizer, texts[: args.latency_n])
    int8_latency = timed(int8, tokenizer, texts[: args.latency_n])
    speedup = fp32_latency[0] / int8_latency[0]

    # The gate is a same-machine comparison: does switching the served model to INT8 change
    # the labels? The recorded Colab run is context, not the bar — a different quantized
    # backend and a different CPU change a few labels, and 1,024 rows carry their own noise.
    checks = {
        "int8_and_fp32_agree_on_this_machine": agreement_fp32 >= 0.99,
        "macro_f1_within_the_one_point_margin": abs(int8_f1 - fp32_f1) <= FP32_MARGIN,
        "still_faster_than_fp32": speedup >= 1.2,
        "agrees_with_the_recorded_int8_run": agreement_recorded >= 0.99,
    }
    receipt = {
        "model_revision": MODEL_REVISION,
        "model_path": _relative(Path(args.model)),
        "split": _relative(Path(args.split)),
        "rows": int(len(labels)),
        "batch_size": args.batch_size,
        "torch_threads": args.threads or torch.get_num_threads(),
        "agreement_with_recorded_int8": round(agreement_recorded, 6),
        "agreement_int8_vs_fp32": round(agreement_fp32, 6),
        "macro_f1": {"int8": round(int8_f1, 4), "fp32": round(fp32_f1, 4),
                     "delta": round(int8_f1 - fp32_f1, 4)},
        "recorded": {"int8_macro_f1": RECORDED_INT8_MACRO_F1, "fp32_macro_f1": RECORDED_FP32_MACRO_F1,
                     "int8_minus_recorded": round(int8_f1 - RECORDED_INT8_MACRO_F1, 4)},
        "latency_ms": {
            "n": args.latency_n,
            "fp32_median": round(fp32_latency[0], 2),
            "fp32_p95": round(fp32_latency[1], 2),
            "int8_median": round(int8_latency[0], 2),
            "int8_p95": round(int8_latency[1], 2),
            "speedup": round(speedup, 2),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")

    print(f"rows {len(labels):,} | INT8 macro-F1 {int8_f1:.4f} vs FP32 {fp32_f1:.4f} on the same rows")
    print(f"INT8 and FP32 agree on {agreement_fp32:.2%}; agreement with the recorded INT8 labels {agreement_recorded:.2%}")
    print(f"latency one review at a time: FP32 {fp32_latency[0]:.1f} ms, INT8 {int8_latency[0]:.1f} ms "
          f"({speedup:.1f}x); recorded {RECORDED_INT8_MACRO_F1} for INT8 on the full test set")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"-> {out.relative_to(ROOT)}")
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
