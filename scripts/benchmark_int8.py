"""Dynamic INT8 and CPU batching for the published DistilBERT, on the full random test set.

docs/experiments/int8_and_batching/PROTOCOL.md (version 1) fixes, before any run:
- the model: krimits/distilbert-hotel-reviews at 7306aeb, whose FP32 labels on the CPU must
  match its committed test logits on at least 99% of the 13,263 random test reviews;
- the primary comparison: macro-F1(INT8) - macro-F1(FP32) on all of them, paired bootstrap,
  2000 resamples, seed 0, and a three-branch rule on a one-point margin and a 10% speed tie;
- speed: the median time per review at batch 1, one PyTorch thread, 3 warm-up calls, on the
  500 test reviews drawn with seed 0, the method of latency.json;
- beside it: serialized sizes, agreement, per-class recall, and batch 1 against batches of
  32 in test order and sorted by token length.

    python scripts/benchmark_int8.py --model models/distilbert --out-dir docs/experiments/results/int8_v1

Model weights are never written to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_latency as latency  # noqa: E402

from reviewnlp.evaluation.metrics import binary_metrics, paired_bootstrap_diff  # noqa: E402
from reviewnlp.utils.experiments import frame_fingerprint  # noqa: E402

PROTOCOL = "docs/experiments/int8_and_batching/PROTOCOL.md"
MODEL_REPO = "krimits/distilbert-hotel-reviews"
MODEL_REVISION = "7306aebcaaebc00d579f5d0a91001ae376f18158"
WEIGHTS_SHA256 = "f2a2bbc9a31c8f3b1d9bee46ec11bb3d8942e7e59c0ef4711c3cb2a6c1cf81c2"
SPLIT = "random"
COMMITTED = ROOT / "docs" / "experiments" / "results" / "distilbert_v2" / "random" / "distilbert" / "test_logits.npy"
SPLIT_VIEWS = latency.SPLIT_VIEWS
PROCESSED = ROOT / "data" / "processed_v2"
OUTPUT = ROOT / "docs" / "experiments" / "results" / "int8_v1"
RESAMPLES = 2000
BATCH = 32
PASS_BATCH = 64  # the untimed full-test pass, sorted by length
SPEED_TIE = 0.9
LABEL2ID = {"negative": 0, "positive": 1}
LABEL_VALUES = (0, 1)
RULE = (
    ("lo > -0.01 and lat_int8 < 0.9 * lat_fp32",
     "INT8 keeps macro-F1 within one point and is meaningfully faster: recommended for single-review CPU serving."),
    ("lo > -0.01 and lat_int8 >= 0.9 * lat_fp32",
     "INT8 keeps macro-F1 within one point but is not meaningfully faster: no reason to switch."),
    ("lo <= -0.01",
     "INT8 may cost a meaningful point of macro-F1: not recommended."),
)
NO_BUILTINS = {"__builtins__": {}}


class Encoder:
    """The checkpoint on the CPU, FP32 or dynamically quantized to INT8: text to logits."""

    def __init__(self, directory: Path, int8: bool = False, max_length: int = latency.MAX_LENGTH):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.max_length = torch, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
        if int8:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model

    def logits(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True,
                                 return_tensors="pt")
        with self.torch.inference_mode():
            return self.model(**encoded).logits.float().numpy()

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.logits(texts).argmax(axis=1)

    def token_lengths(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length)
        return np.array([len(ids) for ids in encoded["input_ids"]])


def length_order(lengths: np.ndarray) -> np.ndarray:
    """Positions sorted by length, shortest first; ties keep their order."""
    return np.argsort(lengths, kind="stable")


def restore(order: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Put values computed in `order` back in the original positions."""
    restored = np.empty_like(values)
    restored[order] = values
    return restored


def all_logits(encoder: Encoder, texts: list[str], batch_size: int = PASS_BATCH) -> np.ndarray:
    """Logits for every text, computed in length-sorted batches and returned in input order."""
    order = length_order(encoder.token_lengths(texts))
    ordered = [texts[i] for i in order]
    chunks = [encoder.logits(ordered[start:start + batch_size]) for start in range(0, len(ordered), batch_size)]
    return restore(order, np.concatenate(chunks))


def timed_sorted(name: str, encoder: Encoder, texts: list[str], scored: np.ndarray, batch_size: int = BATCH) -> dict:
    """Batches of `batch_size` after sorting by token length; sorting time is counted."""
    started = time.perf_counter()
    order = length_order(encoder.token_lengths(texts))
    sort_seconds = time.perf_counter() - started
    report, predicted = latency.time_batches(encoder, [texts[i] for i in order], batch_size)
    total = report["seconds"] + sort_seconds
    return {**report, "sorted_by": "token length", "sort_seconds": round(sort_seconds, 3),
            "seconds": round(total, 3), "ms_per_review": round(1000 * total / len(texts), 3),
            "reviews_per_second": round(len(texts) / total, 1),
            "agrees": latency.agreement(name, restore(order, predicted), scored)}


def state_dict_bytes(model) -> int:
    """Serialized state_dict bytes, packed INT8 weights included."""
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def state_dict_mb(model) -> float:
    return round(state_dict_bytes(model) / 2**20, 1)


def outcome(lo: float, lat_int8: float, lat_fp32: float) -> str:
    # The conditions are the protocol's constant strings (a test matches them verbatim),
    # evaluated without builtins on three numbers, never on input.
    scope = {"lo": lo, "lat_int8": lat_int8, "lat_fp32": lat_fp32}
    (conclusion,) = [text for condition, text in RULE if eval(condition, NO_BUILTINS, scope)]  # noqa: S307
    return conclusion


def batching_verdict(sorted_ms: float, batch_1_ms: float) -> str:
    return "faster" if sorted_ms < SPEED_TIE * batch_1_ms else "not faster"


def weights_sha256(directory: Path) -> str:
    return hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()


def benchmark(test: pd.DataFrame, committed_logits: np.ndarray, model_dir: Path,
              sample_size: int = latency.SAMPLE, seed: int = 0, resamples: int = RESAMPLES) -> tuple[dict, dict]:
    """The protocol's measurements. Returns the report and the arrays to save."""
    import torch

    texts = test["text"].astype(str).tolist()
    labels = test["label"].map(LABEL2ID).to_numpy()
    if len(committed_logits) != len(texts):
        raise latency.BenchmarkError(f"the committed logits cover {len(committed_logits)} reviews, not {len(texts)}")
    threads = torch.get_num_threads()

    encoders = {"fp32": Encoder(model_dir), "int8": Encoder(model_dir, int8=True)}
    logits = {name: all_logits(encoder, texts) for name, encoder in encoders.items()}
    predictions = {name: values.argmax(axis=1) for name, values in logits.items()}
    gate = latency.agreement("the FP32 model", predictions["fp32"], committed_logits.argmax(axis=1))

    d, lo, hi = paired_bootstrap_diff(labels, predictions["fp32"], predictions["int8"], LABEL_VALUES, resamples, seed)
    models = {}
    rows = latency.sample_rows(len(texts), sample_size, seed)
    sample = [texts[row] for row in rows]
    torch.set_num_threads(1)
    try:
        for name, encoder in encoders.items():
            scored = predictions[name][rows]
            batch_1 = latency.timed(name, encoder, sample, scored, 1)
            in_order = latency.timed(name, encoder, sample, scored, BATCH)
            by_length = timed_sorted(name, encoder, sample, scored, BATCH)
            metrics = binary_metrics(labels, predictions[name], label_values=LABEL_VALUES)
            models[name] = {
                "size_mb": state_dict_mb(encoder.model),
                "test": {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                         "recall": {label: metrics["per_class"][label]["recall"] for label in LABEL2ID}},
                "cpu": {"batch_1": batch_1, "batch_32": in_order, "batch_32_sorted": by_length},
                "sorted_batching": batching_verdict(by_length["ms_per_review"], batch_1["ms_per_batch_median"]),
            }
    finally:
        torch.set_num_threads(threads)

    lat_int8, lat_fp32 = (models[name]["cpu"]["batch_1"]["ms_per_batch_median"] for name in ("int8", "fp32"))
    report = {
        "protocol": PROTOCOL, "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "test": frame_fingerprint(test),
        "integrity": {"fp32_agrees_with_committed": gate, "reviews": len(texts)},
        "sample": {"reviews": len(rows), "seed": seed},
        "method": {"threads": 1, "warmup": latency.WARMUP, "batch": BATCH, "resamples": resamples, "seed": seed},
        "hardware": {**latency.hardware(), "torch_threads": 1},
        "models": models,
        "primary": {"d_int8_minus_fp32": d, "paired_ci95": [lo, hi],
                    "agreement": int(np.sum(predictions["int8"] == predictions["fp32"])),
                    "lat_int8": lat_int8, "lat_fp32": lat_fp32,
                    "speedup_batch_1": round(lat_fp32 / lat_int8, 2),
                    "conclusion": outcome(lo, lat_int8, lat_fp32)},
        "selection_frozen": False, "promoted": False,
    }
    arrays = {"fp32_test_logits": logits["fp32"], "int8_test_logits": logits["int8"], "test_labels": labels}
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="the checkpoint downloaded at the pinned revision")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    parser.add_argument("--split-views", type=Path, default=SPLIT_VIEWS)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if weights_sha256(args.model) != WEIGHTS_SHA256:
        raise SystemExit(f"{args.model} is not the published checkpoint at {MODEL_REVISION[:7]}")
    view = json.loads(args.split_views.read_text(encoding="utf-8"))[SPLIT]
    test = pd.read_parquet(args.processed_dir / "test.parquet")
    if frame_fingerprint(test) != view["splits"]["test"]:
        raise SystemExit(f"the {SPLIT} test set differs from the one in {args.split_views.name}")
    try:
        report, arrays = benchmark(test, np.load(COMMITTED), args.model)
    except latency.BenchmarkError as error:
        raise SystemExit(str(error)) from None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(args.out_dir / f"{name}.npy", values)
    (args.out_dir / "int8_benchmark.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    primary = report["primary"]
    for name, entry in report["models"].items():
        cpu = entry["cpu"]
        print(f"{name}: macro-F1 {entry['test']['macro_f1']} | batch 1 {cpu['batch_1']['ms_per_batch_median']} ms"
              f" | batch 32 {cpu['batch_32']['ms_per_review']} ms/review"
              f" | sorted {cpu['batch_32_sorted']['ms_per_review']} ms/review | {entry['size_mb']} MB")
    print(f"d = {primary['d_int8_minus_fp32']:+.4f} {primary['paired_ci95']} -> {primary['conclusion']}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
