"""CPU latency, GPU throughput and size of the phase 2 models, on the same test reviews.

Phase 2, docs/experiments/phase2_analysis_plan.md (section Cost). On one split's test set:
- CPU time from text to label, at batch 1 and at batch 32, on the same 500 test
  reviews (drawn with seed 0), for the dev-selected TF-IDF + Naive Bayes and for
  each DistilBERT checkpoint;
- the whole test set: Naive Bayes on the CPU in one call, DistilBERT on the GPU
  when there is one;
- the size of each model, and the hardware.

Before its times count, each model must give the predictions that were scored:
Naive Bayes those in split_views.json, DistilBERT the argmax of its saved test
logits. The paths come from the split's configs.

    python scripts/benchmark_latency.py --split time --out runs/phase2/distilbert_v2/latency.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.utils.experiments import frame_fingerprint  # noqa: E402
from reviewnlp.utils.seed import load_config  # noqa: E402

PLAN = "docs/experiments/phase2_analysis_plan.md"
SPLIT_VIEWS = ROOT / "docs" / "experiments" / "results" / "split_views.json"
CONFIGS = {"random": ("baselines.yaml", "distilbert_v2.yaml"),
           "time": ("baselines_time.yaml", "distilbert_v2_time.yaml")}
SAMPLE = 500
CPU_BATCHES = (1, 32)
GPU_BATCH = 64  # the trainer's eval batch size
WARMUP = 3
MIN_AGREEMENT = 0.99
MAX_LENGTH = 256
NOTES = {
    "cpu": "text to label on the CPU, tokenization or TF-IDF included. At batch 1, "
           "ms_per_batch_median and ms_per_batch_p95 are the latency of one review.",
    "whole_test": "Naive Bayes on the CPU in one call; DistilBERT on the GPU in batches of 64, "
                  "in fp32 as when its test logits were saved.",
    "size_mb": "Naive Bayes: the pickled pipeline, without the list of dropped terms that older "
               "scikit-learn keeps for inspection. DistilBERT: the weights file. The scratch LoRA "
               "is merged before saving, so its weights match the full fine-tune's in size; "
               "adapter_mb is its adapters and head alone.",
    "agrees": "reviews on which the timed model gave the prediction that was scored.",
}


class BenchmarkError(ValueError):
    """The timed model or test set is not the one that was scored."""


def sample_rows(total: int, size: int = SAMPLE, seed: int = 0) -> np.ndarray:
    """Test rows to time, in test order; the same for every model."""
    if size >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, size, replace=False))


def time_batches(predict: Callable[[list[str]], object], texts: list[str], batch_size: int,
                 warmup: int = WARMUP, clock: Callable[[], float] = time.perf_counter) -> tuple[dict, np.ndarray]:
    """Time predict() on texts in batches, after a few untimed warm-up batches.

    Returns the timings and the predictions of the timed pass, in the order of texts.
    """
    batches = [texts[start:start + batch_size] for start in range(0, len(texts), batch_size)]
    for batch in batches[:warmup]:
        predict(batch)
    seconds, predictions = [], []
    for batch in batches:
        started = clock()
        predictions.append(np.asarray(predict(batch)))
        seconds.append(clock() - started)
    total = float(np.sum(seconds))
    return {"batch_size": batch_size, "reviews": len(texts), "warmup_batches": min(warmup, len(batches)),
            "seconds": round(total, 3), "ms_per_review": round(1000 * total / len(texts), 3),
            "ms_per_batch_median": round(1000 * float(np.median(seconds)), 3),
            "ms_per_batch_p95": round(1000 * float(np.quantile(seconds, 0.95)), 3),
            "reviews_per_second": round(len(texts) / total, 1)}, np.concatenate(predictions)


def agreement(name: str, predicted: np.ndarray, scored: np.ndarray) -> int:
    agree = int(np.sum(predicted == scored))
    if agree < MIN_AGREEMENT * len(scored):
        raise BenchmarkError(f"{name} gives the scored prediction on only {agree} of {len(scored)} "
                             "reviews, so it is not the model that was scored")
    return agree


def timed(name: str, predict, texts: list[str], scored: np.ndarray, batch_size: int, warmup: int = WARMUP) -> dict:
    report, predicted = time_batches(predict, texts, batch_size, warmup)
    return {**report, "agrees": agreement(name, predicted, scored)}


def classical_predictor(pipeline) -> Callable[[list[str]], np.ndarray]:
    return lambda texts: (pipeline.predict(texts) == "positive").astype(int)


def classical_size_mb(pipeline) -> float:
    for step in pipeline.named_steps.values():
        if getattr(step, "stop_words_", None) is not None:
            step.stop_words_ = None  # kept by older scikit-learn for inspection only
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return round(buffer.getbuffer().nbytes / 2**20, 1)


class EncoderPredictor:
    """A saved DistilBERT checkpoint on one device: text to label, tokenization included."""

    def __init__(self, directory: Path, device: str, max_length: int = MAX_LENGTH):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device, self.max_length = torch, torch.device(device), max_length
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory).to(self.device).eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True,
                                 return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).logits
        return logits.argmax(-1).cpu().numpy()


def weights_mb(directory: Path) -> float:
    files = [*directory.glob("*.safetensors"), *directory.glob("pytorch_model*.bin")]
    return round(sum(path.stat().st_size for path in files) / 2**20, 1)


def hardware() -> dict:
    import sklearn
    import torch

    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers_version, "scikit_learn": sklearn.__version__}


def benchmark(test: pd.DataFrame, nb_scored: np.ndarray, classical: Path, encoders: dict[str, Path],
              sample_size: int = SAMPLE, seed: int = 0, gpu: bool | None = None) -> dict:
    import torch

    texts = test["text"].astype(str).tolist()
    rows = sample_rows(len(texts), sample_size, seed)
    sample = [texts[row] for row in rows]
    gpu = torch.cuda.is_available() if gpu is None else gpu

    pipeline = joblib.load(classical)
    predict = classical_predictor(pipeline)
    models = {"tfidf_nb": {
        "path": str(classical), "size_mb": classical_size_mb(pipeline),
        "cpu": {f"batch_{size}": timed("tfidf_nb", predict, sample, nb_scored[rows], size) for size in CPU_BATCHES},
        "cpu_whole_test": timed("tfidf_nb", predict, texts, nb_scored, len(texts), warmup=0)}}

    for name, directory in encoders.items():
        scored = np.load(directory / "test_logits.npy").argmax(axis=1)
        if len(scored) != len(texts):
            raise BenchmarkError(f"{name} was scored on {len(scored)} reviews, not {len(texts)}")
        cpu = EncoderPredictor(directory, "cpu")
        entry = {"path": str(directory), "size_mb": weights_mb(directory),
                 "cpu": {f"batch_{size}": timed(name, cpu, sample, scored[rows], size) for size in CPU_BATCHES}}
        del cpu
        if (directory / "lora_scratch.pt").exists():
            entry["adapter_mb"] = round((directory / "lora_scratch.pt").stat().st_size / 2**20, 1)
        if gpu:
            entry["gpu_whole_test"] = timed(name, EncoderPredictor(directory, "cuda"), texts, scored, GPU_BATCH)
            torch.cuda.empty_cache()
        else:
            entry["gpu_whole_test"] = "no GPU"
        models[name] = entry

    return {"plan": PLAN, "test": frame_fingerprint(test),
            "sample": {"reviews": len(rows), "seed": seed,
                       "mean_characters": round(float(np.mean([len(text) for text in sample])), 1)},
            "hardware": hardware(), "notes": NOTES, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(CONFIGS), default="time")
    parser.add_argument("--split-views", type=Path, default=SPLIT_VIEWS)
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for torch (default: its own)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    classical_cfg, encoder_cfg = (load_config(str(ROOT / "configs" / name)) for name in CONFIGS[args.split])
    view = json.loads(args.split_views.read_text(encoding="utf-8"))[args.split]
    test = pd.read_parquet(ROOT / classical_cfg["data"]["processed_dir"] / "test.parquet")
    if frame_fingerprint(test) != view["splits"]["test"]:
        raise SystemExit(f"the {args.split} test set differs from the one in {args.split_views.name}")
    model_dir = encoder_cfg["output"]["model_dir"]
    encoders = {"distilbert": ROOT / model_dir, "distilbert_lora_scratch": ROOT / f"{model_dir}_lora_scratch"}
    try:
        report = benchmark(test, np.array([{"n": 0, "p": 1}[code] for code in view["test_predictions"]]),
                           ROOT / classical_cfg["output"]["model_dir"] / "best_classical.joblib", encoders,
                           args.sample, args.seed)
    except BenchmarkError as error:
        raise SystemExit(str(error)) from None
    report = {"split": args.split, **report}
    for entry in report["models"].values():
        entry["path"] = os.path.relpath(entry["path"], ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, entry in report["models"].items():
        one, batch = entry["cpu"]["batch_1"], entry["cpu"]["batch_32"]
        print(f"{name:24} CPU batch 1: {one['ms_per_batch_median']:.2f} ms/review (p95 {one['ms_per_batch_p95']:.2f})"
              f" | batch 32: {batch['ms_per_review']:.2f} ms/review | {entry['size_mb']} MB")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
