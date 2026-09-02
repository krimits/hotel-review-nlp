"""Dynamic INT8 quantization benchmark for the DistilBERT checkpoint.

Production story in one script: load the fine-tuned FP32 encoder, quantize
all nn.Linear layers dynamically to int8 (torch.ao.quantization), then
measure per-text latency before/after on the same sample. On CPU this
typically cuts latency by ~2x with <0.1 F1 degradation for this task -
numbers land in runs/quantization/benchmark.json.

Run:  python scripts/quantize_distilbert.py --model runs/distilbert \
        --dataset data/processed/test.parquet --n 256
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.ao.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def measure(model, tokenizer, texts: list[str], repeats: int = 3) -> dict:
    """Median latency over `repeats` full passes (batched inference)."""
    enc = tokenizer(list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt")
    args = {k: v for k, v in enc.items()}
    # warmup
    with torch.no_grad():
        model(**args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**args)
        times.append(time.perf_counter() - t0)
    per_text = [t / len(texts) * 1000 for t in times]
    return {
        "p50_ms_per_text": round(float(np.percentile(per_text, 50)), 2),
        "p95_ms_per_text": round(float(np.percentile(per_text, 95)), 2),
    }


def accuracy(model, tokenizer, texts: list[str], labels: list[int]) -> float:
    preds = []
    for start in range(0, len(texts), 64):
        enc = tokenizer(texts[start : start + 64], truncation=True, max_length=256,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        preds += logits.argmax(-1).tolist()
    return float(np.mean(np.asarray(preds) == np.asarray(labels)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/distilbert")
    parser.add_argument("--dataset", default="data/processed/test.parquet")
    parser.add_argument("--n", type=int, default=256, help="sample size for the benchmark")
    parser.add_argument("--out", default="runs/quantization/benchmark.json")
    args = parser.parse_args()

    device = "cpu"  # dynamic int8 targets CPU deployments
    df = pd.read_parquet(args.dataset).sample(n=args.n, random_state=42)
    label2id = {"negative": 0, "positive": 1}
    texts = df["text"].tolist()
    labels = df["label"].map(label2id).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp32 = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    fp32_latency = measure(fp32, tokenizer, texts)
    fp32_acc = accuracy(fp32, tokenizer, texts, labels)

    int8_model = quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)
    int8_latency = measure(int8_model, tokenizer, texts)
    int8_acc = accuracy(int8_model, tokenizer, texts, labels)

    result = {
        "n_sample": len(texts),
        "fp32": {"latency": fp32_latency, "accuracy": round(fp32_acc, 4),
                 "size_mb": round(_model_size_mb(fp32), 1)},
        "int8_dynamic": {"latency": int8_latency, "accuracy": round(int8_acc, 4),
                         "size_mb": round(_model_size_mb(int8_model), 1)},
        "speedup_p50": round(fp32_latency["p50_ms_per_text"] / max(1e-9, int8_latency["p50_ms_per_text"]), 2),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def _model_size_mb(model) -> float:
    """On-disk-equivalent parameter footprint in MB."""
    total = sum(p.numel() for p in model.parameters())
    # assume mixed dtypes; approximate fp32=4B, int8=1B per param
    bytes_total = 0
    for p in model.parameters():
        bytes_total += p.numel() * (1 if p.dtype == torch.qint8 else 4)
    return bytes_total / 1024 / 1024 if bytes_total else total * 4 / 1024 / 1024


if __name__ == "__main__":
    main()
