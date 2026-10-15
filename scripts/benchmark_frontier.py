"""Time the four models of the BiLSTM protocol's frontier in one session, on one CPU.

docs/experiments/bilstm_clean_splits/PROTOCOL.md (version 3), "Secondary: speed, size and
the measured frontier", fixes the method, which is the one of latency.json:
- the same 500 reviews of the `time` test set (sample seed 0);
- one PyTorch thread, one review at a time, text to label with tokenization included;
- 3 warm-up calls, then the median and 95th percentile per review.

The four models are TF-IDF + Naive Bayes, the BiLSTM (seed 42) and a DistilBERT full
fine-tune and scratch LoRA trained in the same session: the architecture, not the
weights, sets their time. Each timed model must give the prediction it was scored with on
at least 99% of the sample. No training and no Hub access happen here.

    python scripts/benchmark_frontier.py --bilstm runs/bilstm_v2_time \\
        --distilbert runs/distilbert_v2_time_seed43 \\
        --lora runs/distilbert_v2_time_seed43_lora_scratch --out runs/frontier/latency_frontier.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_latency as latency  # noqa: E402

from reviewnlp.utils.experiments import frame_fingerprint  # noqa: E402
from reviewnlp.utils.seed import load_config  # noqa: E402

PROTOCOL = "docs/experiments/bilstm_clean_splits/PROTOCOL.md"
METHOD = {"threads": 1, "batch_size": 1, "sample": latency.SAMPLE, "sample_seed": 0, "warmup": latency.WARMUP}


class BiLSTMPredictor:
    """A saved BiLSTM run on the CPU: text to label, tokenization included."""

    def __init__(self, directory: Path):
        import torch
        import yaml

        from reviewnlp.baselines.bilstm import BiLSTMClassifier
        from reviewnlp.data.dataset import TextVocab

        cfg = yaml.safe_load((directory / "run_config.yaml").read_text(encoding="utf-8"))
        self.torch, self.max_tokens = torch, cfg["data"]["max_tokens"]
        self.vocab = TextVocab.load(str(directory / "vocab.json"))
        model, train = cfg["model"], cfg["train"]
        self.model = BiLSTMClassifier(len(self.vocab), train["embedding_dim"], model["hidden_dim"],
                                      model["num_layers"], model["dropout"], model["bidirectional"], model["pooling"])
        self.model.load_state_dict(torch.load(directory / "bilstm.pt", map_location="cpu"))
        self.model.eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        from reviewnlp.data.dataset import UNK, collate

        unk = self.vocab.itoi[UNK]
        items = []
        for text in texts:
            ids = self.vocab.encode(text, self.max_tokens) or [unk]
            items.append((self.torch.tensor(ids, dtype=self.torch.long), len(ids), self.torch.tensor(0)))
        batch = collate(items)
        with self.torch.inference_mode():
            logits = self.model(batch["input_ids"], batch["lengths"])
        return logits[self.torch.argsort(batch["order"])].argmax(-1).numpy()


def bilstm_size_mb(directory: Path) -> float:
    return round(sum((directory / name).stat().st_size for name in ("bilstm.pt", "vocab.json")) / 2**20, 1)


def frontier(test: pd.DataFrame, nb_scored: np.ndarray, classical: Path, bilstm: Path,
             encoders: dict[str, Path], sample_size: int = latency.SAMPLE, seed: int = 0) -> dict:
    """Batch-1 CPU timings and sizes for every model, on the same sample."""
    texts = test["text"].astype(str).tolist()
    rows = latency.sample_rows(len(texts), sample_size, seed)
    sample = [texts[row] for row in rows]

    def scored(directory: Path) -> np.ndarray:
        predictions = np.load(directory / "test_logits.npy").argmax(axis=1)
        if len(predictions) != len(texts):
            raise latency.BenchmarkError(f"{directory} was scored on {len(predictions)} reviews, not {len(texts)}")
        return predictions[rows]

    pipeline = joblib.load(classical)
    models = {"tfidf_nb": {"path": str(classical), "size_mb": latency.classical_size_mb(pipeline),
                           "cpu": {"batch_1": latency.timed("tfidf_nb", latency.classical_predictor(pipeline),
                                                            sample, nb_scored[rows], 1)}},
              "bilstm": {"path": str(bilstm), "size_mb": bilstm_size_mb(bilstm),
                         "cpu": {"batch_1": latency.timed("bilstm", BiLSTMPredictor(bilstm), sample,
                                                          scored(bilstm), 1)}}}
    for name, directory in encoders.items():
        models[name] = {"path": str(directory), "size_mb": latency.weights_mb(directory),
                        "cpu": {"batch_1": latency.timed(name, latency.EncoderPredictor(directory, "cpu"),
                                                         sample, scored(directory), 1)}}
    return {"protocol": PROTOCOL, "method": METHOD, "test": frame_fingerprint(test),
            "sample": {"reviews": len(rows), "seed": seed}, "hardware": latency.hardware(), "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bilstm", type=Path, required=True)
    parser.add_argument("--distilbert", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--split-views", type=Path, default=latency.SPLIT_VIEWS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(METHOD["threads"])
    classical_cfg = load_config(str(ROOT / "configs" / "baselines_time.yaml"))
    view = json.loads(args.split_views.read_text(encoding="utf-8"))["time"]
    test = pd.read_parquet(ROOT / classical_cfg["data"]["processed_dir"] / "test.parquet")
    if frame_fingerprint(test) != view["splits"]["test"]:
        raise SystemExit(f"the time test set differs from the one in {args.split_views.name}")
    try:
        report = frontier(test, np.array([{"n": 0, "p": 1}[code] for code in view["test_predictions"]]),
                          ROOT / classical_cfg["output"]["model_dir"] / "best_classical.joblib", args.bilstm,
                          {"distilbert": args.distilbert, "distilbert_lora_scratch": args.lora})
    except latency.BenchmarkError as error:
        raise SystemExit(str(error)) from None
    for entry in report["models"].values():
        entry["path"] = os.path.relpath(entry["path"], ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, entry in report["models"].items():
        one = entry["cpu"]["batch_1"]
        print(f"{name:24} {one['ms_per_batch_median']:.2f} ms/review (p95 {one['ms_per_batch_p95']:.2f}) "
              f"| {entry['size_mb']} MB")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
