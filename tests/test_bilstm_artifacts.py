"""The BiLSTM's saved artifacts must line up row by row with its test set.

Paired comparisons (the clean-split BiLSTM protocol) read ``test_logits.npy`` against
``test_labels.npy`` and against other models' logits, row by row. ``collate`` sorts each
batch by length for ``pack_padded_sequence``, so evaluation has to undo that sort.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import yaml

from reviewnlp.baselines.bilstm import _evaluate, train_bilstm
from reviewnlp.data.dataset import TextVocab, collate
from reviewnlp.evaluation.metrics import binary_metrics


class LengthModel(torch.nn.Module):
    """Logits that encode each review's true length, so their order is visible."""

    def forward(self, input_ids, lengths):
        return torch.stack([lengths.float(), -lengths.float()], dim=1)


def test_evaluation_returns_logits_in_dataset_order():
    lengths = [3, 7, 1, 5, 2, 6]
    batch_items = [(torch.ones(n, dtype=torch.long), n, torch.tensor(i % 2)) for i, n in enumerate(lengths)]
    loader = torch.utils.data.DataLoader(batch_items, batch_size=4, collate_fn=collate)
    _, _, logits = _evaluate(LengthModel(), loader, torch.device("cpu"))
    assert logits[:, 0].int().tolist() == lengths


def test_a_review_with_no_word_tokens_does_not_break_a_batch():
    vocab = TextVocab.build(["good room", "bad room"], min_freq=1)
    frame = pd.DataFrame({"text": ["!!!", "good room"], "label": ["negative", "positive"]})
    from reviewnlp.data.dataset import SentimentDataset

    ids, length, _ = SentimentDataset(frame, vocab, max_tokens=10)[0]
    assert length == 1 and ids.tolist() == [vocab.itoi["<unk>"]]


def tiny_split(directory, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    good, bad = ["great", "clean", "friendly", "lovely"], ["dirty", "rude", "noisy", "broken"]
    texts, labels = [], []
    for _ in range(rows):
        positive = bool(rng.integers(2))
        words = rng.choice(good if positive else bad, size=int(rng.integers(1, 12)))
        texts.append("the room was " + " ".join(words))
        labels.append("positive" if positive else "negative")
    pd.DataFrame({"text": texts, "label": labels}).to_parquet(directory, index=False)


def test_a_training_run_writes_aligned_artifacts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split, rows, seed in (("train", 120, 1), ("dev", 40, 2), ("test", 50, 3)):
        tiny_split(data / f"{split}.parquet", rows, seed)
    out = tmp_path / "run"
    config = {
        "seed": 7,
        "data": {"processed_dir": str(data), "max_tokens": 20, "glove": None},
        "model": {"hidden_dim": 8, "num_layers": 1, "bidirectional": True, "dropout": 0.0, "pooling": "last"},
        "train": {"batch_size": 16, "epochs": 2, "embedding_dim": 8, "lr": 0.01, "weight_decay": 0.0,
                  "grad_clip": 5.0, "patience": 2, "num_workers": 0, "min_freq": 1},
        "output": {"model_dir": str(out)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train_bilstm(str(path))

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for split in ("dev", "test"):
        logits, labels = np.load(out / f"{split}_logits.npy"), np.load(out / f"{split}_labels.npy")
        expected = pd.read_parquet(data / f"{split}.parquet")["label"].map({"negative": 0, "positive": 1})
        assert labels.tolist() == expected.tolist()
        recomputed = binary_metrics(labels, logits.argmax(axis=1), label_names=(0, 1))["macro_f1"]
        assert recomputed == metrics[split]["macro_f1"], split
    assert metrics["settings"]["seed"] == 7 and metrics["training_seconds"] > 0
    assert yaml.safe_load((out / "run_config.yaml").read_text(encoding="utf-8")) == config
    assert "TEST" in (out / "train_log.txt").read_text(encoding="utf-8")
    assert {"vocab.json", "bilstm.pt"} <= {p.name for p in out.iterdir()}


def test_the_clean_split_configs_keep_the_protocols_settings():
    """The protocol pins configs/bilstm.yaml: only the data and the output may differ."""
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    base = yaml.safe_load((configs / "bilstm.yaml").read_text(encoding="utf-8"))
    for name, data in (("bilstm_v2", "data/processed_v2"), ("bilstm_v2_time", "data/processed_v2_time"),
                       ("bilstm_v2_hotel", "data/processed_v2_hotel")):
        config = yaml.safe_load((configs / f"{name}.yaml").read_text(encoding="utf-8"))
        assert config["data"].pop("processed_dir") == data
        assert config["output"].pop("model_dir") == f"runs/{name}"
        expected = {**base, "data": {k: v for k, v in base["data"].items() if k != "processed_dir"},
                    "output": {}}
        assert config == expected, name
