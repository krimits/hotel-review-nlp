"""Pure-PyTorch BiLSTM sentiment classifier - a deliberate "no shortcuts" baseline.

Why this file exists (PyTorch requirement):
- a hand-written ``nn.Module`` with an embedding bag over packed sequences,
- a **custom training loop** (optimizer, LR schedule, gradient clipping,
  early stopping) instead of ``Trainer``,
- mixed-pooling head (last hidden state of both directions + max-pool),
- clean CPU/GPU device handling so the same code runs locally and on Colab.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reviewnlp.data.dataset import TextVocab, build_bilstm_datasets, collate
from reviewnlp.evaluation.metrics import (
    metrics_at_threshold,
    positive_probabilities,
    tune_binary_threshold,
)
from reviewnlp.utils.seed import load_config, set_seed


class BiLSTMClassifier(nn.Module):
    """Embedding -> (Bi)LSTM -> pooling head -> logits.

    Supports ``last`` (concat final hidden states of both directions),
    ``mean`` and ``max`` time pooling. Dropout is applied on embeddings
    and on the pooled representation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pooling: str = "last",
        pad_idx: int = 0,
    ):
        super().__init__()
        if pooling not in {"last", "mean", "max"}:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 2)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) padded, lengths: (B,) true lengths (desc order)."""
        emb = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, (h_n, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if self.pooling == "last":
            feats = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.bidirectional else h_n[-1]
        elif self.pooling == "max":
            feats = out.masked_fill(input_ids.unsqueeze(-1) == self.pad_idx, -1e4).max(dim=1).values
        else:  # mean over true (non-pad) tokens
            mask = input_ids.ne(self.pad_idx).unsqueeze(-1).float()
            feats = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.fc(self.dropout(feats))


def _evaluate(
    model: BiLSTMClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict, torch.Tensor, np.ndarray]:
    model.eval()
    logits_all, golds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
            logits_all.append(logits.float().cpu())
            golds += batch["labels"].tolist()
    logits = torch.cat(logits_all)
    gold = np.asarray(golds, dtype=np.int64)
    metrics = metrics_at_threshold(gold, positive_probabilities(logits.numpy()), threshold)
    return metrics, logits, gold


def _threshold_result(gold: np.ndarray, logits: torch.Tensor, cfg: dict) -> dict:
    tuning = cfg.get("evaluation", {}).get("threshold_tuning", {})
    if not tuning.get("enabled", False):
        return {
            "threshold": 0.5,
            "dev_metrics": metrics_at_threshold(gold, positive_probabilities(logits.numpy()), 0.5),
            "objective": "macro_f1",
        }
    return tune_binary_threshold(
        gold,
        positive_probabilities(logits.numpy()),
        minimum=float(tuning.get("minimum", 0.05)),
        maximum=float(tuning.get("maximum", 0.95)),
        step=float(tuning.get("step", 0.005)),
    )


def _init_wandb(cfg: dict):
    tracking = cfg.get("tracking", {}).get("wandb", {})
    if not tracking.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled; install it with `pip install -e '.[tracking]'`"
        ) from exc

    kwargs = {
        "project": tracking.get("project", "hotel-review-nlp"),
        "name": tracking.get("run_name"),
        "tags": tracking.get("tags", []),
        "config": cfg,
    }
    if tracking.get("entity"):
        kwargs["entity"] = tracking["entity"]
    return wandb.init(**kwargs)


def _flatten_metrics(prefix: str, metrics: dict) -> dict:
    """Flatten all scalar metrics for a complete W&B run summary."""
    flat = {
        f"{prefix}/{name}": metrics[name]
        for name in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1")
    }
    for class_name, class_metrics in metrics["per_class"].items():
        for name, value in class_metrics.items():
            flat[f"{prefix}/per_class/{class_name}/{name}"] = value
    flat[f"{prefix}/confusion_matrix"] = metrics["confusion_matrix"]
    return flat


def train_bilstm(config_path: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = _init_wandb(cfg)

    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    train_ds, dev_ds, test_ds, vocab = build_bilstm_datasets(
        d["processed_dir"], d["max_tokens"], d.get("min_freq", 2)
    )
    print(f"vocab={len(vocab):,} train={len(train_ds):,} dev={len(dev_ds):,} test={len(test_ds):,} device={device}")

    loader_kwargs = dict(batch_size=t["batch_size"], collate_fn=collate, num_workers=2)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=d["embedding_dim"],
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        bidirectional=m["bidirectional"],
        pooling=m["pooling"],
    ).to(device)
    if cfg["data"].get("glove"):
        _load_glove(model, vocab, cfg["data"]["glove"], d["embedding_dim"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    total_steps = len(train_loader) * t["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=t["lr"], total_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        running, n_batches, t0 = 0.0, 0, time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n_batches += 1

        dev_default, dev_logits, dev_gold = _evaluate(model, dev_loader, device)
        threshold_result = _threshold_result(dev_gold, dev_logits, cfg)
        dev_selected = threshold_result["dev_metrics"]
        print(
            f"epoch {epoch:02d} | loss {running / n_batches:.4f} | "
            f"dev macro-F1@0.5 {dev_default['macro_f1']:.4f} | "
            f"dev macro-F1@{threshold_result['threshold']:.3f} "
            f"{dev_selected['macro_f1']:.4f} | "
            f"{time.perf_counter() - t0:.1f}s"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": running / n_batches,
                    "dev/default/accuracy": dev_default["accuracy"],
                    "dev/default/macro_f1": dev_default["macro_f1"],
                    "dev/selected/threshold": threshold_result["threshold"],
                    "dev/selected/accuracy": dev_selected["accuracy"],
                    "dev/selected/macro_f1": dev_selected["macro_f1"],
                },
                step=epoch,
            )
        if dev_selected["macro_f1"] > best_f1:
            best_f1, bad_epochs = dev_selected["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                print(f"early stop (no dev improvement for {t['patience']} epochs)")
                break

    model.load_state_dict(best_state)
    dev_default, dev_logits, dev_gold = _evaluate(model, dev_loader, device)
    threshold_result = _threshold_result(dev_gold, dev_logits, cfg)
    selected_threshold = threshold_result["threshold"]
    dev_selected = threshold_result["dev_metrics"]

    test_default, test_logits, test_gold = _evaluate(model, test_loader, device)
    test_selected = metrics_at_threshold(
        test_gold, positive_probabilities(test_logits.numpy()), selected_threshold
    )
    print(
        f"TEST  macro-F1@0.5={test_default['macro_f1']:.4f} | "
        f"macro-F1@{selected_threshold:.3f}={test_selected['macro_f1']:.4f} | "
        f"acc={test_selected['accuracy']:.4f}"
    )

    out_dir = cfg["output"]["model_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "bilstm.pt"))
    vocab.save(os.path.join(out_dir, "vocab.json"))
    # Unified-benchmark artifacts plus dev outputs proving that the threshold
    # was selected without looking at test labels.
    np.save(os.path.join(out_dir, "dev_logits.npy"), dev_logits.numpy())
    np.save(os.path.join(out_dir, "dev_labels.npy"), dev_gold)
    np.save(os.path.join(out_dir, "test_logits.npy"), test_logits.numpy())
    np.save(os.path.join(out_dir, "test_labels.npy"), test_gold)
    artifact = {
        "seed": cfg["seed"],
        "selection_metric": "dev_macro_f1",
        "selected_threshold": selected_threshold,
        "best_dev_macro_f1": best_f1,
        "dev": {
            "threshold_0_5": dev_default,
            "selected_threshold": dev_selected,
        },
        # `test` stays the canonical final result for backwards compatibility.
        "test": test_selected,
        "test_threshold_0_5": test_default,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(artifact, f, indent=2)

    if wandb_run is not None:
        summary = {
            "seed": cfg["seed"],
            "selection_metric": "dev_macro_f1",
            "selected_threshold": selected_threshold,
            "best_dev_macro_f1": best_f1,
        }
        summary.update(_flatten_metrics("dev/default", dev_default))
        summary.update(_flatten_metrics("dev/selected", dev_selected))
        summary.update(_flatten_metrics("test/default", test_default))
        summary.update(_flatten_metrics("test/selected", test_selected))
        wandb_run.summary.update(summary)
        wandb_run.finish()
    return test_selected


def _load_glove(model: BiLSTMClassifier, vocab: TextVocab, path: str, dim: int) -> None:
    """Warm-start embeddings from GloVe; unknown tokens keep random init."""
    print(f"loading GloVe vectors from {path} ...")
    import numpy as np

    vectors = np.random.uniform(-0.1, 0.1, (len(vocab), dim)).astype("float32")
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            token = parts[0]
            if token in vocab.itoi and len(parts) == dim + 1:
                vectors[vocab.itoi[token]] = np.asarray(parts[1:], dtype="float32")
                found += 1
    with torch.no_grad():
        model.embedding.weight.copy_(torch.from_numpy(vectors))
        model.embedding.weight[vocab.itoi["<pad>"]].zero_()
    print(f"initialized {found:,}/{len(vocab):,} tokens from GloVe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bilstm.yaml")
    args = parser.parse_args()
    train_bilstm(args.config)


if __name__ == "__main__":
    main()
