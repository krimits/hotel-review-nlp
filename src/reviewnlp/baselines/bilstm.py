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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reviewnlp.data.dataset import TextVocab, build_bilstm_datasets, collate
from reviewnlp.evaluation.metrics import binary_metrics
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


def _evaluate(model: BiLSTMClassifier, loader: DataLoader, device: torch.device) -> tuple[float, dict, torch.Tensor]:
    """Accuracy, metrics and logits, in the loader's order (collate sorts each batch by length)."""
    model.eval()
    logits_all, preds, golds = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["lengths"].to(device)).float().cpu()
            restore = torch.argsort(batch["order"])
            logits, labels = logits[restore], batch["labels"][restore]
            logits_all.append(logits)
            preds += logits.argmax(dim=-1).tolist()
            golds += labels.tolist()
    acc = sum(p == g for p, g in zip(preds, golds, strict=False)) / max(1, len(golds))
    metrics = binary_metrics(golds, preds, label_names=(0,1))
    return acc, metrics, torch.cat(logits_all)


def train_bilstm(config_path: str) -> dict:
    """Train with early stopping on dev macro-F1, then write the run's artifacts.

    The output directory holds what a paired comparison reads: ``dev_`` and ``test_``
    logits and labels, row-aligned with the processed splits, plus ``metrics.json`` (with
    the data fingerprints and the seed), ``run_config.yaml`` and ``train_log.txt``.
    """
    import numpy as np
    import yaml

    from reviewnlp.data.preprocess import load_processed
    from reviewnlp.utils.experiments import frame_fingerprint

    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log: list[str] = []

    def say(line: str) -> None:
        print(line)
        log.append(line)

    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    started = time.perf_counter()
    train_ds, dev_ds, test_ds, vocab = build_bilstm_datasets(d["processed_dir"], cfg["data"]["max_tokens"], t.get("min_freq", 2))
    say(f"vocab={len(vocab):,} train={len(train_ds):,} dev={len(dev_ds):,} test={len(test_ds):,} device={device}")

    loader_kwargs = dict(batch_size=t["batch_size"], collate_fn=collate, num_workers=t.get("num_workers", 2))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=t["embedding_dim"],
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        bidirectional=m["bidirectional"],
        pooling=m["pooling"],
    ).to(device)
    if cfg["data"].get("glove"):
        _load_glove(model, vocab, cfg["data"]["glove"], t["embedding_dim"])

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

        dev_acc, dev_metrics, _ = _evaluate(model, dev_loader, device)
        say(
            f"epoch {epoch:02d} | loss {running / n_batches:.4f} | "
            f"dev acc {dev_acc:.4f} | dev macro-F1 {dev_metrics['macro_f1']:.4f} | "
            f"{time.perf_counter() - t0:.1f}s"
        )
        if dev_metrics["macro_f1"] > best_f1:
            best_f1, bad_epochs = dev_metrics["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                say(f"early stop (no dev improvement for {t['patience']} epochs)")
                break

    model.load_state_dict(best_state)
    _, dev_metrics, dev_logits = _evaluate(model, dev_loader, device)
    test_acc, test_metrics, test_logits = _evaluate(model, test_loader, device)
    say(f"TEST  acc={test_acc:.4f} macro-F1={test_metrics['macro_f1']:.4f}")
    training_seconds = round(time.perf_counter() - started, 1)

    out_dir = cfg["output"]["model_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "bilstm.pt"))
    vocab.save(os.path.join(out_dir, "vocab.json"))
    # Row-aligned with the processed splits: the same format as the encoders' artifacts.
    for split, logits, dataset in (("dev", dev_logits, dev_ds), ("test", test_logits, test_ds)):
        np.save(os.path.join(out_dir, f"{split}_logits.npy"), logits.numpy())
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), np.asarray(dataset.labels))
    data_fingerprints = {split: frame_fingerprint(frame) for split, frame in load_processed(d["processed_dir"]).items()}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics, "dev": dev_metrics, "best_dev_macro_f1": best_f1,
                   "settings": {"seed": cfg["seed"], "min_freq": t.get("min_freq", 2), "vocab_size": len(vocab),
                                "device": str(device)},
                   "training_seconds": training_seconds, "data": data_fingerprints}, f, indent=2)
    with open(os.path.join(out_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    return test_metrics


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
