"""Torch-side data plumbing for the from-scratch BiLSTM.

Everything here is deliberately implemented without Hugging Face so the
BiLSTM baseline exercises real PyTorch fundamentals: a custom
``torch.utils.data.Dataset``, a vocabulary built from scratch, and a
padding-aware ``collate_fn`` with ``pack_padded_sequence`` support.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator

import pandas as pd
import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Tiny, dependency-free tokenizer (lowercase word tokens)."""
    return _TOKEN_RE.findall(str(text).lower())


class TextVocab:
    """Simple frequency-capped word -> index vocabulary."""

    def __init__(self, itoi: dict[str, int]):
        self.itoi = itoi
        self.itos = {i: t for t, i in itoi.items()}

    @classmethod
    def build(cls, texts: Iterator[str] | list[str], min_freq: int = 2, max_size: int = 50_000) -> TextVocab:
        from collections import Counter

        counts: Counter = Counter()
        for t in texts:
            counts.update(tokenize(t))
        itoi = {PAD: 0, UNK: 1}
        for token, freq in counts.most_common(max_size - len(itoi)):
            if freq < min_freq:
                break
            itoi.setdefault(token, len(itoi))
        return cls(itoi)

    def encode(self, text: str, max_tokens: int) -> list[int]:
        unk = self.itoi[UNK]
        return [self.itoi.get(tok, unk) for tok in tokenize(text)[:max_tokens]]

    def __len__(self) -> int:
        return len(self.itoi)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itoi, f)

    @classmethod
    def load(cls, path: str) -> TextVocab:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


class SentimentDataset(Dataset):
    """(token ids, length, label) triples for the BiLSTM."""

    def __init__(self, df: pd.DataFrame, vocab: TextVocab, max_tokens: int):
        self._label2id = {"negative": 0, "positive": 1}
        self.encoded = [vocab.encode(t, max_tokens) for t in df["text"]]
        self.labels = [self._label2id[lbl] for lbl in df["label"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        ids = self.encoded[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids), torch.tensor(self.labels[idx])


def collate(batch: list[tuple[torch.Tensor, int, torch.Tensor]]) -> dict:
    """Pad token-id sequences to the longest item in the batch.

    Returns inputs sorted by descending length (required by pack_padded_sequence
    when enforce_sorted=True, the faster path we use in training).

    Because that sort reorders the batch, ``"order"`` carries the permutation:
    row ``j`` of the returned tensors is item ``order[j]`` of the incoming
    batch. Evaluation uses it (via ``restore_order``) to hand back logits in
    dataset order - the benchmark caches them against the test parquet rows,
    so emitting them length-sorted would silently misalign every prediction.
    """
    order = sorted(range(len(batch)), key=lambda i: batch[i][1], reverse=True)
    ids_list, lengths, labels = zip(*(batch[i] for i in order), strict=False)
    max_len = lengths[0]
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)  # 0 == <pad>
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = ids
    return {
        "input_ids": padded,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "order": torch.tensor(order, dtype=torch.long),
    }


def restore_order(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Undo the length-sort ``collate`` applied: inverse-permute ``values``."""
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order), device=order.device)
    return values[inverse]


def build_bilstm_datasets(
    processed_dir: str, max_tokens: int, min_freq: int = 2
) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset, TextVocab]:
    """Load parquet splits, build the vocab on train only (no leakage)."""
    frames = {}
    for split in ("train", "dev", "test"):
        path = os.path.join(processed_dir, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run the preprocess step first")
        frames[split] = pd.read_parquet(path)

    vocab = TextVocab.build(frames["train"]["text"], min_freq=min_freq)
    ds = {split: SentimentDataset(df, vocab, max_tokens) for split, df in frames.items()}
    return ds["train"], ds["dev"], ds["test"], vocab
