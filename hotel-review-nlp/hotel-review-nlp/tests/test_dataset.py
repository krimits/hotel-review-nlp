"""Vocab / collate plumbing for the BiLSTM - offline, no dataset needed."""

from __future__ import annotations

import pandas as pd
import torch

from reviewnlp.data.dataset import (
    PAD,
    SentimentDataset,
    TextVocab,
    collate,
    restore_order,
    tokenize,
)


def _tiny_vocab() -> TextVocab:
    texts = ["great room great staff", "dirty room bad bed", "great bed"]
    return TextVocab.build(texts, min_freq=1)


def test_vocab_reserves_pad_and_unk():
    vocab = _tiny_vocab()
    assert vocab.itoi[PAD] == 0
    assert vocab.encode("completely unseen wording", max_tokens=8)[0] == vocab.itoi["<unk>"]


def test_encode_truncates_to_max_tokens():
    vocab = _tiny_vocab()
    assert len(vocab.encode("great room great staff bed", max_tokens=3)) == 3


def test_collate_sorts_by_length_and_pads():
    vocab = _tiny_vocab()
    df = pd.DataFrame(
        {
            "text": ["great bed", "great room great staff", "dirty room bad bed"],
            "label": ["positive", "positive", "negative"],
        }
    )
    ds = SentimentDataset(df, vocab, 32)
    batch = collate([ds[i] for i in range(len(ds))])

    lengths = batch["lengths"].tolist()
    assert lengths == sorted(lengths, reverse=True), "pack_padded_sequence needs desc order"
    assert batch["input_ids"].shape == (3, lengths[0])
    # rows shorter than the longest are pad-filled
    assert batch["input_ids"][-1, lengths[-1] :].eq(0).all()


def test_restore_order_undoes_the_length_sort():
    """Cached logits are matched against test.parquet row-by-row.

    collate reorders each batch by length, so evaluation must invert that
    permutation - otherwise every cached prediction lands on the wrong review.
    """
    vocab = _tiny_vocab()
    df = pd.DataFrame(
        {
            "text": ["great bed", "great room great staff", "dirty room bad bed"],
            "label": ["positive", "negative", "negative"],
        }
    )
    ds = SentimentDataset(df, vocab, 32)
    batch = collate([ds[i] for i in range(len(ds))])

    # labels came back length-sorted; restoring must recover the dataset order
    restored = restore_order(batch["labels"], batch["order"])
    assert restored.tolist() == [1, 0, 0]

    # and the permutation is a true inverse for arbitrary per-row payloads
    payload = torch.arange(3).unsqueeze(1).float()
    sorted_payload = payload[batch["order"]]
    assert torch.equal(restore_order(sorted_payload, batch["order"]), payload)


def test_tokenizer_drops_punctuation():
    assert tokenize("Great hotel!! Really, lovely.") == ["great", "hotel", "really", "lovely"]


def test_encode_never_returns_an_empty_sequence():
    """pack_padded_sequence raises on length 0, so punctuation-only rows
    must still yield one token."""
    vocab = _tiny_vocab()
    assert vocab.encode("!!! ??? ...", max_tokens=8) == [vocab.itoi["<unk>"]]


def test_collate_handles_a_punctuation_only_review():
    vocab = _tiny_vocab()
    df = pd.DataFrame({"text": ["great room great staff", "!!!"], "label": ["positive", "negative"]})
    ds = SentimentDataset(df, vocab, 32)
    batch = collate([ds[i] for i in range(len(ds))])
    assert batch["lengths"].min().item() >= 1
