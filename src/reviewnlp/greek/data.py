"""Dataset loading and preprocessing for the Greek sentiment extension.

Source: DGurgurov/greek_sa — the Tsakalidis et al. (2018) Greek sentiment
corpus (political tweets, 2015 election period), MIT-licensed, pinned to one
revision so results stay reproducible.

Domain-transfer framing (stated, not hidden): the only labeled Greek
sentiment corpus of this size on the Hub is Twitter political sentiment, not
hotel reviews. This pipeline trains the Greek encoder on what exists;
evaluation on Greek hotel text is a future step and is never claimed as an
achieved benchmark here.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

GREEK_SA_DATASET = "DGurgurov/greek_sa"
GREEK_SA_REVISION = "767a5a0737311829f43cb4f4c90a609b6da4ed50"
LABEL_NAMES = ("negative", "positive")

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    """Tweet-aware cleaning: drop URLs and @mentions, collapse whitespace.

    Hashtag content is kept for the tokenizer; diacritics and final sigma
    stay untouched — the nlpaueb tokenizer is trained on unnormalised Greek.
    """
    text = str(text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def load_greek_splits(
    dataset_id: str = GREEK_SA_DATASET,
    revision: str = GREEK_SA_REVISION,
) -> dict[str, pd.DataFrame]:
    """Download the pinned dataset; return cleaned train/validation/test frames.

    Each frame has columns: text (cleaned, non-empty), label (0/1),
    label_name (negative/positive). The heavy `datasets` import stays local
    so that importing reviewnlp.greek.data never pulls torch.
    """
    from datasets import load_dataset

    raw = load_dataset(dataset_id, revision=revision)
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        frame = raw[split].to_pandas()[["text", "label"]].copy()
        labels = {int(value) for value in frame["label"].unique()}
        if not labels <= {0, 1}:
            raise ValueError(f"Unexpected labels in {split}: {sorted(labels)}")
        frame["text"] = frame["text"].map(clean_text)
        frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
        frame["label_name"] = frame["label"].map({0: LABEL_NAMES[0], 1: LABEL_NAMES[1]})
        frames[split] = frame
    return frames


def ordered_fingerprint(frame: pd.DataFrame) -> dict:
    """Order-sensitive (text, label) fingerprint — same convention as the
    legacy frozen-split manifest in reviewnlp.utils.experiments."""
    digest = hashlib.sha256()
    for text, label in frame[["text", "label"]].itertuples(index=False, name=None):
        for value in (str(text), str(int(label))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    counts = frame["label"].value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "negative": int(counts.get(0, 0)),
        "positive": int(counts.get(1, 0)),
        "sha256": digest.hexdigest(),
    }


def dataset_manifest(
    frames: dict[str, pd.DataFrame],
    dataset_id: str = GREEK_SA_DATASET,
    revision: str = GREEK_SA_REVISION,
) -> dict:
    return {
        "purpose": (
            "Greek sentiment training: corpus is Twitter political sentiment "
            "(Tsakalidis et al. 2018), not hotel reviews — cross-domain use is "
            "declared, not presented as an in-domain benchmark"
        ),
        "dataset_id": dataset_id,
        "revision": revision,
        "fingerprints": {
            split: ordered_fingerprint(frame) for split, frame in frames.items()
        },
    }