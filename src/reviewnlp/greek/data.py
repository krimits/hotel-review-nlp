"""Dataset loading, splitting and leakage control for the Greek extension.

Source: DGurgurov/greek_sa — the Tsakalidis et al. (2018) Greek sentiment
corpus (political tweets, 2015 election period), MIT-licensed, pinned to one
revision so results stay reproducible.

Domain-transfer framing (stated, not hidden): the only labeled Greek
sentiment corpus of this size on the Hub is Twitter political sentiment, not
hotel reviews. This pipeline trains the Greek encoder on what exists;
evaluation on Greek hotel text is a future step and is never claimed as an
achieved benchmark here.

Splitting: the upstream splits are 5,936 / 383 / 767 (83.8 / 5.4 / 10.8 %),
which is neither the ratio the config asks for nor a split we control. Since
`duplicate_mode` and `overlap_policy` can only be enforced by whoever draws
the boundary, the pooled rows are re-split here using `test_size`, `val_size`
and `seed`. Near-duplicate tweets are common in this corpus (retweet-like
repetition), so rows sharing a canonical form are assigned as one group and
the result is checked for cross-split overlap before it is returned — the
same discipline the English pipeline applies in reviewnlp.data.preprocess.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import warnings

import pandas as pd

from reviewnlp.data.integrity import canonical_text
from reviewnlp.greek.config import DataConfig

GREEK_SA_DATASET = "DGurgurov/greek_sa"
GREEK_SA_REVISION = "767a5a0737311829f43cb4f4c90a609b6da4ed50"
LABEL_NAMES = ("negative", "positive")
SPLIT_NAMES = ("train", "validation", "test")

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


def strip_diacritics(text: str) -> str:
    """Drop combining marks: 'τέλος' and 'τελος' become one string."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(char for char in decomposed if not unicodedata.combining(char))
    )


def group_key(text: str) -> str:
    """Identity shared by rows the *model* cannot tell apart.

    Starts from the English pipeline's canonical form (NFKC, collapsed
    whitespace, casefold — which also folds final sigma) and additionally
    strips diacritics, because the target encoder is
    bert-base-greek-uncased-v1: its own preprocessing removes accents, so
    'ΚΑΛΟ ΣΧΟΛΙΟ' and 'καλό σχόλιο' reach it as the same token sequence.
    Grouping on the unaccented form is what stops those pairs from being
    split across train and test and quietly inflating the score.
    """
    folded = strip_diacritics(canonical_text(text))
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()


def apply_duplicate_mode(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Resolve repeated texts according to `duplicate_mode`.

    group — keep every row; rows sharing a key are split together (below).
    drop  — keep the first row of each key, discard the rest.
    keep  — treat every row independently, duplicates may cross splits.
    """
    if mode == "drop":
        return frame.loc[~frame["group_key"].duplicated()].reset_index(drop=True)
    return frame.reset_index(drop=True)


def assign_splits(frame: pd.DataFrame, config: DataConfig) -> dict[str, pd.DataFrame]:
    """Deterministically split into train/validation/test. Pure: no network.

    Splitting happens over *groups*, not rows, so duplicated texts never
    straddle a boundary under `duplicate_mode: group`. Groups are bucketed by
    their majority label first, so the label balance of each split tracks the
    corpus even though whole groups move together.
    """
    if frame.empty:
        raise ValueError("cannot split an empty frame")

    working = frame.copy()
    if config.duplicate_mode == "keep":
        # Every row is its own group: duplicates are allowed to separate.
        working["_group"] = working.index.astype(str)
    else:
        working["_group"] = working["group_key"]

    # One stratum per group, from the label most of its rows carry.
    group_label = working.groupby("_group")["label"].agg(lambda s: int(s.mode().iat[0]))

    assignment: dict[str, str] = {}
    for label in sorted(group_label.unique()):
        groups = sorted(group_label.index[group_label == label])
        shuffled = (
            pd.Series(groups)
            .sample(frac=1.0, random_state=config.seed + int(label))
            .tolist()
        )
        total = len(shuffled)
        n_test = int(round(total * config.test_size))
        n_val = int(round(total * config.val_size))
        # Training keeps the remainder, so rounding never overdraws the corpus.
        n_train = total - n_test - n_val
        if n_train <= 0:
            raise ValueError(
                f"label {label}: {total} group(s) cannot fill a "
                f"{config.train_size:.0%}/{config.val_size:.0%}/{config.test_size:.0%} split"
            )
        for offset, group in enumerate(shuffled):
            if offset < n_train:
                assignment[group] = "train"
            elif offset < n_train + n_val:
                assignment[group] = "validation"
            else:
                assignment[group] = "test"

    working["split"] = working["_group"].map(assignment)
    frames = {
        name: working[working["split"] == name]
        .drop(columns=["_group", "split"])
        .reset_index(drop=True)
        for name in SPLIT_NAMES
    }

    # Rounding can starve a split on a small corpus. An empty validation set
    # only surfaces much later, as an opaque Trainer error, so refuse here.
    empty = [name for name, frame in frames.items() if frame.empty]
    if empty:
        raise ValueError(
            f"{len(working)} row(s) in {working['_group'].nunique()} group(s) cannot fill a "
            f"{config.train_size:.0%}/{config.val_size:.0%}/{config.test_size:.0%} split: "
            f"{sorted(empty)} came out empty"
        )
    return frames


def split_overlap(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Count canonical texts shared between each pair of splits."""
    keys = {name: set(frame["group_key"]) for name, frame in frames.items()}
    return {
        f"{a}_{b}": len(keys[a] & keys[b])
        for index, a in enumerate(SPLIT_NAMES)
        for b in SPLIT_NAMES[index + 1 :]
    }


def enforce_overlap_policy(frames: dict[str, pd.DataFrame], policy: str) -> dict[str, int]:
    """Apply `overlap_policy` to the measured cross-split overlap."""
    overlap = split_overlap(frames)
    total = sum(overlap.values())
    if total and policy != "ignore":
        message = f"cross-split text overlap: {overlap}"
        if policy == "raise":
            raise ValueError(
                f"{message}. Set duplicate_mode to 'group' or 'drop', or relax "
                f"overlap_policy, to proceed."
            )
        warnings.warn(message, stacklevel=2)
    return overlap


def build_frame(rows: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Normalise raw rows into the columns the rest of the pipeline expects."""
    missing = {config.text_column, config.label_column} - set(rows.columns)
    if missing:
        raise ValueError(f"dataset is missing column(s) {sorted(missing)}")

    frame = rows[[config.text_column, config.label_column]].copy()
    frame.columns = ["text", "label"]
    labels = {int(value) for value in frame["label"].unique()}
    allowed = set(range(len(LABEL_NAMES)))
    if not labels <= allowed:
        raise ValueError(f"unexpected labels {sorted(labels)}, expected a subset of {sorted(allowed)}")

    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)
    frame["label_name"] = frame["label"].map(dict(enumerate(LABEL_NAMES)))
    frame["group_key"] = frame["text"].map(group_key)
    return frame


def load_greek_splits(config: DataConfig) -> dict[str, pd.DataFrame]:
    """Download the pinned dataset, pool every split, then re-split per config.

    The heavy `datasets` import stays local so importing this module never
    pulls torch — the offline tests exercise the splitting logic directly.
    """
    from datasets import load_dataset

    raw = load_dataset(config.dataset_name, revision=config.dataset_revision)
    pooled = pd.concat(
        [raw[split].to_pandas() for split in raw],
        ignore_index=True,
    )
    frame = build_frame(pooled, config)
    frame = apply_duplicate_mode(frame, config.duplicate_mode)
    frames = assign_splits(frame, config)
    enforce_overlap_policy(frames, config.overlap_policy)
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


def dataset_manifest(frames: dict[str, pd.DataFrame], config: DataConfig) -> dict:
    """Everything needed to tell two runs of this pipeline apart."""
    return {
        "purpose": (
            "Greek sentiment training: corpus is Twitter political sentiment "
            "(Tsakalidis et al. 2018), not hotel reviews — cross-domain use is "
            "declared, not presented as an in-domain benchmark"
        ),
        "dataset_id": config.dataset_name,
        "revision": config.dataset_revision,
        "split_policy": {
            "source": "upstream splits pooled, then re-split by this config",
            "train_size": round(config.train_size, 6),
            "val_size": config.val_size,
            "test_size": config.test_size,
            "seed": config.seed,
            "duplicate_mode": config.duplicate_mode,
            "overlap_policy": config.overlap_policy,
            "grouping": "sha256 of reviewnlp.data.integrity.canonical_text",
        },
        "cross_split_overlap": split_overlap(frames),
        "fingerprints": {
            split: ordered_fingerprint(frame) for split, frame in frames.items()
        },
    }
