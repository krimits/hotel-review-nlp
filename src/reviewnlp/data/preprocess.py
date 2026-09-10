"""Build the binary sentiment dataset from the Booking.com 515K CSV.

Label rule (documented in data/raw/README.md and DESIGN.md):

    Positive_Review written AND Negative_Review == "No Negative"  -> positive
    Positive_Review == "No Positive" AND Negative_Review written  -> negative
    both written                                                   -> dropped

This avoids inventing a score threshold (e.g. "7+/10 is positive") and keeps
the training signal unambiguous. Every row is used exactly once, with the
single free-text field the guest actually wrote.

Outputs (parquet, one row per example):
    data/processed/train.parquet
    data/processed/dev.parquet
    data/processed/test.parquet
    data/processed/label_stats.json

Run:  python -m reviewnlp.data.preprocess --config configs/baselines.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

from reviewnlp.data.integrity import canonical_text, review_id
from reviewnlp.utils.seed import load_config, set_seed

NO_POSITIVE = "No Positive"
NO_NEGATIVE = "No Negative"

_TAG_RE = re.compile(r"</?\s*\w+\s*/?>")  # stray html-ish tags in reviews
_WS_RE = re.compile(r"\s+")


def clean_text(text: str, max_chars: int) -> str:
    """Light cleaning: strip tags, collapse whitespace, truncate."""
    text = _TAG_RE.sub(" ", str(text))
    text = _WS_RE.sub(" ", text).strip()
    return text[:max_chars]


def extract_labeled_reviews(df: pd.DataFrame, max_chars: int, min_chars: int) -> pd.DataFrame:
    """Turn the two-field Booking schema into (text, label) rows."""
    pos = df["Positive_Review"].fillna("").astype(str).str.strip()
    neg = df["Negative_Review"].fillna("").astype(str).str.strip()

    is_pos = pos.ne("") & pos.ne(NO_POSITIVE) & neg.eq(NO_NEGATIVE)
    is_neg = pos.eq(NO_POSITIVE) & neg.ne(NO_NEGATIVE) & neg.ne("")

    out = pd.concat(
        [
            pd.DataFrame({"text": pos[is_pos], "label": "positive"}),
            pd.DataFrame({"text": neg[is_neg], "label": "negative"}),
        ]
    )
    out["text"] = out["text"].map(lambda t: clean_text(t, max_chars))
    out = out[out["text"].str.len() >= min_chars]
    label_counts = out.groupby(out["text"].map(canonical_text))["label"].transform("nunique")
    out = out.loc[label_counts.eq(1)]
    out = out.loc[~out["text"].map(canonical_text).duplicated()].reset_index(drop=True)
    out["review_id"] = out["text"].map(review_id)
    return out


def split_and_cap(
    df: pd.DataFrame,
    seed: int,
    train_frac: float,
    dev_frac: float,
    train_cap: int,
    test_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split using configured fractions, then per-class caps.

    The test split is capped first (it must stay fixed across all model
    comparisons), then dev is sampled from the remainder and train is capped.
    A zero cap means unlimited. Each class must appear in all three splits.
    """
    if not (0 < train_frac < 1 and 0 < dev_frac < 1 and train_frac + dev_frac < 1):
        raise ValueError("train/dev fractions must be positive and sum to less than one")
    if train_cap < 0 or test_cap < 0:
        raise ValueError("split caps must be non-negative; zero means unlimited")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_parts, rest_parts = [], []
    for _, grp in df.groupby("label"):
        grp = grp.reset_index(drop=True)
        test_frac = round(1.0 - train_frac - dev_frac, 10)
        test_n = min(len(grp) - 2, max(1, int(len(grp) * test_frac)))
        if test_cap > 0:
            test_n = min(test_n, test_cap)
        test_parts.append(grp.iloc[:test_n])
        rest_parts.append(grp.iloc[test_n:])

    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rest_df = pd.concat(rest_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_parts, dev_parts = [], []
    for _, grp in rest_df.groupby("label"):
        grp = grp.reset_index(drop=True)
        # dev gets dev_frac/(train_frac+dev_frac) of the post-test remainder,
        # keeping dev and test statistically comparable
        dev_n = min(max(1, int(len(grp) * dev_frac / (train_frac + dev_frac))), len(grp) - 1)
        dev_parts.append(grp.iloc[:dev_n])
        train_parts.append(grp.iloc[dev_n:])

    dev_df = pd.concat(dev_parts).reset_index(drop=True)
    train_df = pd.concat(train_parts).reset_index(drop=True)

    train_df = _cap_per_class(train_df, train_cap, seed)
    return train_df, dev_df, test_df


def _cap_per_class(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0:
        return df
    parts = [grp.sample(n=min(cap, len(grp)), random_state=seed) for _, grp in df.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_dataset(config_path: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    d = cfg["data"]
    raw_csv = d["raw_csv"]
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(
            f"{raw_csv} not found. Download the Booking.com 515K dataset first - "
            "see data/raw/README.md"
        )

    print(f"Loading {raw_csv} ...")
    raw = pd.read_csv(raw_csv)
    labeled = extract_labeled_reviews(raw, d["max_chars"], d["min_chars"])
    print(f"Labeled (unambiguous) reviews: {len(labeled):,}")

    train_df, dev_df, test_df = split_and_cap(
        labeled,
        seed=cfg["seed"],
        train_frac=d["train_frac"],
        dev_frac=d["dev_frac"],
        train_cap=d["train_cap"],
        test_cap=d["test_cap"],
    )

    os.makedirs(d["processed_dir"], exist_ok=True)
    train_df.to_parquet(os.path.join(d["processed_dir"], "train.parquet"))
    dev_df.to_parquet(os.path.join(d["processed_dir"], "dev.parquet"))
    test_df.to_parquet(os.path.join(d["processed_dir"], "test.parquet"))

    stats = {
        split: {
            "rows": int(len(df)),
            "positive": int((df["label"] == "positive").sum()),
            "negative": int((df["label"] == "negative").sum()),
            "avg_chars": float(df["text"].str.len().mean()),
        }
        for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
    }
    with open(os.path.join(d["processed_dir"], "label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    return stats


def load_processed(processed_dir: str) -> dict[str, pd.DataFrame]:
    """Load the three processed splits as {'train': df, 'dev': df, 'test': df}."""
    out = {}
    for split in ("train", "dev", "test"):
        path = os.path.join(processed_dir, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split {path} - run the preprocess step first")
        out[split] = pd.read_parquet(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    build_dataset(args.config)


if __name__ == "__main__":
    main()
