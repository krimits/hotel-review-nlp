"""Build the binary sentiment dataset from the Booking.com 515K CSV.

Label rule (documented in data/raw/README.md and DESIGN.md):

    Positive_Review written AND Negative_Review == "No Negative"  -> positive
    Positive_Review == "No Positive" AND Negative_Review written  -> negative
    both written                                                   -> dropped

This avoids inventing a score threshold (e.g. "7+/10 is positive") and keeps
the training signal unambiguous. Every row is used exactly once, with the
single free-text field the guest actually wrote.

The config's `split` decides what the test set measures:

    random  reviews like the training ones (stratified, the default)
    time    later reviews: train on the oldest, test on the newest
    hotel   hotels the model has never seen: no hotel is in two splits

Outputs (parquet, one row per example), in the config's processed_dir:
    train.parquet, dev.parquet, test.parquet
    label_stats.json, data_manifest.json

Run:  python -m reviewnlp.data.preprocess --config configs/baselines.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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


def _review_context(df: pd.DataFrame) -> pd.DataFrame:
    """The hotel and the review date, when the raw file has them."""
    context = {}
    if "Hotel_Name" in df:
        context["hotel"] = df["Hotel_Name"].astype(str).str.strip()
    if "Review_Date" in df:
        context["review_date"] = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    return pd.DataFrame(context, index=df.index)


def extract_labeled_reviews(df: pd.DataFrame, max_chars: int, min_chars: int) -> pd.DataFrame:
    """Turn the two-field Booking schema into (text, label) rows.

    The hotel and the review date come along when the raw file has them. A
    text that appears more than once keeps only its earliest copy, so a later
    test period never holds a review already seen in training.
    """
    pos = df["Positive_Review"].fillna("").astype(str).str.strip()
    neg = df["Negative_Review"].fillna("").astype(str).str.strip()

    is_pos = pos.ne("") & pos.ne(NO_POSITIVE) & neg.eq(NO_NEGATIVE)
    is_neg = pos.eq(NO_POSITIVE) & neg.ne(NO_NEGATIVE) & neg.ne("")

    context = _review_context(df)
    out = pd.concat(
        [
            pd.DataFrame({"text": pos[is_pos], "label": "positive"}).join(context),
            pd.DataFrame({"text": neg[is_neg], "label": "negative"}).join(context),
        ]
    )
    out["text"] = out["text"].map(lambda t: clean_text(t, max_chars))
    out = out[out["text"].str.len() >= min_chars]
    label_counts = out.groupby(out["text"].map(canonical_text))["label"].transform("nunique")
    out = out.loc[label_counts.eq(1)]
    if "review_date" in out:
        out = out.sort_values("review_date", kind="stable")
    out = out.loc[~out["text"].map(canonical_text).duplicated()].reset_index(drop=True)
    out["review_id"] = out["text"].map(review_id)
    return out


SPLIT_POLICIES = {
    "random": "deduplicate canonical review text globally before stratified split",
    "time": "deduplicate keeping the earliest copy; train on the oldest reviews, dev on the next, "
            "test on the newest; no review date is in two splits",
    "hotel": "deduplicate keeping the earliest copy; shuffle hotels with the seed and give whole "
             "hotels to test, then dev, then train; no hotel is in two splits",
}


def split_and_cap(
    df: pd.DataFrame,
    seed: int,
    train_frac: float,
    dev_frac: float,
    train_cap: int,
    test_cap: int,
    policy: str = "random",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split with the configured fractions and policy, then apply per-class caps.

    random: stratified. The test split is capped first (it must stay fixed
    across all model comparisons), then dev is sampled from the remainder and
    train is capped. time and hotel: the fractions are shares of reviews; the
    test and train splits are then capped by sampling within them, so no
    review crosses a date or hotel boundary. A zero cap means unlimited.
    Each class must appear in all three splits.
    """
    if not (0 < train_frac < 1 and 0 < dev_frac < 1 and train_frac + dev_frac < 1):
        raise ValueError("train/dev fractions must be positive and sum to less than one")
    if train_cap < 0 or test_cap < 0:
        raise ValueError("split caps must be non-negative; zero means unlimited")
    if policy not in SPLIT_POLICIES:
        raise ValueError(f"unknown split policy {policy!r}; use one of {sorted(SPLIT_POLICIES)}")
    if policy != "random":
        splitter = _split_by_time if policy == "time" else _split_by_hotel
        train_df, dev_df, test_df = splitter(df, seed, train_frac, dev_frac)
        return _cap_per_class(train_df, train_cap, seed), dev_df, _cap_per_class(test_df, test_cap, seed)
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


def _split_by_time(df: pd.DataFrame, seed: int, train_frac: float, dev_frac: float):
    """Oldest reviews to train, the next to dev, the newest to test; a date never straddles two splits."""
    if "review_date" not in df:
        raise ValueError("a time split needs the review date")
    ordered = df.sort_values("review_date", kind="stable").reset_index(drop=True)
    dates = ordered["review_date"]
    dev_start = dates.iloc[int(len(ordered) * train_frac)]
    test_start = dates.iloc[int(len(ordered) * (train_frac + dev_frac))]
    if not dev_start < test_start:
        raise ValueError("too few distinct review dates for a time split")
    parts = (dates < dev_start, (dates >= dev_start) & (dates < test_start), dates >= test_start)
    return tuple(ordered[part].reset_index(drop=True) for part in parts)


def _split_by_hotel(df: pd.DataFrame, seed: int, train_frac: float, dev_frac: float):
    """Whole hotels, shuffled with the seed, fill test, then dev; the rest is train."""
    if "hotel" not in df:
        raise ValueError("a hotel split needs the hotel name")
    sizes = df["hotel"].value_counts()
    hotels = sorted(sizes.index)
    random.Random(seed).shuffle(hotels)
    targets = {"test": len(df) * (1 - train_frac - dev_frac), "dev": len(df) * dev_frac}
    filled = {"test": 0, "dev": 0}
    side = {}
    for hotel in hotels:
        name = next((split for split in ("test", "dev") if filled[split] < targets[split]), "train")
        side[hotel] = name
        if name != "train":
            filled[name] += int(sizes[hotel])
    assigned = df["hotel"].map(side)
    return tuple(df[assigned == name].reset_index(drop=True) for name in ("train", "dev", "test"))


def _cap_per_class(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0:
        return df
    parts = [grp.sample(n=min(cap, len(grp)), random_state=seed) for _, grp in df.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_dataset(config_path: str) -> dict:
    # Keep the experiment import local: experiments imports data.integrity,
    # and importing the data package initializes this module.
    from reviewnlp.utils.experiments import assert_clean_splits, frame_fingerprint

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

    policy = d.get("split", "random")
    train_df, dev_df, test_df = split_and_cap(
        labeled,
        seed=cfg["seed"],
        train_frac=d["train_frac"],
        dev_frac=d["dev_frac"],
        train_cap=d["train_cap"],
        test_cap=d["test_cap"],
        policy=policy,
    )

    frames = {"train": train_df, "dev": dev_df, "test": test_df}
    assert_clean_splits(frames)
    if policy == "hotel" and _shared(frames, "hotel"):
        raise ValueError("a hotel appears in more than one split")

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

    # A new split is a new experiment. Keep its provenance next to the parquet
    # files so a cached prediction cannot silently be compared to another test.
    digest = hashlib.sha256()
    with open(raw_csv, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "schema_version": 2,
        "raw_csv_sha256": digest.hexdigest(),
        "raw_csv_bytes": os.path.getsize(raw_csv),
        "config": {"seed": cfg["seed"], **d},
        "split": policy,
        "split_policy": SPLIT_POLICIES[policy],
        "splits": {split: frame_fingerprint(frame) for split, frame in frames.items()},
        "cross_split_overlap": {"train_dev": 0, "train_test": 0, "dev_test": 0},
    }
    if "review_date" in labeled:
        manifest["review_dates"] = {split: [frame["review_date"].min(), frame["review_date"].max()]
                                    for split, frame in frames.items()}
    if "hotel" in labeled:
        manifest["hotels"] = {split: int(frame["hotel"].nunique()) for split, frame in frames.items()}
        manifest["hotels_in_more_than_one_split"] = len(_shared(frames, "hotel"))
    with open(os.path.join(d["processed_dir"], "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(stats, indent=2))
    return stats


def _shared(frames: dict[str, pd.DataFrame], column: str) -> set:
    """Values of `column` that occur in more than one split."""
    seen: dict[object, str] = {}
    shared = set()
    for split, frame in frames.items():
        for value in frame[column].unique():
            if seen.setdefault(value, split) != split:
                shared.add(value)
    return shared


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
