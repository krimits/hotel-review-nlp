"""Reproducible split preparation and fingerprints for GPU experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "dev", "test")
LABELS = ("negative", "positive")


def frame_fingerprint(frame: pd.DataFrame) -> dict:
    """Return an order-sensitive, serialization-independent split fingerprint."""
    _validate_frame(frame)
    digest = hashlib.sha256()
    for text, label in frame[["text", "label"]].itertuples(index=False, name=None):
        for value in (str(text), str(label)):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big"))
            digest.update(encoded)

    counts = frame["label"].value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "negative": int(counts.get("negative", 0)),
        "positive": int(counts.get("positive", 0)),
        "sha256": digest.hexdigest(),
    }


def fingerprint_splits(processed_dir: str | Path) -> dict:
    """Fingerprint the ordered train/dev/test parquet files in a directory."""
    directory = Path(processed_dir)
    return {
        split: frame_fingerprint(pd.read_parquet(directory / f"{split}.parquet"))
        for split in SPLITS
    }


def prepare_experiment_splits(
    source_dir: str | Path,
    destination_dir: str | Path,
    caps: dict[str, int | None],
    seed: int,
) -> dict:
    """Create deterministic, stratified experiment splits from frozen source splits.

    ``None`` keeps a complete split. Integer caps select an order-stable,
    stratified subset. Source files are never modified and test labels are not
    used for model or hyperparameter selection.
    """
    source = Path(source_dir)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    unknown = set(caps) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown split caps: {sorted(unknown)}")

    source_frames: dict[str, pd.DataFrame] = {}
    experiment_frames: dict[str, pd.DataFrame] = {}
    for split_index, split in enumerate(SPLITS):
        path = source / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path).reset_index(drop=True)
        _validate_frame(frame)
        source_frames[split] = frame

        cap = caps.get(split)
        if cap is not None and cap <= 0:
            raise ValueError(f"{split} cap must be positive or None")
        selected = _stratified_cap(frame, cap, seed + split_index)
        selected.to_parquet(destination / f"{split}.parquet", index=False)
        experiment_frames[split] = selected

    _validate_no_cross_split_overlap(source_frames)

    manifest = {
        "seed": int(seed),
        "caps": {split: caps.get(split) for split in SPLITS},
        "source_splits": {
            split: frame_fingerprint(frame) for split, frame in source_frames.items()
        },
        "splits": {
            split: frame_fingerprint(frame) for split, frame in experiment_frames.items()
        },
    }
    manifest_path = destination.parent / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _stratified_cap(frame: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or cap >= len(frame):
        return frame.copy()

    counts = frame["label"].value_counts()
    negative_count = int(round(cap * counts["negative"] / len(frame)))
    negative_count = min(max(1, negative_count), int(counts["negative"]))
    positive_count = cap - negative_count
    if positive_count <= 0 or positive_count > int(counts["positive"]):
        raise ValueError(f"cannot draw a stratified cap of {cap} rows")

    selected_indices = []
    for offset, (label, sample_size) in enumerate(
        (("negative", negative_count), ("positive", positive_count))
    ):
        sampled = frame.index[frame["label"] == label].to_series().sample(
            n=sample_size,
            random_state=seed + offset,
            replace=False,
        )
        selected_indices.extend(sampled.tolist())
    return frame.loc[sorted(selected_indices)].reset_index(drop=True)


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[["text", "label"]].isna().any().any():
        raise ValueError("text and label columns must not contain nulls")
    if frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("reviews must not be empty")
    observed = set(frame["label"].astype(str).unique())
    if observed != set(LABELS):
        raise ValueError(f"expected labels {LABELS}, got {sorted(observed)}")


def _validate_no_cross_split_overlap(frames: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        normalized = frames[split]["text"].astype(str).str.strip().str.casefold()
        for text in normalized:
            previous = seen.get(text)
            if previous is not None and previous != split:
                raise ValueError(f"review text overlaps between {previous} and {split}")
            seen[text] = split
