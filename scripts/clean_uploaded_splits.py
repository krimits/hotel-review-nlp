"""Make independently identified, clean splits from uploaded legacy parquets.

The original processed files remain untouched. This path documents only the
provenance available from the uploaded parquets; it cannot reconstruct the
missing raw Booking CSV or the original row sampling decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewnlp.data.integrity import canonical_text  # noqa: E402
from reviewnlp.utils.experiments import SPLITS, assert_clean_splits, frame_fingerprint  # noqa: E402


def clean_uploaded_splits(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.exists():
        raise ValueError("output must be a new directory, separate from the uploaded source")
    cleaned = {}
    provenance = {}
    for split in SPLITS:
        path = source / f"{split}.parquet"
        frame = pd.read_parquet(path).reset_index(drop=True)
        frame_fingerprint(frame)  # reject malformed labels or missing columns
        keys = frame["text"].map(canonical_text)
        contradictory = frame.groupby(keys)["label"].transform("nunique").gt(1)
        unambiguous = frame.loc[~contradictory].copy()
        normalized = unambiguous["text"].map(canonical_text)
        duplicate = normalized.duplicated(keep="first")
        cleaned[split] = unambiguous.loc[~duplicate].reset_index(drop=True)
        provenance[split] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_rows": len(frame),
            "conflicting_label_groups": int(keys[contradictory].nunique()),
            "conflicting_rows_removed": int(contradictory.sum()),
            "duplicate_rows_removed": int(duplicate.sum()),
        }

    assert_clean_splits(cleaned)
    manifest = {
        "schema_version": 3,
        "source_type": "uploaded_processed_parquets_without_raw_csv",
        "source_files_sha256": {name: details["sha256"] for name, details in provenance.items()},
        "source_audit": provenance,
        "split_policy": "preserve the original split assignment and row order; drop conflicting labels, then keep the first copy of a normalized review within each split",
        "splits": {name: frame_fingerprint(frame) for name, frame in cleaned.items()},
        "cross_split_overlap": {"train_dev": 0, "train_test": 0, "dev_test": 0},
        "raw_csv_provenance_available": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clean-splits-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        for split, frame in cleaned.items():
            frame.to_parquet(stage / f"{split}.parquet", index=False)
        (stage / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # A complete directory becomes visible in one operation.
        os.replace(stage, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed_clean"))
    args = parser.parse_args()
    manifest = clean_uploaded_splits(args.source, args.output)
    print(json.dumps({"source_audit": manifest["source_audit"],
                      "clean_splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
