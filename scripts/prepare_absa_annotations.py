"""Create a deterministic human-annotation template from the frozen test parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=25)
    args = parser.parse_args()
    if args.per_class <= 0:
        parser.error("--per-class must be positive")
    frame = pd.read_parquet(args.test_parquet).reset_index().rename(columns={"index": "row_id"})
    if not {"text", "label"} <= set(frame):
        parser.error("test parquet needs text and label columns")
    parts = [frame[frame["label"].astype(str).str.lower().isin(names)].head(args.per_class)
             for names in ({"positive", "1"}, {"negative", "0"})]
    if any(len(part) != args.per_class for part in parts):
        parser.error("the test split has too few examples of one class")
    sampled = pd.concat(parts).sort_values("row_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sampled.itertuples(index=False):
            handle.write(json.dumps({"row_id": int(row.row_id), "text": str(row.text),
                                     "status": "unlabeled", "aspects": []}, ensure_ascii=False) + "\n")
    print("Unlabeled template written:", args.output)
    print("Frozen source SHA256:", hashlib.sha256(args.test_parquet.read_bytes()).hexdigest())
    print("Have two reviewers label independently, reconcile disagreements, then set status=annotated.")


if __name__ == "__main__":
    main()
