"""CLI inference:
python scripts/predict_greek.py --model-dir runs/greek_bert \
  --text "Εξαιρετικό δωμάτιο και εξυπηρετικό προσωπικό"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewnlp.greek.predict import predict_texts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Greek sentiment inference")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", action="append", default=[], help="Repeatable")
    parser.add_argument("--file", help="One review per line")
    parser.add_argument("--max-length", type=int, default=160)
    args = parser.parse_args()

    texts = list(args.text)
    if args.file:
        texts.extend(
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not texts:
        parser.error("provide --text or --file")

    predictions = predict_texts(texts, args.model_dir, max_length=args.max_length)
    print(json.dumps(
        [
            {"text": text, "sentiment": label}
            for text, label in zip(texts, predictions, strict=True)
        ],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
