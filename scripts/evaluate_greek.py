"""CLI evaluation:
python scripts/evaluate_greek.py --config configs/greek_bert.yaml --model-dir runs/greek_bert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewnlp.greek.evaluate import evaluate_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Greek sentiment checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    report = evaluate_checkpoint(
        model_dir=args.model_dir,
        config_path=args.config,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
