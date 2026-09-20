"""CLI training:
python scripts/train_greek.py --config configs/greek_bert.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewnlp.greek.train import train_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GreekBERT on the configured splits")
    parser.add_argument("--config", default="configs/greek_bert.yaml")
    args = parser.parse_args()

    metrics = train_from_config(args.config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
