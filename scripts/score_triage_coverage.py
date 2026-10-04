"""Score a completed paired development run's human coverage counts without selecting a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reviewnlp.triage.workflow_review import score_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_coverage(args.run, args.ratings)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Coverage scored. No selection frozen; final evaluation has not been opened.")


if __name__ == "__main__":
    main()
