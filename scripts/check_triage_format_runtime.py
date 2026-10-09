"""Write a model-free preflight receipt; refuse incompatible or non-isolated Colab runtimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp.triage.format_runtime import RuntimePreflightError, check_runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new_preflight_receipt_required")
    try:
        report = check_runtime(expected_prefix=args.expected_prefix, require_cuda=args.require_cuda)
    except RuntimePreflightError as error:
        report = {"runtime_ready": False, "error": error.code, "details": error.details,
                  "model_loaded": False, "hub_requests": False}
    except Exception as error:
        report = {"runtime_ready": False, "error": "preflight_failed", "error_type": type(error).__name__,
                  "model_loaded": False, "hub_requests": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["runtime_ready"] else 1)


if __name__ == "__main__":
    main()
