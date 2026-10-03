"""Normalize unambiguous human-rating tokens; export partial judgments for adjudication, never freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from reviewnlp.triage.generator_review import audit_ratings, write_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="intact, extracted development ZIP")
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "runs"
    output = args.output.resolve()
    if not output.is_relative_to(root.resolve()) or output == root.resolve():
        parser.error("--output must be a new folder inside runs/")
    report, sheet = audit_ratings(args.run, args.ratings)
    write_audit(output, report, sheet)
    print("Outputs:", report["outputs"], "Pending judgments:", len(report["pending_judgments"]))
    print("Original execution fields restored in the separate sheet:", len(report["execution_issue_overwrites"]))
    print("This audit selects no candidate and does not open the reserved set.")
    return 0 if report["ready_for_freeze"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
