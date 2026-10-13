"""Apply adjudication decisions to unreconciled paired rating sheets; refresh the manifest; never promote.

Reads the captured run, the two unreconciled completed sheets, and a decisions CSV
holding one row per disagreement (blind_id, field, agreed_value, adjudication_notes).
Produces reconciled v3 sheets, the filled disagreement record, the scored reports before
and after adjudication, and a manifest whose hashes cover the final files *and* pin the
inputs they came from (the capture, the two unreconciled sheets, the decisions CSV).
Inputs are pinned by hash rather than copied, so the output directory stays small; the
hashes are what prove which sheets this adjudication started from.

Semantics:
- Sheets are copied unchanged except for the adjudicated fields and their notes. A cell
  whose agreed value equals the annotator's keeps its note; a changed cell's note
  becomes "Μετά τη συζήτηση: <field> <old>→<new>. <decision note>" (joined with " | "
  when one review carries several decisions).
- The disagreement record keeps every column of the supplied discussion sheet and only
  fills agreed_value/adjudication_notes.
- The original capture and the unreconciled sheets are never modified. No selection is
  frozen and no candidate is promoted.

The archived adjudication of 2026-10-06
(docs/experiments/triage_space_publication/e1e37e5/human_review_v3) predates this tool:
its values, disagreement record and scored reports match, while three of its notes carry
hand-merged phrasing this deterministic rule does not reproduce (blinds 0010, 0020 and
0022: the note there summarises the whole decision, whereas this rule reports only the
cell that changed in that sheet). The archive stays the recorded evidence; this script
defines the forward-looking convention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp.triage.generator_experiment import RATINGS  # noqa: E402
from reviewnlp.triage.workflow_review import COUNTS  # noqa: E402
from scripts.score_triage_review import reconcile  # noqa: E402

JUDGED = (*COUNTS, *RATINGS)
SHORT_LABELS = {"actual_problem_count": "actual", "missed_problem_count": "missed",
                "false_exclusion_count": "false_exclusion", "unaddressed_problem_count": "unaddressed",
                "invented_problem_count": "invented", "unnecessary_action_count": "unnecessary",
                "useful_action_count": "useful_action", "wrong_department_count": "wrong_department",
                "no_invented_facts": "nif"}


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_decisions(path: Path, disagreements: list[dict]) -> dict[tuple[str, str], dict]:
    expected = {(item["blind_id"], item["field"]) for item in disagreements}
    rows = read_csv_rows(path)
    seen: dict[tuple[str, str], dict] = {}
    for row in rows:
        identity = (row["blind_id"], row["field"])
        if identity not in expected:
            raise ValueError("decision outside the reported disagreements: " + str(identity))
        if identity in seen:
            raise ValueError("duplicated decision for " + str(identity))
        if not row["adjudication_notes"].strip():
            raise ValueError("adjudication note required for " + str(identity))
        value = row["agreed_value"].strip()
        if row["field"] in RATINGS:
            allowed = {"0", "1"}
        else:
            allowed = {str(n) for n in range(1000)}
        if value not in allowed:
            raise ValueError("agreed_value must be a valid rating or count: " + str(identity))
        seen[identity] = {"value": value, "note": row["adjudication_notes"].strip()}
    missing = expected - set(seen)
    if missing:
        raise ValueError("missing decisions for " + ", ".join(sorted(map(str, missing))))
    return seen


def apply_decisions(sheet_path: Path, decisions: dict[tuple[str, str], dict], target: Path) -> int:
    """Write the sheet with only adjudicated fields and their notes changed; return changed cells."""
    raw = sheet_path.read_text(encoding="utf-8-sig")
    bom = sheet_path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(io.StringIO(raw, newline="")))
    fields = list(rows[0])
    changed = 0
    for row in rows:
        notes = []
        for (blind_id, field), decision in decisions.items():
            if row["blind_id"] != blind_id:
                continue
            if row[field] != decision["value"]:
                changed += 1
                label = SHORT_LABELS.get(field, field)
                notes.append(f"Μετά τη συζήτηση: {label} {row[field]}→{decision['value']}. {decision['note']}")
            row[field] = decision["value"]
        if notes:
            row["notes"] = " | ".join(notes)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    target.write_bytes((b"\xef\xbb\xbf" + data) if bom else data)
    return changed


def write_disagreements(template: Path, decisions: dict[tuple[str, str], dict],
                        target: Path) -> None:
    """Fill agreed_value/adjudication_notes in a copy of the supplied discussion sheet."""
    rows = read_csv_rows(template)
    fields = list(rows[0])
    for row in rows:
        decision = decisions[(row["blind_id"], row["field"])]
        row["agreed_value"] = decision["value"]
        row["adjudication_notes"] = decision["note"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    target.write_bytes(b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8"))


def submitted_adjudications_hold(output: Path, decisions: dict[tuple[str, str], dict]) -> bool:
    """Read the written v3 sheets back and confirm every agreed value landed in both."""
    for name in ("completed_A_v3.csv", "completed_B_v3.csv"):
        rows = {row["blind_id"]: row for row in read_csv_rows(output / name)}
        for (blind_id, field), decision in decisions.items():
            if rows.get(blind_id, {}).get(field) != decision["value"]:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="intact captured run folder")
    parser.add_argument("--ratings-a", type=Path, required=True, help="unreconciled sheet A")
    parser.add_argument("--ratings-b", type=Path, required=True, help="unreconciled sheet B")
    parser.add_argument("--decisions", type=Path, required=True,
                        help="CSV: blind_id, field, agreed_value, adjudication_notes")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--date", default="", help="manifest date stamp (e.g. 2026-10-13)")
    parser.add_argument("--space-commit", default="", help="Space commit recorded in the manifest")
    parser.add_argument("--source-commit", default="", help="source commit recorded in the manifest")
    args = parser.parse_args()

    report = reconcile(args.run, args.ratings_a, args.ratings_b)
    if report["judgments_reconciled"]:
        parser.error("the sheets are already reconciled; nothing to adjudicate")
    decisions = load_decisions(args.decisions, report["disagreements"])

    args.output.mkdir(parents=True, exist_ok=False)
    changed_a = apply_decisions(args.ratings_a, decisions, args.output / "completed_A_v3.csv")
    changed_b = apply_decisions(args.ratings_b, decisions, args.output / "completed_B_v3.csv")
    reconciled = reconcile(args.run, args.output / "completed_A_v3.csv", args.output / "completed_B_v3.csv")
    if not reconciled["judgments_reconciled"]:
        parser.error("the supplied decisions did not reconcile the sheets")
    write_disagreements(args.decisions, decisions,
                        args.output / "triage_review_disagreements.csv")
    for name, payload in (("scored_v2.json", report), ("scored_v3.json", reconciled)):
        (args.output / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "scope": "adjudicated ratings of a captured development snapshot; no production accuracy",
        "date": args.date or None, "space_commit": args.space_commit or None,
        "source_commit": args.source_commit or None,
        "source_run_sha256": reconciled["source_run_sha256"],
        "prior_disagreement_fields": len(report["disagreements"]),
        "prior_disagreement_reviews": len({item["blind_id"] for item in report["disagreements"]}),
        # The three flags below are measured, not asserted: a manifest that merely claims
        # them cannot be used to verify anything.
        "all_submitted_adjudications_match_v3": submitted_adjudications_hold(args.output, decisions),
        "original_capture_unchanged": report["source_run_sha256"] == reconciled["source_run_sha256"],
        "judgments_reconciled": bool(reconciled["judgments_reconciled"]),
        "reviewer_independence_verified": False,
        "reserved_evaluation_used": False,
        "selection_frozen": False,
        "promoted": False,
        # Hashes of what went in: without these the manifest cannot show which
        # unreconciled sheets produced these v3 sheets.
        "inputs": {"ratings_A_v2_sha256": report["ratings_A_sha256"],
                   "ratings_B_v2_sha256": report["ratings_B_sha256"],
                   "decisions_sha256": sha256(args.decisions)},
        "files": {name: sha256(args.output / name) for name in
                  ("completed_A_v3.csv", "completed_B_v3.csv",
                   "triage_review_disagreements.csv", "scored_v2.json", "scored_v3.json")},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Applied decisions:", len(decisions), "· cells changed A/B:", changed_a, "/", changed_b)
    print("Manifest refreshed over the final sheets; judgments reconciled: True.")
    print("No selection frozen; no candidate promoted.")


if __name__ == "__main__":
    main()
