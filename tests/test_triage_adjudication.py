"""Adjudication applies only reported disagreements, reconciles the sheets, and refreshes the manifest."""
from __future__ import annotations

import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

from reviewnlp.triage.generator_experiment import RATINGS
from reviewnlp.triage.workflow_review import COUNTS
from scripts.adjudicate_triage_review import load_decisions, submitted_adjudications_hold
from scripts.adjudicate_triage_review import main as adjudicate_main
from scripts.capture_triage_review import capture
from scripts.score_triage_review import reconcile

SOURCE = "a" * 40
ROWS = [{"id": "dev-01", "text": "The lobby was comfortable."},
        {"id": "dev-02", "text": "The room was comfortable."}]
MANIFEST = {"source_commit": SOURCE, "selection_status": "experimental_not_selected_not_promoted"}
DECISION_FIELDS = ["blind_id", "field", "agreed_value", "adjudication_notes"]


class FakeClient:
    def predict(self, *, api_name):
        assert api_name == "/model_info"
        return {"source_snapshot": MANIFEST, "prompt_version": "actions-v6-source-spans",
                "jev_enabled": True, "runtime": {"ready": True}}

    def submit(self, text, jev, *, api_name):
        assert api_name == "/analyze"
        def result(timeout):
            return "summary", [], [], {"deployment": {"source_snapshot": MANIFEST}, "stored": False,
                "api_cost": {"attempts": 0}, "complaints": {"status": "disabled"}, "sentiment": {"label": "positive"},
                "issue_assessments": [], "actions": {"actions": []}, "status": "complete",
                "stage_reports": [], "stage_failure": None, "timings": {"total_ms": 10}}
        def cancel():
            pass
        return SimpleNamespace(result=result, cancel=cancel)


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def divergent_sheets(folder):
    """A captured run plus two completed sheets that disagree on one count for one review."""
    report = capture(FakeClient(), ROWS, folder / "capture", source_commit=SOURCE)
    assert report["execution_complete"]
    with (folder / "capture" / "coverage_A.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(dict.fromkeys(COUNTS, "0"))
        row.update(dict.fromkeys(RATINGS, "na"))
        row["useful"] = "1"
    write(folder / "completed_A.csv", rows)
    divergent = [dict(row) for row in rows]
    divergent[-1]["useful"] = "0"
    write(folder / "completed_B.csv", divergent)
    return folder / "capture", folder / "completed_A.csv"


def decision_csv(path, rows):
    csv_path = path / f"decisions_{len(rows)}_{len(list(path.glob('decisions_*')))}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_decisions_reconcile_both_sheets_and_refresh_the_manifest(tmp_path, monkeypatch):
    run, sheet_a = divergent_sheets(tmp_path)
    report = reconcile(run, sheet_a, tmp_path / "completed_B.csv")
    assert not report["judgments_reconciled"] and len(report["disagreements"]) == 1
    item = report["disagreements"][0]
    note = "No action was justified; the review praises the stay."
    decisions_path = decision_csv(tmp_path, [{"blind_id": item["blind_id"], "field": item["field"],
                                              "agreed_value": "0", "adjudication_notes": note}])
    output = tmp_path / "adjudicated"
    original_bytes = {name: (tmp_path / name).read_bytes()
                      for name in ("completed_A.csv", "completed_B.csv")}
    monkeypatch.setattr("sys.argv", ["adjudicate_triage_review.py", "--run", str(run),
        "--ratings-a", str(sheet_a), "--ratings-b", str(tmp_path / "completed_B.csv"),
        "--decisions", str(decisions_path), "--output", str(output), "--date", "2026-10-13"])
    adjudicate_main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["judgments_reconciled"] and manifest["all_submitted_adjudications_match_v3"]
    assert manifest["selection_frozen"] is manifest["promoted"] is False
    assert manifest["prior_disagreement_fields"] == 1 and manifest["date"] == "2026-10-13"
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    v3 = reconcile(run, output / "completed_A_v3.csv", output / "completed_B_v3.csv")
    assert v3 == json.loads((output / "scored_v3.json").read_text(encoding="utf-8"))
    assert v3["judgments_reconciled"] and v3["disagreements"] == []
    recorded = read_rows(output / "triage_review_disagreements.csv")
    assert len(recorded) == 1 and recorded[0]["agreed_value"] == "0" \
        and recorded[0]["adjudication_notes"] == note
    sheet_a_v3 = {row["blind_id"]: row for row in read_rows(output / "completed_A_v3.csv")}
    assert sheet_a_v3[item["blind_id"]]["useful"] == "0", "A held 1, the agreed value is 0"
    assert "Μετά τη συζήτηση: useful 1→0" in sheet_a_v3[item["blind_id"]]["notes"]
    assert note in sheet_a_v3[item["blind_id"]]["notes"]
    sheet_b_v3 = {row["blind_id"]: row for row in read_rows(output / "completed_B_v3.csv")}
    assert sheet_b_v3[item["blind_id"]]["useful"] == "0", "B already held the agreed value"
    assert "Μετά τη συζήτηση" not in sheet_b_v3[item["blind_id"]]["notes"], "unchanged cells keep their note"
    assert original_bytes["completed_A.csv"] == sheet_a.read_bytes()
    assert original_bytes["completed_B.csv"] == (tmp_path / "completed_B.csv").read_bytes()


def test_no_op_decision_keeps_the_annotator_note_but_still_reconciles(tmp_path, monkeypatch):
    run, sheet_a = divergent_sheets(tmp_path)
    report = reconcile(run, sheet_a, tmp_path / "completed_B.csv")
    item = report["disagreements"][0]
    decisions_path = decision_csv(tmp_path, [{"blind_id": item["blind_id"], "field": item["field"],
                                              "agreed_value": item["A"],
                                              "adjudication_notes": "A's reading stands."}])
    output = tmp_path / "adjudicated"
    monkeypatch.setattr("sys.argv", ["adjudicate_triage_review.py", "--run", str(run),
        "--ratings-a", str(sheet_a), "--ratings-b", str(tmp_path / "completed_B.csv"),
        "--decisions", str(decisions_path), "--output", str(output)])
    adjudicate_main()
    sheet_a_v3 = {row["blind_id"]: row for row in read_rows(output / "completed_A_v3.csv")}
    assert sheet_a_v3[item["blind_id"]]["notes"] == "", "an unchanged cell keeps its original note"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["judgments_reconciled"]


def test_incomplete_or_outside_disagreements_decisions_are_rejected(tmp_path):
    run, sheet_a = divergent_sheets(tmp_path)
    report = reconcile(run, sheet_a, tmp_path / "completed_B.csv")
    assert len(report["disagreements"]) == 1
    identity = (report["disagreements"][0]["blind_id"], report["disagreements"][0]["field"])
    good = {identity: {"value": "0", "note": "clear"}}
    assert load_decisions(
        decision_csv(tmp_path, [{"blind_id": identity[0], "field": identity[1],
                                 "agreed_value": "0", "adjudication_notes": "clear"}]),
        report["disagreements"]) == good
    with pytest.raises(ValueError, match="outside the reported disagreements"):
        load_decisions(decision_csv(tmp_path, [{"blind_id": identity[0], "field": "grounded",
                                                "agreed_value": "1", "adjudication_notes": "x"}]),
                       report["disagreements"])
    with pytest.raises(ValueError, match="missing decisions"):
        load_decisions(decision_csv(tmp_path, []), report["disagreements"])
    with pytest.raises(ValueError, match="note required"):
        load_decisions(decision_csv(tmp_path, [{"blind_id": identity[0], "field": identity[1],
                                                "agreed_value": "1", "adjudication_notes": " "}]),
                       report["disagreements"])


def test_manifest_pins_its_inputs_and_keeps_the_pre_adjudication_report(tmp_path, monkeypatch):
    """A manifest that hashes only its outputs cannot show which sheets produced them."""
    run, sheet_a = divergent_sheets(tmp_path)
    sheet_b = tmp_path / "completed_B.csv"
    report = reconcile(run, sheet_a, sheet_b)
    item = report["disagreements"][0]
    decisions_path = decision_csv(tmp_path, [{"blind_id": item["blind_id"], "field": item["field"],
                                              "agreed_value": "0", "adjudication_notes": "agreed"}])
    output = tmp_path / "adjudicated"
    monkeypatch.setattr("sys.argv", ["adjudicate_triage_review.py", "--run", str(run),
        "--ratings-a", str(sheet_a), "--ratings-b", str(sheet_b),
        "--decisions", str(decisions_path), "--output", str(output)])
    adjudicate_main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    assert manifest["inputs"]["ratings_A_v2_sha256"] == digest(sheet_a)
    assert manifest["inputs"]["ratings_B_v2_sha256"] == digest(sheet_b)
    assert manifest["inputs"]["decisions_sha256"] == digest(decisions_path)
    # the report from before the adjudication is kept and covered by the manifest
    assert json.loads((output / "scored_v2.json").read_text(encoding="utf-8")) == report
    assert manifest["files"]["scored_v2.json"] == digest(output / "scored_v2.json")


def test_the_match_flag_is_measured_rather_than_asserted(tmp_path, monkeypatch):
    """`all_submitted_adjudications_match_v3` must come from reading the sheets back."""
    run, sheet_a = divergent_sheets(tmp_path)
    sheet_b = tmp_path / "completed_B.csv"
    item = reconcile(run, sheet_a, sheet_b)["disagreements"][0]
    decisions_path = decision_csv(tmp_path, [{"blind_id": item["blind_id"], "field": item["field"],
                                              "agreed_value": "0", "adjudication_notes": "agreed"}])
    output = tmp_path / "adjudicated"
    monkeypatch.setattr("sys.argv", ["adjudicate_triage_review.py", "--run", str(run),
        "--ratings-a", str(sheet_a), "--ratings-b", str(sheet_b),
        "--decisions", str(decisions_path), "--output", str(output)])
    adjudicate_main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_submitted_adjudications_match_v3"] is True
    assert manifest["original_capture_unchanged"] is True
    # the same check against a value the sheets do not hold must come back False
    identity = (item["blind_id"], item["field"])
    assert submitted_adjudications_hold(output, {identity: {"value": "1", "note": "n"}}) is False
