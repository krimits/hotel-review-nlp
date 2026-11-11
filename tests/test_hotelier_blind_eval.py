"""The hoteliers' blind evaluation: groups, the draw of the rated set and its files, on synthetic and committed data."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load("hotelier_blind_eval")


def test_groups_split_by_the_urgent_types_only():
    out = pd.DataFrame({"teacher": [("cleanliness",), ("cleanliness", "wifi"), (), ("wifi",), ("pests",)],
                        "student": [("cleanliness",), ("cleanliness",), (), ("bed_comfort",), ("hvac_failure",)]},
                       index=[10, 11, 12, 13, 14])
    # 11 differs only in a routine type, 13 in routine types with nothing urgent: neither is in D.
    assert evaluation.groups(out).to_dict() == {10: "C", 11: "C", 12: "Z", 13: "Z", 14: "D"}


def test_the_rated_set_keeps_all_of_d_draws_c_per_stratum_and_weighs_it(monkeypatch):
    monkeypatch.setattr(evaluation, "C_SIZES", {"H": 2, "N": 1})
    rows = list(range(100, 120))
    test = pd.DataFrame({"stratum": ["H", "N"] * 10, "weight": [3.0, 7.0] * 10, "item": 0, "set": "test",
                         "pool": 95, "text_sha256": [f"h{r}" for r in rows]}, index=pd.Index(rows, name="raw_row"))
    group = pd.Series(["D"] * 4 + ["C"] * 10 + ["Z"] * 6, index=rows)
    rated = evaluation.draw_rated(test, group, seed=1)
    assert sorted(rated["item"]) == list(range(1, 8)) and set(rated[rated["group"] == "D"]["raw_row"]) == set(rows[:4])
    c = rated[rated["group"] == "C"]
    assert c["stratum"].value_counts().to_dict() == {"H": 2, "N": 1}
    assert (c.loc[c["stratum"] == "H", "sample_weight"] == 5 / 2).all()  # 5 H reviews in C, 2 drawn
    assert (c.loc[c["stratum"] == "N", "sample_weight"] == 5 / 1).all()
    assert (rated[rated["group"] == "D"]["sample_weight"] == 1.0).all()
    assert rated.equals(evaluation.draw_rated(test, group, seed=1))
    assert not rated.equals(evaluation.draw_rated(test, group, seed=2))


def test_build_on_the_committed_files_writes_60_items_without_text_and_never_redraws(tmp_path):
    manifest = evaluation.build(tmp_path)
    assert manifest["rated"]["items"] == 60 and manifest["rated"]["D"] == 49 and manifest["rated"]["C"] == 11
    rated = pd.read_csv(tmp_path / "rated_key.csv")
    assert "text" not in rated.columns and sorted(rated["item"]) == list(range(1, 61))
    outputs = json.loads((tmp_path / "outputs.json").read_text(encoding="utf-8"))
    assert set(outputs) == {str(i) for i in range(1, 61)}
    urgent = {"cleanliness", "pests", "security_fault", "safety_fault", "safety_incident", "hvac_failure",
              "plumbing_major"}
    for row in rated.itertuples():
        teacher, student = ({t for t in outputs[str(row.item)][s] if t in urgent} for s in ("teacher", "student"))
        assert (teacher != student) == (row.group == "D") and (teacher or student)
    first = (tmp_path / "rated_key.csv").read_bytes()
    with pytest.raises(SystemExit, match="never redrawn"):
        evaluation.build(tmp_path)
    evaluation.build(tmp_path, overwrite=True)
    assert (tmp_path / "rated_key.csv").read_bytes() == first


def test_the_rule_and_its_boundaries():
    assert evaluation.outcome(0.79, 0.0, 0.1) == "Neither ready"
    assert evaluation.outcome(0.80, -0.10, 0.0) == "Student non-inferior"
    assert evaluation.outcome(0.95, -0.25, -0.11) == "Student inferior"
    assert evaluation.outcome(0.95, -0.15, -0.05) == "Not established"
    assert [c for c, _ in evaluation.RULE] == ["p_teacher < 0.80", "lo_d >= -0.10", "hi_d < -0.10", "otherwise"]
    assert (evaluation.MARGIN, evaluation.FLOOR, evaluation.MIN_COMPLETE) == (-0.10, 0.80, 2)


def write_ratings(folder: Path, hoteliers, right, protocol_sha256: str, items=range(1, 61)):
    """One file per (hotelier, item) as the Space writes them; right(hotelier, item, system) -> bool."""
    for hotelier in hoteliers:
        for item in items:
            by_system = {neutral: {"q1_urgent_right": right(hotelier, item, system),
                                   "q2_what_is_wrong": [] if right(hotelier, item, system) else ["missed_urgent"],
                                   "q3_others": "yes", "q4_useful": 4}
                         for neutral, system in evaluation.NEUTRAL.items()}
            path = folder / "ratings" / hotelier / f"item_{item:02d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"hotelier": hotelier, "item": item, "by_system": by_system,
                                        "preferred_system": "x", "protocol_sha256": protocol_sha256}),
                            encoding="utf-8")


@pytest.fixture
def protocol_sha():
    return evaluation.sample.trends._sha256(evaluation.PROTOCOL)


def run(tmp_path, right, protocol_sha, hoteliers=("H01", "H02"), **kwargs):
    write_ratings(tmp_path / "export", hoteliers, right, protocol_sha, **kwargs)
    return evaluation.analyse(tmp_path / "export", out=tmp_path / "result.json", require_commits=False, record=None)


def test_equal_judgements_make_the_student_non_inferior(tmp_path, protocol_sha):
    result = run(tmp_path, lambda h, i, s: True, protocol_sha)
    decision = result["decision"]
    assert decision["d"] == 0.0 and decision["lo_d"] == 0.0 and decision["p_teacher"] == 1.0
    assert decision["outcome"] == "Student non-inferior" and decision["u_reviews"] == 146
    assert decision["d_reviews"] == 49 and result["reported"]["identical_outputs_judged_differently"] == 0
    assert result["reported"]["fleiss_kappa_q1"] is None  # every answer yes: no variation to agree on
    with pytest.raises(SystemExit, match="applied once"):
        evaluation.analyse(tmp_path / "export", out=tmp_path / "result.json", require_commits=False, record=None)


def test_a_student_wrong_on_every_contested_review_is_inferior(tmp_path, protocol_sha):
    rated = pd.read_csv(evaluation.SAMPLES / "rated_key.csv").set_index("item")
    contested = set(rated.index[rated["group"] == "D"])
    result = run(tmp_path, lambda h, i, s: not (s == "student" and i in contested), protocol_sha)
    decision = result["decision"]
    assert decision["d"] < -0.10 and decision["hi_d"] < -0.10 and decision["outcome"] == "Student inferior"
    assert result["reported"]["right_by_group"]["D"] == {"teacher": 1.0, "student": 0.0}


def test_a_teacher_judged_wrong_too_often_means_neither_is_ready(tmp_path, protocol_sha):
    result = run(tmp_path, lambda h, i, s: i % 3 == 0, protocol_sha)
    assert result["decision"]["p_teacher"] < 0.80 and result["decision"]["outcome"] == "Neither ready"


def test_one_complete_hotelier_is_not_decided_and_the_partial_one_is_reported_apart(tmp_path, protocol_sha):
    write_ratings(tmp_path / "export", ["H02"], lambda h, i, s: True, protocol_sha, items=range(1, 11))
    result = run(tmp_path, lambda h, i, s: True, protocol_sha, hoteliers=("H01",))
    assert result["decision"]["outcome"].startswith("not decided") and result["hoteliers_partial"] == ["H02"]


def test_ratings_given_under_another_protocol_version_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="another version of the protocol"):
        run(tmp_path, lambda h, i, s: True, "0" * 64)


def make_export(folder: Path, protocol_sha256: str, hoteliers=("H01", "H02"), items=range(1, 61), bundle="b1"):
    """An export as `hf download` leaves it: the Space's own records, plus the files a download adds beside them."""
    logic = evaluation.space_logic()
    right, wrong = logic.answers("Ναι", [], "Ναι", "4"), logic.answers("Όχι", ["Λείπει επείγον πρόβλημα"], "Όχι", "2")
    for hotelier in hoteliers:
        for item in items:
            rating = logic.record(hotelier, item, right, wrong if item % 4 == 0 else right, "Το ίδιο",
                                  {"protocol_sha256": protocol_sha256, "bundle": bundle})
            path = folder / logic.path_of(hotelier, item)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rating, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (folder / ".gitattributes").write_text("*.json text\n", encoding="utf-8")
    (folder / ".cache" / "huggingface").mkdir(parents=True)
    return folder


def edit(path: Path, **changes):
    rating = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**rating, **changes}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


def swap_sides(path: Path):
    rating = json.loads(path.read_text(encoding="utf-8"))
    edit(path, system_1=rating["system_2"], system_2=rating["system_1"])


def test_a_clean_export_is_recorded_by_one_digest_and_no_rating_enters_the_repository(tmp_path, protocol_sha):
    export, out = make_export(tmp_path / "export", protocol_sha), tmp_path / "ratings"
    with pytest.raises(SystemExit, match="--source is required"):
        evaluation.record_ratings(" ", export, out)
    record = evaluation.record_ratings("two hoteliers, themselves", export, out)
    assert record["hoteliers"]["H01"]["ratings"] == 60 and record["hoteliers"]["H02"]["complete"]
    assert record["files"] == 120 and record["bundle"] == "b1" and record["source"] == "two hoteliers, themselves"
    assert (record["ratings_sha256"], 120) == evaluation.ratings_digest(export)
    assert [p.name for p in out.iterdir()] == ["record.json"]  # the rating files are not copied
    text = (out / "record.json").read_text(encoding="utf-8")
    assert "q1_urgent_right" not in text and "preference" not in text and "item_07" not in text
    with pytest.raises(SystemExit, match="never replaced"):
        evaluation.record_ratings("again", export, out)
    # The analysis reads the export, and only while it is the recorded one.
    swap_sides(export / "ratings/H02/item_09.json")
    with pytest.raises(SystemExit, match="does not hold the recorded ratings"):
        evaluation.analyse(export, out=tmp_path / "result.json", require_commits=False, record=out / "record.json")
    swap_sides(export / "ratings/H02/item_09.json")
    (export / "ratings/H02/item_09.json").rename(tmp_path / "taken_out.json")
    with pytest.raises(SystemExit, match="does not hold the recorded ratings"):
        evaluation.analyse(export, out=tmp_path / "result.json", require_commits=False, record=out / "record.json")
    (tmp_path / "taken_out.json").rename(export / "ratings/H02/item_09.json")
    result = evaluation.analyse(export, out=tmp_path / "result.json", require_commits=False,
                                record=out / "record.json")
    assert result["hoteliers_complete"] == ["H01", "H02"] and "d" in result["decision"]
    assert result["ratings"] == {"files": 120, "sha256": record["ratings_sha256"], "kept": evaluation.RATINGS_KEPT}
    written = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert "item_07" not in written and "q1_urgent_right" not in written  # aggregates only


def test_the_digest_changes_with_any_file_and_not_with_what_a_download_adds(tmp_path, protocol_sha):
    export = make_export(tmp_path / "export", protocol_sha, hoteliers=("H01",), items=range(1, 4))
    digest, count = evaluation.ratings_digest(export)
    (export / ".cache" / "huggingface" / "x.lock").write_text("", encoding="utf-8")
    assert evaluation.ratings_digest(export) == (digest, 3) and count == 3
    edit(export / "ratings/H01/item_02.json", submitted_at="2026-11-06T00:00:00+00:00")
    assert evaluation.ratings_digest(export)[0] != digest


@pytest.mark.parametrize("damage, message", [
    (lambda e: swap_sides(e / "ratings/H01/item_07.json"), "the sides are not the ones"),
    (lambda e: edit(e / "ratings/H01/item_07.json", protocol_sha256="0" * 64), "another version of the protocol"),
    (lambda e: edit(e / "ratings/H01/item_07.json", hotelier="H02"), "differs from its path"),
    (lambda e: edit(e / "ratings/H01/item_07.json", preferred_system="x", preference="same"), "the preference"),
    (lambda e: edit(e / "ratings/H02/item_09.json", bundle="b2"), "more than one bundle"),
    (lambda e: (e / "ratings/H01/item_07.json").rename(e / "ratings/H01/item_61.json"), "not an item of the rated set"),
    (lambda e: (e / "ratings/DEMO").mkdir(), "not a hotelier's folder"),
    (lambda e: (e / "ratings/H01/notes.txt").write_text("x"), "not a rating file"),
])
def test_one_problem_refuses_the_whole_export(tmp_path, protocol_sha, damage, message):
    export = make_export(tmp_path / "export", protocol_sha)
    damage(export)
    with pytest.raises(SystemExit, match=message):
        evaluation.record_ratings("two hoteliers", export, tmp_path / "ratings")
    assert not (tmp_path / "ratings").exists()


def test_an_inconsistent_answer_is_refused(tmp_path, protocol_sha):
    export = make_export(tmp_path / "export", protocol_sha)
    path = export / "ratings/H02/item_05.json"
    rating = json.loads(path.read_text(encoding="utf-8"))
    rating["shown"]["1"]["q2_what_is_wrong"] = ["missed_urgent"]  # says the urgent part is right, and what is wrong
    rating["by_system"][rating["system_1"]] = rating["shown"]["1"]
    path.write_text(json.dumps(rating), encoding="utf-8")
    with pytest.raises(SystemExit, match="Q1 and Q2 are not a complete, consistent answer"):
        evaluation.record_ratings("two hoteliers", export, tmp_path / "ratings")
