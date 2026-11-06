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
    return evaluation.analyse(tmp_path / "export", out=tmp_path / "result.json", require_commits=False)


def test_equal_judgements_make_the_student_non_inferior(tmp_path, protocol_sha):
    result = run(tmp_path, lambda h, i, s: True, protocol_sha)
    decision = result["decision"]
    assert decision["d"] == 0.0 and decision["lo_d"] == 0.0 and decision["p_teacher"] == 1.0
    assert decision["outcome"] == "Student non-inferior" and decision["u_reviews"] == 146
    assert decision["d_reviews"] == 49 and result["reported"]["identical_outputs_judged_differently"] == 0
    assert result["reported"]["fleiss_kappa_q1"] is None  # every answer yes: no variation to agree on
    with pytest.raises(SystemExit, match="applied once"):
        evaluation.analyse(tmp_path / "export", out=tmp_path / "result.json", require_commits=False)


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
