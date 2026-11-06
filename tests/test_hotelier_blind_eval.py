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
