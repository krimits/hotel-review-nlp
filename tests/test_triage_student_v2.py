"""The data steps of the triage student v2 protocol on a tiny synthetic CSV, and its teacher with a fake
OpenRouter; no network and no real data."""

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


v2 = _load("triage_student_v2")
sample, labels, teacher = v2.sample, v2.labels, v2.teacher
TYPES = sample.catalogue_types(v2.CATALOGUE)
# The first protocol's draws, small, as its own tests run them.
FIRST_SAMPLES = (("development", range(80, 90), 20261021, {"H": 3, "N": 4, "B": 2}),
                 ("test", range(90, 100), 20261022, {"H": 5, "N": 6, "B": 2}))
FIRST_CHECK = ("check", range(90, 100), 20261024, {"H": 3, "N": 4, "B": 2})
FIRST_TEACHER = ("teacher", range(0, 80), 20261030, {"H": 5, "N": 6, "B": 3})
# This protocol's, with the same development sizes as the first's, so the 18 weigh N_s / (2 x n_s).
DEVELOPMENT = ("development", range(80, 90), 20261103, {"H": 3, "N": 4, "B": 2})
TEST = ("test", range(90, 100), 20261102, {"H": 5, "N": 6, "B": 2})
TEACHER = ("teacher", range(0, 80), 20261104, {"H": 40, "N": 40, "B": 20})


def raw_csv(path: Path, n: int = 3000) -> Path:
    negatives = ["The room was dirty {i}", "Staff were slow at desk {i}", "No Negative"]
    positives = ["Great breakfast {i}", "No Positive", "Lovely view {i}"]
    rows = [{"Hotel_Address": f"{i % 7} Street London United Kingdom", "Review_Date": "8/3/2017",
             "Reviewer_Nationality": " UK ", "Negative_Review": negatives[i % 3].format(i=i),
             "Positive_Review": positives[i % 3].format(i=i), "Reviewer_Score": 7.5,
             "Hotel_Name": f"Hotel {i % 7}", "Tags": "[' Leisure trip ', ' Couple ']"} for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def drawn(tmp_path, monkeypatch):
    """The first protocol's keys drawn small, then this protocol's draw on the same data."""
    for name, value in (("SAMPLES", FIRST_SAMPLES), ("CHECK", FIRST_CHECK), ("TEACHER", FIRST_TEACHER),
                        ("SEEN_KEYS", ())):
        monkeypatch.setattr(sample, name, value)
    for name, value in (("DEVELOPMENT", DEVELOPMENT), ("TEST", TEST), ("TEACHER", TEACHER)):
        monkeypatch.setattr(v2, name, value)
    csv = raw_csv(tmp_path / "raw.csv")
    first, first_runs = tmp_path / "first", tmp_path / "first_runs"
    sample.build(csv, first_runs, first, check_population=False)
    sample.build_check(csv, first_runs, first, check_population=False)
    sample.build_teacher(csv, first_runs, first, check_population=False)
    runs, record = tmp_path / "runs", tmp_path / "samples"
    manifest = v2.build(csv, runs, record, first, check_population=False)
    return csv, first, runs, record, manifest


def test_constants_are_the_protocols():
    assert v2.DEVELOPMENT == ("development", range(80, 90), 20261103, {"H": 90, "N": 180, "B": 30})
    assert v2.TEST == ("test", range(90, 100), 20261102, {"H": 150, "N": 300, "B": 50})
    assert v2.TEACHER == ("teacher", range(0, 80), 20261104, {"H": 12000, "N": 15000, "B": 3000})
    assert sum(v2.TEACHER[3].values()) == 30000 and sum(v2.TEST[3].values()) == 500
    assert 2 * sum(v2.DEVELOPMENT[3].values()) == 600 == v2.BATCHES["development"] * v2.BATCH_SIZE
    assert 500 == v2.BATCHES["test"] * v2.BATCH_SIZE and v2.BATCHES == {"development": 12, "test": 10}
    assert teacher.MODELS[v2.TEACHER_MODEL] == "anthropic/claude-haiku-5.5-20261007" and teacher.BATCH == 25
    assert v2.CATALOGUE.name == "catalogue_v2.yaml" and len(TYPES) == 21 and "safety_hazard" not in TYPES


def test_development_keeps_the_first_300_adds_new_ones_and_weighs_by_the_whole_pool(drawn):
    _, first, _, record, manifest = drawn
    key, first_key = pd.read_csv(record / "key.csv"), pd.read_csv(first / "key.csv")
    development = key[key["set"] == "development"]
    assert sorted(development["item"]) == list(range(1, 19))
    old = set(first_key.loc[first_key["set"] == "development", "raw_row"])
    assert set(development.loc[development["first_protocol"] == 1, "raw_row"]) == old
    new = development[development["first_protocol"] == 0]
    assert len(new) == 9 and not set(new["raw_row"]) & old
    assert new["stratum"].value_counts().to_dict() == {"N": 4, "H": 3, "B": 2}
    strata = manifest["samples"]["development"]["strata"]
    for stratum, wanted in DEVELOPMENT[3].items():
        in_stratum = development[development["stratum"] == stratum]
        assert len(in_stratum) == strata[stratum]["n_s"] == 2 * wanted
        assert (in_stratum["weight"] == strata[stratum]["N_s"] / (2 * wanted)).all()
        assert strata[stratum]["N_s"] > manifest["samples"]["development_new"]["strata"][stratum]["N_s"]


def test_the_test_set_is_new_and_counted_after_the_first_protocols_reviews(drawn):
    _, first, _, record, manifest = drawn
    key = pd.read_csv(record / "key.csv")
    test = key[key["set"] == "test"]
    assert sorted(test["item"]) == list(range(19, 32)) and (test["first_protocol"] == 0).all()
    first_key, first_check = pd.read_csv(first / "key.csv"), pd.read_csv(first / "check_key.csv")
    taken = set(first_key.loc[first_key["set"] == "test", "raw_row"]) | set(first_check["raw_row"])
    assert not set(test["raw_row"]) & taken and test["pool"].between(90, 99).all()
    assert manifest["samples"]["test"]["left_out_already_drawn"] == len(taken) == 13 + 9
    for stratum, wanted in TEST[3].items():
        assert (test.loc[test["stratum"] == stratum, "weight"]
                == manifest["samples"]["test"]["strata"][stratum]["N_s"] / wanted).all()


def test_the_teacher_sample_and_its_held_out_tenth(drawn):
    _, _, runs, record, manifest = drawn
    teacher_key = pd.read_csv(record / "teacher_key.csv")
    texts = pd.read_csv(runs / "teacher_sample.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert len(teacher_key) == len(texts) == 100 and sorted(teacher_key["item"]) == list(range(1, 101))
    assert teacher_key["pool"].between(0, 79).all()
    assert (teacher_key["held_out"] == (teacher_key["pool"] % 10 == 0).astype(int)).all()
    assert 0 < manifest["teacher_held_out"] == teacher_key["held_out"].sum() < 100
    by_item = dict(zip(texts["item"].astype(int), texts["text"], strict=True))
    assert [sample.text_hash(by_item[i]) for i in teacher_key["item"]] == teacher_key["text_sha256"].tolist()
    assert all(v2.held_out(by_item[i]) == bool(h) for i, h in zip(teacher_key["item"], teacher_key["held_out"],
                                                                  strict=True))


def test_text_stays_under_runs_the_sheet_is_catalogue_v2s_and_files_are_lf(drawn):
    _, _, runs, record, manifest = drawn
    for name in ("key.csv", "teacher_key.csv"):
        assert "text" not in pd.read_csv(record / name).columns, name
    sheet = pd.read_csv(runs / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(sheet.columns) == ["item", "text", *TYPES, "none", "done", "note"] == manifest["sheet_columns"]
    assert (sheet.drop(columns=["item", "text"]) == "").all().all() and len(sheet) == 31
    for path in (runs / "sheet.csv", runs / "teacher_sample.csv", record / "key.csv", record / "teacher_key.csv",
                 record / "manifest.json"):
        assert b"\r\n" not in path.read_bytes(), path


def test_the_draw_is_reproducible_and_never_redrawn_by_accident(drawn):
    csv, first, runs, record, _ = drawn
    before = {name: (record / name).read_bytes() for name in ("key.csv", "teacher_key.csv")}
    with pytest.raises(SystemExit, match="never redrawn"):
        v2.build(csv, runs, record, first, check_population=False)
    v2.build(csv, runs, record, first, overwrite=True, check_population=False)
    assert before == {name: (record / name).read_bytes() for name in before}


@pytest.mark.parametrize("file", ["key.csv", "check_key.csv", "teacher_key.csv"])
def test_the_draw_stops_unless_every_first_key_redraws(drawn, file):
    csv, first, runs, record, _ = drawn
    path = first / file
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8", newline="")
    with pytest.raises(SystemExit, match="byte for byte|does not give"):
        v2.build(csv, runs, record, first, overwrite=True, check_population=False)


def test_batches_split_development_from_test_and_carry_the_prompt(drawn, monkeypatch, tmp_path):
    _, _, runs, record, _ = drawn
    monkeypatch.setattr(v2, "BATCH_SIZE", 9)
    monkeypatch.setattr(v2, "BATCHES", {"development": 2, "test": 2})
    prompt = tmp_path / "reference_prompt_v2.md"
    prompt.write_text(labels.reference_prompt(v2.CATALOGUE), encoding="utf-8", newline="\n")
    result = v2.write_batches(runs, record, prompt)
    assert [b["items"] for b in result["batches"].values()] == [[1, 9], [10, 18], [19, 27], [28, 31]]
    assert result["development_batches"] == [1, 2] and result["test_batches"] == [3, 4]
    assert (runs / "reference" / prompt.name).read_bytes() == prompt.read_bytes()
    assert json.loads((record / "reference_batches.json").read_text(encoding="utf-8")) == result
    prompt.write_text("an edited prompt", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match catalogue_v2"):
        v2.write_batches(runs, record, prompt)


class FakeOpenRouter:
    """Labels odd items `safety_fault` and even items nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, body, api_key, timeout):
        self.calls.append(body)
        reviews = json.loads(body["messages"][1]["content"])
        entries = [{"item": r["item"], "types": ["safety_fault"] if r["item"] % 2 else []} for r in reviews]
        answer = {"choices": [{"message": {"content": json.dumps({"labels": entries})}}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.01}}
        return 200, {}, json.dumps(answer).encode()


def test_the_teacher_runs_the_first_protocols_code_on_this_protocols_files(drawn, monkeypatch, tmp_path):
    _, _, runs, record, _ = drawn
    prompt = tmp_path / "teacher_prompt_v2.md"
    prompt.write_text(labels.teacher_prompt(v2.CATALOGUE), encoding="utf-8", newline="\n")
    out = tmp_path / "labels"
    monkeypatch.setattr(v2, "RUNS", runs)
    monkeypatch.setattr(v2, "SAMPLES", record)
    monkeypatch.setattr(v2, "LABELS", out)
    monkeypatch.setattr(v2, "SETUP", teacher.Setup(v2.CATALOGUE, prompt, v2.PROTOCOL, v2.SUBSETS))
    fake = FakeOpenRouter()
    result = v2.run_teacher("development", 1.0, transport=fake, api_key="sk-or-test", sleep=lambda _: None)
    body = fake.calls[0]
    assert body["model"] == "anthropic/claude-haiku-5.5-20261007"
    assert body["messages"][0]["content"] == labels.teacher_prompt(v2.CATALOGUE)
    assert body["response_format"]["json_schema"]["schema"]["properties"]["labels"]["items"]["properties"][
        "types"]["items"]["enum"] == TYPES
    assert [r["item"] for r in json.loads(body["messages"][1]["content"])] == list(range(1, 19))
    table = pd.read_csv(out / "teacher_haiku_development.csv")
    assert list(table.columns) == ["raw_row", "set", *TYPES, "none", "failed"] and len(table) == 18
    assert table["safety_fault"].sum() == 9 and (table["set"] == "development").all()
    assert result["protocol_sha256"] == sample.trends._sha256(v2.PROTOCOL)
    assert result["teacher_prompt_sha256"] == sample.trends._sha256(prompt)
    with pytest.raises(teacher.Stop, match="never replaced"):
        v2.run_teacher("development", 1.0, transport=fake, api_key="sk-or-test")
