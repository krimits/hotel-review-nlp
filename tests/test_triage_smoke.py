"""The triage smoke test's script, offline: its fixture, its guards, its checks and where it writes.

The real models are not run here. These tests use the fake stages of the other triage tests, and check what the
script does with what the stages return.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from triage_fakes import NEGATIVE, POSITIVE, FakeGenerator, FakeJev, FakeWrapper

from reviewnlp.triage import jev_client
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.questions import QUESTIONS_SHA256
from reviewnlp.triage.schemas import OTHER, TOPICS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("triage_smoke", ROOT / "scripts" / "triage_smoke.py")
smoke_script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = smoke_script
spec.loader.exec_module(smoke_script)

REVIEWS = smoke_script.load_reviews(smoke_script.DEFAULT_REVIEWS)


def two_arms(wrapper=None, jev=None, generator=None) -> dict:
    wrapper = wrapper or FakeWrapper(NEGATIVE)
    generator = generator if generator is not None else FakeGenerator()
    return {"without-jev": TriagePipeline(wrapper, None, generator),
            "with-jev": TriagePipeline(wrapper, jev if jev is not None else FakeJev(yes=["bathroom"]), generator)}


# --- The invented reviews -------------------------------------------------------------------------------------------

def test_the_fixture_has_positive_negative_and_mixed_reviews_and_ids_are_unique():
    kinds = [review["kind"] for review in REVIEWS]
    assert len(REVIEWS) >= 12 and len({review["id"] for review in REVIEWS}) == len(REVIEWS)
    for kind in ("positive", "negative", "mixed"):
        assert kinds.count(kind) >= 3, kind
    assert set(kinds) <= set(smoke_script.KINDS)
    for review in REVIEWS:  # what the author reads in each kind of review
        if review["kind"] == "positive":
            assert review["expected_sentiment"] == "positive" and review["expected_topics"] == [], review["id"]
        if review["kind"] == "negative":
            assert review["expected_sentiment"] == "negative" and review["expected_topics"], review["id"]
        if review["kind"] == "mixed":
            assert review["expected_topics"], review["id"]  # praise and at least one complaint


def test_the_fixture_is_invented_text_with_nothing_that_looks_like_a_contact():
    for review in REVIEWS:  # no address, link, e-mail or number; names are for a person to look for, not a pattern
        assert not re.search(r"[@/\\]|https?:|www\.|\d{3,}", review["text"]), review["id"]
    data = json.loads(smoke_script.DEFAULT_REVIEWS.read_text(encoding="utf-8"))
    assert "Invented" in data["note"] and "may be tuned on them" in data["note"]


def test_a_bad_fixture_is_refused_with_the_review_and_the_reason(tmp_path):
    good = {"id": "a", "kind": "positive", "expected_sentiment": "positive", "expected_topics": [], "text": "Fine stay."}
    for change, reason in [({"kind": "angry"}, "kind"), ({"expected_topics": ["noise"]}, "expected_topics"),
                           ({"text": "ok"}, "text length"), ({"expected_sentiment": "good"}, "expected_sentiment")]:
        path = tmp_path / "reviews.json"
        path.write_text(json.dumps({"reviews": [{**good, **change}]}), encoding="utf-8")
        with pytest.raises(SystemExit, match=reason):
            smoke_script.load_reviews(path)
    path.write_text(json.dumps({"reviews": [good, good]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="used twice"):
        smoke_script.load_reviews(path)


# --- What may be sent -----------------------------------------------------------------------------------------------

def test_jev_is_off_without_the_flag_even_when_a_key_is_set():
    client = smoke_script.build_jev(False, "openrouter", "jev-latest", {"OPENROUTER_API_KEY": "sk-secret"})
    assert client.enabled is False and client.config.api_key == ""
    with pytest.raises(jev_client.JevError) as raised:  # and calling it sends nothing
        client.classify("an invented review")
    assert raised.value.kind == "disabled"


def test_the_flag_without_the_routes_key_stops_before_anything_is_sent():
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY first.*nothing was sent"):
        smoke_script.build_jev(True, "openrouter", "jev-latest", {"TYPESAFE_API_KEY": "the other route's key"})


def test_the_flag_with_the_routes_key_turns_jev_on_and_the_key_is_not_in_its_repr():
    client = smoke_script.build_jev(True, "typesafe", "jev-latest", {"TYPESAFE_API_KEY": "sk-secret"})
    assert client.enabled and client.config.route == "typesafe"
    assert "sk-secret" not in repr(client.config)


def args_for(**changes):
    base = dict(model_path=None, hub_repo="r/m", revision="rev", allow_external_api=False, route="openrouter",
                jev_model="jev-latest", no_qwen=False, qwen_model="Qwen/x", device=None, arms="auto")
    return SimpleNamespace(**{**base, **changes})


def test_the_stages_are_built_without_loading_anything(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    arms = smoke_script.build_arms(args_for(model_path=str(tmp_path)))
    assert list(arms) == ["without-jev"]  # no flag: only the arm that sends nothing
    pipeline = arms["without-jev"]
    assert pipeline.wrapper.model_path == str(tmp_path) and pipeline.wrapper._loaded is False
    assert pipeline.generator is not None and pipeline.generator._bundle is None
    assert smoke_script.build_arms(args_for(model_path=str(tmp_path), no_qwen=True))["without-jev"].generator is None


def test_both_arms_share_the_sentiment_model_and_the_generator(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    arms = smoke_script.build_arms(args_for(model_path=str(tmp_path), allow_external_api=True))
    assert list(arms) == ["without-jev", "with-jev"]
    assert arms["with-jev"].wrapper is arms["without-jev"].wrapper
    assert arms["with-jev"].generator is arms["without-jev"].generator
    assert arms["without-jev"].jev is None and arms["with-jev"].jev.enabled
    with pytest.raises(SystemExit, match="needs --allow-external-api"):
        smoke_script.build_arms(args_for(model_path=str(tmp_path), arms="with-jev"))


def test_the_model_path_is_a_folder_or_the_hub_at_the_pinned_revision(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="is not a folder"):
        smoke_script.resolve_model_path(str(tmp_path / "nope"), "r/m", "rev")
    assert smoke_script.resolve_model_path(str(tmp_path), "r/m", "rev") == str(tmp_path)
    calls = []
    fake_hub = SimpleNamespace(snapshot_download=lambda **kw: calls.append(kw) or "/cache/model")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert smoke_script.resolve_model_path(None, "r/m", "abc") == "/cache/model"
    assert calls == [{"repo_id": "r/m", "revision": "abc"}]
    assert smoke_script.HUB_REPO == "krimits/distilbert-hotel-reviews" and len(smoke_script.HUB_REVISION) == 40


# --- The run and its checks -----------------------------------------------------------------------------------------

def test_every_review_goes_through_every_arm():
    records = smoke_script.smoke(two_arms(), REVIEWS)
    assert [(r["id"], r["arm"]) for r in records] == [(review["id"], arm) for review in REVIEWS
                                                       for arm in ("without-jev", "with-jev")]
    first = records[1]
    assert first["result"]["complaints"]["status"] == "ok" and records[0]["result"]["complaints"]["status"] == "disabled"
    assert first["crashed"] is None and first["expected_topics"] == REVIEWS[0]["expected_topics"]


def test_a_crashing_pipeline_is_recorded_by_the_kind_of_error_and_the_run_goes_on():
    arms = two_arms(wrapper=FakeWrapper(error=RuntimeError("boom")))
    records = smoke_script.smoke(arms, REVIEWS[:2])
    assert len(records) == 4 and all(r["result"] is None and r["crashed"].startswith("RuntimeError") for r in records)
    problems, _ = smoke_script.check(records)
    assert len(problems) == 4 and all("the pipeline raised RuntimeError" in p for p in problems)


def test_a_healthy_run_has_no_problems():
    records = smoke_script.smoke(two_arms(jev=FakeJev(yes=["bathroom"])), REVIEWS)
    problems, notes = smoke_script.check(records)
    assert problems == []
    assert any("expected" in note for note in notes)  # the fake does not agree with the fixture's reading everywhere


def test_labels_the_routing_cannot_read_are_a_problem_once_and_first():
    wrapper = FakeWrapper({"LABEL_0": 0.05, "LABEL_1": 0.95})
    problems, _ = smoke_script.check(smoke_script.smoke(two_arms(wrapper=wrapper), REVIEWS[:3]))
    assert "LABEL_0" in problems[0] and "every review as not negative" in problems[0]
    assert sum("not negative and positive" in p for p in problems) == 1


def test_probabilities_that_do_not_add_up_are_a_problem():
    wrapper = FakeWrapper({"negative": 0.5, "positive": 0.2})
    problems, _ = smoke_script.check(smoke_script.smoke(two_arms(wrapper=wrapper), REVIEWS[:1]))
    assert any("add up to 0.7000" in p for p in problems)


def test_a_jev_stage_that_was_switched_on_and_failed_is_a_problem_not_a_note():
    arms = two_arms(jev=FakeJev(error=jev_client.JevError("http_402")))
    problems, _ = smoke_script.check(smoke_script.smoke(arms, REVIEWS[:2]))
    assert problems and all("the Jev stage is error: http_402" in p for p in problems)


def test_what_a_model_said_is_a_note_not_a_problem():
    arms = two_arms(generator=FakeGenerator(raw="I cannot help."))
    records = smoke_script.smoke(arms, [REVIEWS[4]])  # a negative review: asked for actions
    problems, notes = smoke_script.check(records)
    assert problems == []
    assert any("the suggestion stage gave error (invalid_output)" in n for n in notes)
    empty = smoke_script.check(smoke_script.smoke(two_arms(generator=FakeGenerator(raw='{"actions": []}')), [REVIEWS[4]]))
    assert any("gave an empty list" in n for n in empty[1])


def test_the_fixtures_reading_is_compared_with_what_jev_found():
    review = next(r for r in REVIEWS if r["id"] == "neg-3")  # the author reads: air_conditioning
    records = smoke_script.smoke(two_arms(jev=FakeJev(yes=["bathroom"])), [review])
    _, notes = smoke_script.check(records)
    assert any("neg-3 (with-jev): expected ['air_conditioning'], not found" in n for n in notes)
    assert any("neg-3 (with-jev): found ['bathroom'], not expected" in n for n in notes)
    agreeing = smoke_script.smoke(two_arms(jev=FakeJev(yes=["air_conditioning"])), [review])
    assert not any("expected" in n for n in smoke_script.check(agreeing)[1])


def test_the_sentiment_that_disagrees_with_the_fixture_is_a_note():
    review = next(r for r in REVIEWS if r["kind"] == "positive")
    _, notes = smoke_script.check(smoke_script.smoke(two_arms(wrapper=FakeWrapper(NEGATIVE)), [review]))
    assert any("sentiment is negative, the fixture reads positive" in n for n in notes)
    _, quiet = smoke_script.check(smoke_script.smoke(two_arms(wrapper=FakeWrapper(POSITIVE)), [review]))
    assert not any("sentiment is" in n for n in quiet)


# --- The report and where it is written -----------------------------------------------------------------------------

def test_the_table_has_a_row_per_review_and_arm_and_says_what_each_stage_did():
    table = smoke_script.format_table(smoke_script.smoke(two_arms(), REVIEWS[:2]))
    lines = table.splitlines()
    assert lines[0].startswith("id") and set(lines[1]) <= {"-", " "} and len(lines) == 2 + 4
    assert "without-jev" in lines[2] and "with-jev" in lines[3] and "bath" in lines[3]
    assert "(author: none)" in lines[2]


@pytest.fixture
def runs_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke_script, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(smoke_script, "ROOT", tmp_path)
    return tmp_path / "runs"


def test_the_output_must_be_inside_runs(runs_folder):
    assert smoke_script.output_dir(Path("runs/smoke_1")) == (runs_folder / "smoke_1").resolve()
    for outside in (Path("docs/experiments"), Path("../elsewhere"), Path("runs"), Path("/tmp/anywhere")):
        with pytest.raises(SystemExit, match="inside"):
            smoke_script.output_dir(outside)


def test_the_command_writes_its_three_files_and_returns_zero_for_a_healthy_run(runs_folder, capsys):
    code = smoke_script.main(["--limit", "3", "--output", "runs/t1"], build=lambda args: two_arms())
    out = capsys.readouterr().out
    assert code == 0 and "0 problem(s)" in out and "[3/3] pos-3 (with-jev)" in out
    folder = runs_folder / "t1"
    assert sorted(path.name for path in folder.iterdir()) == ["results.jsonl", "run.json", "summary.md"]
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and {row["arm"] for row in rows} == {"without-jev", "with-jev"}
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert info["problems"] == [] and info["reviews"] == 3 and info["arms"] == ["with-jev", "without-jev"]
    assert info["jev"]["questions_sha256"] == QUESTIONS_SHA256 and info["jev"]["models_answered"] == ["typesafe/jev-test"]
    assert info["script_sha256"] == smoke_script.sha256_file(ROOT / "scripts" / "triage_smoke.py")
    assert info["reviews_sha256"] == smoke_script.sha256_file(smoke_script.DEFAULT_REVIEWS)
    assert info["sentiment"] == {"hub_repo": smoke_script.HUB_REPO, "revision": smoke_script.HUB_REVISION}
    assert "## Problems" in (folder / "summary.md").read_text(encoding="utf-8")


def test_the_command_returns_one_when_the_chain_has_a_problem(runs_folder, capsys):
    broken = lambda args: two_arms(wrapper=FakeWrapper({"LABEL_0": 0.1, "LABEL_1": 0.9}))  # noqa: E731
    assert smoke_script.main(["--limit", "1", "--output", "runs/t2"], build=broken) == 1
    assert "PROBLEM:" in capsys.readouterr().out
    assert json.loads((runs_folder / "t2" / "run.json").read_text(encoding="utf-8"))["problems"]


def test_nothing_in_the_results_holds_a_key(runs_folder):
    smoke_script.main(["--limit", "2", "--output", "runs/t3"], build=lambda args: two_arms())
    for path in (runs_folder / "t3").iterdir():
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in ("sk-", "Bearer", "API_KEY")), path.name


def test_the_default_output_is_a_timestamped_folder_inside_runs(runs_folder):
    smoke_script.main(["--limit", "1"], build=lambda args: two_arms())
    (folder,) = list(runs_folder.iterdir())
    assert re.fullmatch(r"triage_smoke_\d{8}T\d{6}Z", folder.name)


def test_every_topic_the_fixture_expects_is_one_the_questions_ask_about():
    expected = {topic for review in REVIEWS for topic in review["expected_topics"]}
    assert expected <= {*TOPICS, OTHER} and OTHER in expected
