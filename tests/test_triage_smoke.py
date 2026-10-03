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
from triage_fakes import GOOD_ACTIONS, NEGATIVE, POSITIVE, FakeGenerator, FakeJev, FakeWrapper

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
    assert isinstance(pipeline.generator, smoke_script.RecordingGenerator) and pipeline.generator.inner._bundle is None
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


# --- The suggestion model: that it ran, what it wrote, and how it failed -----------------------------------------------

def recorded(**kwargs):
    return smoke_script.RecordingGenerator(FakeGenerator(**kwargs))


def test_the_recorder_keeps_what_was_written_and_how_it_failed_and_the_pipeline_sees_no_difference():
    generator = recorded(raw="some text", hit_token_budget=True)
    assert generator.model_name == "fake/qwen"
    result = generator.generate("a review", None)
    assert (result.raw, result.hit_token_budget) == ("some text", True)
    (call,) = generator.drain()
    assert (call["raw"], call["hit_token_budget"], call["model"]) == ("some text", True, "fake/qwen")
    assert generator.drain() == []  # drained
    failing = recorded(error=RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):  # raised again, as it would have been
        failing.generate("a review", None)
    assert failing.drain()[0]["error"] == "RuntimeError: CUDA out of memory"


def test_a_model_that_has_never_worked_is_not_tried_again_but_one_that_has_is():
    class Flaky:
        model_name = "fake/qwen"

        def __init__(self, outcomes):
            self.outcomes, self.calls = list(outcomes), 0

        def generate(self, review, signals):
            self.calls += 1
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    ok = FakeGenerator().generate("r", None)
    never = smoke_script.RecordingGenerator(Flaky([OSError("cannot load"), ok, ok]))
    with pytest.raises(OSError):
        never.generate("a", None)
    for _ in range(2):  # the second and third calls do not reach the model
        with pytest.raises(RuntimeError, match="not tried again after its first failure: OSError: cannot load"):
            never.generate("b", None)
    assert never.inner.calls == 1
    assert [c["error"] for c in never.drain()] == ["OSError: cannot load"] * 3  # all three are the same, first failure
    worked = smoke_script.RecordingGenerator(Flaky([ok, OSError("this review only"), ok]))
    worked.generate("a", None)
    with pytest.raises(OSError):
        worked.generate("b", None)
    assert worked.generate("c", None) is ok and worked.inner.calls == 3  # it had worked, so it is tried again


def test_each_record_carries_what_the_suggestion_model_wrote_for_that_review():
    arms = two_arms(generator=recorded(raw="I cannot help."))
    records = smoke_script.smoke(arms, REVIEWS[:2])
    assert all(r["generation"]["raw"] == "I cannot help." for r in records)  # asked in both arms, for both reviews
    plain = smoke_script.smoke(two_arms(), REVIEWS[:1])  # a generator that is not recorded
    assert all(r["generation"] is None for r in plain)
    quiet = smoke_script.smoke(two_arms(wrapper=FakeWrapper(POSITIVE), jev=FakeJev(), generator=recorded()), REVIEWS[:1])
    assert all(r["generation"] is None for r in quiet)  # not asked, so nothing was written


def test_a_suggestion_model_that_raises_on_every_review_is_a_problem_with_the_exception_in_it():
    arms = two_arms(generator=recorded(error=RuntimeError("CUDA out of memory")))
    records = smoke_script.smoke(arms, REVIEWS)
    problems, notes = smoke_script.check(records)
    assert problems == ["the suggestion model did not run on 28 of the 28 reviews it was asked about "
                        "(first: pos-1 (without-jev)): RuntimeError: CUDA out of memory"]  # one line, not 28
    assert not any("suggestion stage gave" in n for n in notes)
    unrecorded = smoke_script.check(smoke_script.smoke(two_arms(generator=FakeGenerator(error=RuntimeError("x"))),
                                                        REVIEWS[:2]))[0]
    assert len(unrecorded) == 1 and unrecorded[0].endswith("): generation_failed")  # still a problem without the recorder


def test_a_suggestion_stage_that_was_never_asked_has_not_been_tested_and_that_is_a_problem():
    arms = two_arms(wrapper=FakeWrapper(POSITIVE), jev=FakeJev(), generator=recorded())  # nothing is flagged
    problems, _ = smoke_script.check(smoke_script.smoke(arms, REVIEWS))
    assert any("without-jev: the suggestion stage was not asked about any of the 14 reviews" in p for p in problems)
    assert any("with-jev: the suggestion stage was not asked" in p for p in problems)
    flagged = two_arms(wrapper=FakeWrapper(POSITIVE), jev=FakeJev(yes=["bathroom"]), generator=recorded())
    problems, _ = smoke_script.check(smoke_script.smoke(flagged, REVIEWS))
    assert len(problems) == 1 and problems[0].startswith("without-jev:")  # Jev flagged reviews, so that arm asked


def test_a_suggestion_stage_that_is_off_is_a_problem_only_if_it_was_meant_to_be_on():
    off = {"without-jev": TriagePipeline(FakeWrapper(NEGATIVE), None, None)}
    records = smoke_script.smoke(off, REVIEWS[:3])
    assert any("off in every review, though it was not switched off" in p for p in smoke_script.check(records)[0])
    assert smoke_script.check(records, qwen_enabled=False)[0] == []


def test_what_the_model_wrote_is_a_note_whether_it_was_invalid_cut_off_or_empty():
    for generator, wanted in [(recorded(raw="I cannot help."), "error (invalid_output)"),
                              (recorded(raw=GOOD_ACTIONS, hit_token_budget=True), "error (hit_token_budget)"),
                              (recorded(raw='{"actions": []}'), "gave an empty list")]:
        problems, notes = smoke_script.check(smoke_script.smoke(two_arms(generator=generator), REVIEWS[4:5]))
        assert problems == [], wanted
        assert any(wanted in note for note in notes), wanted


def test_the_stage_counts_say_how_many_reviews_the_suggestion_model_wrote_for():
    arms = two_arms(generator=recorded(raw="I cannot help."))
    counts = smoke_script.stage_counts(smoke_script.smoke(arms, REVIEWS[:3]))
    assert counts["without-jev"] == {"reviews": 3, "crashed": 0, "complaints": {"disabled": 3},
                                     "actions": {"error:invalid_output": 3}, "suggestion_model_wrote": 3}
    assert counts["with-jev"]["complaints"] == {"ok": 3}
    mixed = smoke_script.stage_counts(smoke_script.smoke(two_arms(generator=recorded(error=RuntimeError("x"))),
                                                         REVIEWS[:2]))
    assert mixed["with-jev"]["actions"] == {"error:generation_failed": 2} and mixed["with-jev"]["suggestion_model_wrote"] == 0
    crashed = smoke_script.stage_counts(smoke_script.smoke(two_arms(wrapper=FakeWrapper(error=RuntimeError("b"))),
                                                           REVIEWS[:1]))
    assert crashed["with-jev"]["crashed"] == 1 and crashed["with-jev"]["actions"] == {}


def info_for(records, **changes):
    args = SimpleNamespace(**{**dict(hub_repo="r/m", revision="rev", model_path=None, route="openrouter",
                                     jev_model="jev-latest", no_qwen=False, qwen_model="Qwen/x"), **changes})
    return smoke_script.run_info(args, records, smoke_script.DEFAULT_REVIEWS, "t0", "t1")


def test_the_summary_shows_the_stages_and_what_the_model_wrote_only_where_it_gave_nothing_usable():
    bad = smoke_script.smoke(two_arms(generator=recorded(raw="```json\nnot what was asked\n```")), REVIEWS[:2])
    text = smoke_script.summary_markdown(info_for(bad), bad, *smoke_script.check(bad))
    assert "## Stages" in text and "| with-jev | 2 | ok 2 | error:invalid_output 2 | 2 |" in text
    assert "## What the suggestion model wrote where it gave no usable actions" in text and "not what was asked" in text
    assert "````\n```json" in text  # the raw text has backticks, so its fence has four
    good = smoke_script.smoke(two_arms(), REVIEWS[:2])
    clean = smoke_script.summary_markdown(info_for(good), good, *smoke_script.check(good))
    assert "## Stages" in clean and "What the suggestion model wrote" not in clean


def test_a_long_output_is_cut_in_the_summary_and_kept_whole_in_the_results():
    records = smoke_script.smoke(two_arms(generator=recorded(raw="x" * 2000)), REVIEWS[:1])
    text = smoke_script.summary_markdown(info_for(records), records, *smoke_script.check(records))
    assert "(1200 more characters)" in text and "x" * 801 not in text
    assert len(records[0]["generation"]["raw"]) == 2000


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


def test_the_command_returns_one_and_says_why_when_the_suggestion_model_cannot_run(runs_folder, capsys):
    broken = lambda args: two_arms(generator=recorded(error=OSError("cannot load the weights")))  # noqa: E731
    assert smoke_script.main(["--limit", "2", "--output", "runs/t4"], build=broken) == 1
    out = capsys.readouterr().out
    assert "PROBLEM:" in out and "OSError: cannot load the weights" in out
    assert "the suggestion model wrote text for 0 of 2 reviews" in out
    info = json.loads((runs_folder / "t4" / "run.json").read_text(encoding="utf-8"))
    assert info["stages"]["with-jev"]["suggestion_model_wrote"] == 0
    assert len(info["problems"]) == 1 and "did not run on 4 of the 4 reviews" in info["problems"][0]


def test_a_run_without_the_suggestion_model_by_choice_is_not_a_problem(runs_folder):
    off = lambda args: {"without-jev": TriagePipeline(FakeWrapper(NEGATIVE), None, None)}  # noqa: E731
    assert smoke_script.main(["--limit", "2", "--no-qwen", "--output", "runs/t5"], build=off) == 0
    info = json.loads((runs_folder / "t5" / "run.json").read_text(encoding="utf-8"))
    assert info["qwen"] is None and info["stages"]["without-jev"]["actions"] == {"disabled": 2}


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


# --- The Colab notebook ---------------------------------------------------------------------------------------------

NOTEBOOK = ROOT / "notebooks" / "13_triage_smoke_colab.ipynb"


def _notebook_code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_the_notebook_runs_the_pinned_script_with_flags_that_script_has():
    import subprocess

    revision = re.search(r'REVISION = "([0-9a-f]{40})"', _notebook_code()).group(1)
    shown = subprocess.run(["git", "show", f"{revision}:scripts/triage_smoke.py"], cwd=ROOT, capture_output=True)
    if shown.returncode != 0:
        pytest.skip("the pinned commit is not in this clone")
    pinned_flags = set(re.findall(r'add_argument\("(--[a-z-]+)"', shown.stdout.decode("utf-8")))
    used = set(re.findall(r"--[a-z][a-z-]+", _notebook_code())) - {"--quiet", "--no-deps", "--oneline"}  # pip's, git's
    assert {"--output", "--allow-external-api", "--route"} <= used <= pinned_flags


def test_the_notebook_reads_the_key_from_colab_secrets_and_never_shows_it():
    code = _notebook_code()
    assert 'userdata.get("OPENROUTER_API_KEY")' in code
    assert jev_client.ROUTES["openrouter"]["key_env"] == "OPENROUTER_API_KEY"
    assert not re.search(r"sk-[A-Za-z0-9]", NOTEBOOK.read_text(encoding="utf-8"))
    assert not re.search(r"print\([^)]*(environ|userdata)", code)  # the value is never printed
    assert "--allow-external-api" in code and "if USE_JEV" in code  # and Jev is off unless a key was found


def test_the_pinned_distilbert_is_the_one_in_the_publication_record():
    record = (ROOT / "docs" / "experiments" / "results" / "distilbert_v2" / "README.md").read_text(encoding="utf-8")
    assert f"`{smoke_script.HUB_REPO}` at `{smoke_script.HUB_REVISION}`" in record
