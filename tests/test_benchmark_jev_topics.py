"""The Jev harness, offline: which texts it sends, when it sends nothing, and how it scores.

No test reaches the network. The sender is replaced by a fake that answers in the form the
System One API documents (the wire schema of the provider's typesafe-sdk 0.7.2).
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reviewnlp.evaluation.metrics import wilson_interval
from reviewnlp.evaluation.significance import mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs" / "experiments" / "jev_topic_benchmark"
GUIDELINE = ROOT / "docs" / "annotation" / "complaint_topics_guideline.md"
DRAFT_COMMIT = "2c0847b3293945cfd875dd96163f2925dfaea0bc"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("benchmark_jev_topics")
score = _load("score_annotations")
KEY = bench.read_key(bench.RESULTS / "key.csv")
GOLD = bench.read_gold(bench.RESULTS / "labels_final.csv")
RANDOM = sorted(item for item, row in KEY.items() if row["in_R"])
KEY_ENV = bench.ROUTES["typesafe"]["key_env"]
OPENROUTER_ENV = bench.ROUTES["openrouter"]["key_env"]


def text_of(item: int) -> str:
    return f"made-up review {item} TEXTMARK"


def answer(choice: str) -> dict:
    probabilities = {label: (0.8 if label == choice else 0.1) for label in bench.LABELS}
    return {"type": "choice", "choice": choice, "confidence": 0.7, "probabilities": probabilities}


class FakeAPI:
    """Answers like the System One API and records every request.

    By default it answers responsiveness as the final labels do and 0 for the other topics.
    `plan` maps a call number to the (status, headers, body) to return instead.
    """

    def __init__(self, plan: dict | None = None):
        self.plan, self.calls, self.urls, self.sleeps = plan or {}, [], [], []

    def __call__(self, url: str, body: dict, api_key: str, timeout: float):
        self.calls.append((body, api_key))
        self.urls.append(url)
        if len(self.calls) in self.plan:
            planned = self.plan[len(self.calls)]
            if isinstance(planned, Exception):
                raise planned
            status, headers, payload = planned
            return status, headers, json.dumps(payload).encode()
        item = int(body["state"].split()[2])
        choices = {t: GOLD[item][t] if t == "responsiveness" else "0" for t in bench.TOPICS}
        response = {"model": "jev-2026-09-15", "answers": {t: answer(c) for t, c in choices.items()},
                    "usage": {"input_tokens": 900, "output_tokens": 5}}
        return 200, {}, json.dumps(response).encode()


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RUNS", tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def texts(tmp_path):
    """A sheet laid out like sheet A, with a made-up text for each of the 300 items."""
    path = tmp_path / "sheet.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "text", *bench.TOPICS, "done", "note"])
        for item in sorted(KEY):
            writer.writerow([item, text_of(item), *["0"] * len(bench.TOPICS), "yes", ""])
    return path


def run(monkeypatch, fake: FakeAPI, texts: Path, output: Path, *extra: str) -> None:
    monkeypatch.setattr(bench, "post", fake)
    monkeypatch.setattr(bench.time, "sleep", fake.sleeps.append)
    bench.main(["--texts", str(texts), "--output", str(output), *extra])


def test_the_committed_labels_give_the_pilot_baseline():
    assert len(RANDOM) == 200
    positives = {t: sum(GOLD[i][t] == "1" for i in RANDOM) for t in bench.TOPICS}
    assert positives == {"bathroom": 13, "cleanliness": 11, "air_conditioning": 18, "pests": 2,
                         "responsiveness": 16}
    assert sum(GOLD[i]["responsiveness"] == "1" and KEY[i]["responsiveness"] for i in RANDOM) == 2


def test_without_the_flag_nothing_is_sent(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    fake = FakeAPI()
    run(monkeypatch, fake, texts, runs / "jev")
    assert fake.calls == []
    assert [path.name for path in (runs / "jev").iterdir()] == ["request_example.json"]


def test_the_flag_without_a_key_sends_nothing(runs, texts, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    fake = FakeAPI()
    with pytest.raises(SystemExit, match="set TYPESAFE_API_KEY"):
        run(monkeypatch, fake, texts, runs / "jev", "--allow-external-api")
    assert fake.calls == []


def test_the_run_folder_must_be_inside_runs(runs, texts, tmp_path, monkeypatch):
    for output in (tmp_path / "elsewhere", runs):
        with pytest.raises(SystemExit, match="inside"):
            run(monkeypatch, FakeAPI(), texts, output)


def test_a_run_sends_each_random_text_once_and_writes_no_text_or_key(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    fake = FakeAPI()
    run(monkeypatch, fake, texts, runs / "jev", "--allow-external-api")
    assert [body["state"] for body, _ in fake.calls] == [text_of(item) for item in RANDOM]
    assert all(body == bench.request_body(body["state"]) for body, _ in fake.calls)
    assert {api_key for _, api_key in fake.calls} == {"sk-test-secret"}
    folder = runs / "jev"
    for path in folder.iterdir():
        assert "sk-test-secret" not in path.read_text(encoding="utf-8"), path.name
    for name in ("summary.json", "predictions.csv", "run.json"):
        assert "TEXTMARK" not in (folder / name).read_text(encoding="utf-8"), name
    with open(folder / "predictions.csv", newline="", encoding="utf-8") as handle:
        assert [int(row["item"]) for row in csv.DictReader(handle)] == RANDOM


def test_the_openrouter_route_has_its_own_endpoint_and_key(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-typesafe-secret")
    monkeypatch.setenv(OPENROUTER_ENV, "sk-or-secret")
    fake = FakeAPI()
    run(monkeypatch, fake, texts, runs / "jev", "--route", "openrouter", "--allow-external-api",
        "--limit", "2")
    assert set(fake.urls) == {"https://openrouter.ai/api/v1/systemone"}
    assert {api_key for _, api_key in fake.calls} == {"sk-or-secret"}
    assert all(body == bench.request_body(body["state"]) for body, _ in fake.calls)
    record = json.loads((runs / "jev" / "run.json").read_text(encoding="utf-8"))
    assert (record["route"], record["endpoint"], record["model_requested"]) == (
        "openrouter", "https://openrouter.ai/api/v1/systemone", "jev-latest")
    assert record["questions_sha256"] == bench.QUESTIONS_SHA256
    assert json.loads((runs / "jev" / "summary.json").read_text(encoding="utf-8"))["inputs"] == record
    for path in (runs / "jev").iterdir():
        for secret in ("sk-or-secret", "sk-typesafe-secret"):
            assert secret not in path.read_text(encoding="utf-8"), path.name


def test_a_model_name_for_the_route_changes_the_request_and_not_the_questions(runs, texts, monkeypatch):
    monkeypatch.setenv(OPENROUTER_ENV, "sk-or-secret")
    fake = FakeAPI()
    run(monkeypatch, fake, texts, runs / "jev", "--route", "openrouter", "--model", "~typesafe/jev-latest",
        "--allow-external-api", "--limit", "1")
    body, _ = fake.calls[0]
    assert body["model"] == "~typesafe/jev-latest" and body["questions"] == bench.QUESTIONS
    record = json.loads((runs / "jev" / "run.json").read_text(encoding="utf-8"))
    assert record["model_requested"] == "~typesafe/jev-latest"
    assert record["questions_sha256"] == bench.QUESTIONS_SHA256


def test_a_key_is_never_sent_to_the_other_provider(runs, texts, monkeypatch):
    fake = FakeAPI()
    monkeypatch.setenv(KEY_ENV, "sk-typesafe-secret")
    monkeypatch.delenv(OPENROUTER_ENV, raising=False)
    with pytest.raises(SystemExit, match="set OPENROUTER_API_KEY"):
        run(monkeypatch, fake, texts, runs / "one", "--route", "openrouter", "--allow-external-api")
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setenv(OPENROUTER_ENV, "sk-or-secret")
    with pytest.raises(SystemExit, match="set TYPESAFE_API_KEY"):
        run(monkeypatch, fake, texts, runs / "two", "--allow-external-api")
    assert fake.calls == []


def test_a_run_cannot_go_on_with_another_route_or_model(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-typesafe-secret")
    monkeypatch.setenv(OPENROUTER_ENV, "sk-or-secret")
    stopped = FakeAPI(plan={3: (401, {}, {"error": "invalid key"})})
    with pytest.raises(SystemExit, match="HTTP 401"):
        run(monkeypatch, stopped, texts, runs / "jev", "--allow-external-api")
    other = FakeAPI()
    for extra in (("--route", "openrouter"), ("--model", "another-model")):
        with pytest.raises(SystemExit, match="differ"):
            run(monkeypatch, other, texts, runs / "jev", "--allow-external-api", *extra)
    assert other.calls == []


def test_an_unknown_route_is_refused(runs, texts, monkeypatch):
    fake = FakeAPI()
    with pytest.raises(SystemExit):
        run(monkeypatch, fake, texts, runs / "jev", "--route", "elsewhere", "--allow-external-api")
    assert fake.calls == []


def test_the_default_run_folder_depends_on_the_route_and_the_dry_run_names_the_endpoint(
        runs, texts, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    bench.main(["--texts", str(texts), "--route", "openrouter"])
    assert (runs / "jev_topic_benchmark_openrouter" / "request_example.json").exists()
    assert not (runs / "jev_topic_benchmark_typesafe").exists()
    printed = capsys.readouterr().out
    assert "Nothing was sent" in printed and "https://openrouter.ai/api/v1/systemone" in printed


@pytest.fixture
def local_servers(monkeypatch):
    """An 'api' that records what it gets or redirects, and an 'other' host that records any visit."""
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    seen = {"api": [], "other": []}

    def handler(name: str, redirect_to: str | None = None):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                seen[name].append((self.path, dict(self.headers), json.loads(body)))
                if redirect_to and self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", f"{redirect_to}/stolen")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, *args):
                pass
        return Handler

    other = ThreadingHTTPServer(("127.0.0.1", 0), handler("other"))
    api = ThreadingHTTPServer(("127.0.0.1", 0), handler("api", f"http://127.0.0.1:{other.server_port}"))
    threads = [threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
               for server in (api, other)]
    for thread in threads:
        thread.start()
    yield f"http://127.0.0.1:{api.server_port}", seen
    for server in (api, other):
        server.shutdown()
        server.server_close()


def test_a_request_is_sent_as_the_providers_documents_and_a_redirect_does_not_carry_the_key(local_servers):
    base, seen = local_servers
    status, _, raw = bench.post(f"{base}/v1/systemone", {"state": "hello"}, "sk-secret", 10)
    assert (status, json.loads(raw)) == (200, {"ok": True})
    path, headers, body = seen["api"][0]
    assert (path, body) == ("/v1/systemone", {"state": "hello"})
    assert headers["Authorization"] == "Bearer sk-secret" and headers["Content-Type"] == "application/json"
    status, _, _ = bench.post(f"{base}/redirect", {"state": "hello"}, "sk-secret", 10)
    assert status == 302 and seen["other"] == []


def test_a_run_on_texts_other_than_the_handed_in_sheet_does_not_decide(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    run(monkeypatch, FakeAPI(), texts, runs / "jev", "--allow-external-api")
    summary = json.loads((runs / "jev" / "summary.json").read_text(encoding="utf-8"))
    decision = summary["decision"]
    assert (decision["only_jev"], decision["only_lexicon"]) == (14, 0)
    assert decision["jev_better"] and decision["guard_holds"] and not decision["rule_applies"]
    assert decision["outcome"] == "not decided: the texts file is not the handed-in sheet A"
    assert summary["texts"] == {**summary["texts"], "random": 200, "selected": 200, "answered": 200}
    assert summary["models_answered"] == ["jev-2026-09-15"]
    assert summary["tokens"] == {"input": 200 * 900, "output": 200 * 5}


def test_a_smoke_run_is_marked_and_does_not_decide(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    fake = FakeAPI()
    run(monkeypatch, fake, texts, runs / "jev", "--allow-external-api", "--limit", "3")
    assert len(fake.calls) == 3
    decision = json.loads((runs / "jev" / "summary.json").read_text(encoding="utf-8"))["decision"]
    assert decision["outcome"].startswith("not decided: smoke run on 3 of 200 texts")


def test_a_stopped_run_resumes_without_sending_a_text_twice(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    first = FakeAPI(plan={51: (401, {}, {"error": "invalid key"})})
    with pytest.raises(SystemExit, match="HTTP 401"):
        run(monkeypatch, first, texts, runs / "jev", "--allow-external-api")
    assert len(first.calls) == 51 and first.sleeps == []
    second = FakeAPI()
    run(monkeypatch, second, texts, runs / "jev", "--allow-external-api")
    assert [body["state"] for body, _ in second.calls] == [text_of(item) for item in RANDOM[50:]]
    summary = json.loads((runs / "jev" / "summary.json").read_text(encoding="utf-8"))
    assert summary["texts"]["answered"] == 200 and summary["requests"]["sent"] == 200


def test_retries_wait_as_the_server_asks_and_a_client_error_stops_at_once(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    busy = FakeAPI(plan={1: (429, {"Retry-After-Ms": "1500"}, {"error": "slow down"}),
                         2: (503, {}, {"error": "busy"}),
                         3: http.client.IncompleteRead(b"")})
    run(monkeypatch, busy, texts, runs / "busy", "--allow-external-api", "--limit", "1")
    assert len(busy.calls) == 4 and busy.sleeps == [1.5, 4.0, 8.0]
    assert json.loads((runs / "busy" / "summary.json").read_text(encoding="utf-8"))["requests"]["retried"] == 3
    refused = FakeAPI(plan={1: (422, {}, {"detail": [{"loc": ["body", "state"], "msg": "bad"}]})})
    with pytest.raises(SystemExit, match="HTTP 422"):
        run(monkeypatch, refused, texts, runs / "refused", "--allow-external-api")
    assert len(refused.calls) == 1 and refused.sleeps == []


def test_an_answer_in_another_form_stops_after_that_text(runs, texts, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    no_confidence = {t: {**answer("0"), "confidence": None} for t in bench.TOPICS}
    for number, response in enumerate(({"model": "jev", "answers": {"bathroom": answer("1")}, "usage": {}},
                                       {"model": "jev", "answers": no_confidence, "usage": {}})):
        fake = FakeAPI(plan={1: (200, {}, response)})
        with pytest.raises(SystemExit, match="not in the documented form"):
            run(monkeypatch, fake, texts, runs / f"jev{number}", "--allow-external-api")
        assert len(fake.calls) == 1
        assert len((runs / f"jev{number}" / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_scoring_only_reproduces_the_summary_and_refuses_other_inputs(runs, texts, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no saved answers"):
        run(monkeypatch, FakeAPI(), texts, runs / "jev", "--score-only")
    monkeypatch.setenv(KEY_ENV, "sk-test-secret")
    run(monkeypatch, FakeAPI(), texts, runs / "jev", "--allow-external-api")
    summary = (runs / "jev" / "summary.json").read_text(encoding="utf-8")
    fake = FakeAPI()
    run(monkeypatch, fake, texts, runs / "jev", "--score-only")
    assert fake.calls == [] and (runs / "jev" / "summary.json").read_text(encoding="utf-8") == summary
    other = tmp_path / "other.csv"
    other.write_text(texts.read_text(encoding="utf-8").replace("made-up", "changed"), encoding="utf-8")
    with pytest.raises(SystemExit, match="differ"):
        run(monkeypatch, fake, other, runs / "jev", "--score-only")


def test_scoring_on_a_known_example():
    items = list(range(1, 31))
    key = {i: {"in_R": True, **{t: i in (1, 2, 14) and t == "responsiveness" for t in bench.TOPICS}} for i in items}
    gold = {i: {t: "0" for t in bench.TOPICS} for i in items}
    choices = dict.fromkeys(items, "0")
    for i in range(1, 13):
        gold[i]["responsiveness"] = "1"
    gold[13]["responsiveness"] = "unsure"
    choices.update({i: "1" for i in (1, *range(3, 11), 15)} | {11: "unsure", 16: "unsure"})
    answers = {i: bench.read_answers({"answers": {t: answer(choices[i] if t == "responsiveness" else "0")
                                                  for t in bench.TOPICS}}) for i in items}
    report = bench.score(items, key, gold, answers)
    result = report["responsiveness"]
    assert (result["texts"], result["gold_unsure_left_out"]) == (29, 1)
    lexicon, jev = result["lexicon"], result["jev"]
    assert (lexicon["true_positives"], lexicon["false_positives"], lexicon["missed"]) == (2, 1, 10)
    assert (jev["true_positives"], jev["false_positives"], jev["missed"]) == (9, 1, 3)
    assert jev["recall"]["wilson95"] == [round(x, 4) for x in wilson_interval(9, 12)]
    assert jev["precision"]["estimate"] == 0.9
    assert jev["unsure"] == {"on_gold_1": 1, "on_gold_0": 1}
    assert jev["recall_if_unsure_counted_as_found"] == round(10 / 12, 4)
    assert result["paired_recall"] == {"both": 1, "only_jev": 8, "only_lexicon": 1, "neither": 2,
                                       "mcnemar_p": round(mcnemar_exact(8, 1)["p_value"], 4)}
    assert result["paired_false_positives"] == {"both": 0, "only_jev": 1, "only_lexicon": 1,
                                                "neither": 15, "mcnemar_p": 1.0}
    assert result["jev_calibration_error"] == pytest.approx(abs(24 / 27 - 0.8))
    assert report["pests"]["jev"]["precision"] == {"numerator": 0, "denominator": 0,
                                                   "status": "insufficient sample"}
    decision = bench.decide(report, [])
    assert decision["jev_better"] and decision["guard_holds"] and decision["outcome"].startswith("Jev finds more")
    assert bench.decide(report, ["a reason"])["outcome"] == "not decided: a reason"


def _verdict(only_jev: int, only_lexicon: int, true_positives: int, false_positives: int) -> dict:
    report = {"responsiveness": {"paired_recall": {"only_jev": only_jev, "only_lexicon": only_lexicon},
                                 "jev": {"true_positives": true_positives, "false_positives": false_positives}}}
    return bench.decide(report, [])


def test_the_decision_rule_at_its_edges():
    assert not _verdict(5, 0, 7, 0)["jev_better"]  # p = 0.0625
    assert _verdict(6, 0, 8, 0)["jev_better"]  # p = 0.03125
    assert not _verdict(7, 1, 9, 0)["jev_better"] and _verdict(8, 1, 10, 0)["jev_better"]
    assert _verdict(6, 0, 8, 8)["guard_holds"]  # exactly half of the flagged texts are right
    assert _verdict(6, 0, 8, 9)["outcome"] == "keep the lexicon"
    assert _verdict(0, 0, 0, 0)["outcome"] == "keep the lexicon"


def test_the_statistics_match_the_project_code():
    for total in range(1, 40):
        for successes in range(total + 1):
            assert bench.wilson(successes, total) == wilson_interval(successes, total)
            assert bench.proportion(successes, total) == score._proportion(successes, total)
    for b in range(25):
        for c in range(25):
            assert bench.mcnemar_p(b, c) == pytest.approx(mcnemar_exact(b, c)["p_value"], rel=1e-9)


def test_the_questions_carry_the_guidelines_rules():
    questions = json.dumps(bench.QUESTIONS).lower()
    guideline = " ".join(GUIDELINE.read_text(encoding="utf-8").lower().split())
    for rule in ("a suggestion or wish that implies a lack", "a mild complaint is still 1",
                 "only the setting of another complaint", "neutral mention",
                 "mark bathroom 1 only if something else about the bathroom is also criticised",
                 "booking.com, not the hotel, failing to pass something on"):
        assert rule in guideline and rule in questions, rule
    assert all(set(q["criteria"]) == set(bench.LABELS) for q in bench.QUESTIONS.values())


def test_the_questions_and_the_rule_are_the_ones_fixed_in_the_decision_note():
    note = (FOLDER / "DECISION_v2.md").read_text(encoding="utf-8")
    assert f"`{bench.QUESTIONS_SHA256}`" in note
    assert (bench.PRIMARY_TOPIC, bench.ALPHA, bench.MIN_PRECISION, bench.MODEL) == (
        "responsiveness", 0.05, 0.5, "jev-latest")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def test_the_first_draft_is_kept_as_committed():
    draft = _git("show", f"{DRAFT_COMMIT}:docs/experiments/jev_topic_benchmark/DECISION.md")
    if draft.returncode != 0:
        pytest.skip("the draft's commit is not in this clone")
    assert (FOLDER / "DECISION.md").read_bytes() == draft.stdout


def test_the_default_run_folders_are_ignored_by_git():
    for route in bench.ROUTES:
        for name in ("responses.jsonl", "request_example.json", "run.json", "summary.json", "predictions.csv"):
            assert _git("check-ignore", "-q", f"runs/jev_topic_benchmark_{route}/{name}").returncode == 0, name


# --- The confirmation stage: new texts, responsiveness alone -------------------------------------------------------

NEW_ITEMS = list(range(1, 41))


class ScriptedAPI(FakeAPI):
    """Answers responsiveness `1` for the items it is told to flag, `unsure` for some, and `0` for the rest."""

    def __init__(self, flagged, unsure=()):
        super().__init__()
        self.flagged, self.unsure = set(flagged), set(unsure)

    def __call__(self, url: str, body: dict, api_key: str, timeout: float):
        self.calls.append((body, api_key))
        self.urls.append(url)
        item = int(body["state"].split()[2])
        choice = "1" if item in self.flagged else "unsure" if item in self.unsure else "0"
        response = {"model": "jev-2026-09-15", "usage": {"input_tokens": 900, "output_tokens": 5},
                    "answers": {t: answer(choice if t == "responsiveness" else "0") for t in bench.TOPICS}}
        return 200, {}, json.dumps(response).encode()


@pytest.fixture
def new_files(tmp_path):
    """Forty new texts: 1-12 are complaints, 13 is `unsure`, the rest are not; the lexicon matches 1, 2 and 14."""
    sheet = tmp_path / "new_sheet.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "text", "responsiveness", "done", "note"])
        writer.writerows([item, text_of(item), "", "yes", ""] for item in NEW_ITEMS)
    key = tmp_path / "new_key.csv"
    with open(key, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "in_R", *(f"lex_{t}" for t in bench.TOPICS)])
        for item in NEW_ITEMS:
            writer.writerow([item, 1, *[int(item in (1, 2, 14) and t == "responsiveness") for t in bench.TOPICS]])
    gold = tmp_path / "new_labels.csv"
    with open(gold, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "annotator", "responsiveness"])
        writer.writerows([item, "final", "1" if item <= 12 else "unsure" if item == 13 else "0"]
                         for item in NEW_ITEMS)
    return sheet, key, gold


def run_confirmation(monkeypatch, runs, files, api, *extra, sha: str | None = None) -> dict:
    sheet, key, gold = files
    sha = sha or hashlib.sha256(sheet.read_bytes()).hexdigest()
    monkeypatch.setenv(OPENROUTER_ENV, "sk-or-secret")
    run(monkeypatch, api, sheet, runs / "new", "--stage", "confirmation", "--key", str(key), "--gold", str(gold),
        "--expected-texts-sha256", sha, "--route", "openrouter", "--allow-external-api", *extra)
    return json.loads((runs / "new" / "summary.json").read_text(encoding="utf-8"))


def test_the_confirmation_stage_scores_responsiveness_alone_on_the_new_texts(runs, new_files, monkeypatch):
    api = ScriptedAPI(flagged=[*range(1, 11), 15, 16])
    summary = run_confirmation(monkeypatch, runs, new_files, api)
    assert [body["state"] for body, _ in api.calls] == [text_of(item) for item in NEW_ITEMS]
    assert summary["design"] == "docs/annotation/confirmation_protocol.md"
    assert summary["inputs"]["stage"] == "confirmation" and summary["inputs"]["topics_scored"] == ["responsiveness"]
    assert summary["texts"]["random"] == summary["texts"]["selected"] == summary["texts"]["answered"] == 40
    assert list(summary["topics"]) == ["responsiveness"]
    result = summary["topics"]["responsiveness"]
    assert (result["texts"], result["gold_unsure_left_out"]) == (39, 1)
    assert (result["jev"]["true_positives"], result["jev"]["false_positives"]) == (10, 2)
    assert (result["lexicon"]["true_positives"], result["lexicon"]["false_positives"]) == (2, 1)
    decision = summary["decision"]
    assert (decision["only_jev"], decision["only_lexicon"], decision["mcnemar_p"]) == (8, 0, 0.0078)
    assert decision["jev_precision"] == 0.8333 and decision["rule_applies"]
    assert decision["outcome"].startswith("confirmed: Jev finds more responsiveness complaints on new texts too")


@pytest.mark.parametrize("flagged, outcome", [
    ([*range(1, 11), *range(15, 29)], "not confirmed as a whole: Jev finds more, but fewer than half"),
    ([1, 2], "not confirmed: the pilot's result did not repeat; keep the lexicon"),
    ([1, 2, 3, 4, 5, 15], "not confirmed: the pilot's result did not repeat; keep the lexicon"),
])
def test_the_confirmation_outcomes_follow_the_protocols_table(runs, new_files, monkeypatch, flagged, outcome):
    # 1-10 plus 14 false flags: precision 10/24, below the guard. Only the two the lexicon has: nothing more.
    # Three more than the lexicon (3, 4, 5) and one false flag: b = 3, c = 0, p = 0.25, so Jev does not find more.
    summary = run_confirmation(monkeypatch, runs, new_files, ScriptedAPI(flagged=flagged))
    assert summary["decision"]["outcome"].startswith(outcome)
    assert summary["decision"]["rule_applies"]


def test_a_confirmation_on_another_sheet_than_the_recorded_one_does_not_decide(runs, new_files, monkeypatch):
    summary = run_confirmation(monkeypatch, runs, new_files, ScriptedAPI(flagged=range(1, 11)), sha="0" * 64)
    assert summary["decision"]["outcome"] == "not decided: the texts file is not the sheet the labels came back on"
    assert not summary["decision"]["rule_applies"]


def test_a_confirmation_smoke_run_does_not_decide(runs, new_files, monkeypatch):
    summary = run_confirmation(monkeypatch, runs, new_files, ScriptedAPI(flagged=range(1, 11)), "--limit", "3")
    assert summary["decision"]["outcome"] == "not decided: smoke run on 3 of 40 texts"


def test_the_topics_scored_must_include_the_one_the_rule_is_about(runs, new_files, monkeypatch):
    sheet, key, gold = new_files
    with pytest.raises(SystemExit, match="must include responsiveness"):
        run(monkeypatch, FakeAPI(), sheet, runs / "new", "--stage", "confirmation", "--topics", "bathroom",
            "--key", str(key), "--gold", str(gold))


def test_a_key_and_labels_for_one_topic_are_read_only_when_one_topic_is_asked_for(new_files):
    _, key, gold = new_files
    with pytest.raises(SystemExit, match="missing columns"):
        bench.read_gold(gold)
    assert bench.read_gold(gold, ("responsiveness",))[13] == {"responsiveness": "unsure"}
    assert bench.read_key(key)[14]["responsiveness"] is True
    assert bench.read_key(key, ("responsiveness",))[3] == {"in_R": True, "responsiveness": False}


def test_a_dry_run_of_the_confirmation_names_responsiveness_only(runs, new_files, monkeypatch, capsys):
    sheet, key, gold = new_files
    run(monkeypatch, FakeAPI(), sheet, runs / "new", "--stage", "confirmation", "--key", str(key),
        "--gold", str(gold), "--route", "openrouter")
    printed = capsys.readouterr().out
    assert "responsiveness" in printed and "bathroom" not in printed and "Nothing was sent" in printed


def test_the_stages_differ_only_in_what_they_read_and_say():
    pilot, confirmation = bench.STAGES["pilot"], bench.STAGES["confirmation"]
    assert pilot["topics"] == bench.TOPICS and confirmation["topics"] == (bench.PRIMARY_TOPIC,)
    assert pilot["design"].endswith("DECISION_v2.md") and confirmation["design"].endswith("confirmation_protocol.md")
    assert confirmation["key"].parent == confirmation["gold"].parent == bench.COMMITTED
    assert (ROOT / confirmation["design"]).exists()
    protocol = (ROOT / confirmation["design"]).read_text(encoding="utf-8")
    assert f"`{bench.QUESTIONS_SHA256}`" in protocol  # the confirmation asks the same questions
    for outcome in ("**Confirmed.**", "**Not confirmed as a whole.**", "**Not confirmed.**"):
        assert outcome in protocol
