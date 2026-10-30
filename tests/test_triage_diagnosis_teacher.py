"""The teacher of the triage diagnosis protocol (version 4) and its metric, with a fake OpenRouter; no network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from reviewnlp.triage import diagnosis_metrics as metrics

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


teacher = _load("triage_diagnosis_teacher")
labels = teacher.labels
TYPES = labels.sample.catalogue_types()
KEY = "sk-or-test-secret"


class FakeOpenRouter:
    """Answers each call from a queue of behaviours, then labels odd items `cleanliness` and even items nothing."""

    def __init__(self, behaviours=(), cost=0.01):
        self.behaviours, self.cost, self.calls = list(behaviours), cost, []

    def __call__(self, url, body, api_key, timeout):
        self.calls.append((url, body, api_key))
        reviews = json.loads(body["messages"][1]["content"])
        entries = [{"item": r["item"], "types": ["cleanliness"] if r["item"] % 2 else []} for r in reviews]
        behaviour = self.behaviours.pop(0) if self.behaviours else "ok"
        if isinstance(behaviour, int):
            return behaviour, {"retry-after": "0"}, b'{"error": "status"}'
        if behaviour == "garbled":
            content = "I cannot help with that"
        elif behaviour == "missing":
            content = json.dumps({"labels": entries[1:]})
        elif behaviour == "fenced":
            content = "```json\n" + json.dumps({"labels": entries}) + "\n```"
        else:
            content = json.dumps({"labels": entries})
        answer = {"choices": [{"message": {"content": content}}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": self.cost}}
        return 200, {}, json.dumps(answer).encode()


@pytest.fixture
def layout(tmp_path):
    """Runs, samples and labels for 6 development and 6 test reviews, with ChatGPT reference labels."""
    runs, samples, out = tmp_path / "runs", tmp_path / "samples", tmp_path / "labels"
    for folder in (runs, samples, out):
        folder.mkdir()
    texts = [f"Disliked: the room was dirty {i}" for i in range(12)]
    key = pd.DataFrame({"item": range(1, 13), "set": ["development", "test"] * 6,
                        "raw_row": [500 + i for i in range(12)], "stratum": "H", "weight": [1.0, 2.0, 3.0] * 4,
                        "pool": 85, "text_sha256": [labels.sample.text_hash(t) for t in texts]})
    key.to_csv(samples / "key.csv", index=False)
    pd.DataFrame({"item": range(1, 13), "text": texts}).to_csv(runs / "sheet.csv", index=False, encoding="utf-8-sig")
    reference = pd.DataFrame({"raw_row": key["raw_row"], "set": key["set"], **{t: 0 for t in TYPES}, "none": 0})
    reference["cleanliness"] = 1  # every review: one urgent pair, housekeeping
    reference.to_csv(out / "reference_chatgpt.csv", index=False)
    return runs, samples, out


def run(layout, transport, model="haiku", subset="development", max_usd=5.0, sleeps=None):
    runs, samples, out = layout
    return teacher.label(model, subset, max_usd, transport, KEY, runs, samples, out,
                         sleep=(sleeps.append if sleeps is not None else lambda _: None))


def test_the_metric_on_a_hand_worked_example():
    urgent = {"cleanliness": "housekeeping", "pests": "housekeeping", "hvac_failure": "maintenance"}
    gold = pd.DataFrame({"cleanliness": [1, 1, 0, 0], "pests": [0, 0, 0, 0], "hvac_failure": [1, 0, 0, 0]},
                        index=[10, 11, 12, 13])
    pred = pd.DataFrame({"cleanliness": [0, 0, 1, 0], "pests": [1, 0, 0, 0], "hvac_failure": [0, 1, 0, 0]},
                        index=[10, 11, 12, 13])
    counts = metrics.per_review(gold, pred, urgent)
    # Review 10: cleanliness caught through pests (same department), hvac missed. Review 11: missed, and its
    # hvac flag is false. Review 12: one false flag. Review 13: nothing.
    assert counts.to_dict("list") == {"pairs": [2, 1, 0, 0], "caught": [1, 0, 0, 0], "false_urgent": [0, 1, 1, 0]}
    weights = pd.Series([1.0, 3.0, 2.0, 2.0], index=[10, 11, 12, 13])
    assert metrics.department_urgent_recall(counts, weights) == pytest.approx(1 / (2 + 3))
    assert metrics.false_urgent_per_100(counts, weights) == pytest.approx(100 * (3 + 2) / 8)
    missing = metrics.per_review(gold, pred.drop(index=10), urgent)
    assert missing.loc[10, "caught"] == 0 and missing.loc[10, "false_urgent"] == 0


def test_a_call_carries_the_prompt_the_reviews_and_the_schema_and_nothing_else(layout):
    fake = FakeOpenRouter()
    run(layout, fake)
    url, body, api_key = fake.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions" and api_key == KEY
    assert body["model"] == "anthropic/claude-haiku-5.5-20261007"
    assert body["messages"][0] == {"role": "system", "content": labels.teacher_prompt()}
    assert json.loads(body["messages"][1]["content"]) == [
        {"item": i, "text": f"Disliked: the room was dirty {i - 1}"} for i in (1, 3, 5, 7, 9, 11)]
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["labels"]["items"]["properties"]["types"]["items"]["enum"] == TYPES
    assert set(body) == {"model", "messages", "response_format", "usage"}  # no sampling parameters


def test_a_finished_run_writes_labels_without_text_and_a_record_of_every_response(layout, monkeypatch):
    monkeypatch.setattr(teacher, "BATCH", 2)
    _, _, out = layout
    record = run(layout, FakeOpenRouter())
    table = pd.read_csv(out / "teacher_haiku_development.csv")
    assert list(table.columns) == ["raw_row", "set", *TYPES, "none", "failed"]
    assert table["cleanliness"].tolist() == [1, 1, 1, 1, 1, 1]  # development items are 1, 3, ... 11: all odd
    assert (record["batches"], record["calls"], record["retries"], record["failed_reviews"]) == (3, 3, 0, 0)
    assert record["cost_usd"] == pytest.approx(0.03) and record["cost_sources"] == ["reported"]
    assert len(record["responses"]) == 3 and all(len(v) == 1 for v in record["responses"].values())
    written = [p.read_bytes() for p in out.parent.rglob("*") if p.is_file()]
    assert not any(KEY.encode() in data for data in written)
    assert b"Disliked" not in (out / "teacher_haiku_development.csv").read_bytes()
    with pytest.raises(teacher.Stop, match="never replaced"):
        run(layout, FakeOpenRouter())


def test_bad_outputs_are_retried_twice_then_the_reviews_fail(layout, monkeypatch):
    monkeypatch.setattr(teacher, "BATCH", 2)
    _, _, out = layout
    fake = FakeOpenRouter(["garbled", "missing", "fenced", "garbled", "garbled", "garbled"])
    record = run(layout, fake)
    # Batch 1: garbled, missing, then a fenced answer that parses. Batch 2: three garbled tries. Batch 3: ok.
    assert record["calls"] == 7 and record["retries"] == 4 and record["failed_reviews"] == 2
    table = pd.read_csv(out / "teacher_haiku_development.csv").set_index("raw_row")
    failed = table[table["failed"] == 1]
    assert len(failed) == 2 and (failed["none"] == 0).all() and (failed[TYPES].sum(axis=1) == 0).all()


def test_rate_limits_are_waited_out_and_other_errors_stop_the_run_until_it_resumes(layout, monkeypatch):
    monkeypatch.setattr(teacher, "BATCH", 2)
    sleeps = []
    fake = FakeOpenRouter([429, 503, "ok", 402])
    with pytest.raises(teacher.Stop, match="402"):
        run(layout, fake, sleeps=sleeps)
    assert len(sleeps) == 2 and len(fake.calls) == 4
    resumed = FakeOpenRouter()
    record = run(layout, resumed)
    assert len(resumed.calls) == 2  # batch 1 is not sent again
    assert record["calls"] == 3 and record["retries"] == 0


def test_the_spending_limit_stops_before_a_call_and_the_run_resumes(layout, monkeypatch):
    monkeypatch.setattr(teacher, "BATCH", 1)
    first = FakeOpenRouter(cost=0.4)
    with pytest.raises(teacher.Stop, match="max-usd 1.0 reached"):
        run(layout, first, max_usd=1.0)
    assert len(first.calls) == 3  # 0.4 + 0.4 + 0.4 >= 1.0 before a fourth call
    with pytest.raises(teacher.Stop, match="max-usd 1.0 reached"):
        run(layout, FakeOpenRouter(cost=0.4), max_usd=1.0)  # what was spent counts on resuming
    second = FakeOpenRouter(cost=0.4)
    record = run(layout, second, max_usd=5.0)
    assert len(second.calls) == 3 and record["calls"] == 6 and record["cost_usd"] == pytest.approx(2.4)


def test_nothing_is_sent_without_a_key_or_with_a_changed_text(layout, monkeypatch):
    runs, samples, out = layout
    fake = FakeOpenRouter()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(teacher.Stop, match="OPENROUTER_API_KEY"):
        teacher.label("haiku", "development", 5.0, fake, None, runs, samples, out)
    sheet = pd.read_csv(runs / "sheet.csv", dtype=str, encoding="utf-8-sig")
    sheet.loc[0, "text"] = "something else"
    sheet.to_csv(runs / "sheet.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(teacher.Stop, match="does not hold the texts"):
        run(layout, fake)
    assert fake.calls == []


def write_candidate(out: Path, model_key: str, caught_rows: list[int]) -> None:
    reference = pd.read_csv(out / "reference_chatgpt.csv")
    dev = reference[reference["set"] == "development"].copy()
    dev[TYPES] = 0
    dev["cleanliness"] = dev["raw_row"].isin(caught_rows).astype(int)
    dev["none"], dev["failed"] = 1 - dev["cleanliness"], 0
    dev.to_csv(out / f"teacher_{model_key}_development.csv", index=False)
    (out / f"teacher_{model_key}_development_record.json").write_text(
        json.dumps({"cost_usd": 0.1, "failed_reviews": 0}), encoding="utf-8")


@pytest.mark.parametrize(("sonnet_rows", "chosen"), [
    ([500, 502, 504, 506, 508, 510], "sonnet"),  # Sonnet catches all; Haiku misses enough
    ([500, 502, 504, 506, 508], "haiku"),        # a lead below 0.05 keeps Haiku
])
def test_choose_applies_the_rule_on_development(layout, sonnet_rows, chosen):
    _, samples, out = layout
    # Development raw rows 500, 502, ... 510 carry weights 1, 3, 2, 1, 3, 2.
    write_candidate(out, "haiku", [500, 502, 504, 506, 508])
    write_candidate(out, "sonnet", sonnet_rows)
    result = teacher.choose(out, samples)
    assert result["rec_dev"]["haiku"] == pytest.approx(10 / 12, abs=1e-6)
    assert result["teacher"] == chosen and result["model"] == teacher.MODELS[chosen]
    assert teacher.model_key_of("chosen", out) == chosen
    with pytest.raises(teacher.Stop, match="chosen once"):
        teacher.choose(out, samples)


def test_the_chosen_model_is_unknown_until_choose_has_run(layout):
    _, _, out = layout
    with pytest.raises(teacher.Stop, match="run `choose`"):
        teacher.model_key_of("chosen", out)
