"""The hoteliers' blind evaluation Space: blinding, order, answers, storage, the app's two steps, the bundle and the
deployment. No network; the app is built but not launched."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "spaces" / "hotelier-blind-eval"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


logic = _load("hotelier_space_logic", SPACE / "logic.py")
_saved = sys.modules.get("logic")
sys.modules["logic"] = logic  # app.py imports its sibling as `logic`
try:
    app = _load("hotelier_space_app", SPACE / "app.py")
finally:
    if _saved is None:
        sys.modules.pop("logic", None)
    else:
        sys.modules["logic"] = _saved
evaluation = _load("hotelier_blind_eval", ROOT / "scripts" / "hotelier_blind_eval.py")
deploy = _load("deploy_hotelier_space", ROOT / "scripts" / "deploy_hotelier_space.py")
DISPLAY = evaluation.display_table()
REVEALING = ("teacher", "student", "claude", "haiku", "distilbert", "openrouter", "llm", "chatgpt")


def complete(q1="Ναι"):
    return (q1, [] if q1 == "Ναι" else ["Λείπει επείγον πρόβλημα"], "Εν μέρει", "4")


def test_nothing_the_hotelier_can_open_names_a_system():
    texts = [(SPACE / name).read_text(encoding="utf-8") for name in ("app.py", "logic.py", "README.md")]
    texts.append(json.dumps(DISPLAY, ensure_ascii=False))
    for text in texts:
        assert not [word for word in REVEALING if word in text.lower()]
    assert logic.SYSTEMS == ("x", "y") and set(evaluation.NEUTRAL) == {"x", "y"}
    assert set(evaluation.NEUTRAL.values()) == {"teacher", "student"}


def test_each_hotelier_has_a_fixed_order_and_balanced_sides():
    items = list(range(1, 61))
    first = logic.item_order("H01", items)
    assert sorted(first) == items and first == logic.item_order("H01", items) != logic.item_order("H02", items)
    left = [logic.system_1("H01", i) for i in items]
    assert 20 <= left.count("x") <= 40  # a hash, so about half each way
    assert logic.remaining("H01", items, {first[0], first[1]}) == first[2:]


def test_codes_are_checked_by_their_hash_only():
    codes = {hashlib.sha256(b"k3y-one").hexdigest(): "H01"}
    assert logic.hotelier_for(" k3y-one ", codes) == "H01" and logic.hotelier_for("k3y-two", codes) is None


def test_both_outputs_render_alike_urgent_first():
    shown = logic.render(["wifi", "cleanliness", "price_value"], DISPLAY)
    assert shown == logic.render(["price_value", "cleanliness", "wifi"], DISPLAY)
    lines = shown.splitlines()
    assert len(lines) == 3 and all(line.startswith("- ") for line in lines)  # one line per problem, no nesting
    assert "**ΕΠΕΙΓΟΝ**" in lines[0] and "Καθαριότητα" in lines[0] and "Υπηρεσία ορόφων" in lines[0]
    assert "<br>Ενέργεια: Νέος καθαρισμός" in lines[0] and "(πριν την επόμενη άφιξη)" in lines[0]
    assert "Wi-Fi" in lines[1] and "Τιμή και αξία" in lines[2]
    assert len(logic.render(["wifi"], DISPLAY).splitlines()) == 1  # a single problem shows as one, not two
    assert logic.render([], DISPLAY) == logic.NO_PROBLEM


def test_the_display_table_covers_the_catalogue_in_greek():
    assert len(DISPLAY["order"]) == 21 and set(DISPLAY["types"]) == set(DISPLAY["order"])
    for entry in DISPLAY["types"].values():
        assert entry["label"] and entry["measure"] and entry["department_el"] and entry["priority_el"]
    assert DISPLAY["types"]["safety_fault"]["deadline_el"] == "σήμερα"
    assert DISPLAY["types"]["noise_external"]["deadline_el"] is None


def test_answers_must_be_complete_and_consistent():
    assert logic.problems(*complete(), "Σύστημα 1") == [] and logic.problems(*complete("Όχι"), "Σύστημα 1") == []
    assert len(logic.problems(None, [], None, None, "Σύστημα 1")) == 3
    assert logic.problems("Όχι", [], "Ναι", "3", "Σύστημα 2") == ["Σύστημα 2: σημείωσε τι είναι λάθος στα επείγοντα."]
    assert "άφησε κενό" in logic.problems("Ναι", ["Λάθος τμήμα σε επείγον πρόβλημα"], "Ναι", "3", "1")[0]


def test_a_record_says_which_system_was_shown_where_and_is_saved_once():
    first = logic.system_1("H01", 7)
    one, two = logic.answers(*complete()), logic.answers(*complete("Όχι"))
    rating = logic.record("H01", 7, one, two, "Σύστημα 2", {"protocol_sha256": "p", "bundle": "b"})
    other = "y" if first == "x" else "x"
    assert rating["system_1"] == first and rating["by_system"] == {first: one, other: two}
    assert rating["preference"] == "2" and rating["preferred_system"] == other
    assert two == {"q1_urgent_right": False, "q2_what_is_wrong": ["missed_urgent"], "q3_others": "partly",
                   "q4_useful": 4}
    store = logic.MemoryStore()
    assert store.save(rating) and not store.save({**rating, "preference": "same"})
    assert store.done("H01") == {7} and json.loads(store.files["ratings/H01/item_07.json"])["preference"] == "2"


@pytest.fixture
def built():
    items = {i: {"item": i, "text": f"Disliked: the room was dirty {i}", "x": ["cleanliness"], "y": []}
             for i in (1, 2)}
    store = logic.MemoryStore()
    codes = {hashlib.sha256(b"abc").hexdigest(): "H01"}
    demo = app.build_app(items, DISPLAY, {"protocol_sha256": "p", "bundle": "b"}, store, codes, preview=False)
    steps = {f.fn.__name__: f for f in demo.fns.values() if getattr(f.fn, "__name__", "") in ("on_start", "on_submit")}
    return steps, store


def test_the_app_refuses_a_wrong_code_and_shows_the_first_review(built):
    steps, _ = built
    start = steps["on_start"]
    refused = start.fn("nope", {"hotelier": None, "queue": []})
    assert len(refused) == len(start.outputs) and refused[-1] == "Ο κωδικός δεν αναγνωρίστηκε."
    accepted = start.fn("abc", {"hotelier": None, "queue": []})
    state = accepted[0]
    assert len(accepted) == len(start.outputs) and state["hotelier"] == "H01" and len(state["queue"]) == 2
    assert accepted[4] == "Κριτική **1** από 2" and accepted[5].startswith("Disliked:")


def test_the_app_saves_a_complete_answer_and_resumes_where_the_hotelier_stopped(built):
    steps, store = built
    state = steps["on_start"].fn("abc", {"hotelier": None, "queue": []})[0]
    submit = steps["on_submit"]
    incomplete = submit.fn(*complete(), None, [], None, None, None, state)
    assert len(incomplete) == len(submit.outputs) and "Σύστημα 2" in incomplete[-1] and incomplete[0] == state
    after = submit.fn(*complete(), *complete("Όχι"), "Το ίδιο", state)
    assert len(after) == len(submit.outputs) and len(after[0]["queue"]) == 1 and store.done("H01") == {state["queue"][0]}
    resumed = steps["on_start"].fn("abc", {"hotelier": None, "queue": []})[0]
    assert resumed["queue"] == after[0]["queue"]


def test_the_bundle_rebuilds_and_checks_the_texts_and_names_the_systems_neutrally(tmp_path):
    rows = [{"Negative_Review": f"The bath was broken {i}", "Positive_Review": "No Positive"} for i in range(5)]
    raw = pd.DataFrame(rows)
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, index=False)
    samples = tmp_path / "samples"
    samples.mkdir()
    texts = [evaluation.sample.whole_review(r["Negative_Review"], r["Positive_Review"]) for r in rows]
    pd.DataFrame({"item": [1, 2], "raw_row": [3, 1], "stratum": "H", "group": "D", "test_weight": 1.0,
                  "sample_weight": 1.0, "text_sha256": [evaluation.sample.text_hash(texts[3]),
                                                        evaluation.sample.text_hash(texts[1])]}).to_csv(
        samples / "rated_key.csv", index=False)
    (samples / "outputs.json").write_text(json.dumps({"1": {"teacher": ["safety_fault"], "student": []},
                                                      "2": {"teacher": [], "student": ["wifi"]}}), encoding="utf-8")
    folder = evaluation.bundle(csv, samples, tmp_path / "runs")
    items = json.loads((folder / "items.json").read_text(encoding="utf-8"))
    assert items[0] == {"item": 1, "text": texts[3], "x": ["safety_fault"], "y": []} and items[1]["y"] == ["wifi"]
    assert {p.name for p in folder.iterdir()} == set(deploy.FILES)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["protocol_sha256"] == hashlib.sha256(evaluation.PROTOCOL.read_bytes()).hexdigest()
    (samples / "rated_key.csv").write_text((samples / "rated_key.csv").read_text().replace(
        evaluation.sample.text_hash(texts[3]), "0" * 64), encoding="utf-8")
    with pytest.raises(SystemExit, match="differ from the key"):
        evaluation.bundle(csv, samples, tmp_path / "runs2")


class RecordingApi:
    """Records the writes; `existing` maps the repos that already exist to whether they are private."""

    def __init__(self, name, existing=None):
        self.name, self.calls, self.existing = name, [], dict(existing or {})

    def whoami(self):
        return {"name": self.name}

    def repo_info(self, repo_id, repo_type):
        if repo_id not in self.existing:
            raise deploy.RepositoryNotFoundError(f"{repo_id} not found")
        return type("Info", (), {"private": self.existing[repo_id]})()

    def create_repo(self, **kwargs):
        self.calls.append(("create", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs))


@pytest.fixture
def bundle_dir(tmp_path):
    for name in deploy.FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    meta = {"protocol_sha256": hashlib.sha256(deploy.PROTOCOL.read_bytes()).hexdigest(), "bundle": "abc"}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_publishing_refuses_another_account_or_a_stale_bundle(bundle_dir, tmp_path):
    api = RecordingApi("someone-else")
    with pytest.raises(RuntimeError, match="Nothing was uploaded"):
        deploy.deploy(api, bundle=bundle_dir)
    assert api.calls == []
    other = tmp_path / "other_protocol.md"
    other.write_text("another version", encoding="utf-8")
    api = RecordingApi("krimits")
    with pytest.raises(RuntimeError, match="another version of the protocol"):
        deploy.deploy(api, bundle=bundle_dir, protocol=other)
    assert api.calls == []


def test_publishing_makes_a_private_dataset_a_public_space_and_uploads_only_the_bundle(bundle_dir):
    api = RecordingApi("krimits")
    result = deploy.deploy(api, bundle=bundle_dir)
    assert api.calls[0] == ("create", {"repo_id": "krimits/hotelier-blind-eval-ratings", "repo_type": "dataset",
                                       "private": True, "exist_ok": False})
    assert api.calls[1] == ("create", {"repo_id": "krimits/hotelier-blind-eval", "repo_type": "space",
                                       "space_sdk": "gradio", "private": False, "exist_ok": False})
    assert api.calls[2][1]["allow_patterns"] == deploy.FILES and api.calls[2][1]["delete_patterns"] == ["*"]
    assert result["space"] == "https://huggingface.co/spaces/krimits/hotelier-blind-eval"


def test_a_run_that_stopped_after_the_dataset_is_finished_only_with_resume(bundle_dir):
    # The owner's first run created the dataset and stopped before the Space (6 November 2026).
    halfway = {"krimits/hotelier-blind-eval-ratings": True}
    api = RecordingApi("krimits", halfway)
    with pytest.raises(RuntimeError, match="Rerun with --resume"):
        deploy.deploy(api, bundle=bundle_dir)
    assert api.calls == []
    api = RecordingApi("krimits", halfway)
    deploy.deploy(api, bundle=bundle_dir, resume=True)
    assert [call[0] for call in api.calls] == ["create", "upload"]
    assert api.calls[0][1]["repo_id"] == "krimits/hotelier-blind-eval" and api.calls[0][1]["private"] is False


def test_resume_refuses_a_public_ratings_dataset(bundle_dir):
    api = RecordingApi("krimits", {"krimits/hotelier-blind-eval-ratings": False})
    with pytest.raises(RuntimeError, match="must be private"):
        deploy.deploy(api, bundle=bundle_dir, resume=True)
    assert api.calls == []


def test_codes_are_random_and_the_secret_holds_only_their_hashes():
    tokens = iter(["aaa", "bbb"])
    pairs, secret = deploy.make_codes(2, token=lambda _: next(tokens))
    assert pairs == [("H01", "aaa"), ("H02", "bbb")]
    assert secret == {hashlib.sha256(b"aaa").hexdigest(): "H01", hashlib.sha256(b"bbb").hexdigest(): "H02"}
    assert logic.hotelier_for("bbb", secret) == "H02"
