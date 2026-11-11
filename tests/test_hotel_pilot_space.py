"""The hotel pilot Space: the call to the model as tested, the urgent set, the judgements, storage without text, the
app's steps, the bundle and the deployment. No network; the app is built but not launched."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "spaces" / "hotel-pilot"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


logic = _load("hotel_pilot_space_logic", SPACE / "logic.py")
_saved = sys.modules.get("logic")
sys.modules["logic"] = logic  # app.py imports its sibling as `logic`
try:
    app = _load("hotel_pilot_space_app", SPACE / "app.py")
finally:
    if _saved is None:
        sys.modules.pop("logic", None)
    else:
        sys.modules["logic"] = _saved
pilot = _load("hotel_pilot", ROOT / "scripts" / "hotel_pilot.py")
deploy = _load("deploy_hotel_pilot_space", ROOT / "scripts" / "deploy_hotel_pilot_space.py")
teacher = _load("triage_diagnosis_teacher", ROOT / "scripts" / "triage_diagnosis_teacher.py")
DISPLAY = pilot.display_table()
TYPES = DISPLAY["order"]
URGENT = logic.urgent_types(DISPLAY)
PROMPT = pilot.PROMPT.read_text(encoding="utf-8")
META = {"model": logic.MODEL, "protocol_sha256": "p", "prompt_sha256": "q", "bundle": "b"}
SALT = "salt"


def answer(reviews: list[dict], rule=None, cost=0.002) -> bytes:
    """An OpenRouter answer: cleanliness for a review that says dirty, wifi for one that says wifi."""
    rule = rule or (lambda text: [t for t, word in (("cleanliness", "dirty"), ("wifi", "wifi")) if word in text])
    labels = [{"item": r["item"], "types": rule(r["text"])} for r in reviews]
    return json.dumps({"choices": [{"message": {"content": json.dumps({"labels": labels})}}],
                       "usage": {"cost": cost}}).encode()


class FakeOpenRouter:
    """Answers each call from a queue of behaviours (a status, an exception, 'garbled'), then properly."""

    def __init__(self, behaviours=()):
        self.behaviours, self.calls = list(behaviours), []

    def __call__(self, url, body, api_key, timeout):
        self.calls.append((url, body, api_key))
        behaviour = self.behaviours.pop(0) if self.behaviours else "ok"
        if isinstance(behaviour, Exception):
            raise behaviour
        if isinstance(behaviour, int):
            return behaviour, {}, b'{"error": "status"}'
        if behaviour == "garbled":
            return 200, {}, json.dumps({"choices": [{"message": {"content": "I cannot"}}]}).encode()
        return 200, {}, answer(json.loads(body["messages"][1]["content"]))


def review(text: str) -> dict:
    return {"key": logic.text_key(text, SALT), "text": text, "script": logic.script_of(text)}


# --- the system is the tested teacher -----------------------------------------------------------------------------------

def test_the_call_is_the_tested_teachers_byte_for_byte():
    assert logic.MODEL == teacher.MODELS["haiku"] == "anthropic/claude-haiku-5.5-20261007"
    assert logic.URL == teacher.URL and logic.BATCH == teacher.BATCH == 25 and logic.TRIES == teacher.TRIES
    reviews = [{"item": 1, "text": "Το δωμάτιο ήταν βρώμικο"}, {"item": 2, "text": "Great stay"}]
    assert logic.request_body(PROMPT, reviews, TYPES) == teacher.request_body(logic.MODEL, PROMPT, reviews, TYPES)
    good = answer([{"item": 1, "text": "dirty"}, {"item": 2, "text": "fine"}])
    assert logic.parse(good, [1, 2], TYPES) == teacher.parse(good, [1, 2], TYPES) == {1: ["cleanliness"], 2: []}
    for bad, items in ((good, [1, 2, 3]), (b"not json", [1]),
                       (answer([{"item": 1, "text": "x"}], rule=lambda _: ["made_up"]), [1])):
        with pytest.raises(logic.OutputError):
            logic.parse(bad, items, TYPES)
        with pytest.raises(teacher.OutputError):
            teacher.parse(bad, items, TYPES)


def test_reviews_go_25_per_call_and_each_carries_its_calls_cost_and_time():
    fake, ticks = FakeOpenRouter(), iter(range(100))
    reviews = [review(f"dirty room {i}") for i in range(30)]
    done = logic.analyse(reviews, PROMPT, TYPES, "sk-test", fake, sleep=lambda _: None, clock=lambda: next(ticks))
    assert [len(json.loads(c[1]["messages"][1]["content"])) for c in fake.calls] == [25, 5]
    assert all(c[2] == "sk-test" and c[0] == logic.URL for c in fake.calls)
    assert [r["types"] for r in done] == [["cleanliness"]] * 30 and done[0]["key"] == reviews[0]["key"]
    assert done[0]["call"] == {"seconds": 1, "cost_usd": 0.002, "reviews": 25} and done[29]["call"]["reviews"] == 5


def test_waits_retries_and_failures():
    sleeps: list[float] = []
    fake = FakeOpenRouter([429, OSError("down"), "garbled"])
    done = logic.analyse([review("dirty")], PROMPT, TYPES, "k", fake, sleep=sleeps.append)
    assert done[0]["types"] == ["cleanliness"] and sleeps == [1.0, 3.0] and len(fake.calls) == 4
    for status in (401, 402, 403):  # a wrong key or an empty balance: no second try
        fake = FakeOpenRouter([status])
        with pytest.raises(logic.AnalysisError, match=f"status {status}"):
            logic.analyse([review("dirty")], PROMPT, TYPES, "k", fake, sleep=sleeps.append)
        assert len(fake.calls) == 1
    with pytest.raises(logic.AnalysisError, match="status 503"):
        logic.analyse([review("dirty")], PROMPT, TYPES, "k", FakeOpenRouter([503] * 4), sleep=lambda _: None)
    fake = FakeOpenRouter(["garbled"] * 3)
    with pytest.raises(logic.AnalysisError, match="could not be used"):
        logic.analyse([review("dirty")], PROMPT, TYPES, "k", fake, sleep=lambda _: None)
    assert len(fake.calls) == 3


# --- the hotel's reviews -------------------------------------------------------------------------------------------------

def test_reviews_are_split_read_from_csv_and_given_a_script():
    assert logic.split_reviews("Dirty room.\nBroken AC.\n\n\n  Great stay  \r\n\r\nΩραία θέα") == [
        "Dirty room. Broken AC.", "Great stay", "Ωραία θέα"]
    assert logic.reviews_from_csv("date,Review\n1/1,\"Dirty, very\"\n2/1,\n3/1,Fine") == ["Dirty, very", "Fine"]
    assert logic.reviews_from_csv("Dirty room\nFine") == ["Dirty room", "Fine"]
    assert [logic.script_of(t) for t in ("Το δωμάτιο ήταν ok", "Zimmer war schmutzig", "部屋が汚い", "12 !!")] == [
        "greek", "latin", "other", "other"]


def test_a_long_review_refuses_the_submission_and_repeats_and_the_cap_are_respected(monkeypatch):
    with pytest.raises(ValueError, match="ξεπερνούν"):
        logic.check_reviews(["fine", "x" * (logic.MAX_CHARS + 1)], set(), set(), SALT)
    rated = {logic.text_key("already rated", SALT)}
    kept, note = logic.check_reviews(["already rated", "new one", "new  one", "another"], rated, set(), SALT)
    assert [r["text"] for r in kept] == ["new one", "another"] and "2 κριτικές" in note
    monkeypatch.setattr(logic, "CAP", 3)
    analysed = {logic.text_key("seen", SALT), logic.text_key("lost", SALT)}
    kept, note = logic.check_reviews(["seen", "a", "b"], set(), analysed, SALT)
    assert [r["text"] for r in kept] == ["seen", "a"] and "2 από τις 3" in note  # one place left; a re-analysis is free
    monkeypatch.setattr(logic, "SUBMISSION", 1)
    assert len(logic.check_reviews(["a", "b"], set(), set(), SALT)[0]) == 1


def test_the_key_is_salted_and_ignores_white_space():
    assert logic.text_key("a  b\n", SALT) == logic.text_key("a b", SALT) != logic.text_key("a b", "other")
    codes = {"h1": "P01"}
    assert logic.salt_from(codes) == logic.salt_from(dict(codes)) != logic.salt_from({"h2": "P01"})


# --- the urgent set and the judgements -----------------------------------------------------------------------------------

def test_a_switched_off_urgent_type_is_not_a_flag_and_shows_as_routine():
    assert URGENT == ["cleanliness", "pests", "security_fault", "safety_fault", "safety_incident", "hvac_failure",
                      "plumbing_major"]
    reported = ["wifi", "hvac_failure", "cleanliness"]
    assert logic.flags(reported, URGENT, DISPLAY) == ["cleanliness", "hvac_failure"]
    mine = [t for t in URGENT if t != "cleanliness"]
    assert logic.flags(reported, mine, DISPLAY) == ["hvac_failure"]
    shown = logic.render_others(reported, mine, DISPLAY).splitlines()
    assert len(shown) == 2 and shown[0].startswith("- Τακτικό · **Καθαριότητα**") and "Wi-Fi" in shown[1]
    assert "ΕΠΕΙΓΟΝ" not in "\n".join(shown) and logic.render_others([], mine, DISPLAY).startswith("_Κανένα")


def test_answers_must_be_complete_and_the_record_holds_no_text():
    item = {**review("The room was dirty and the AC broken"), "types": ["cleanliness", "hvac_failure", "wifi"],
            "call": {"seconds": 2.0, "cost_usd": 0.001, "reviews": 3}}
    flagged = logic.flags(item["types"], URGENT, DISPLAY)
    assert len(logic.problems(flagged, {}, None, None, None, None, DISPLAY)) == 5
    verdicts = {"cleanliness": "Υπάρχει, αλλά δεν είναι επείγον", "hvac_failure": "Σωστό και επείγον"}
    assert logic.problems(flagged, verdicts, "Ναι", None, "Ναι", "4", DISPLAY) == [
        "Διαλέξτε ποιο επείγον πρόβλημα λείπει."]
    assert logic.problems(flagged, verdicts, "Όχι", None, "Εν μέρει", "4", DISPLAY) == []
    saved = logic.record("P01", item, URGENT, verdicts, "Ναι", "Έντομα ή τρωκτικά", "Εν μέρει", "4", META, DISPLAY)
    assert saved["flags"] == {"cleanliness": "not_urgent", "hvac_failure": "accepted"}
    assert (saved["missing_urgent"], saved["missing_type"], saved["others"], saved["useful"]) == (
        True, "pests", "partly", 4)
    assert saved["urgent_set"] == URGENT and saved["script"] == "latin" and saved["model"] == logic.MODEL
    assert "text" not in saved and "dirty" not in json.dumps(saved)
    other = logic.record("P01", item, URGENT, verdicts, "Ναι", logic.OTHER_TYPE, "Ναι", "5", META, DISPLAY)
    none = logic.record("P01", item, URGENT, verdicts, "Όχι", "Έντομα ή τρωκτικά", "Ναι", "5", META, DISPLAY)
    assert other["missing_type"] == "other" and none["missing_type"] is None
    assert "text" not in logic.analysed_note("P01", item)


def test_the_store_keeps_the_first_judgement_and_the_hotels_urgent_set():
    store = logic.MemoryStore()
    rating = {"hotel": "P01", "key": "abc", "useful": 4}
    assert store.save_record(rating) and not store.save_record({**rating, "useful": 1})
    assert json.loads(store.files["records/P01/abc.json"])["useful"] == 4
    store.save_analysed([{"hotel": "P01", "key": "abc"}, {"hotel": "P01", "key": "def"}, {"hotel": "P02", "key": "x"}])
    assert store.keys("P01", "records") == {"abc"} and store.keys("P01", "analysed") == {"abc", "def"}
    assert store.urgent_set("P01") is None
    store.save_urgent_set("P01", ["pests"])
    assert store.urgent_set("P01") == ["pests"] and store.urgent_set("P02") is None


class FakeHub:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def list_repo_files(self, repo_id, repo_type):
        return list(self.files)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.files[path_in_repo] = path_or_fileobj

    def create_commit(self, repo_id, operations, repo_type, commit_message):
        for operation in operations:
            self.files[operation.path_in_repo] = operation.path_or_fileobj


def test_the_hub_store_writes_json_under_the_datasets_paths():
    store = logic.HubStore("krimits/hotel-triage-pilot-records", "not-a-real-token")
    store.api = FakeHub()
    assert store.save_record({"hotel": "P01", "key": "abc"}) and not store.save_record({"hotel": "P01", "key": "abc"})
    store.save_analysed([{"hotel": "P01", "key": "abc"}, {"hotel": "P01", "key": "def"}])
    store.save_analysed([{"hotel": "P01", "key": "def"}])
    assert sorted(store.api.files) == ["analysed/P01/abc.json", "analysed/P01/def.json", "records/P01/abc.json"]
    assert store.keys("P01", "analysed") == {"abc", "def"}
    assert json.loads(store.api.files["records/P01/abc.json"]) == {"hotel": "P01", "key": "abc"}


# --- the app ---------------------------------------------------------------------------------------------------------------

def steps_of(settings, transport):
    demo = app.build_app(DISPLAY, PROMPT, META, settings, transport)
    names = ("on_start", "on_save_urgent", "on_analyse", "on_submit")
    return {f.fn.__name__: f for f in demo.fns.values() if getattr(f.fn, "__name__", "") in names}


@pytest.fixture
def live():
    settings = {"preview": False, "store": logic.MemoryStore(), "api_key": "sk-test",
                "codes": {hashlib.sha256(b"abc").hexdigest(): "P01"}}
    fake = FakeOpenRouter()
    return steps_of(settings, fake), settings["store"], fake


EMPTY = {"hotel": None, "urgent": list(URGENT), "queue": [], "rated": 0}


def call(step, *args):
    out = step.fn(*args)
    assert len(out) == len(step.outputs), (step.fn.__name__, len(out), len(step.outputs))
    return out


# Where the note under the entry form is among each step's outputs: `screen` starts with entry, rating, status,
# entry_message; on_analyse returns the state and the two inputs before it, on_submit only the state.
NOTE_AFTER_ANALYSE, NOTE_AFTER_SUBMIT = 6, 4


def test_a_hotel_enters_sets_its_urgent_types_and_judges_review_after_review(live):
    steps, store, fake = live
    assert call(steps["on_start"], "nope", EMPTY)[-1] == "Ο κωδικός δεν αναγνωρίστηκε."
    state = call(steps["on_start"], "abc", EMPTY)[0]
    assert state["hotel"] == "P01" and state["urgent"] == URGENT
    state = call(steps["on_save_urgent"], [DISPLAY["types"][t]["label"] for t in URGENT if t != "pests"], state)[0]
    assert "pests" not in state["urgent"] and store.urgent_set("P01") == state["urgent"]
    out = call(steps["on_analyse"], "The room was dirty\n\nGood wifi but dirty\n\nLovely", None, state)
    state = out[0]
    assert len(fake.calls) == 1 and [r["types"] for r in state["queue"]] == [["cleanliness"], ["cleanliness", "wifi"], []]
    assert store.keys("P01", "analysed") == {r["key"] for r in state["queue"]}
    assert "dirty" not in json.dumps(store.files)  # no text reaches the store
    blank = [None] * len(URGENT)
    refused = call(steps["on_submit"], *blank, "Όχι", None, "Ναι", "5", state)
    assert refused[0] == state and "Καθαριότητα" in refused[-1] and not store.keys("P01", "records")
    first = ["Σωστό και επείγον", *blank[1:]]
    state = call(steps["on_submit"], *first, "Όχι", None, "Ναι", "5", state)[0]
    assert state["rated"] == 1 and len(state["queue"]) == 2
    state = call(steps["on_submit"], "Δεν υπάρχει τέτοιο πρόβλημα", *blank[1:], "Όχι", None, "Εν μέρει", "3", state)[0]
    done = call(steps["on_submit"], *blank, "Ναι", logic.OTHER_TYPE, "Όχι", "2", state)
    state = done[0]
    assert state["queue"] == [] and state["rated"] == 3 and "Βαθμολογήσατε 3" in done[NOTE_AFTER_SUBMIT]
    saved = [json.loads(v) for k, v in sorted(store.files.items()) if k.startswith("records/")]
    assert sorted(len(r["flags"]) for r in saved) == [0, 1, 1] and all("text" not in r for r in saved)
    assert {r["protocol_sha256"] for r in saved} == {"p"} and all(r["urgent_set"] == state["urgent"] for r in saved)
    # The same reviews again: all were rated, so nothing is analysed and nothing is charged.
    again = call(steps["on_analyse"], "The room was dirty\n\nLovely", None, state)
    assert again[0]["queue"] == [] and len(fake.calls) == 1 and "παραλείφθηκαν" in again[NOTE_AFTER_ANALYSE]
    assert call(steps["on_start"], "abc", EMPTY)[0]["rated"] == 3  # the code resumes with its progress


def test_the_urgent_set_cannot_change_while_reviews_wait_to_be_judged(live):
    steps, _, _ = live
    state = call(steps["on_start"], "abc", EMPTY)[0]
    state = call(steps["on_analyse"], "dirty", None, state)[0]
    out = call(steps["on_save_urgent"], [], state)
    assert out[0] == state and out[-1] == app.BUSY


def test_a_failed_analysis_shows_a_message_and_records_nothing(capsys):
    settings = {"preview": False, "store": logic.MemoryStore(), "api_key": "sk-test",
                "codes": {hashlib.sha256(b"abc").hexdigest(): "P01"}}
    steps = steps_of(settings, FakeOpenRouter([402]))
    state = call(steps["on_start"], "abc", EMPTY)[0]
    out = call(steps["on_analyse"], "dirty", None, state)
    assert out[0] == state and out[NOTE_AFTER_ANALYSE] == app.ANALYSIS_DOWN and not settings["store"].files
    logs = capsys.readouterr().err
    assert "status 402" in logs and "sk-test" not in logs and "dirty" not in logs


def test_the_preview_makes_up_analyses_and_never_calls_the_model(monkeypatch):
    for name in app.SECRETS:
        monkeypatch.delenv(name, raising=False)
    settings = app.settings_from_env()
    assert settings["preview"] and isinstance(settings["store"], logic.MemoryStore)
    fake = FakeOpenRouter()
    steps = steps_of(settings, fake)
    state = call(steps["on_start"], "demo", EMPTY)[0]
    state = call(steps["on_analyse"], "Anything at all", None, state)[0]
    assert len(state["queue"]) == 1 and fake.calls == []
    monkeypatch.setenv("OPENROUTER_API_KEY", " sk-x \n")
    assert app.settings_from_env()["preview"]  # one secret of four is still the preview
    monkeypatch.setenv("RATINGS_REPO", " krimits/hotel-triage-pilot-records ")
    monkeypatch.setenv("HF_TOKEN", "hf_not_real")
    monkeypatch.setenv("HOTEL_CODES", ' {"h": "P01"} ')
    live_settings = app.settings_from_env()
    assert not live_settings["preview"] and live_settings["api_key"] == "sk-x"
    assert live_settings["store"].repo_id == "krimits/hotel-triage-pilot-records"


# --- the bundle and the deployment -------------------------------------------------------------------------------------------

def test_the_bundle_carries_the_tested_prompt_and_names_what_it_was_built_from(tmp_path):
    folder = pilot.bundle(tmp_path)
    assert {p.name for p in folder.iterdir()} == set(deploy.FILES)
    assert (folder / "prompt.md").read_bytes() == pilot.PROMPT.read_bytes()
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == logic.MODEL and meta["protocol_sha256"] == hashlib.sha256(pilot.PROTOCOL.read_bytes()).hexdigest()
    assert meta["prompt_sha256"] == "f731026f001ea8262444edc7c3847b6729d0414588369a409de8f5717c98245a"
    display, prompt, loaded = app.load(folder)
    assert display == DISPLAY and prompt == PROMPT and loaded == meta and display["routine_el"] == "Τακτικό"


class RecordingApi:
    def __init__(self, name, existing=None):
        self.name, self.calls, self.existing = name, [], dict(existing or {})

    def whoami(self):
        return {"name": self.name}

    def repo_info(self, repo_id, repo_type):
        if repo_id not in self.existing:
            raise deploy.ensure_repo.__globals__["RepositoryNotFoundError"](f"{repo_id} not found")
        return type("Info", (), {"private": self.existing[repo_id]})()

    def create_repo(self, **kwargs):
        self.calls.append(("create", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs))


def test_publishing_checks_the_account_the_protocol_and_the_prompt(tmp_path):
    folder = pilot.bundle(tmp_path)
    api = RecordingApi("someone-else")
    with pytest.raises(RuntimeError, match="Nothing was uploaded"):
        deploy.deploy(api, bundle=folder)
    (folder / "prompt.md").write_text("another prompt", encoding="utf-8")
    api = RecordingApi("krimits")
    with pytest.raises(RuntimeError, match="not the tested teacher prompt"):
        deploy.deploy(api, bundle=folder)
    assert api.calls == []
    folder = pilot.bundle(tmp_path)
    api = RecordingApi("krimits")
    result = deploy.deploy(api, bundle=folder)
    assert api.calls[0] == ("create", {"repo_id": "krimits/hotel-triage-pilot-records", "repo_type": "dataset",
                                       "exist_ok": False, "private": True})
    assert api.calls[1][1]["repo_id"] == "krimits/hotel-triage-pilot" and api.calls[1][1]["private"] is False
    assert api.calls[2][1]["allow_patterns"] == deploy.FILES and result["space"].endswith("/hotel-triage-pilot")
    api = RecordingApi("krimits", {"krimits/hotel-triage-pilot-records": False})
    with pytest.raises(RuntimeError, match="must be private"):
        deploy.deploy(api, bundle=folder, resume=True)


def test_codes_are_pseudonyms_p01_and_the_secret_holds_only_hashes():
    tokens = iter(["aaa", "bbb"])
    pairs, value = deploy.make_codes(2, token=lambda _: next(tokens))
    assert pairs == [("P01", "aaa"), ("P02", "bbb")] and logic.hotel_for(" bbb ", value) == "P02"
    assert "aaa" not in json.dumps(value)
