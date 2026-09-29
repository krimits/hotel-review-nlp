"""Publishing the clean-split model: every check, and resuming after a partial success.

Offline: an in-memory Hub, a fake demo client and a tiny local DistilBERT stand in
for the network and the real weights.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("publish_model", ROOT / "scripts" / "publish_model.py")
publish = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = publish
_spec.loader.exec_module(publish)

FIXTURES = ROOT / "tests" / "fixtures" / "hotel_review_demo"
MODEL, SPACE = publish.MODEL_REPO, publish.SPACE_REPO
LFS = (".safetensors", ".npy")
WORDS = ["great", "clean", "friendly", "lovely", "dirty", "noisy", "rude", "broken"]


# --- an in-memory Hub and demo -------------------------------------------------------------------------------


class FakeHub:
    """Commits of files, a main branch, tags, and which Space commit the demo is running."""

    def __init__(self, tmp: Path):
        self.tmp, self.repos, self.commits_made, self.serving = tmp, {}, [], None
        self.runtime = SimpleNamespace(stage="RUNNING", raw={})
        self.fail_next_commit = set()

    def add_repo(self, repo_id: str, files: dict[str, bytes]) -> str:
        self.repos[repo_id] = {"commits": {}, "main": None, "tags": {}}
        return self._store(repo_id, files)

    def _store(self, repo_id: str, files: dict[str, bytes]) -> str:
        repo = self.repos[repo_id]
        sha = hashlib.sha1(f"{repo_id}{len(repo['commits'])}".encode()).hexdigest()
        repo["commits"][sha], repo["main"] = dict(files), sha
        return sha

    def _resolve(self, repo_id: str, revision: str | None) -> str:
        repo = self.repos[repo_id]
        if revision in (None, "main"):
            return repo["main"]
        return repo["tags"].get(revision, revision)

    def files(self, repo_id: str, revision: str | None = None) -> dict[str, bytes]:
        return self.repos[repo_id]["commits"][self._resolve(repo_id, revision)]

    def model_info(self, repo_id, revision=None, files_metadata=False):
        commit = self._resolve(repo_id, revision)
        siblings = [SimpleNamespace(rfilename=path, blob_id=publish.git_blob_sha1(data),
                                    lfs=SimpleNamespace(sha256=publish.sha256_of(data)) if path.endswith(LFS) else None)
                    for path, data in self.repos[repo_id]["commits"][commit].items()]
        return SimpleNamespace(sha=commit, siblings=siblings)

    def space_info(self, repo_id, revision=None):
        return SimpleNamespace(sha=self._resolve(repo_id, revision))

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(tags=[SimpleNamespace(name=name, target_commit=commit)
                                     for name, commit in self.repos[repo_id]["tags"].items()])

    def create_tag(self, repo_id, *, tag, revision):
        self.repos[repo_id]["tags"][tag] = self._resolve(repo_id, revision)

    def create_commit(self, repo_id, operations, commit_message, parent_commit=None, repo_type=None,
                      commit_description=None):
        if repo_id in self.fail_next_commit:
            self.fail_next_commit.discard(repo_id)
            raise ConnectionError("the connection dropped")
        if parent_commit != self.repos[repo_id]["main"]:
            raise RuntimeError("412: the parent commit is not the head")
        files = dict(self.files(repo_id))
        for operation in operations:
            if hasattr(operation, "path_or_fileobj"):
                source = operation.path_or_fileobj
                files[operation.path_in_repo] = source if isinstance(source, bytes) else Path(source).read_bytes()
            else:
                files.pop(operation.path_in_repo)
        self.commits_made.append((repo_id, parent_commit, [type(op).__name__ for op in operations]))
        return SimpleNamespace(oid=self._store(repo_id, files))

    def hf_hub_download(self, repo_id, filename, revision=None, repo_type=None):
        path = self.tmp / "downloads" / self._resolve(repo_id, revision) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files(repo_id, revision)[filename])
        return str(path)

    def get_space_runtime(self, repo_id):
        return self.runtime


class FakeDemo:
    """The demo's gradio client: it answers from the app of the Space commit being served."""

    def __init__(self, hub: FakeHub):
        self.hub, self.config = hub, {"version": "5.49.1"}

    def __call__(self, space_id):
        return self

    def predict(self, *args, api_name):
        app = self.hub.files(SPACE, self.hub.serving or "main")["app.py"].decode()
        if api_name == "/predict_distilbert":
            return {"label": "positive" if "spotless" in args[0] else "negative"}, "5 ms"
        if 'api_name="model_info"' not in app:
            raise ValueError("Cannot find a function with api_name /model_info")
        revision = re.search(r'DISTILBERT_REVISION = "([0-9a-f]+)"', app).group(1)
        weights = self.hub.files(MODEL, revision)["model.safetensors"]
        return {"distilbert_repo": MODEL, "distilbert_revision": revision, "loaded_commit": revision,
                "weights_sha256": publish.sha256_of(weights)}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A Hub holding the legacy model and the demo, a new run, and a state path."""
    hub = FakeHub(tmp_path)
    old = {".gitattributes": b"*.safetensors filter=lfs\n", "README.md": b"old card", "model.safetensors": b"old weights",
           "config.json": b"{}", "train.log": b"old log", "old_extra.txt": b"only in the legacy version"}
    monkeypatch.setattr(publish, "OLD_MODEL_SHA256", publish.sha256_of(old["model.safetensors"]))
    legacy = hub.add_repo(MODEL, old)
    hub.add_repo(SPACE, {"app.py": (FIXTURES / "app.py.txt").read_bytes(),
                         "README.md": (FIXTURES / "README.md").read_bytes()})
    hub.serving = hub.repos[SPACE]["main"]
    new = {"model.safetensors": b"new weights", "config.json": b'{"model_type": "distilbert"}',
           "README.md": b"new card", "train.log": b"new log"}
    files = {path: data for path, data in new.items()}
    state = {"new_model_sha256": publish.sha256_of(new["model.safetensors"]), "code_commit": "c0de" * 10,
             "gradio_version": "5.49.1"}
    assert publish.read_hub(hub, state) is True
    state["manifest"] = publish.manifest_of(files)
    return SimpleNamespace(hub=hub, demo=FakeDemo(hub), files=files, state=state, legacy=legacy,
                           path=tmp_path / "state.json")


# --- the run and its weights -------------------------------------------------------------------------------


def _tiny_run(directory: Path, texts: list[str]) -> None:
    """A tiny DistilBERT saved like a run, with its test logits scored as the trainer scores them."""
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    from reviewnlp.llm.encoder_trainer import EncodedReviews, pad_collate, predict_logits

    directory.mkdir(parents=True)
    (directory / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=5 + len(WORDS), dim=16, n_layers=1, n_heads=2, hidden_dim=32, max_position_embeddings=64))
    model.save_pretrained(directory)
    labels = [int(any(word in text for word in WORDS[:4])) for text in texts]
    loader = torch.utils.data.DataLoader(EncodedReviews(texts, labels, tokenizer, 32), batch_size=4,
                                         collate_fn=pad_collate)
    np.save(directory / "test_logits.npy", predict_logits(model, loader, torch.device("cpu")))
    np.save(directory / "test_labels.npy", np.array(labels))
    (directory / "metrics.json").write_text(json.dumps({"settings": {"max_length": 32, "eval_batch_size": 4}}))


def _test_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    texts = [" ".join(rng.choice(WORDS, size=rng.integers(2, 7))) for _ in range(12)]
    return pd.DataFrame({"text": texts,
                         "label": ["positive" if any(w in t for w in WORDS[:4]) else "negative" for t in texts]})


def test_the_weights_reproduce_the_saved_logits(tmp_path):
    test = _test_frame()
    _tiny_run(tmp_path / "run", test["text"].tolist())
    report = publish.agreement_check(tmp_path / "run", test, publish.frame_fingerprint(test), examples=10)
    assert report["passed"] and report["examples"] == 10 and report["argmax_agree"] == 10
    assert report["max_abs_diff"] <= 1e-6 and report["name"] == "agreement check on 10 examples"
    with pytest.raises(publish.PublishError, match="test set differs"):
        publish.agreement_check(tmp_path / "run", test.iloc[1:], publish.frame_fingerprint(test))


def test_the_logit_comparison_is_strict():
    saved = np.array([[1.0, -1.0], [-0.5, 0.5], [0.0002, 0.0]])
    assert publish.compare_logits(saved + 5e-4, saved)["passed"]
    assert not publish.compare_logits(saved + 2e-3, saved)["passed"]  # outside the tolerance
    flipped = saved.copy()
    flipped[2] = [0.0, 0.0006]  # within the tolerance, but the prediction changes
    report = publish.compare_logits(flipped, saved)
    assert report["within_tolerance"] and report["argmax_agree"] == 2 and not report["passed"]
    assert not publish.compare_logits(np.full_like(saved, np.nan), saved)["passed"]
    assert not publish.compare_logits(saved[:2], saved)["passed"]


def test_the_run_folder_must_hold_the_scored_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for run_name, bundle_name in publish.RECORDS.items():
        shutil.copy(publish.BUNDLE / bundle_name, run / run_name)
    for name in publish.MODEL_FILES:
        (run / name).write_bytes(b"weights or tokenizer")
    assert set(publish.check_run_records(run)) == set(publish.RECORDS)
    (run / "test_logits.npy").write_bytes((run / "test_logits.npy").read_bytes()[:-1] + b"\x00")
    with pytest.raises(publish.PublishError, match="test_logits.npy differs"):
        publish.check_run_records(run)


# --- the card --------------------------------------------------------------------------------------------


def _card() -> str:
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    return publish.build_card(read(publish.COMPARISON), read(publish.LATENCY), read(publish.SPLIT_VIEWS),
                              read(publish.BUNDLE / "metrics.json"), code_commit="c0de" * 10,
                              new_model_sha256="ab" * 32, old_hub_commit="0123456789" * 4,
                              agreement={"name": "agreement check on 1,024 examples", "atol": 1e-3,
                                         "max_abs_diff": 2.1e-5, "argmax_agree": 1024, "examples": 1024})


def test_the_card_claims_only_this_checkpoints_result():
    card = _card()
    before, related = card.split("## Related experiment, with a different checkpoint")
    assert "**0.9642** (0.9605–0.9677)" in before and "0.9351 (0.9303–0.9398)" in before
    assert "0.9491" not in before and "0.9491 macro-F1 (0.9450–0.9531)" in related
    assert "up to 9 March 2017" in related and "26 May 2017 to 3 August 2017" in related
    assert "not a\n  measurement of this model" in related
    assert "+0.0291" in before and "387 reviews only this model gets right" in before
    assert "0.953 / 0.939" in before and "354" in before and "117,880 reviews" in before
    assert "The timings come from the related checkpoint above" in related
    assert "legacy-split-v1" in card and "c0de" * 10 in card and publish.TRAINING_COMMIT in card
    assert "the largest difference was\n  2.1e-05" in card.replace("The largest", "the largest")


# --- the Hub ---------------------------------------------------------------------------------------------


def test_a_fresh_upload_tags_the_legacy_commit_and_makes_one_checked_commit(world):
    commit = publish.upload(world.hub, world.state, world.path, world.files)
    hub = world.hub
    assert hub.repos[MODEL]["tags"][publish.LEGACY_TAG] == world.legacy
    assert hub.repos[MODEL]["main"] == commit == world.state["model_commit"]
    [(repo, parent, operations)] = hub.commits_made
    assert (repo, parent) == (MODEL, world.legacy)
    assert operations.count("CommitOperationDelete") == 1  # old_extra.txt, never .gitattributes
    assert set(hub.files(MODEL)) == {*world.files, ".gitattributes"}
    assert publish.load_state(world.path)["model_commit"] == commit


def test_a_tag_pointing_elsewhere_stops_before_any_commit(world):
    world.hub.repos[MODEL]["tags"][publish.LEGACY_TAG] = "somewhere else"
    with pytest.raises(publish.PublishError, match="tag points to"):
        publish.upload(world.hub, world.state, world.path, world.files)
    assert world.hub.commits_made == []


def test_a_hub_that_holds_neither_model_stops_the_check(world):
    world.hub._store(MODEL, {"model.safetensors": b"someone else's weights"})
    with pytest.raises(publish.PublishError, match="neither the legacy nor the new"):
        publish.read_hub(world.hub, {"new_model_sha256": world.state["new_model_sha256"]})


def test_weights_that_differ_on_the_hub_are_not_recorded_as_published(world, monkeypatch):
    original = world.hub.create_commit

    def corrupting(*args, **kwargs):
        info = original(*args, **kwargs)
        world.hub.files(MODEL)["model.safetensors"] = b"corrupted"
        return info

    monkeypatch.setattr(world.hub, "create_commit", corrupting)
    with pytest.raises(publish.PublishError):
        publish.upload(world.hub, world.state, world.path, world.files)
    assert "model_commit" not in world.state and world.state["model_commit_unverified"]


def test_upload_done_demo_failed_and_the_rerun_goes_on(world):
    publish.upload(world.hub, world.state, world.path, world.files)
    world.hub.fail_next_commit.add(SPACE)
    with pytest.raises(ConnectionError):
        publish.pin_demo(world.hub, world.state, world.path)
    state = publish.load_state(world.path)  # the rerun starts from what was saved
    assert state["model_commit"] and "space_commit" not in state
    assert publish.upload(world.hub, state, world.path, world.files) == state["model_commit"]
    space_commit = publish.pin_demo(world.hub, state, world.path)
    assert [repo for repo, _, _ in world.hub.commits_made] == [MODEL, SPACE]  # no second model commit
    assert world.hub.repos[SPACE]["main"] == space_commit


def test_a_commit_whose_state_was_lost_is_adopted_only_if_it_is_exactly_ours(world):
    commit = publish.upload(world.hub, world.state, world.path, world.files)
    state = dict(world.state)
    del state["model_commit"]
    assert publish.read_hub(world.hub, state) is False  # the legacy commit is found through the tag
    assert publish.upload(world.hub, state, world.path, world.files) == commit
    assert len(world.hub.commits_made) == 1
    state.pop("model_commit")
    state["manifest"] = dict(state["manifest"], **{"README.md": {"sha256": "x", "git_sha1": "x"}})
    with pytest.raises(publish.PublishError, match="does not hold the new version"):
        publish.upload(world.hub, state, world.path, world.files)


# --- the demo --------------------------------------------------------------------------------------------


def test_the_fixtures_are_the_demo_as_published():
    assert (FIXTURES / "app.py.txt").stat().st_size == 9783
    assert (FIXTURES / "README.md").stat().st_size == 1483


def test_the_demo_patch_pins_the_model_and_corrects_the_texts():
    app, readme = publish.patch_demo((FIXTURES / "app.py.txt").read_text(encoding="utf-8"),
                                     (FIXTURES / "README.md").read_text(encoding="utf-8"), "ab" * 20, "5.49.1")
    tree = ast.parse(app)
    assert 'DISTILBERT_REVISION = "' + "ab" * 20 + '"' in app
    assert "snapshot_download(DISTILBERT_ID, revision=DISTILBERT_REVISION)" in app
    assert "from_pretrained(DISTILBERT_ID" not in app
    assert 'api_name="model_info"' in app and "def model_info" in app
    assert any(isinstance(node, ast.FunctionDef) and node.name == "model_info" for node in tree.body)
    assert "0.9642" in app and "0.9634" not in app + readme and "0.525" not in app + readme
    assert "sdk_version: 5.49.1\n" in readme and "not comparable" in readme
    assert publish.patch_demo(*[(FIXTURES / name).read_text(encoding="utf-8") for name in ("app.py.txt", "README.md")],
                              "ab" * 20, "5.49.1") == (app, readme)  # deterministic


def test_a_demo_that_changed_is_not_patched():
    app = (FIXTURES / "app.py.txt").read_text(encoding="utf-8").replace("EXAMPLES = [", "SAMPLES = [")
    with pytest.raises(publish.PublishError, match="EXAMPLES"):
        publish.patch_demo(app, (FIXTURES / "README.md").read_text(encoding="utf-8"), "ab" * 20, "5.49.1")


def test_the_check_waits_for_the_new_app_and_then_classifies(world):
    publish.upload(world.hub, world.state, world.path, world.files)
    publish.pin_demo(world.hub, world.state, world.path)
    now = [0.0]

    def sleep(seconds):  # the rebuild finishes while we wait
        now[0] += seconds
        world.hub.serving = world.state["space_commit"]

    result = publish.check_demo(world.hub, world.demo, world.state, world.path, clock=lambda: now[0], sleep=sleep)
    assert result["model_info"]["loaded_commit"] == world.state["model_commit"]
    assert result["predictions"] == {"positive": "positive", "negative": "negative"}


def test_the_previous_app_never_passes_the_check(world):
    publish.upload(world.hub, world.state, world.path, world.files)
    publish.pin_demo(world.hub, world.state, world.path)
    now = [0.0]  # the old commit keeps being served: it has no model_info endpoint
    with pytest.raises(publish.PublishError, match="did not serve the new model"):
        publish.check_demo(world.hub, world.demo, world.state, world.path, timeout=90,
                           clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    world.hub.runtime = SimpleNamespace(stage="BUILD_ERROR", raw={"errorMessage": "pip failed"})
    with pytest.raises(publish.PublishError, match="BUILD_ERROR: pip failed"):
        publish.check_demo(world.hub, world.demo, world.state, world.path)


def test_the_patched_demo_builds_and_reports_what_it_loaded(tmp_path, monkeypatch):
    """Run the patched app.py with the models replaced by stand-ins, on the Gradio of the CI."""
    pytest.importorskip("gradio")
    peft = pytest.importorskip("peft")
    import huggingface_hub
    import transformers

    revision = "ab" * 20
    snapshot = tmp_path / "snapshots" / revision  # the Hub cache names a snapshot after its commit
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"new weights")
    (tmp_path / "qwen" / "adapter").mkdir(parents=True)

    class StandIn:
        config = SimpleNamespace(id2label={0: "negative", 1: "positive"})
        pad_token, eos_token = "<pad>", "<eos>"

        def eval(self):
            return self

    def snapshot_download(repo_id, revision=None, **kwargs):
        return str(snapshot if repo_id == MODEL else tmp_path / "qwen")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    for loader in (transformers.AutoTokenizer, transformers.AutoModelForSequenceClassification,
                   transformers.AutoModelForCausalLM, peft.PeftModel):
        monkeypatch.setattr(loader, "from_pretrained", lambda *args, **kwargs: StandIn())
    app, _ = publish.patch_demo((FIXTURES / "app.py.txt").read_text(encoding="utf-8"),
                                (FIXTURES / "README.md").read_text(encoding="utf-8"), revision, "5.49.1")
    namespace = {"__name__": "patched_demo"}
    exec(compile(app, "app.py", "exec"), namespace)
    assert namespace["model_info"]() == {"distilbert_repo": MODEL, "distilbert_revision": revision,
                                         "loaded_commit": revision,
                                         "weights_sha256": publish.sha256_of(b"new weights")}
    endpoints = {dependency.get("api_name") for dependency in namespace["demo"].config["dependencies"]}
    assert {"model_info", "predict_distilbert"} <= endpoints
