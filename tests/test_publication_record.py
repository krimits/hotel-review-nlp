"""The recorded publication of the clean-split DistilBERT agrees with the repository.

Notebook 11 published the phase 2 run on the random split to the Hugging Face
Hub, and pinned the demo to it, on 29 September 2026. It left three records,
all committed:

- published.json, the state file the publishing script kept in Google Drive;
- model_card.md, the card it committed to the Hub;
- the executed notebook, with the log of every step.

These checks are offline. They show that the records agree with each other and
with the committed results. They do not show that the Hub and the demo are
still in that state today.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "experiments" / "results"
BUNDLE = RESULTS / "distilbert_v2"
RUN = BUNDLE / "random" / "distilbert"
NOTEBOOK = ROOT / "docs" / "experiments" / "notebooks" / "11_publish_model_colab_executed.ipynb"
MODEL_FILES = {"config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.txt"}
# The run's records on the Hub, and their names in the bundle.
RECORDS = {"metrics.json": "metrics.json", "request.json": "request.json", "run_config.yaml": "run_config.yaml",
           "dev_logits.npy": "dev_logits.npy", "dev_labels.npy": "dev_labels.npy",
           "test_logits.npy": "test_logits.npy", "test_labels.npy": "test_labels.npy",
           "train.log": "train_log.txt"}
RELATED = "## Related experiment, with a different checkpoint"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _published() -> dict:
    return _json(BUNDLE / "published.json")


def _card() -> bytes:
    return (BUNDLE / "model_card.md").read_bytes()


def test_the_record_is_the_state_file_as_the_script_wrote_it():
    text = (BUNDLE / "published.json").read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), indent=1, sort_keys=True) + "\n"


def test_the_published_files_are_the_committed_ones():
    published = _published()
    manifest = published["manifest"]
    assert set(manifest) == MODEL_FILES | set(RECORDS) | {"README.md"}
    card = _card()
    assert hashlib.sha256(card).hexdigest() == manifest["README.md"]["sha256"]
    assert hashlib.sha1(b"blob %d\0" % len(card) + card).hexdigest() == manifest["README.md"]["git_sha1"]
    for on_the_hub, in_the_bundle in RECORDS.items():
        assert hashlib.sha256((RUN / in_the_bundle).read_bytes()).hexdigest() == manifest[on_the_hub]["sha256"], on_the_hub
    assert manifest["model.safetensors"]["sha256"] == published["new_model_sha256"]


def test_the_publication_used_the_training_run_and_kept_the_legacy_model():
    published = _published()
    assert published["training_commit"] == _json(BUNDLE / "artifact_manifest.json")["commit"]
    legacy = {entry["path"]: entry["sha256"]
              for entry in _json(RESULTS / "distilbert_legacy_full_v1" / "artifact_manifest.json")["files"]}
    assert published["old_model_sha256"] == legacy["distilbert/model.safetensors"]
    assert published["legacy_tag"] == {"name": "legacy-split-v1", "commit": published["old_hub_commit"]}
    assert published["old_hub_commit"] != published["model_commit"]
    assert published["old_model_sha256"] != published["new_model_sha256"]


def test_the_weights_passed_the_agreement_check():
    check = _published()["logits_check"]
    assert check["passed"] and check["same_shape"] and check["finite"] and check["within_tolerance"]
    assert (check["examples"], check["atol"], check["rtol"], check["argmax_agree"]) == (1024, 1e-3, 0.0, 1024)
    assert check["max_abs_diff"] <= check["atol"]


def test_the_demo_served_the_published_model():
    published = _published()
    demo = published["demo_check"]
    loaded = demo["model_info"]
    assert loaded["distilbert_repo"] == published["model_repo"]
    assert loaded["distilbert_revision"] == loaded["loaded_commit"] == published["model_commit"]
    assert loaded["weights_sha256"] == published["new_model_sha256"]
    assert demo["runtime_commit"] == published["space_commit"] != published["space_parent_commit"]
    assert demo["stage"] == "RUNNING"
    assert demo["predictions"] == {"negative": "negative", "positive": "positive"}


def test_the_card_gives_only_this_checkpoints_score_as_its_result():
    published, card = _published(), _card().decode("utf-8")
    splits = _json(RESULTS / "model_comparison_v2.json")["splits"]
    ours, other = splits["random"]["models"]["distilbert"], splits["time"]["models"]["distilbert"]
    assert card.count(RELATED) == 1
    own, related = card.split(RELATED)
    lo, hi = ours["macro_f1_ci95"]
    assert f"**{ours['macro_f1']:.4f}** ({lo:.4f}–{hi:.4f})" in own
    assert f"{other['macro_f1']:.4f}" not in own and f"{other['macro_f1']:.4f}" in related
    for value in (published["code_commit"], published["training_commit"], published["new_model_sha256"],
                  published["old_hub_commit"][:7]):
        assert value in card


def test_the_executed_notebook_logged_the_same_publication():
    published = _published()
    logs = ["".join("".join(output.get("text", "")) for output in cell.get("outputs", []))
            for cell in _json(NOTEBOOK)["cells"]]
    everything = "\n".join(logs)
    assert _card().decode("utf-8") in everything  # the check step printed the card
    assert f"upload: committed and verified {published['model_commit']}" in everything
    assert f"pin-demo: committed {published['space_commit']}" in everything
    assert f"check-demo: the demo serves {published['model_commit']} and labels both examples correctly" in everything
    printed = json.loads(logs[-1])  # the last cell prints the state's main entries
    assert "demo_check" in printed and printed == {key: published[key] for key in printed}
