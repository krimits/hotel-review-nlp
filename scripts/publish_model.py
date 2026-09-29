"""Publish the clean-split DistilBERT to the Hugging Face Hub, and pin the demo to it.

No training. The weights are those of the phase 2 run on the random split
(full fine-tune), which notebook 10 saved to Google Drive. Notebook 11 runs
these steps on a Colab CPU, with the owner's write token:

  check       No writes.
              - The run's records are byte-identical to the committed bundle.
              - Its weights give the saved test logits on the first 1,024 test
                reviews.
              - The Hub still holds the legacy model.
              - The demo answers, its Gradio version is read, and its patch
                applies.
  upload      Tag the legacy commit. Then make one explicit commit of the new
              files, and check the commit it returns.
  pin-demo    The demo loads the new model at that commit, says what it is,
              and reports what it loaded.
  check-demo  The running demo must report that commit and those weights. Then
              it must classify one positive and one negative review.

Each completed write is recorded in a state file. A rerun checks the remote
state again and skips only what was done exactly as expected.

    python scripts/publish_model.py check --run-dir RUN --state STATE
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewnlp.utils.experiments import frame_fingerprint  # noqa: E402

RESULTS = ROOT / "docs" / "experiments" / "results"
BUNDLE = RESULTS / "distilbert_v2" / "random" / "distilbert"
COMPARISON = RESULTS / "model_comparison_v2.json"
LATENCY = RESULTS / "distilbert_v2" / "latency.json"
SPLIT_VIEWS = RESULTS / "split_views.json"
TEST = ROOT / "data" / "processed_v2" / "test.parquet"
MODEL_REPO = "krimits/distilbert-hotel-reviews"
SPACE_REPO = "krimits/hotel-review-demo"
GITHUB = "https://github.com/krimits/hotel-review-nlp"
LEGACY_TAG = "legacy-split-v1"
OLD_MODEL_SHA256 = "f30daa44e795b7015151892ff5ec285e14a916e122902aa0617a15bf96ec54b7"
TRAINING_COMMIT = "994a103574d07833d466e9553680a9cab46bfe76"
CHECK_EXAMPLES = 1024
ATOL = 1e-3
PACKAGES = ("huggingface_hub", "transformers", "tokenizers", "gradio_client", "torch", "numpy", "pandas")
# The run's records, which must be byte-identical to the committed bundle (name in the run: name in the bundle).
RECORDS = {"metrics.json": "metrics.json", "request.json": "request.json",
           "dev_logits.npy": "dev_logits.npy", "dev_labels.npy": "dev_labels.npy",
           "test_logits.npy": "test_logits.npy", "test_labels.npy": "test_labels.npy",
           "train.log": "train_log.txt"}
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt")
KEEP_REMOTE = frozenset({".gitattributes"})
EXAMPLES = {"positive": "The room was spotless, the staff were kind and the breakfast was excellent.",
            "negative": "Filthy bathroom, broken air conditioning and rude staff. Never again."}
FAILED_STAGES = frozenset({"BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR", "NO_APP_FILE", "DELETING", "STOPPED",
                           "PAUSED"})
PLACEHOLDER_REVISION = "0" * 40


class PublishError(RuntimeError):
    """A check failed; nothing after it runs."""


# --- state -----------------------------------------------------------------------------------------------


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    """Write the state atomically, so an interruption leaves the previous one intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# --- hashes ----------------------------------------------------------------------------------------------


def _chunks(data: bytes | Path):
    if isinstance(data, Path):
        with data.open("rb") as handle:
            yield from iter(lambda: handle.read(1 << 20), b"")
    else:
        yield data


def sha256_of(data: bytes | Path) -> str:
    digest = hashlib.sha256()
    for chunk in _chunks(data):
        digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data: bytes | Path) -> str:
    """The id the Hub reports for a file kept in git rather than LFS."""
    size = data.stat().st_size if isinstance(data, Path) else len(data)
    digest = hashlib.sha1(f"blob {size}\0".encode())
    for chunk in _chunks(data):
        digest.update(chunk)
    return digest.hexdigest()


def manifest_of(files: dict[str, bytes | Path]) -> dict[str, dict]:
    return {path: {"sha256": sha256_of(source), "git_sha1": git_blob_sha1(source)}
            for path, source in sorted(files.items())}


def manifest_problems(remote: dict[str, dict], manifest: dict[str, dict]) -> list[str]:
    """How a commit's files differ from the manifest: LFS files by sha256, the others by git blob id."""
    problems = [f"missing {path}" for path in sorted(set(manifest) - set(remote))]
    problems += [f"unexpected {path}" for path in sorted(set(remote) - set(manifest) - KEEP_REMOTE)]
    for path in sorted(set(remote) & set(manifest)):
        key = "sha256" if "sha256" in remote[path] else "git_sha1"
        if remote[path][key] != manifest[path][key]:
            problems.append(f"{path} has {key} {remote[path][key]}, expected {manifest[path][key]}")
    return problems


# --- check: the run ----------------------------------------------------------------------------------------


def check_run_records(run_dir: Path, bundle: Path = BUNDLE) -> dict[str, str]:
    """The run's records are byte-identical to the committed bundle of the scored run."""
    problems = [f"missing {name}" for name in MODEL_FILES if not (run_dir / name).is_file()]
    hashes = {}
    for run_name, bundle_name in RECORDS.items():
        if not (run_dir / run_name).is_file():
            problems.append(f"missing {run_name}")
            continue
        hashes[run_name] = sha256_of(run_dir / run_name)
        if hashes[run_name] != sha256_of(bundle / bundle_name):
            problems.append(f"{run_name} differs from {bundle_name} in the committed bundle")
    if problems:
        raise PublishError("the run folder is not the scored run: " + "; ".join(problems))
    return hashes


def compare_logits(new: np.ndarray, saved: np.ndarray, atol: float = ATOL) -> dict:
    """Same shape, finite values, every value within atol (rtol 0), and every argmax equal."""
    report = {"examples": int(len(saved)), "atol": atol, "rtol": 0.0,
              "same_shape": new.shape == saved.shape, "finite": bool(np.isfinite(new).all())}
    if report["same_shape"] and report["finite"]:
        report["max_abs_diff"] = float(np.max(np.abs(new - saved)))
        report["within_tolerance"] = bool(np.allclose(new, saved, rtol=0.0, atol=atol))
        report["argmax_agree"] = int(np.sum(new.argmax(axis=1) == saved.argmax(axis=1)))
    report["passed"] = bool(report["same_shape"] and report["finite"] and report.get("within_tolerance")
                            and report.get("argmax_agree") == len(saved))
    return report


def agreement_check(run_dir: Path, test: pd.DataFrame, fingerprint: dict, examples: int = CHECK_EXAMPLES,
                    atol: float = ATOL) -> dict:
    """An agreement check on the first test reviews: do the weights give the saved test logits?

    It scores them as training scored the test set: the run's tokenizer, the same
    maximum length, dynamic zero padding, batches in test order, eval mode and fp32,
    with no autocast; here on the CPU.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from reviewnlp.llm.encoder_trainer import EncodedReviews, pad_collate, predict_logits

    if frame_fingerprint(test) != fingerprint:
        raise PublishError("the test set differs from the one in split_views.json")
    settings = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["settings"]
    head = test.iloc[:examples]
    labels = head["label"].map({"negative": 0, "positive": 1}).tolist()
    if not np.array_equal(np.load(run_dir / "test_labels.npy")[:examples], labels):
        raise PublishError("the saved test labels are not in the order of the test set")
    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    loader = DataLoader(EncodedReviews(head["text"].astype(str).tolist(), labels, tokenizer, settings["max_length"]),
                        batch_size=settings["eval_batch_size"], collate_fn=pad_collate)
    model = AutoModelForSequenceClassification.from_pretrained(run_dir).float()
    new = predict_logits(model, loader, torch.device("cpu"))
    report = compare_logits(new, np.load(run_dir / "test_logits.npy")[:examples], atol)
    report["name"] = f"agreement check on {len(head):,} examples"
    return report


# --- the model card ----------------------------------------------------------------------------------------


def _day(iso: str) -> str:
    day = datetime.date.fromisoformat(iso)
    return f"{day.day} {day:%B %Y}"


def _negatives(model: dict) -> dict:
    (tn, fp), (fn, _tp) = model["confusion_matrix"]
    return {"recall": tn / (tn + fp), "precision": tn / (tn + fn), "errors": fp + fn}


def _power_of_ten(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def build_card(comparison: dict, latency: dict, split_views: dict, metrics: dict, *, code_commit: str,
               new_model_sha256: str, old_hub_commit: str, agreement: dict) -> str:
    """The model card. Every number comes from the committed results, not from memory."""
    random, time_split = comparison["splits"]["random"], comparison["splits"]["time"]
    model, nb = random["models"]["distilbert"], random["models"]["tfidf_nb"]
    pair = random["pairs"]["distilbert vs tfidf_nb"]
    other = time_split["models"]["distilbert"]
    ours, theirs = _negatives(model), _negatives(nb)
    test = random["test"]
    dates = split_views["time"]["review_dates"]
    timed = latency["models"]["distilbert"]
    settings = metrics["settings"]
    run_link = f"{GITHUB}/tree/{code_commit}/docs/experiments/results/distilbert_v2/random/distilbert"
    note_link = f"{GITHUB}/blob/{code_commit}/docs/experiments/decision_distilbert_vs_nb.md"
    lo, hi = model["macro_f1_ci95"]
    return f"""---
license: mit
base_model: distilbert-base-uncased
language:
  - en
tags:
  - sentiment-analysis
  - text-classification
  - hotel-reviews
library_name: transformers
pipeline_tag: text-classification
---

# DistilBERT hotel-review sentiment classifier

A full fine-tune of `distilbert-base-uncased` that labels an English hotel review as negative or
positive. It is part of the [`krimits/hotel-review-nlp`]({GITHUB}) project. It was trained and
tested on splits of Booking.com reviews that share no review text.

## Result of this model

It was scored on its test set of {test['rows']:,} reviews ({test['negative']:,} negative). That set
shares no review text with training. It was scored once, with the epoch that did best on dev.

| Measure | This model | TF-IDF + Naive Bayes, same test |
|---|---:|---:|
| Macro-F1 (95% bootstrap interval) | **{model['macro_f1']:.4f}** ({lo:.4f}–{hi:.4f}) | {nb['macro_f1']:.4f} ({nb['macro_f1_ci95'][0]:.4f}–{nb['macro_f1_ci95'][1]:.4f}) |
| Accuracy | {model['accuracy']:.2%} | {nb['accuracy']:.2%} |
| Negative reviews: recall / precision | {ours['recall']:.3f} / {ours['precision']:.3f} | {theirs['recall']:.3f} / {theirs['precision']:.3f} |
| Errors | {ours['errors']:,} | {theirs['errors']:,} |

- **Against Naive Bayes.** On the same reviews, the gain is
  {pair['macro_f1_difference_b_minus_a']:+.4f} macro-F1, with a paired 95% interval of
  {pair['paired_ci95'][0]:.4f} to {pair['paired_ci95'][1]:.4f}. Exact McNemar test:
  - {pair['mcnemar']['a_wrong_b_right']} reviews only this model gets right;
  - {pair['mcnemar']['a_right_b_wrong']} only Naive Bayes gets right;
  - p = {_power_of_ten(pair['mcnemar']['p_value'])}.
- **Training:** {metrics['data']['train']['rows']:,} reviews. The best dev macro-F1 was
  {metrics['best_dev_macro_f1']:.4f}.

## Related experiment, with a different checkpoint

A second model, with the same architecture and training recipe, was trained on another split of the
same reviews.
- **Training data:** reviews written up to {_day(dates['train'][1])}.
- **Result:** {other['macro_f1']:.4f} macro-F1 ({other['macro_f1_ci95'][0]:.4f}–{other['macro_f1_ci95'][1]:.4f})
  on reviews written from {_day(dates['test'][0])} to {_day(dates['test'][1])}.
- **What it means.** That is a different checkpoint, trained on different data. It is not a
  measurement of this model, nor an estimate of it. It shows only that scores can be lower on
  reviews from another period.

## Cost

- **Size:** {metrics['total_params']:,} parameters, {timed['size_mb']:.0f} MiB of weights.
- **Speed.** The timings come from the related checkpoint above. It has the same architecture and
  size as this model.
  - {timed['cpu']['batch_1']['ms_per_batch_median']:.0f} ms per review (median, one review at a time,
    one thread of an {latency['hardware']['cpu']});
  - {timed['gpu_whole_test']['reviews_per_second']:.0f} reviews per second on a {latency['hardware']['gpu']}.
- **Training time:** {comparison['cost']['training']['random']['distilbert']['training_minutes']} minutes
  on a {latency['hardware']['gpu']}.

## Limits

- **Data.** English Booking.com reviews of European hotels, written from {_day(split_views['random']['review_dates']['train'][0])}
  to {_day(split_views['random']['review_dates']['train'][1])}.
- **Two classes.** Reviews with both positive and negative text were left out of training and
  test.
- **Test set.** Each test set holds at most 10,000 positive reviews.
- **One training seed ({settings['seed']}).** The interval covers the sampling of test reviews, not
  the randomness of training.

## Training settings

`{settings['model_name']}`, seed {settings['seed']}, {settings['epochs']} epochs, batch
{settings['batch_size']}, maximum length {settings['max_length']}, learning rate {settings['lr']:g},
warm-up {settings['warmup_ratio']:.0%}, weight decay {settings['weight_decay']}, fp16 training. The
checkpoint kept is the epoch with the best dev macro-F1.

## Provenance

- **Training.** Trained by notebook 10 of the project, from commit
  [`{TRAINING_COMMIT[:7]}`]({GITHUB}/commit/{TRAINING_COMMIT}). Its metrics, logits and logs are
  [committed]({run_link}). The CI recomputes its scores from those logits. The
  [decision note]({note_link}) compares it with Naive Bayes.
- **Weights:** `model.safetensors` SHA-256 `{new_model_sha256}`.
- **Check before publishing.** An {agreement['name']}: the weights reproduced the saved test logits
  within an absolute tolerance of {agreement['atol']:g}. The largest difference was
  {agreement['max_abs_diff']:.1e}, and {agreement['argmax_agree']:,} of {agreement['examples']:,}
  predictions were equal.
- **Publishing code:** commit [`{code_commit[:7]}`]({GITHUB}/commit/{code_commit}) of the project
  (`scripts/publish_model.py`).
- **The previous model** is kept under the tag
  [`{LEGACY_TAG}`](https://huggingface.co/{MODEL_REPO}/tree/{LEGACY_TAG}) (commit
  `{old_hub_commit[:7]}`). It was trained on an earlier split whose test set shared 170 texts
  with training.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{MODEL_REPO}")
model = AutoModelForSequenceClassification.from_pretrained("{MODEL_REPO}")
inputs = tokenizer("The room was spotless and the staff was wonderful.", return_tensors="pt")
print(model.config.id2label[model(**inputs).logits.argmax(-1).item()])  # positive
```
"""


def upload_files(run_dir: Path, card: str, bundle: Path = BUNDLE) -> dict[str, bytes | Path]:
    """What the new commit holds: the model, the run's records, its config and the card."""
    files: dict[str, bytes | Path] = {name: run_dir / name for name in (*MODEL_FILES, *RECORDS)}
    files["run_config.yaml"] = bundle / "run_config.yaml"
    files["README.md"] = card.encode("utf-8")
    return files


# --- the Hub -----------------------------------------------------------------------------------------------


def remote_files(api, repo_id: str, revision: str) -> tuple[str, dict[str, dict]]:
    """The commit a model revision resolves to, and each file's id: LFS sha256, or git blob sha1."""
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    files = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        files[sibling.rfilename] = {"sha256": lfs.sha256} if lfs else {"git_sha1": sibling.blob_id}
    return info.sha, files


def tag_commit(api, repo_id: str, tag: str) -> str | None:
    return next((ref.target_commit for ref in api.list_repo_refs(repo_id).tags if ref.name == tag), None)


def read_hub(api, state: dict, repo_id: str = MODEL_REPO) -> bool:
    """Record the legacy commit: main while it holds the legacy model, else where the legacy tag points.

    Returns whether main still holds the legacy model, that is, whether nothing has been published yet.
    """
    head, files = remote_files(api, repo_id, "main")
    weights = files.get("model.safetensors", {}).get("sha256")
    if weights == OLD_MODEL_SHA256:
        legacy = head
    elif weights == state.get("new_model_sha256"):
        legacy = tag_commit(api, repo_id, LEGACY_TAG)
        if legacy is None:
            raise PublishError(f"{repo_id} holds the new weights but has no {LEGACY_TAG} tag")
        if remote_files(api, repo_id, legacy)[1].get("model.safetensors", {}).get("sha256") != OLD_MODEL_SHA256:
            raise PublishError(f"the {LEGACY_TAG} tag does not point to the legacy model")
    else:
        raise PublishError(f"main of {repo_id} ({head}) holds neither the legacy nor the new weights ({weights})")
    if state.get("old_hub_commit") not in (None, legacy):
        raise PublishError(f"the legacy commit is now {legacy}, not {state['old_hub_commit']} as recorded")
    state["old_hub_commit"], state["old_model_sha256"] = legacy, OLD_MODEL_SHA256
    return legacy == head


def verify_published(api, state: dict, commit: str, repo_id: str = MODEL_REPO) -> None:
    """The commit is main, holds exactly the manifest, has our weights, and the tag still marks the legacy commit."""
    head, _ = remote_files(api, repo_id, "main")
    if head != commit:
        raise PublishError(f"main of {repo_id} is {head}, not the published {commit}")
    problems = manifest_problems(remote_files(api, repo_id, commit)[1], state["manifest"])
    if problems:
        raise PublishError(f"commit {commit} does not hold the new version: " + "; ".join(problems))
    if tag_commit(api, repo_id, LEGACY_TAG) != state["old_hub_commit"]:
        raise PublishError(f"the {LEGACY_TAG} tag does not point to {state['old_hub_commit']}")
    weights = Path(api.hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=commit))
    if sha256_of(weights) != state["new_model_sha256"]:
        raise PublishError(f"the weights downloaded from {commit} are not the local ones")


def upload(api, state: dict, state_path: Path, files: dict[str, bytes | Path] | None,
           repo_id: str = MODEL_REPO) -> str:
    """Tag the legacy commit, then one explicit commit of the new files. Resumable."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    if "manifest" not in state:
        raise PublishError("no manifest: run the check first")
    if state.get("model_commit"):
        verify_published(api, state, state["model_commit"], repo_id)
        print("upload: already done, and verified at", state["model_commit"])
        return state["model_commit"]
    head, remote = remote_files(api, repo_id, "main")
    if remote.get("model.safetensors", {}).get("sha256") == state["new_model_sha256"]:
        verify_published(api, state, head, repo_id)  # an earlier run's commit, adopted only if it is exactly ours
        state["model_commit"] = head
        save_state(state_path, state)
        print("upload: an earlier run's commit is exactly this version; adopted", head)
        return head
    if head != state["old_hub_commit"] or remote.get("model.safetensors", {}).get("sha256") != OLD_MODEL_SHA256:
        raise PublishError(f"main of {repo_id} moved to {head} since the check")
    if files is None:
        raise PublishError("nothing to upload: run the check first")
    tagged = tag_commit(api, repo_id, LEGACY_TAG)
    if tagged is None:
        api.create_tag(repo_id, tag=LEGACY_TAG, revision=state["old_hub_commit"])
        tagged = tag_commit(api, repo_id, LEGACY_TAG)
    if tagged != state["old_hub_commit"]:
        raise PublishError(f"the {LEGACY_TAG} tag points to {tagged}, not to the legacy {state['old_hub_commit']}")
    state["legacy_tag"] = {"name": LEGACY_TAG, "commit": tagged}
    save_state(state_path, state)
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=source if isinstance(source, bytes)
                                     else str(source)) for path, source in sorted(files.items())]
    operations += [CommitOperationDelete(path_in_repo=path) for path in sorted(remote)
                   if path not in files and path not in KEEP_REMOTE]
    info = api.create_commit(
        repo_id=repo_id, operations=operations, parent_commit=state["old_hub_commit"],
        commit_message="Clean-split DistilBERT: full fine-tune on the v2 random split",
        commit_description=(f"Trained at {GITHUB}/commit/{TRAINING_COMMIT}; published with "
                            f"{GITHUB}/commit/{state['code_commit']}. model.safetensors SHA-256 "
                            f"{state['new_model_sha256']}. The previous model is under the tag {LEGACY_TAG}."))
    state["model_commit_unverified"] = info.oid
    save_state(state_path, state)
    verify_published(api, state, info.oid, repo_id)
    state["model_commit"] = state.pop("model_commit_unverified")
    save_state(state_path, state)
    print("upload: committed and verified", info.oid)
    return info.oid


# --- the demo ----------------------------------------------------------------------------------------------

APP_REPLACEMENTS = (
    ("- DistilBERT: krimits/distilbert-hotel-reviews (full fine-tune, macro-F1 0.9634).\n",
     "- DistilBERT: krimits/distilbert-hotel-reviews, pinned to the clean-split model (full\n"
     "  fine-tune; macro-F1 0.9642 on a test set that shares no review text with training).\n"),
    ("import time\n", "import hashlib\nimport time\n"),
    ('DISTILBERT_ID = "krimits/distilbert-hotel-reviews"\n',
     'DISTILBERT_ID = "krimits/distilbert-hotel-reviews"\n'
     'DISTILBERT_REVISION = "<<REVISION>>"  # the clean-split model, pinned so the texts match the weights\n'),
    ('print("Loading DistilBERT encoder ...")\n'
     "dtok = AutoTokenizer.from_pretrained(DISTILBERT_ID)\n"
     "dmodel = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_ID)\n"
     "dmodel.eval()\n",
     'print("Loading DistilBERT encoder ...")\n'
     "distilbert_dir = Path(snapshot_download(DISTILBERT_ID, revision=DISTILBERT_REVISION))\n"
     "dtok = AutoTokenizer.from_pretrained(distilbert_dir)\n"
     "dmodel = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)\n"
     "dmodel.eval()\n"
     'DISTILBERT_WEIGHTS_SHA256 = hashlib.sha256((distilbert_dir / "model.safetensors").read_bytes()).hexdigest()\n'),
    ("EXAMPLES = [\n",
     "def model_info():\n"
     '    """Which DistilBERT weights this running app loaded, for the check after a deploy."""\n'
     "    return {\n"
     '        "distilbert_repo": DISTILBERT_ID,\n'
     '        "distilbert_revision": DISTILBERT_REVISION,\n'
     '        "loaded_commit": distilbert_dir.name,  # the snapshot folder is named after its commit\n'
     '        "weights_sha256": DISTILBERT_WEIGHTS_SHA256,\n'
     "    }\n"
     "\n"
     "\n"
     "EXAMPLES = [\n"),
    ('        "The same review, three model families. "\n'
     '        "Full FT macro-F1 **0.9634** · Qwen QLoRA **0.9571** · McNemar **p = 0.525** "\n'
     '        "(statistically indistinguishable) · GreekBERT **0.9086** (Greek political-tweet "\n'
     '        "corpus — cross-domain on hotel text). "\n'
     '        "[Project repo](https://github.com/krimits/hotel-review-nlp) · "\n'
     '        "[Benchmark results](https://huggingface.co/datasets/krimits/hotel-review-nlp-frozen-splits/blob/main/runs/benchmark/results.json)"\n',
     '        "The same review, three model families. "\n'
     '        "DistilBERT full fine-tune: macro-F1 **0.9642** on a test set that shares no review "\n'
     '        "text with training. Qwen QLoRA (**0.9571**) and GreekBERT (**0.9086**, Greek "\n'
     '        "political-tweet corpus, cross-domain on hotel text) were measured on other test "\n'
     '        "sets, so the three numbers are not comparable. "\n'
     '        "[Project repo](https://github.com/krimits/hotel-review-nlp) · "\n'
     '        "[DistilBERT model card](https://huggingface.co/krimits/distilbert-hotel-reviews)"\n'),
    ("    gr.Examples(examples=EXAMPLES, inputs=[text_in])\n",
     "    gr.Examples(examples=EXAMPLES, inputs=[text_in])\n"
     "    # Not shown: lets a deploy check ask which DistilBERT weights this process loaded.\n"
     "    info_button = gr.Button(visible=False)\n"
     "    info_out = gr.JSON(visible=False)\n"
     '    info_button.click(model_info, outputs=info_out, api_name="model_info")\n'),
)
README_REPLACEMENTS = (
    ("sdk: gradio\n", "sdk: gradio\nsdk_version: <<GRADIO>>\n"),
    ("- **DistilBERT full fine-tune** (macro-F1 0.9634, ~25 ms/review) — production-speed encoder.\n",
     "- **DistilBERT full fine-tune** (macro-F1 0.9642 on a test set that shares no review text with\n"
     "  training, ~25 ms/review), pinned to one model commit — production-speed encoder.\n"),
    ("In the unified benchmark the two LoRA-based English approaches were **statistically\n"
     "indistinguishable** (exact McNemar p = 0.525) — try both and compare their errors.\n",
     "The Qwen and GreekBERT numbers were measured on other test sets, so they are not comparable\n"
     "with DistilBERT's — try the three side by side and compare their errors.\n"),
)


def _replace_each_once(name: str, text: str, replacements) -> str:
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise PublishError(f"{name}: expected exactly one {old.splitlines()[0].strip()!r}, found {count}")
        text = text.replace(old, new)
    return text


def patch_demo(app: str, readme: str, revision: str, gradio_version: str) -> tuple[str, str]:
    """The demo's app.py and README with DistilBERT pinned to `revision`. Deterministic."""
    if "sdk_version:" in readme:
        raise PublishError("README.md already pins an sdk_version")
    new_app = _replace_each_once("app.py", app, [(old, new.replace("<<REVISION>>", revision))
                                                 for old, new in APP_REPLACEMENTS])
    new_readme = _replace_each_once("README.md", readme, [(old, new.replace("<<GRADIO>>", gradio_version))
                                                          for old, new in README_REPLACEMENTS])
    ast.parse(new_app)
    if "from_pretrained(DISTILBERT_ID" in new_app or new_app.count("revision=DISTILBERT_REVISION") != 1:
        raise PublishError("app.py would still load DistilBERT without the pinned revision")
    for stale in ("0.9634", "0.525"):
        if stale in new_app or stale in new_readme:
            raise PublishError(f"the patched demo still says {stale}")
    return new_app, new_readme


def space_files(api, space_id: str, revision: str) -> tuple[str, str, str]:
    """(commit, app.py, README.md) of the Space at a revision."""
    commit = api.space_info(space_id, revision=revision).sha
    read = lambda name: Path(api.hf_hub_download(  # noqa: E731
        repo_id=space_id, filename=name, repo_type="space", revision=commit)).read_text(encoding="utf-8")
    return commit, read("app.py"), read("README.md")


def demo_status(api, state: dict, space_id: str = SPACE_REPO) -> tuple[str, str, str, str]:
    """("unpatched" or "pinned", commit, app.py, README.md): the only two states from which to go on."""
    head, app, readme = space_files(api, space_id, "main")
    parent = state.get("space_parent_commit")
    if parent is None or head == parent:
        patch_demo(app, readme, PLACEHOLDER_REVISION, state.get("gradio_version") or "0")  # it must apply
        return "unpatched", head, app, readme
    if not state.get("model_commit"):
        raise PublishError(f"the demo moved from {parent} to {head} before the model was published")
    _, original_app, original_readme = space_files(api, space_id, parent)
    if (app, readme) != patch_demo(original_app, original_readme, state["model_commit"], state["gradio_version"]):
        raise PublishError(f"the demo's main ({head}) is neither {parent} nor this patch of it")
    if state.get("space_commit") not in (None, head):
        raise PublishError(f"the demo moved from our commit {state['space_commit']} to {head}")
    return "pinned", head, app, readme


def _as_dict(value) -> dict:
    """A JSON output as gradio_client returns it: a dict, a JSON string, or the path of a JSON file."""
    if isinstance(value, str):
        value = Path(value).read_text(encoding="utf-8") if Path(value).is_file() else value
        value = json.loads(value)
    return value


def label_of(result) -> str:
    value = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(value, dict):
        return value["label"] if "label" in value else max(value, key=value.get)
    return str(value)


def connect_demo(space_id: str, attempts: int = 10, wait: float = 30.0):
    """A gradio_client for the demo, waiting for it to wake up if it sleeps."""
    from gradio_client import Client

    last = None
    for _ in range(attempts):
        try:
            return Client(space_id, verbose=False)
        except Exception as error:  # noqa: BLE001 - reported below if every attempt fails
            last = error
            time.sleep(wait)
    raise PublishError(f"cannot reach the demo {space_id}: {last}")


def pin_demo(api, state: dict, state_path: Path, space_id: str = SPACE_REPO) -> str:
    """Commit the patched app.py and README with parent_commit. Resumable."""
    from huggingface_hub import CommitOperationAdd

    if not state.get("model_commit"):
        raise PublishError("the model is not published yet")
    status, head, app, readme = demo_status(api, state, space_id)
    if status == "pinned":
        state["space_commit"] = head
        save_state(state_path, state)
        print("pin-demo: already done at", head)
        return head
    new_app, new_readme = patch_demo(app, readme, state["model_commit"], state["gradio_version"])
    state["space_parent_commit"] = head
    save_state(state_path, state)
    info = api.create_commit(
        repo_id=space_id, repo_type="space", parent_commit=head,
        operations=[CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=new_app.encode("utf-8")),
                    CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=new_readme.encode("utf-8"))],
        commit_message="Pin DistilBERT to the clean-split model",
        commit_description=f"{MODEL_REPO} at {state['model_commit']}; published with "
                           f"{GITHUB}/commit/{state['code_commit']}.")
    state["space_commit"] = info.oid
    save_state(state_path, state)
    if space_files(api, space_id, info.oid)[1:] != (new_app, new_readme):
        raise PublishError(f"the demo's files at {info.oid} are not the patched ones")
    print("pin-demo: committed", info.oid)
    return info.oid


def check_demo(api, connect: Callable, state: dict, state_path: Path, space_id: str = SPACE_REPO,
               timeout: float = 25 * 60, poll: float = 30.0, clock: Callable = time.monotonic,
               sleep: Callable = time.sleep) -> dict:
    """The running demo reports the new commit and weights, then classifies two reviews. Reads only."""
    expected = {"distilbert_revision": state["model_commit"], "loaded_commit": state["model_commit"],
                "weights_sha256": state["new_model_sha256"]}
    deadline, last = clock() + timeout, "no answer yet"
    while True:
        runtime = api.get_space_runtime(space_id)
        runtime_commit = (runtime.raw or {}).get("sha")
        if runtime.stage in FAILED_STAGES:
            raise PublishError(f"the demo is {runtime.stage}: {(runtime.raw or {}).get('errorMessage', '')}")
        if runtime.stage == "RUNNING" and runtime_commit in (None, state["space_commit"]):
            try:
                client = connect(space_id)
                info = _as_dict(client.predict(api_name="/model_info"))
                if all(info.get(key) == value for key, value in expected.items()):
                    break
                last = f"the demo reports {info}"
            except Exception as error:  # noqa: BLE001 - the previous app has no model_info yet
                last = f"{type(error).__name__}: {error}"
        else:
            last = f"stage {runtime.stage}, runtime commit {runtime_commit}"
        if clock() > deadline:
            raise PublishError(f"the demo did not serve the new model in {timeout / 60:.0f} minutes; last: {last}")
        print("check-demo: waiting;", last, flush=True)
        sleep(poll)
    predictions = {want: label_of(client.predict(text, api_name="/predict_distilbert"))
                   for want, text in EXAMPLES.items()}
    if any(got != want for want, got in predictions.items()):
        raise PublishError(f"the demo labels the examples {predictions}")
    state["demo_check"] = {"model_info": info, "runtime_commit": runtime_commit, "stage": runtime.stage,
                           "predictions": predictions,
                           "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    save_state(state_path, state)
    print("check-demo: the demo serves", state["model_commit"], "and labels both examples correctly")
    return state["demo_check"]


# --- the check, and the command line --------------------------------------------------------------------


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def versions() -> dict[str, str | None]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def check(api, connect: Callable, run_dir: Path, state: dict, state_path: Path, test_path: Path = TEST,
          space_id: str = SPACE_REPO) -> dict[str, bytes | Path]:
    """Everything that can be checked before the first write. Returns the files of the new commit."""
    state.update(code_commit=code_commit(), packages=versions(), model_repo=MODEL_REPO, space_repo=space_id,
                 training_commit=TRAINING_COMMIT)
    check_run_records(run_dir)
    state["new_model_sha256"] = sha256_of(run_dir / "model.safetensors")
    split_views = json.loads(SPLIT_VIEWS.read_text(encoding="utf-8"))
    report = agreement_check(run_dir, pd.read_parquet(test_path), split_views["random"]["splits"]["test"])
    state["logits_check"] = report
    save_state(state_path, state)
    print(f"check: {report['name']}: largest difference {report.get('max_abs_diff')}, "
          f"{report.get('argmax_agree')} of {report['examples']} predictions equal")
    if not report["passed"]:
        raise PublishError(f"the weights do not give the saved logits: {report}")
    unpublished = read_hub(api, state)
    print("check:", "the legacy model is on main" if unpublished else "the new model is already on main")
    client = connect(space_id)
    state["gradio_version"] = state.get("gradio_version") or client.config.get("version")
    if not state["gradio_version"]:
        raise PublishError("cannot read the demo's Gradio version")
    if label_of(client.predict(EXAMPLES["positive"], api_name="/predict_distilbert")) != "positive":
        raise PublishError("the demo does not label the positive example as positive")
    status, head, _, _ = demo_status(api, state, space_id)
    if status == "unpatched":
        state["space_parent_commit"] = head
    save_state(state_path, state)
    print(f"check: demo {status} at {head}, Gradio {state['gradio_version']}")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    card = build_card(json.loads(COMPARISON.read_text(encoding="utf-8")),
                      json.loads(LATENCY.read_text(encoding="utf-8")), split_views, metrics,
                      code_commit=state["code_commit"], new_model_sha256=state["new_model_sha256"],
                      old_hub_commit=state["old_hub_commit"], agreement=report)
    files = upload_files(run_dir, card)
    if unpublished:  # once committed, only the manifest saved before the commit may describe it
        state["manifest"] = manifest_of(files)
        save_state(state_path, state)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("check", "upload", "pin-demo", "check-demo", "all"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=TEST)
    parser.add_argument("--card-out", type=Path, help="also write the model card here")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api, state = HfApi(), load_state(args.state)
    try:
        files = None
        if args.step in ("check", "upload", "all"):
            files = check(api, connect_demo, args.run_dir, state, args.state, args.test)
            if args.card_out:
                args.card_out.write_bytes(files["README.md"])
        if args.step in ("upload", "all"):
            upload(api, state, args.state, files)
        if args.step in ("pin-demo", "all"):
            pin_demo(api, state, args.state)
        if args.step in ("check-demo", "all"):
            check_demo(api, connect_demo, state, args.state)
    except PublishError as error:
        raise SystemExit(f"stopped: {error}") from None


if __name__ == "__main__":
    main()
