"""The committed phase 2 results: intact, trained as the plan says, and reproducible from their logits.

The runs came back from Colab (notebook 10) as a bundle in
docs/experiments/results/distilbert_v2/. These checks need no GPU and no data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reviewnlp.evaluation.compare import matches
from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.utils.seed import load_config

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "experiments" / "results" / "distilbert_v2"
COMPARISON = ROOT / "docs" / "experiments" / "results" / "model_comparison_v2.json"
SPLIT_VIEWS = ROOT / "docs" / "experiments" / "results" / "split_views.json"
PLAN = "docs/experiments/phase2_analysis_plan.md"
CONFIGS = {"random": "configs/distilbert_v2.yaml", "time": "configs/distilbert_v2_time.yaml"}
RUNS = [(split, name) for split in CONFIGS for name in ("distilbert", "distilbert_lora_scratch")]
LORA = {"r": 8, "alpha": 16.0, "dropout": 0.0, "target_modules": ["q_lin", "v_lin"],
        "modules_to_save": ["pre_classifier", "classifier"]}
LABELS = {"label_names": ("negative", "positive"), "label_values": (0, 1)}

_spec = importlib.util.spec_from_file_location("compare_split_models", ROOT / "scripts" / "compare_split_models.py")
compare = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = compare
_spec.loader.exec_module(compare)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return _json(BUNDLE / "artifact_manifest.json")


def test_every_file_matches_the_manifest():
    listed = {entry["path"] for entry in _manifest()["files"]}
    for entry in _manifest()["files"]:
        content = (BUNDLE / entry["path"]).read_bytes()
        assert len(content) == entry["bytes"], entry["path"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], entry["path"]
    present = {path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob("*") if path.is_file()}
    assert present - listed == {"artifact_manifest.json", "README.md"}


def test_every_run_comes_from_the_same_commit():
    commit = _manifest()["commit"]
    assert _json(BUNDLE / "environment.json")["commit"] == commit
    for split, name in RUNS:
        assert _json(BUNDLE / split / name / "request.json")["commit"] == commit, f"{split}/{name}"


def _fixed_part(plan: str) -> str:
    """From the question to the rule for a meaningful gain: the part fixed before training."""
    return plan[plan.index("## Question"):plan.index("## Cost")]


def test_the_plan_was_in_the_training_commit_and_its_rules_have_not_changed():
    commit = _manifest()["commit"]
    run = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)  # noqa: E731
    if run("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        pytest.skip("the training commit is not in this clone")
    at_training = run("show", f"{commit}:{PLAN}")
    assert at_training.returncode == 0, "the plan was not in the commit the runs were trained from"
    assert _fixed_part(at_training.stdout) == _fixed_part((ROOT / PLAN).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("split", "name"), RUNS)
def test_each_run_used_the_settings_of_the_plan(split, name):
    config = load_config(str(ROOT / CONFIGS[split]))
    metrics = _json(BUNDLE / split / name / "metrics.json")
    lora = name == "distilbert_lora_scratch"
    assert {key: value for key, value in metrics["settings"].items() if key != "processed_dir"} == {
        "model_name": config["model"]["name"], "seed": config["seed"], "max_length": config["model"]["max_length"],
        "batch_size": config["train"]["batch_size"], "eval_batch_size": config["train"]["eval_batch_size"],
        "epochs": config["train"]["epochs"], "lr": 1e-4 if lora else config["train"]["lr"],
        "weight_decay": config["train"]["weight_decay"], "warmup_ratio": config["train"]["warmup_ratio"],
        "fp16": config["train"]["fp16"], "lora": LORA if lora else None}
    assert metrics["data"] == _json(SPLIT_VIEWS)[split]["splits"]


@pytest.mark.parametrize(("split", "name"), RUNS)
def test_the_recorded_scores_come_from_the_saved_logits(split, name):
    directory = BUNDLE / split / name
    metrics = _json(directory / "metrics.json")
    for part in ("dev", "test"):
        labels = np.load(directory / f"{part}_labels.npy", allow_pickle=False)
        logits = np.load(directory / f"{part}_logits.npy", allow_pickle=False)
        fingerprint = metrics["data"][part]
        assert logits.shape == (fingerprint["rows"], 2) and np.isfinite(logits).all()
        assert [int((labels == value).sum()) for value in (0, 1)] == [fingerprint["negative"], fingerprint["positive"]]
        scored = binary_metrics(labels, logits.argmax(axis=1), **LABELS)
        if part == "dev":
            assert matches(scored["macro_f1"], metrics["best_dev_macro_f1"])
        else:
            assert matches(scored, metrics["test"])


def test_the_comparison_is_reproduced_from_the_logits():
    recomputed = json.loads(json.dumps(compare.compare(BUNDLE, _json(SPLIT_VIEWS))))
    committed = _json(COMPARISON)
    assert matches(recomputed, committed)
    assert committed == _json(BUNDLE / "model_comparison_v2.json")  # as the notebook wrote it on Colab
    assert committed["primary"]["verdict"] == {"better": "distilbert", "gain": "meaningful"}
