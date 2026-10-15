"""The notebook 27 results: intact, from the pinned commit, and reproducible from their logits.

The bundle came back from Colab as seeds_and_bilstm_results.zip, in repository paths. It
holds the BiLSTM runs (docs/experiments/results/bilstm_v2/) and the DistilBERT seed 43 and
44 runs (docs/experiments/results/distilbert_v2/time/*_seed4*). No GPU and no data needed.
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "experiments" / "results"
BILSTM = RESULTS / "bilstm_v2"
DISTILBERT = RESULTS / "distilbert_v2"
SPLIT_VIEWS = RESULTS / "split_views.json"
COMMIT = "ed4070c72c452d8e9ce80568e30170e0cafdd32e"
PROTOCOLS = ("docs/experiments/bilstm_clean_splits/PROTOCOL.md",
             "docs/experiments/distilbert_seed_robustness/PROTOCOL.md")
RUNS = [(BILSTM / view / "bilstm", view, 42) for view in ("random", "time", "hotel")]
RUNS += [(BILSTM / "time" / f"bilstm_seed{seed}", "time", seed) for seed in (43, 44)]
RUNS += [(DISTILBERT / "time" / f"{name}_seed{seed}", "time", seed)
         for name in ("distilbert", "distilbert_lora_scratch") for seed in (43, 44)]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_file_matches_the_manifest():
    manifest = _json(BILSTM / "artifact_manifest.json")
    assert manifest["commit"] == COMMIT == _json(BILSTM / "environment.json")["commit"]
    listed = {entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        content = (ROOT / entry["path"]).read_bytes()
        assert len(content) == entry["bytes"], entry["path"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], entry["path"]
    present = {path.relative_to(ROOT).as_posix() for path in BILSTM.rglob("*") if path.is_file()}
    present |= {path.relative_to(ROOT).as_posix() for path in DISTILBERT.glob("time/*_seed4*/*")}
    present.add("docs/experiments/results/distilbert_seed_robustness.json")
    assert present - listed == {"docs/experiments/results/bilstm_v2/artifact_manifest.json"}
    assert not any(path.endswith((".safetensors", ".bin", ".pt", ".joblib")) for path in listed)  # no weights


def test_the_protocols_were_in_the_training_commit_and_have_not_changed():
    run = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)  # noqa: E731
    if run("cat-file", "-e", f"{COMMIT}^{{commit}}").returncode != 0:
        pytest.skip("the training commit is not in this clone")
    for protocol in PROTOCOLS:
        at_training = run("show", f"{COMMIT}:{protocol}")
        assert at_training.returncode == 0, protocol
        assert at_training.stdout == (ROOT / protocol).read_text(encoding="utf-8"), protocol


@pytest.mark.parametrize(("directory", "view", "seed"), RUNS, ids=lambda value: getattr(value, "name", str(value)))
def test_each_run_has_its_seed_its_data_and_scores_from_its_logits(directory, view, seed):
    metrics = _json(directory / "metrics.json")
    assert metrics["settings"]["seed"] == seed
    assert metrics["data"] == _json(SPLIT_VIEWS)[view]["splits"]
    assert _json(directory / "request.json")["commit"] == COMMIT
    for part in ("dev", "test"):
        labels = np.load(directory / f"{part}_labels.npy", allow_pickle=False)
        logits = np.load(directory / f"{part}_logits.npy", allow_pickle=False)
        fingerprint = metrics["data"][part]
        assert logits.shape == (fingerprint["rows"], 2) and np.isfinite(logits).all()
        assert [int((labels == value).sum()) for value in (0, 1)] == [fingerprint["negative"], fingerprint["positive"]]
        # The BiLSTM names its classes "0" and "1"; the DistilBERT trainer names them.
        names = ("0", "1") if directory.name.startswith("bilstm") else ("negative", "positive")
        scored = binary_metrics(labels, logits.argmax(axis=1), label_names=names, label_values=(0, 1))
        if part == "dev":
            assert matches(scored["macro_f1"], metrics["best_dev_macro_f1"])
        else:
            assert matches(scored, metrics["test"])


def test_the_bilstm_rule_is_reproduced_from_the_logits():
    compare = _script("compare_bilstm_clean")
    committed = _json(BILSTM / "bilstm_comparison_v2.json")
    recomputed = json.loads(json.dumps(compare.compare(BILSTM, _json(SPLIT_VIEWS),
                                                       _json(BILSTM / "latency_frontier.json"))))
    assert matches(recomputed, committed)
    assert committed["primary"]["branch"] == "DistilBERT meaningfully better"
    assert committed["seed_check"]["statement"] == "the conclusion holds across three training seeds of the BiLSTM"
    assert committed["frontier"]["frontier"] == ["tfidf_nb", "bilstm", "distilbert", "distilbert_lora_scratch"]
    assert committed["selection_frozen"] is False and committed["promoted"] is False


def test_the_seed_rule_is_reproduced_from_the_logits():
    compare = _script("compare_seeds")
    committed = _json(RESULTS / "distilbert_seed_robustness.json")
    assert matches(json.loads(json.dumps(compare.compare(DISTILBERT, _json(SPLIT_VIEWS)))), committed)
    assert committed["outcome"] == "the lead holds across seeds"
    assert committed["size"] == "clear but small"
    assert committed["selection_frozen"] is False and committed["promoted"] is False
