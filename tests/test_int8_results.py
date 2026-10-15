"""The notebook 28 results: intact, from the pinned commit and model, and reproducible from their logits."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reviewnlp.evaluation.metrics import binary_metrics, paired_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "experiments" / "results" / "int8_v1"
COMMIT = "93dfdff49545faf81a71386baaeaf8a45d889d21"
PROTOCOL = "docs/experiments/int8_and_batching/PROTOCOL.md"

_spec = importlib.util.spec_from_file_location("benchmark_int8", ROOT / "scripts" / "benchmark_int8.py")
int8 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = int8
_spec.loader.exec_module(int8)


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> dict:
    return _json("int8_benchmark.json")


def test_every_file_matches_the_manifest():
    manifest = _json("artifact_manifest.json")
    assert manifest["commit"] == COMMIT == _json("environment.json")["commit"]
    for entry in manifest["files"]:
        content = (ROOT / entry["path"]).read_bytes()
        assert len(content) == entry["bytes"] and hashlib.sha256(content).hexdigest() == entry["sha256"], entry["path"]
    present = {path.relative_to(ROOT).as_posix() for path in RESULTS.iterdir() if path.is_file()}
    listed = {entry["path"] for entry in manifest["files"]}
    assert present - listed == {"docs/experiments/results/int8_v1/artifact_manifest.json"}
    assert not any(path.endswith((".safetensors", ".bin", ".pt")) for path in present)


def test_the_run_used_the_pinned_model_data_and_protocol(report):
    model = _json("environment.json")["model"]
    assert (model["revision"], model["weights_sha256"]) == (int8.MODEL_REVISION, int8.WEIGHTS_SHA256)
    views = json.loads((ROOT / "docs" / "experiments" / "results" / "split_views.json").read_text(encoding="utf-8"))
    assert report["test"] == views["random"]["splits"]["test"]
    assert report["protocol"] == PROTOCOL and report["hardware"]["torch_threads"] == 1
    run = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)  # noqa: E731
    if run("cat-file", "-e", f"{COMMIT}^{{commit}}").returncode != 0:
        pytest.skip("the run's commit is not in this clone")
    assert run("show", f"{COMMIT}:{PROTOCOL}").stdout == (ROOT / PROTOCOL).read_text(encoding="utf-8")


def test_the_comparison_and_rule_are_reproduced_from_the_logits(report):
    labels = np.load(RESULTS / "test_labels.npy")
    fp32, quantized = (np.load(RESULTS / f"{name}_test_logits.npy").argmax(1) for name in ("fp32", "int8"))
    committed = np.load(int8.COMMITTED).argmax(1)
    assert np.array_equal(labels, np.load(int8.COMMITTED.parent / "test_labels.npy"))
    assert int(np.sum(fp32 == committed)) == report["integrity"]["fp32_agrees_with_committed"] == len(labels)
    d, lo, hi = paired_bootstrap_diff(labels, fp32, quantized, (0, 1), 2000, 0)
    primary = report["primary"]
    assert (d, [lo, hi]) == (primary["d_int8_minus_fp32"], primary["paired_ci95"])
    assert int(np.sum(fp32 == quantized)) == primary["agreement"]
    for name, predicted in (("fp32", fp32), ("int8", quantized)):
        assert binary_metrics(labels, predicted, label_values=(0, 1))["macro_f1"] == report["models"][name]["test"]["macro_f1"]
    assert int8.outcome(lo, primary["lat_int8"], primary["lat_fp32"]) == primary["conclusion"] == int8.RULE[0][1]
    for name, verdict in (("fp32", "faster"), ("int8", "not faster")):
        cpu = report["models"][name]["cpu"]
        assert int8.batching_verdict(cpu["batch_32_sorted"]["ms_per_review"],
                                     cpu["batch_1"]["ms_per_batch_median"]) == report["models"][name]["sorted_batching"] == verdict
    assert report["selection_frozen"] is False and report["promoted"] is False
