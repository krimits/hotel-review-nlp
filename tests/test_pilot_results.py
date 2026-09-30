"""The pilot's committed agreement: no review text, the drawn items, and kappa reproduced from the labels.

The labels came back on 30 September 2026 (docs/case_study/results/annotation/README.md). The
sheets hold review text and are not committed, so these checks use only the committed ids, labels
and scorer outputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "case_study" / "results" / "annotation"
PROTOCOL = "docs/annotation/pilot_protocol.md"
CHANGES = "## Changes after locking"

_spec = importlib.util.spec_from_file_location("score_annotations", ROOT / "scripts" / "score_annotations.py")
score = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = score
_spec.loader.exec_module(score)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(key: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each annotator's labels, read back through the scorer's own check."""
    raw = pd.read_csv(RESULTS / "labels_raw.csv", dtype=str, keep_default_na=False)
    expected = {"A": set(key.index), "B": set(key.index[key["in_B"] == 1])}
    return {who: score.check_sheet(frame.assign(done="yes"), expected[who], f"committed labels of {who}")
            for who, frame in raw.groupby("annotator")}


def test_no_review_text_is_committed():
    assert list(pd.read_csv(RESULTS / "labels_raw.csv", nrows=0).columns) == ["item", "annotator", *score.TOPICS]
    assert {"text", "note"}.isdisjoint(pd.read_csv(RESULTS / "key.csv", nrows=0).columns)
    assert sorted(path.name for path in RESULTS.iterdir()) == [
        "README.md", "agreement.json", "key.csv", "labels_raw.csv", "sample_manifest.json"]


def test_the_labels_cover_exactly_the_drawn_items():
    key = score.read_key(RESULTS / "key.csv")
    manifest = _json(RESULTS / "sample_manifest.json")
    assert len(key) == manifest["items"] and int(key["in_B"].sum()) == manifest["second_annotator_items"]
    labels = _labels(key)  # the check refuses a missing, repeated or unknown item
    sheets = _json(RESULTS / "agreement.json")["sheets"]
    assert (len(labels["A"]), len(labels["B"])) == (sheets["A"]["items"], sheets["B"]["items"]) == (300, 80)


def test_kappa_is_reproduced_from_the_committed_labels():
    key = score.read_key(RESULTS / "key.csv")
    labels = _labels(key)
    recomputed = json.loads(json.dumps(score.agreement(key, labels["A"], labels["B"], seed=0)))
    assert recomputed == _json(RESULTS / "agreement.json")["topics"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def test_the_protocol_was_locked_at_the_draw_and_only_its_changes_have_grown():
    manifest = _json(RESULTS / "sample_manifest.json")
    commit = manifest["code_commit"]
    if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        pytest.skip("the draw's commit is not in this clone")
    at_draw = _git("show", f"{commit}:{PROTOCOL}")
    assert at_draw.returncode == 0, "the protocol was not in the commit the sample was drawn from"
    assert hashlib.sha256(at_draw.stdout).hexdigest() == manifest["protocol_sha256"]
    locked, now = at_draw.stdout.decode("utf-8"), (ROOT / PROTOCOL).read_text(encoding="utf-8")
    assert now[:now.index(CHANGES)] == locked[:locked.index(CHANGES)]
