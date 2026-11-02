"""Shape checks on the committed experiment artifacts.

The suite used to spend thirteen modules asserting that notebook *source text* still
contains given strings. This replaces that with what actually matters: every committed
result file parses, is an object, and carries plausible values for the keys it declares.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "experiments"

# Files whose name marks them as a recorded result rather than an input.
RESULT_NAMES = {"metrics.json", "results.json", "score.json", "verdict.json", "verification.json", "run.json"}

PROBABILITY_KEYS = ("macro_f1", "accuracy", "precision", "recall", "f1", "weighted_f1")
COUNT_KEYS = ("n_test", "n_train", "n_dev", "test_size", "n_samples", "n_reviews")


def _result_files() -> list[Path]:
    return sorted(
        path
        for path in RESULTS.rglob("*.json")
        if path.name in RESULT_NAMES
    )


def test_the_repository_still_commits_its_result_files():
    files = _result_files()
    assert len(files) >= 30, f"expected the committed result set, found {len(files)}"


@pytest.mark.parametrize("path", _result_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_a_result_file_parses_as_an_object(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} should hold a JSON object"
    assert payload, f"{path} should not be empty"


@pytest.mark.parametrize("path", _result_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_declared_metrics_are_in_range(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if key in PROBABILITY_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
            assert 0.0 <= float(value) <= 1.0, f"{path}: {key}={value} is not a metric in [0, 1]"
        if key in COUNT_KEYS and isinstance(value, int) and not isinstance(value, bool):
            assert value > 0, f"{path}: {key}={value} should be a positive count"
