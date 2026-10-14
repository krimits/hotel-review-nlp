"""The README's Qwen row is bound to an archived artifact that says how it was scored."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from reviewnlp.llm.prompt_format import parse_label

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "experiments" / "results" / "qwen_legacy_v1"


def metrics() -> dict:
    return json.loads((RECORD / "metrics.json").read_bytes())


def test_the_archived_metrics_match_the_hash_their_record_states():
    stated = re.search(r"SHA-256 of the archived file: `([0-9a-f]{64})`, ([\d,]+) bytes",
                       (RECORD / "README.md").read_text(encoding="utf-8"))
    data = (RECORD / "metrics.json").read_bytes()
    assert stated and hashlib.sha256(data).hexdigest() == stated[1]
    assert len(data) == int(stated[2].replace(",", "")) == 4561


def test_the_two_scores_are_kept_apart():
    evaluation = metrics()["evaluation"]
    assert evaluation["generation_metrics"] == {"accuracy": 0.0, "macro_f1": 0.0}
    assert evaluation["invalid_output_count"] == metrics()["dataset"]["eval_rows"] == 13278
    assert evaluation["first_step_score_metrics"] == {"accuracy": 0.9671, "macro_f1": 0.9571}
    assert metrics()["dataset"]["train_cap_total"] == 20000
    # The run's code commit is part of this repository's history.
    commit = metrics()["repository"]["commit"]
    assert subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT).returncode == 0


def test_the_recorded_rejections_begin_with_their_gold_label():
    # Consistent with, not a measurement of, close agreement between the two rules.
    samples = metrics()["evaluation"]["invalid_output_samples"]
    assert len(samples) == 25
    assert all(parse_label(sample["raw_generation"]) == sample["gold"] for sample in samples)
    assert all(sample["raw_generation"].strip() != sample["gold"] for sample in samples)


def test_the_readme_row_points_to_the_record_and_names_the_rule():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    row = next(line for line in readme.splitlines() if line.startswith("| Qwen2.5-0.5B + QLoRA |"))
    assert "0.9571" in row and "first-step label scores" in row
    assert "docs/experiments/results/qwen_legacy_v1/README.md" in readme
    assert "20k subset" not in readme
