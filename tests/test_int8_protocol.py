"""Lock test for docs/experiments/int8_and_batching/PROTOCOL.md.

The protocol is frozen once committed. This test fails if its sections, its data fingerprint,
its model pin or its decision rule drift from what the repository records.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "experiments" / "int8_and_batching" / "PROTOCOL.md"
MANIFEST = ROOT / "docs" / "experiments" / "results" / "v2_splits" / "random" / "data_manifest.json"
PUBLISHED = ROOT / "docs" / "experiments" / "results" / "distilbert_v2" / "published.json"
RUN = ROOT / "docs" / "experiments" / "results" / "distilbert_v2" / "random" / "distilbert"
RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
RULE = [
    "lo > -0.01 and lat_int8 < 0.9 * lat_fp32",
    "lo > -0.01 and lat_int8 >= 0.9 * lat_fp32",
    "lo <= -0.01",
]


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(protocol) -> str:
    return " ".join(protocol.split())


def table_rows(text: str) -> list[list[str]]:
    """Body rows of every markdown table (after its `---` separator), markers stripped."""
    rows, in_body = [], False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            in_body = False
            continue
        if set(line) <= set("|- :"):
            in_body = True
            continue
        if in_body:
            rows.append([c.strip().strip("*").strip().strip("`") for c in line.strip("|").split("|")])
    return rows


def test_the_five_required_sections_are_present(protocol):
    for heading in ("## 1. Question", "## 2. Data", "## 3. One primary comparison",
                    "## 4. Decision rule", "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"


def test_the_random_test_set_is_pinned_by_its_recorded_fingerprint(protocol):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert RAW_SHA256 in protocol and manifest["raw_csv_sha256"] == RAW_SHA256
    assert "238,154,765" in protocol and manifest["raw_csv_bytes"] == 238154765
    (row,) = [cells for cells in table_rows(protocol) if len(cells) == 6 and cells[0] == "random"]
    info = manifest["splits"][row[1]]
    assert row[1] == "test"
    assert [int(row[2]), int(row[3]), int(row[4])] == [info["rows"], info["negative"], info["positive"]]
    assert info["sha256"].startswith(row[5])


def test_the_model_is_the_published_checkpoint(prose):
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    assert published["model_commit"] in prose and len(published["model_commit"]) == 40
    assert published["manifest"]["model.safetensors"]["sha256"] in prose
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    assert f"macro-F1 {metrics['test']['macro_f1']}" in prose
    assert "at least 99% of the test reviews" in prose


def test_the_decision_rule_partitions_every_outcome(protocol):
    rule = protocol.split("## 4. Decision rule", 1)[1].split("## 5. Limits", 1)[0]
    conditions = [cells[0] for cells in table_rows(rule) if len(cells) == 2]
    assert conditions == RULE, conditions
    outcomes = [  # (lo, lat_int8, lat_fp32)
        (-0.002, 30.0, 57.0),   # kept, faster
        (-0.002, 55.0, 57.0),   # kept, a tie on speed
        (-0.0099, 51.3, 57.0),  # kept at the margin; exactly 0.9 x is not faster
        (-0.01, 30.0, 57.0),    # lo exactly at the margin
        (-0.03, 30.0, 57.0),    # a meaningful loss
    ]
    fired = dict.fromkeys(conditions, 0)
    for lo, lat_int8, lat_fp32 in outcomes:
        scope = {"lo": lo, "lat_int8": lat_int8, "lat_fp32": lat_fp32}
        matches = [c for c in conditions if eval(c, {"__builtins__": {}}, scope)]  # noqa: S307
        assert len(matches) == 1, f"{scope} matched {matches}"
        fired[matches[0]] += 1
    assert all(fired.values()), f"a branch was never exercised: {fired}"


def test_the_method_is_the_one_the_existing_timings_used(prose):
    for required in ("13,263 test reviews", "2000 resamples, seed 0", "one PyTorch thread",
                     "3 warm-up calls", "500 test reviews drawn with seed 0", "quantize_dynamic",
                     "under 0.9 times batch 1's", "at least 99% of the reviews"):
        assert required in prose, required


def test_the_limits_name_what_the_run_does_not_show(protocol):
    limits = protocol.split("## 5. Limits", 1)[1]
    for required in ("No production", "One CPU and one thread", "linear layers only",
                     "The random split", "One checkpoint, seed 42", "selection_frozen"):
        assert required in limits, f"limits must mention {required}"
