"""Lock test for docs/experiments/triage_diagnosis/PROTOCOL.md.

The protocol is frozen once committed. This test fails if its sections, its data fingerprint, the catalogue
or planning scripts it pins, its trigger words or its decision rule drift from what the repository records.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs" / "experiments" / "triage_diagnosis"
PROTOCOL = FOLDER / "PROTOCOL.md"
RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
PINNED = {  # file -> SHA-256 at lock
    "catalogue_v1.yaml": "2575996133afaa6e0178060df3bd5288d01ce9b7d842a6b3bf67d362d29445e9",
    "planning/population.py": "73813486e0289d70e883f0180e0c63d92d8a21adfbd8c3f32502ba00e2404950",
    "planning/power.py": "979d02f4a0007c5ebd858e34ef37dcbb35c1fca839ec59fe26f6dacadeb46ec4",
}
RULE = [
    "rec >= 0.90 and lo >= 0.80 and fa <= 5",
    "rec >= 0.90 and lo >= 0.80 and fa > 5",
    "rec >= 0.90 and lo < 0.80",
    "rec < 0.90",
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


def section(protocol: str, start: str, end: str) -> str:
    return protocol.split(start, 1)[1].split(end, 1)[0]


def test_the_five_required_sections_are_present_and_it_is_locked(protocol):
    for heading in ("## 1. Question", "## 2. Data", "## 3. One primary comparison",
                    "## 4. Decision rule", "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"
    assert protocol.startswith("# Triage diagnosis and routing: protocol — version 1, locked\n")
    assert "## 9. Fields filled at lock" in protocol and "open field" not in protocol.lower()


def test_the_raw_data_and_the_pinned_files_match_their_fingerprints(prose):
    assert RAW_SHA256 in prose and "238,154,765 bytes" in prose
    for name, digest in PINNED.items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == digest, name
    assert PINNED["catalogue_v1.yaml"] in prose


def test_models_seeds_and_sample_sizes_are_fixed(prose):
    assert "`distilbert/distilbert-base-uncased` at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`" in prose
    teacher, second = "anthropic/claude-sonnet-5.5-20260928", "openai/gpt-6.1-sol-20260929"
    assert f"**Teacher:** `{teacher}`" in prose and f"**Second rater:** `{second}`" in prose
    assert teacher.split("/")[0] != second.split("/")[0]
    assert "seed 20261021 for development and 20261022 for test" in prose
    assert "300 reviews from the development pool, split H 90, N 180, B 30" in prose and 90 + 180 + 30 == 300
    assert "500 reviews from the test pool, split H 150, N 300, B 50" in prose and 150 + 300 + 50 == 500
    assert "2,000 bootstrap resamples, seed 0" in prose


def trigger_probes(protocol: str) -> list[str]:
    """Words the protocol's trigger list names: `x*` gives x and an extension, `x(s)` gives x and xs."""
    block = section(section(protocol, "**Trigger words.**", "### Exclusions"), "```\n", "\n```")
    # Two entries contain a space; "a/c" is one spelling, not two alternatives.
    tokens = block.replace("bed bug", "bed_bug").replace("air con", "air_con").split()
    probes = []
    for token in tokens:
        for alt in [token] if token == "a/c" else token.split("/"):
            alt = alt.replace("_", " ")
            if alt.endswith("*"):
                probes += [alt[:-1], alt[:-1] + "ed"]
            elif alt.endswith("(s)"):
                probes += [alt[:-3], alt[:-3] + "s"]
            else:
                probes.append(alt)
    return probes


def test_the_trigger_words_agree_with_the_stratifier(protocol):
    tree = ast.parse((FOLDER / "planning" / "population.py").read_text(encoding="utf-8"))
    (call,) = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(getattr(target, "id", None) == "TRIGGER" for target in node.targets)]
    assert isinstance(call, ast.Call)
    pattern = ast.literal_eval(call.args[0])
    assert pattern.startswith(r"\b(") and pattern.endswith(r")\b")
    alternatives = pattern[3:-3].split("|")
    probes = trigger_probes(protocol)
    for probe in probes:
        assert re.fullmatch(pattern[2:-2], probe, re.I), f"protocol word not in the regex: {probe}"
    for alt in alternatives:
        assert any(re.fullmatch(alt, probe, re.I) for probe in probes), f"regex word not in the protocol: {alt}"


def test_the_decision_rule_partitions_every_outcome(protocol):
    rule = section(protocol, "## 4. Decision rule", "**Feasibility")
    conditions = [cells[0] for cells in table_rows(rule) if len(cells) == 2]
    assert conditions == RULE, conditions
    assert "**`F = 5` false urgent flags per 100 reviews**" in " ".join(protocol.split())
    assert "**`L = 0.80`**" in protocol
    outcomes = [  # (rec, lo, fa)
        (0.93, 0.85, 3.0),   # met
        (0.90, 0.80, 5.0),   # met at every margin
        (0.93, 0.85, 5.1),   # over budget
        (0.91, 0.79, 2.0),   # not established
        (0.899, 0.85, 1.0),  # not met
        (0.50, 0.30, 9.0),   # not met
    ]
    # eval only ever sees the four RULE strings above: the equality check has already pinned them.
    fired = dict.fromkeys(conditions, 0)
    for rec, lo, fa in outcomes:
        hits = [c for c in conditions if eval(c, {}, {"rec": rec, "lo": lo, "fa": fa})]  # noqa: S307
        assert len(hits) == 1, (rec, lo, fa, hits)
        fired[hits[0]] += 1
    assert all(fired.values()), fired


def test_the_catalogue_matches_the_protocols_label_rules(protocol):
    import yaml

    types = {entry["id"]: entry for entry in yaml.safe_load(
        (FOLDER / "catalogue_v1.yaml").read_text(encoding="utf-8"))["types"]}
    assert "20 types" in protocol and len(types) == 20
    for name, priority, department in (("plumbing_major", "urgent", "maintenance"),
                                       ("plumbing_minor", "routine", "maintenance"),
                                       ("noise_internal", "routine", "reception"),
                                       ("noise_external", "structural", "management")):
        assert (types[name]["priority"], types[name]["department"]) == (priority, department), name
        assert f"`{name}`" in protocol
