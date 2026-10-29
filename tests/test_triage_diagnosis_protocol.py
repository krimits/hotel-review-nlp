"""Lock test for docs/experiments/triage_diagnosis/PROTOCOL.md, version 3.

The protocol is frozen once committed. This test fails if its sections, its data fingerprint, the files it
pins (catalogue, planning scripts, drawn samples, the reference prompt and batches), its models, seeds and
sample sizes, its decision rule or its version history drift from what the repository records.

Version 1 compared its trigger-word list with the stratifier's regex; version 2 cites the stratifier instead of
repeating the list, and the stratifier is pinned by hash below.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs" / "experiments" / "triage_diagnosis"
PROTOCOL = FOLDER / "PROTOCOL.md"
RAW_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
FIRST_CHATGPT_SHEET = "6a9fa2bb32cf44cd91d5e182fbc8287cc7603c0278f51e53a2649568543a0c6a"
PINNED = {  # file -> SHA-256 at lock
    "catalogue_v1.yaml": "2575996133afaa6e0178060df3bd5288d01ce9b7d842a6b3bf67d362d29445e9",
    "planning/population.py": "73813486e0289d70e883f0180e0c63d92d8a21adfbd8c3f32502ba00e2404950",
    "planning/power.py": "979d02f4a0007c5ebd858e34ef37dcbb35c1fca839ec59fe26f6dacadeb46ec4",
    "reference_prompt.md": "1147d9a9278717394d060d24c0e6618712165b7449e39f965f3a79048d7a6064",
    "samples/key.csv": "7b5e04415cee8db324214f4d45bebec8a56d1e3a480f68d93c2ff25519a815bb",
    "samples/manifest.json": "ea6468bc4dea6011c2afa6a9ce650ad5b318ef86f88ee85e5efabec694b4f804",
    "samples/reference_batches.json": "b84c6f93ae0289bb18fb0cb84b4b202490e52d5f4c82cca50266d363acb8e499",
    "samples/check_key.csv": "2ca63686b36cce36534d57648fc789a0d4b8f72ed42cef0c9cefe0e9f25e39da",
    "samples/check_manifest.json": "957761d794416074c504aea1c5bd05fbf822f2131e4a59a05d49e117cce1395c",
    "samples/second_rater_batches.json": "a0ad9de8516e92de950aabc44a08c13bbb3f7dafe81eb200b93a8735b31aaf03",
}
IN_TEXT = ("catalogue_v1.yaml", "reference_prompt.md")  # pinned files whose hash the protocol itself states
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


def test_the_five_required_sections_are_present_and_version_3_is_locked(protocol):
    for heading in ("## 1. Question", "## 2. Data", "## 3. One primary comparison",
                    "## 4. Decision rule", "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"
    assert protocol.startswith("# Triage diagnosis and routing: protocol — version 3, locked\n")
    assert "draft" not in protocol.lower() and "open field" not in protocol.lower()
    assert not (FOLDER / "PROTOCOL_v2_draft.md").exists()


def test_the_question_is_agreement_with_the_reference_llm(prose):
    assert "does it catch at least 90% of the urgent problems that a reference LLM (ChatGPT, OpenAI) labels?" in prose
    assert "agreement with one OpenAI model's labels, made through ChatGPT. Not agreement with the owner" in prose
    assert "against `reference_chatgpt`" in prose


def test_the_raw_data_and_the_pinned_files_match_their_fingerprints(prose):
    assert RAW_SHA256 in prose and "238,154,765 bytes" in prose
    for name, digest in PINNED.items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == digest, name
    for name in IN_TEXT:
        assert PINNED[name] in prose, name


def test_the_reference_prompt_is_the_catalogues():
    spec = importlib.util.spec_from_file_location("triage_diagnosis_labels",
                                                  ROOT / "scripts" / "triage_diagnosis_labels.py")
    labels = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = labels
    spec.loader.exec_module(labels)
    assert (FOLDER / "reference_prompt.md").read_text(encoding="utf-8") == labels.reference_prompt()


def test_models_seeds_and_sample_sizes_are_fixed(protocol, prose):
    fields = {cells[0]: cells[1] for cells in table_rows(section(protocol, "## 9. Fields", "## 10."))}
    assert fields["Teacher"] == "anthropic/claude-sonnet-5.5-20260928"
    assert fields["Second rater"].startswith("Gemini (Google) web app, model \"gemini-4-argon\"")
    assert "**18 batches of 50**" in prose and "`check_batch_01`–`02`, the 100 check reviews" in prose
    assert fields["Reference labels"].startswith("ChatGPT (OpenAI)")
    # Three providers: the student is never scored against its teacher's family, nor the reference against its own.
    assert len({"anthropic", "google", "openai"}) == 3
    assert "`distilbert/distilbert-base-uncased` at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`" in prose
    assert "Development:** 300 reviews, H 90, N 180, B 30, seed 20261021" in prose and 90 + 180 + 30 == 300
    assert "Test:** 500 reviews, H 150, N 300, B 50, seed 20261022" in prose and 150 + 300 + 50 == 500
    assert "100 reviews from the test pool, none of them among the 500 test reviews: H 30, N 60, B 10" in prose
    assert "**seed 20261024**" in prose and 30 + 60 + 10 == 100
    assert "**16 batches of 50**" in prose and 16 * 50 == 800
    assert "2,000 bootstrap resamples, seed 0" in prose


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


def test_the_catalogue_is_the_approved_one():
    import yaml

    types = {entry["id"]: entry for entry in yaml.safe_load(
        (FOLDER / "catalogue_v1.yaml").read_text(encoding="utf-8"))["types"]}
    assert len(types) == 20
    for name, priority, department in (("plumbing_major", "urgent", "maintenance"),
                                       ("plumbing_minor", "routine", "maintenance"),
                                       ("noise_internal", "routine", "reception"),
                                       ("noise_external", "structural", "management")):
        assert (types[name]["priority"], types[name]["department"]) == (priority, department), name


def test_the_version_history_records_what_happened_under_versions_1_and_2(protocol, prose):
    history = protocol.split("## 10. Version history", 1)[1]
    rows = {cells[0]: cells for cells in table_rows(history)}
    assert set(rows) == {"1", "2", "3"}
    assert (rows["1"][1], rows["2"][1], rows["3"][1]) == ("6a6dc63", "8d35b8b", "this commit")
    assert "c2780fe" in rows["3"][2] and "44812dc" in rows["3"][2] and "no second-rater output existed" in rows["3"][2]
    assert "No model was trained, no threshold set and nothing scored under version 1." in rows["1"][2]
    for commit in ("a61b51f", "02092f4", "3a5d7f6", "8d0c8bf"):
        assert commit in rows["1"][2], commit
    assert FIRST_CHATGPT_SHEET in prose and "**The first ChatGPT sheet is not used.**" in prose
