"""Lock test for docs/experiments/hotelier_blind_eval/PROTOCOL.md, version 1 (approved 6 November 2026).

The protocol is frozen once committed. This test pins its SHA-256 and the triage student v2 files it reads, checks
the group sizes it states against the outputs recomputed from those files, and checks that the rule's numbers and
the draw's settings are the code's. A change is a version 2 beside this one and a new pin here, never an edit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs" / "experiments" / "hotelier_blind_eval"
PROTOCOL = FOLDER / "PROTOCOL.md"
PROTOCOL_SHA256 = "6aa74caf65f0f34e7410bcafea4c893b3c5a3edbb55a460a7299336df9495cae"
V2 = ROOT / "docs" / "experiments" / "triage_student_v2"
INPUTS = {  # the v2 protocol's committed files this evaluation reads, as its lock test pins them
    V2 / "samples" / "key.csv": "14d4f955ffa0fecc482b4d2f31c48fce2d6a9db5413d6b8873b8d7b5fe42b134",
    V2 / "labels" / "teacher_haiku_test.csv": "d68eac256f9109e88d1b00b0adb5cd7c5317b700942ee1774fe7e4b47639187a",
    V2 / "student" / "logits_B_test.csv": "72e0ad3d0af6388bf6b8492110587da02a018966b94570d61c720b307ba7bc14",
    V2 / "student" / "thresholds_locked.json": "a03f45b38466f0de2501768464b687bbb69d3a0914a77ad2679d137e73520b4e",
    ROOT / "docs" / "experiments" / "triage_diagnosis" / "catalogue_v2.yaml":
        "5c592af14016cba5f97a0ee021af0a54e2196d1f50ad1f01b938be3a384f966d",
}
RULE = ["p_teacher < 0.80", "lo_d >= -0.10", "hi_d < -0.10"]
DRAWN = {  # step 2, drawn at ce4b0af: the rated set, both outputs, the manifest, and the Greek label table
    FOLDER / "samples" / "rated_key.csv": "bdc98b58b340c383cf9c6ccaf184ff3737a4cb10ed92e77a4bb71cc9561a3315",
    FOLDER / "samples" / "outputs.json": "7424b85b65de6542a37594b93b548e522e9c3195957d2cde76fc807ff3a323e4",
    FOLDER / "samples" / "manifest.json": "1a5e3560e06c865ab8783ee09c0fa16612bf54a328d1edfe7143a0ea85ade1f7",
    FOLDER / "labels_el.yaml": "d0f992b9fe9fc8251da5afd20dbb233785992a36de79361467befcb5b3698a47",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load("hotelier_blind_eval")


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(protocol) -> str:
    return " ".join(protocol.split())


def test_version_1_is_locked_and_unchanged(protocol):
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256, (
        "PROTOCOL.md changed after lock: write a version 2 beside it instead of editing version 1")
    assert protocol.startswith("# Hoteliers' blind evaluation of the triage teacher and student: protocol — "
                               "version 1, locked\n")
    for heading in ("## 1. Question", "## 2. Data", "## 3. One primary comparison", "## 4. Decision rule",
                    "## 5. Limits"):
        assert heading in protocol, f"missing section: {heading}"
    assert "draft" not in protocol.lower() and "proposed" not in protocol.lower()


def test_the_owners_fields_are_in_the_rule(prose):
    for condition in RULE:
        assert f"`{condition}`" in prose, condition
    assert "The run is decided only when **at least 2 hoteliers** are complete" in prose
    assert "approved by the owner on 6 November 2026, with a margin of 10 points, a floor of 80% and at least two" in prose
    assert "2,000 bootstrap resamples (seed 0) of the U reviews within strata" in prose


def test_the_inputs_are_the_v2_protocols_committed_files():
    for path, digest in INPUTS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path.name
    assert {Path(p).name for p in evaluation.inputs()} == {p.name for p in INPUTS}


def test_the_group_sizes_the_protocol_states_are_the_datas(prose):
    test = evaluation.test_reviews()
    group = evaluation.groups(evaluation.outputs())
    counts = {g: {s: int(((group == g) & (test["stratum"] == s)).sum()) for s in ("H", "N", "B")} for g in "DCZ"}
    assert counts["D"] == {"H": 33, "N": 16, "B": 0} and counts["C"] == {"H": 75, "N": 22, "B": 0}
    assert sum(counts["Z"].values()) == 354
    assert "| **D**, differ | the two urgent outputs differ | 49 (H 33, N 16) | **all 49** |" in prose
    assert "| **C**, same | the same urgent output, not empty | 97 (H 75, N 22) | **11** (H 8, N 3), seed **20261106** |" in prose


def test_the_drawn_set_is_the_committed_one_and_the_rule_is_the_codes():
    for path, digest in DRAWN.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path.name
    assert [condition for condition, _ in evaluation.RULE][:3] == RULE
    assert (evaluation.MARGIN, evaluation.FLOOR, evaluation.MIN_COMPLETE) == (-0.10, 0.80, 2)
    assert (evaluation.BOOTSTRAP, evaluation.BOOTSTRAP_SEED) == (2000, 0)


def test_the_draw_settings_are_the_codes(prose):
    assert evaluation.SEED == 20261106 and evaluation.C_SIZES == {"H": 8, "N": 3} and evaluation.VARIANT == "B"
    assert "`DataFrame.sample(n, random_state=20261106)`" in prose
