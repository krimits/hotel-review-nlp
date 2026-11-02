"""The student's training run (PROTOCOL.md version 6) with a tiny local DistilBERT: no download, CPU only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


student = _load("train_triage_student")
sample = student.sample
TYPES = sample.catalogue_types()
WORDS = ["dirty", "room", "noisy", "street", "great", "staff", "broken", "ac", "breakfast", "cold", "nice", "bed"]
SMALL = {**student.SETTINGS, "epochs": 2, "batch_size": 4, "eval_batch_size": 8, "max_length": 32}


def tiny(tmp_path: Path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "disliked", "liked", ":", *WORDS]) + "\n",
                     encoding="utf-8")

    def load(types):
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
        config = DistilBertConfig(vocab_size=len(tokenizer.vocab), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
                                  max_position_embeddings=64, num_labels=len(types),
                                  problem_type="multi_label_classification")
        return DistilBertForSequenceClassification(config), tokenizer
    return load


@pytest.fixture
def layout(tmp_path):
    """A raw CSV of 60 reviews; 24 teacher reviews (one failed), 6 development, 6 test and 4 check reviews."""
    rows = []
    for i in range(60):
        rows.append({"Negative_Review": f" {WORDS[i % 6]} {WORDS[(i + 3) % 12]} " if i % 5 else "No Negative",
                     "Positive_Review": f"{WORDS[(i + 4) % 12]} staff" if i % 3 else "No Positive"})
    raw = pd.DataFrame(rows)
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, index=False)
    samples, labels = tmp_path / "samples", tmp_path / "labels"
    samples.mkdir()
    labels.mkdir()

    def key(raw_rows, sets):
        texts = [sample.whole_review(raw.loc[r, "Negative_Review"], raw.loc[r, "Positive_Review"]) for r in raw_rows]
        return pd.DataFrame({"item": range(1, len(raw_rows) + 1), "set": sets, "raw_row": raw_rows, "stratum": "N",
                             "weight": 1.0, "pool": 0, "text_sha256": [sample.text_hash(t) for t in texts]})
    key(list(range(24)), "teacher").to_csv(samples / "teacher_key.csv", index=False)
    key(list(range(30, 42)), ["development", "test"] * 6).to_csv(samples / "key.csv", index=False)
    key(list(range(50, 54)), "check").to_csv(samples / "check_key.csv", index=False)
    teacher = pd.DataFrame({"raw_row": range(24), "set": "teacher", **{t: 0 for t in TYPES}, "none": 0, "failed": 0})
    teacher.loc[teacher["raw_row"] % 6 == 0, "cleanliness"] = 1
    teacher.loc[teacher["raw_row"] % 6 != 0, "none"] = 1
    teacher.loc[23, ["none", "failed"]] = [0, 1]
    teacher.to_csv(labels / "teacher_haiku_teacher.csv", index=False)
    return csv, samples, labels


def test_the_settings_are_version_6s():
    prose = " ".join((ROOT / "docs/experiments/triage_diagnosis/PROTOCOL.md").read_text(encoding="utf-8").split())
    assert ("**Settings, fixed here with no search:** 4 epochs, batch 32, learning rate 2e-5 with linear decay after "
            "a 6% warmup, weight decay 0.01, fp16 on a GPU, seed 42.") in prose
    s = student.SETTINGS
    assert (s["epochs"], s["batch_size"], s["lr"], s["warmup_ratio"], s["weight_decay"], s["fp16"], s["seed"]) == (
        4, 32, 2e-5, 0.06, 0.01, True, 42)
    assert s["max_length"] == 256 and "truncated at 256 tokens" in prose
    assert f"`{s['base']}` at revision `{s['revision']}`" in prose


def test_a_text_that_does_not_match_its_key_stops_everything(layout):
    csv, samples, _ = layout
    raw = pd.read_csv(csv).fillna("")
    raw.loc[3, "Negative_Review"] = "something else"
    with pytest.raises(student.Stop, match="differ from the key's hash"):
        student.rebuild_texts(raw, pd.read_csv(samples / "teacher_key.csv"))


def test_the_targets_follow_the_key_and_leave_failed_reviews_out(layout):
    _, samples, labels = layout
    key = pd.read_csv(samples / "teacher_key.csv")
    teacher = pd.read_csv(labels / "teacher_haiku_teacher.csv")
    y, keep = student.targets(teacher.sample(frac=1, random_state=0), key, TYPES)
    assert y.shape == (24, 20) and keep.sum() == 23 and not keep[23]
    assert y[:, TYPES.index("cleanliness")].tolist() == [float(r % 6 == 0) for r in range(24)]
    with pytest.raises(student.Stop, match="not for the teacher sample"):
        student.targets(teacher.iloc[1:], key, TYPES)


def test_one_run_writes_logits_for_the_three_sets_and_a_record_and_runs_once(layout, tmp_path):
    csv, samples, labels = layout
    out = tmp_path / "student"
    record = student.run(csv, out, samples, labels, SMALL, tiny(tmp_path), device="cpu")
    for name, rows in (("development", 6), ("test", 6), ("check", 4)):
        table = pd.read_csv(out / f"logits_{name}.csv")
        assert list(table.columns) == ["raw_row", "set", *TYPES] and len(table) == rows
        assert table["raw_row"].is_monotonic_increasing
    assert record["trained_on"] == 23 and record["left_out_failed"] == 1
    assert len(record["loss_per_epoch"]) == 2 and all(loss > 0 for loss in record["loss_per_epoch"])
    assert record["logits"] == {"development": 6, "test": 6, "check": 4}
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["settings"]["epochs"] == 2
    assert not list(out.glob("*.safetensors")) and not list(out.glob("*.bin"))
    with pytest.raises(student.Stop, match="trained once"):
        student.run(csv, out, samples, labels, SMALL, tiny(tmp_path), device="cpu")


def test_the_same_seed_gives_the_same_logits(layout, tmp_path):
    csv, samples, labels = layout
    first, second = tmp_path / "a", tmp_path / "b"
    student.run(csv, first, samples, labels, SMALL, tiny(tmp_path), device="cpu")
    student.run(csv, second, samples, labels, SMALL, tiny(tmp_path), device="cpu")
    for name in ("development", "test", "check"):
        assert (first / f"logits_{name}.csv").read_bytes() == (second / f"logits_{name}.csv").read_bytes()
