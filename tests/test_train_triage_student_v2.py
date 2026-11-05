"""The triage student v2 training run (variants A and B) with a tiny local DistilBERT: no download, CPU only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


student = _load("train_triage_student_v2")
first, sample = student.first, student.sample
TYPES = sample.catalogue_types(student.v2.CATALOGUE)
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
    """A raw CSV of 60 reviews; 30 teacher reviews (6 held out, one failed), 6 development and 6 test reviews."""
    rows = [{"Negative_Review": f" {WORDS[i % 6]} {WORDS[(i + 3) % 12]} " if i % 5 else "No Negative",
             "Positive_Review": f"{WORDS[(i + 4) % 12]} staff" if i % 3 else "No Positive"} for i in range(60)]
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
    key(list(range(30)), "teacher").assign(held_out=[int(r % 5 == 0) for r in range(30)]).to_csv(
        samples / "teacher_key.csv", index=False)
    key(list(range(40, 52)), ["development", "test"] * 6).to_csv(samples / "key.csv", index=False)
    teacher = pd.DataFrame({"raw_row": range(30), "set": "teacher", **{t: 0 for t in TYPES}, "none": 0, "failed": 0})
    teacher.loc[teacher["raw_row"] % 4 == 0, "cleanliness"] = 1
    teacher.loc[teacher["raw_row"] % 7 == 0, "safety_fault"] = 1
    teacher["none"] = (teacher[TYPES].sum(axis=1) == 0).astype(int)
    teacher.loc[29, ["none", "failed"]] = [0, 1]
    teacher.to_csv(labels / "teacher_haiku_teacher.csv", index=False)
    return csv, samples, labels


def test_the_settings_and_the_cap_are_the_protocols():
    prose = " ".join((ROOT / "docs/experiments/triage_student_v2/PROTOCOL.md").read_text(encoding="utf-8").split())
    assert student.SETTINGS is first.SETTINGS
    s = student.SETTINGS
    assert (s["revision"], s["max_length"], s["epochs"], s["batch_size"], s["lr"], s["warmup_ratio"],
            s["weight_decay"], s["fp16"], s["seed"]) == ("12040accade4e8a0f71eabdb258fecc2e7e948be", 256, 4, 32,
                                                         2e-5, 0.06, 0.01, True, 42)
    assert "`pos_weight` per type = min(negatives / positives in the training labels, 20)" in prose
    assert student.POS_WEIGHT_CAP == 20 and student.VARIANTS == ("A", "B")


def test_positive_weights_are_capped_and_a_type_without_positives_gets_the_cap():
    y = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    assert student.positive_weights(y).tolist() == [3.0, 20.0, 1.0]
    assert student.positive_weights(y, cap=2.0).tolist() == [2.0, 2.0, 1.0]


def test_a_and_b_train_once_on_the_training_reviews_and_write_their_logits(layout, tmp_path):
    csv, samples, labels = layout
    out, weights = tmp_path / "student", tmp_path / "drive"
    record = student.run(csv, out, samples, labels, weights, SMALL, tiny(tmp_path), "cpu")
    assert record["trained_on"] == 23 and record["held_out"] == 6 and record["left_out_failed"] == 1
    assert record["pos_weight_b"]["cleanliness"] == pytest.approx(17 / 6)  # 6 of the 23 training reviews
    assert record["pos_weight_b"]["wifi"] == 20.0 and record["weights_kept"]
    for variant in ("A", "B"):
        assert record["variants"][variant]["logits"] == {"development": 6, "test": 6, "held_out": 6}
        files = record["variants"][variant]["weights"]["files"]
        assert "config.json" in files and any(name.endswith(".safetensors") for name in files)
        for name in ("development", "test", "held_out"):
            table = pd.read_csv(out / f"logits_{variant}_{name}.csv")
            assert list(table.columns) == ["raw_row", "set", *TYPES] and (table["set"] == name).all()
    held = pd.read_csv(out / "logits_A_held_out.csv")
    assert held["raw_row"].tolist() == [0, 5, 10, 15, 20, 25]
    a, b = (pd.read_csv(out / f"logits_{v}_development.csv")[TYPES].to_numpy() for v in ("A", "B"))
    assert not np.allclose(a, b)
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["variants"]["A"]["loss"] == "BCE, model head"
    with pytest.raises(student.Stop, match="trained once"):
        student.run(csv, out, samples, labels, None, SMALL, tiny(tmp_path), "cpu")


def test_variant_a_is_the_first_protocols_recipe(layout, tmp_path):
    csv, samples, labels = layout
    out = tmp_path / "student"
    student.run(csv, out, samples, labels, None, SMALL, tiny(tmp_path), "cpu")
    raw = pd.read_csv(csv, usecols=["Negative_Review", "Positive_Review"]).fillna("").astype(str)
    teacher_key = pd.read_csv(samples / "teacher_key.csv")
    y, answered = first.targets(pd.read_csv(labels / "teacher_haiku_teacher.csv"), teacher_key, TYPES)
    keep = answered & (teacher_key["held_out"] == 0).to_numpy()
    texts = [t for t, k in zip(first.rebuild_texts(raw, teacher_key), keep, strict=True) if k]
    first.seed_everything(SMALL["seed"])
    model, tokenizer = tiny(tmp_path)(TYPES)
    first.train(model, tokenizer, texts, y[keep], SMALL, "cpu")
    key = pd.read_csv(samples / "key.csv")
    development = key[key["set"] == "development"]
    expected = first.predict(model, tokenizer, first.rebuild_texts(raw, development), SMALL, "cpu")
    order = np.argsort(development["raw_row"].to_numpy())
    written = pd.read_csv(out / "logits_A_development.csv")[TYPES].to_numpy()
    assert np.allclose(written, expected[order], atol=1e-5)


def test_weights_are_never_written_inside_the_repository(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--weights", str(ROOT / "runs" / "weights")])
    with pytest.raises(SystemExit, match="outside the repository"):
        student.main()
