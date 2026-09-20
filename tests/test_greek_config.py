"""The config/code contract for the Greek pipeline — no network, no torch.

These tests exist because configs/greek_bert.yaml and greek/train.py had
drifted so far apart that 12 of 13 config lookups raised KeyError and the
pipeline could not run at all. A test that loads the shipped file turns that
failure mode into a red CI run instead of a crash at training time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reviewnlp.greek.config import DataConfig, GreekConfig, TrainingConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "greek_bert.yaml"


def test_shipped_config_loads():
    config = GreekConfig.from_yaml(CONFIG_PATH)
    assert config.model.name == "nlpaueb/bert-base-greek-uncased-v1"
    assert config.data.dataset_name == "DGurgurov/greek_sa"
    assert config.data.duplicate_mode == "group"
    assert config.training.early_stopping_patience == 2
    assert config.baseline.ngram_range == (1, 2)
    assert config.evaluation.output_dir == "runs/greek_bert/evaluation"


def test_every_key_in_the_shipped_file_is_consumed():
    """No silently-ignored settings: the parser rejects unknown keys, so a
    clean load proves each key reached a dataclass field."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = GreekConfig.from_yaml(CONFIG_PATH)
    for section, values in raw.items():
        parsed = getattr(config, section)
        for key in values:
            assert hasattr(parsed, key), f"{section}.{key} is not honoured by the code"


def test_unknown_key_is_rejected():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["training"]["lr"] = 1e-5  # the name the old code used
    with pytest.raises(ValueError, match="unsupported key"):
        GreekConfig.from_mapping(raw)


def test_missing_required_key_is_rejected():
    with pytest.raises(ValueError, match="missing required key"):
        GreekConfig.from_mapping({"model": {}, "data": {"dataset_name": "x"}})


def test_split_fractions_are_validated():
    with pytest.raises(ValueError, match="test_size must be in"):
        DataConfig(dataset_name="x", dataset_revision="y", test_size=0.0)
    with pytest.raises(ValueError, match="leave room for training"):
        DataConfig(dataset_name="x", dataset_revision="y", test_size=0.6, val_size=0.5)
    assert DataConfig(dataset_name="x", dataset_revision="y").train_size == pytest.approx(0.8)


def test_enum_like_fields_are_validated():
    with pytest.raises(ValueError, match="duplicate_mode must be one of"):
        DataConfig(dataset_name="x", dataset_revision="y", duplicate_mode="dedupe")
    with pytest.raises(ValueError, match="overlap_policy must be one of"):
        DataConfig(dataset_name="x", dataset_revision="y", overlap_policy="explode")


def test_best_checkpoint_requires_matching_strategies():
    """HF Trainer silently misbehaves when these disagree; refuse up front."""
    with pytest.raises(ValueError, match="load_best_model_at_end requires"):
        TrainingConfig(eval_strategy="epoch", save_strategy="steps")
    TrainingConfig(eval_strategy="steps", save_strategy="steps")  # consistent: fine
