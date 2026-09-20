"""Typed view over configs/greek_bert.yaml.

The Greek pipeline previously read the YAML key by key at the point of use,
which let the file and the code drift apart silently until a run crashed.
Parsing the whole document once into frozen dataclasses makes the contract
explicit: every supported key appears exactly once here, `from_yaml` rejects
anything it does not recognise, and `tests/test_greek_config.py` asserts the
shipped YAML round-trips. A key added to the file without code to honour it
now fails fast instead of being ignored.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DUPLICATE_MODES = ("group", "drop", "keep")
OVERLAP_POLICIES = ("raise", "warn", "ignore")


def _take(section: str, mapping: dict, cls: type) -> dict:
    """Pull exactly the keys `cls` declares; complain about the rest."""
    known = {f.name for f in fields(cls)}
    unknown = set(mapping) - known
    if unknown:
        raise ValueError(
            f"{section}: unsupported key(s) {sorted(unknown)}. "
            f"Supported: {sorted(known)}"
        )
    required = {f.name for f in fields(cls) if f.default is MISSING}
    absent = required - set(mapping)
    if absent:
        raise ValueError(f"{section}: missing required key(s) {sorted(absent)}")
    return dict(mapping)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_labels: int = 2


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    dataset_revision: str
    text_column: str = "text"
    label_column: str = "label"
    test_size: float = 0.10
    val_size: float = 0.10
    seed: int = 42
    max_length: int = 128
    overlap_policy: str = "raise"
    duplicate_mode: str = "group"

    def __post_init__(self) -> None:
        if self.duplicate_mode not in DUPLICATE_MODES:
            raise ValueError(
                f"duplicate_mode must be one of {DUPLICATE_MODES}, got {self.duplicate_mode!r}"
            )
        if self.overlap_policy not in OVERLAP_POLICIES:
            raise ValueError(
                f"overlap_policy must be one of {OVERLAP_POLICIES}, got {self.overlap_policy!r}"
            )
        for name in ("test_size", "val_size"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"test_size + val_size must leave room for training, got "
                f"{self.test_size} + {self.val_size}"
            )

    @property
    def train_size(self) -> float:
        return 1.0 - self.test_size - self.val_size


@dataclass(frozen=True)
class BaselineConfig:
    max_features: int = 100_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "runs/greek_bert"
    num_epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    seed: int = 42
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    early_stopping_patience: int = 2

    def __post_init__(self) -> None:
        if self.load_best_model_at_end and self.eval_strategy != self.save_strategy:
            raise ValueError(
                "load_best_model_at_end requires eval_strategy == save_strategy, got "
                f"{self.eval_strategy!r} and {self.save_strategy!r}"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    output_dir: str = "runs/greek_bert/evaluation"


@dataclass(frozen=True)
class GreekConfig:
    model: ModelConfig
    data: DataConfig
    baseline: BaselineConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> GreekConfig:
        sections = {
            "model": ModelConfig,
            "data": DataConfig,
            "baseline": BaselineConfig,
            "training": TrainingConfig,
            "evaluation": EvaluationConfig,
        }
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(
                f"unsupported top-level section(s) {sorted(unknown)}. "
                f"Supported: {sorted(sections)}"
            )
        built: dict[str, Any] = {}
        for name, section_cls in sections.items():
            payload = raw.get(name, {}) or {}
            if not isinstance(payload, dict):
                raise ValueError(f"{name}: expected a mapping, got {type(payload).__name__}")
            kwargs = _take(name, payload, section_cls)
            if name == "baseline" and "ngram_range" in kwargs:
                kwargs["ngram_range"] = tuple(kwargs["ngram_range"])
            built[name] = section_cls(**kwargs)
        return cls(**built)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GreekConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_mapping(raw)
