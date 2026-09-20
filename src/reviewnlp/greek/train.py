"""Training loop: GreekBERT fine-tune on the config-defined splits.

transformers.Trainer with validation-split checkpoint selection — the same
selection discipline as the English pipeline. Artifacts follow the
unified-benchmark convention: test_logits.npy / test_labels.npy saved in
original test order, so this family can join reviewnlp.evaluation.benchmark
without format changes. The classical baseline is trained alongside on the
identical frames, so metrics.json always carries a floor to compare against.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from reviewnlp.greek.baseline import train_baseline
from reviewnlp.greek.config import GreekConfig, TrainingConfig
from reviewnlp.greek.data import dataset_manifest, load_greek_splits
from reviewnlp.greek.model import build_model, build_tokenizer


def _score(predictions: np.ndarray, labels: np.ndarray) -> dict:
    predictions = np.asarray(predictions).argmax(axis=-1)
    labels = np.asarray(labels)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "macro_f1": float(macro_f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def compute_metrics(eval_pred) -> dict:
    return _score(eval_pred.predictions, eval_pred.label_ids)


def build_training_arguments(config: TrainingConfig, output_dir: Path):
    """Translate the `training:` section into TrainingArguments.

    Split out so the mapping is testable without downloading an encoder.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        logging_steps=config.logging_steps,
        seed=config.seed,
        data_seed=config.seed,
        fp16=config.fp16,
        report_to=[],
    )


def train_from_config(config_path: str) -> dict:
    """Run the full Greek pipeline: baseline + encoder, artifacts on disk."""
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        set_seed,
    )

    config = GreekConfig.from_yaml(config_path)
    set_seed(config.training.seed)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_greek_splits(config.data)
    manifest = dataset_manifest(frames, config.data)

    baseline_metrics = train_baseline(frames, config.baseline)

    tokenizer = build_tokenizer(config.model)
    model = build_model(config.model)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"], truncation=True, max_length=config.data.max_length
        )

    def as_dataset(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(
            frame[["text", "label"]], preserve_index=False
        ).map(tokenize, batched=True, remove_columns=["text"])

    train_tok = as_dataset(frames["train"])
    dev_tok = as_dataset(frames["validation"])
    test_frame = frames["test"][["text", "label"]].copy()

    callbacks = []
    if config.training.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=build_training_arguments(config.training, output_dir),
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - started

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    test_output = trainer.predict(as_dataset(test_frame))
    test_logits = np.asarray(test_output.predictions, dtype=np.float32)
    test_labels = np.asarray(test_frame["label"], dtype=np.int64)
    np.save(output_dir / "test_logits.npy", test_logits)
    np.save(output_dir / "test_labels.npy", test_labels)
    test_scores = _score(test_logits, test_labels)

    dev_scores = [
        float(entry["eval_macro_f1"])
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]
    best_validation_macro_f1 = (
        float(trainer.state.best_metric)
        if trainer.state.best_metric is not None
        else (max(dev_scores) if dev_scores else None)
    )

    metrics = {
        "schema_version": 2,
        "model_family": f"{config.model.name} sequence classification (Greek)",
        "base_model": config.model.name,
        "data_manifest": manifest,
        "best_validation_macro_f1": best_validation_macro_f1,
        "epochs_run": len(dev_scores),
        "test": {
            "macro_f1": test_scores["macro_f1"],
            "accuracy": test_scores["accuracy"],
            "rows": int(len(test_labels)),
        },
        "baseline": baseline_metrics,
        "encoder_minus_baseline_macro_f1_pp": round(
            100 * (test_scores["macro_f1"] - baseline_metrics["test"]["macro_f1"]), 4
        ),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": round(training_seconds, 3),
        "settings": {
            "epochs": config.training.num_epochs,
            "lr": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "max_length": config.data.max_length,
            "seed": config.training.seed,
            "dataset_revision": config.data.dataset_revision,
            "fp16": config.training.fp16,
            "early_stopping_patience": config.training.early_stopping_patience,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
