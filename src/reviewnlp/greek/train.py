"""Training loop: GreekBERT fine-tune on the pinned greek_sa splits.

transformers.Trainer with validation-split checkpoint selection — the same
selection discipline as the English pipeline. Artifacts follow the
unified-benchmark convention: test_logits.npy / test_labels.npy saved in
original test order, so this family can join reviewnlp.evaluation.benchmark
without format changes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

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


def train_from_config(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    set_seed(seed)

    model_name = cfg["model"]["name"]
    max_length = int(cfg["model"]["max_length"])
    output_dir = Path(cfg["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_greek_splits(
        dataset_id=cfg["data"]["dataset_id"],
        revision=cfg["data"]["revision"],
    )
    manifest = dataset_manifest(frames, cfg["data"]["dataset_id"], cfg["data"]["revision"])

    tokenizer = build_tokenizer(model_name)
    model = build_model(model_name)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_tok = (
        Dataset.from_pandas(frames["train"][["text", "label"]], preserve_index=False)
        .map(tokenize, batched=True, remove_columns=["text"])
    )
    dev_tok = (
        Dataset.from_pandas(frames["validation"][["text", "label"]], preserve_index=False)
        .map(tokenize, batched=True, remove_columns=["text"])
    )
    test_frame = frames["test"][["text", "label"]].copy()

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=int(cfg["train"]["epochs"]),
        learning_rate=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        per_device_train_batch_size=int(cfg["train"]["batch_size"]),
        per_device_eval_batch_size=int(cfg["train"]["eval_batch_size"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        seed=seed,
        data_seed=seed,
        fp16=bool(cfg["train"].get("fp16", False)),
        report_to=[],
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    started = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - started

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    test_tok = (
        Dataset.from_pandas(test_frame[["text", "label"]], preserve_index=False)
        .map(tokenize, batched=True, remove_columns=["text"])
    )
    test_output = trainer.predict(test_tok)
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
        else max(dev_scores)
    )
    metrics = {
        "schema_version": 1,
        "model_family": f"{model_name} sequence classification (Greek)",
        "base_model": model_name,
        "data_manifest": manifest,
        "best_validation_macro_f1": best_validation_macro_f1,
        "test": {
            "macro_f1": test_scores["macro_f1"],
            "accuracy": test_scores["accuracy"],
            "rows": int(len(test_labels)),
        },
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": round(training_seconds, 3),
        "settings": {
            "epochs": cfg["train"]["epochs"],
            "lr": cfg["train"]["lr"],
            "batch_size": cfg["train"]["batch_size"],
            "max_length": max_length,
            "seed": seed,
            "dataset_revision": cfg["data"]["revision"],
            "fp16": bool(cfg["train"].get("fp16", False)),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics