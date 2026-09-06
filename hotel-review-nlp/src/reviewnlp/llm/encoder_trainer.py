"""Custom PyTorch training loop for encoder sentiment models (DistilBERT).

The same loop powers full fine-tuning and the from-scratch LoRA experiment,
which keeps the data path, checkpoint selection, and evaluation identical.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.lora.lora import (
    inject_lora,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_and_unload_lora,
)
from reviewnlp.utils.experiments import frame_fingerprint
from reviewnlp.utils.seed import set_seed

_LABEL2ID = {"negative": 0, "positive": 1}
_ID2LABEL = {value: key for key, value in _LABEL2ID.items()}
_DEFAULT_HEAD_MODULES = ("pre_classifier", "classifier")


@dataclass
class LoopResult:
    test_metrics: dict
    best_dev_f1: float
    trainable_params: int
    total_params: int


class EncodedReviews(Dataset):
    """Tokenized reviews held in memory as integer tensors."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
        self.attention = [
            torch.tensor(mask, dtype=torch.long) for mask in encoded["attention_mask"]
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention[index], self.labels[index]


def pad_collate(batch):
    """Right-pad ``(input_ids, attention_mask, label)`` tuples."""
    max_length = max(len(item[0]) for item in batch)
    input_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_length, dtype=torch.long)
    labels = torch.stack([item[2] for item in batch])
    for row, (ids, mask, _label) in enumerate(batch):
        input_ids[row, : len(ids)] = ids
        attention_mask[row, : len(mask)] = mask
    return input_ids, attention_mask, labels


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for input_ids, attention_mask, _labels in loader:
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits
        chunks.append(logits.float().cpu())
    return torch.cat(chunks).numpy()


def _warmup_cosine(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def run_encoder_training(
    model_name: str,
    processed_dir: str,
    out_dir: str,
    seed: int = 42,
    max_length: int = 256,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    epochs: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    fp16: bool = True,
    lora: dict | None = None,
    train_cap: int | None = None,
) -> LoopResult:
    """Train and evaluate full-FT or from-scratch-LoRA DistilBERT."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    frames = {
        split: pd.read_parquet(os.path.join(processed_dir, f"{split}.parquet"))
        for split in ("train", "dev", "test")
    }
    if train_cap:
        frames["train"] = _stratified_train_cap(frames["train"], train_cap, seed)
        print(f"train capped to {len(frames['train']):,} rows")

    loaders = _build_loaders(frames, tokenizer, max_length, batch_size, eval_batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=_ID2LABEL,
        label2id=_LABEL2ID,
    )

    head_modules = tuple((lora or {}).get("modules_to_save", _DEFAULT_HEAD_MODULES))
    if lora:
        replaced = inject_lora(
            model,
            target_modules=lora["target_modules"],
            r=lora["r"],
            alpha=lora["alpha"],
            dropout=lora.get("dropout", 0.0),
        )
        mark_only_lora_trainable(model, modules_to_save=head_modules)
        print(
            f"from-scratch LoRA injected into {len(replaced)} modules "
            f"(r={lora['r']}); task head remains trainable"
        )

    model.to(device)
    parameter_counts = _parameter_counts(model, head_modules)
    print(
        f"params: {parameter_counts['total_params']:,} total | "
        f"{parameter_counts['trainable_params']:,} trainable "
        f"({parameter_counts['trainable_params'] / parameter_counts['total_params']:.2%})"
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    scaler = torch.amp.GradScaler(enabled=fp16 and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    best_f1, best_state = -1.0, None
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, epoch_started = 0.0, time.perf_counter()
        for input_ids, attention_mask, labels in loaders["train"]:
            learning_rate = _warmup_cosine(global_step, total_steps, warmup_steps, lr)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                ).logits
                loss = criterion(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {global_step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

        dev_logits = predict_logits(model, loaders["dev"], device)
        dev_gold = _numeric_labels(frames["dev"])
        dev_metrics = binary_metrics(
            dev_gold,
            dev_logits.argmax(axis=-1),
            label_names=("negative", "positive"),
            label_values=(0, 1),
        )
        print(
            f"epoch {epoch} | loss {running_loss / steps_per_epoch:.4f} | "
            f"dev macro-F1 {dev_metrics['macro_f1']:.4f} | "
            f"lr {learning_rate:.2e} | {time.perf_counter() - epoch_started:.0f}s"
        )
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state)

    dev_logits = predict_logits(model, loaders["dev"], device)
    test_logits = predict_logits(model, loaders["test"], device)
    dev_gold = _numeric_labels(frames["dev"])
    test_gold = _numeric_labels(frames["test"])
    test_metrics = binary_metrics(
        test_gold,
        test_logits.argmax(axis=-1),
        label_names=("negative", "positive"),
        label_values=(0, 1),
    )
    peak_cuda_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    print(f"TEST macro-F1 {test_metrics['macro_f1']:.4f} acc {test_metrics['accuracy']:.4f}")

    output = os.path.abspath(out_dir)
    os.makedirs(output, exist_ok=True)
    if lora:
        head_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if "lora_" not in name and _belongs_to_any(name, head_modules)
        }
        torch.save(
            {"lora": lora_state_dict(model), "head": head_state, "config": lora},
            os.path.join(output, "lora_scratch.pt"),
        )
        with open(
            os.path.join(output, "lora_scratch_config.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(lora, file, indent=2)
        merged = merge_and_unload_lora(model)
        print(f"merged and unloaded {len(merged)} LoRA modules for portable inference")

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    np.save(os.path.join(output, "dev_logits.npy"), dev_logits)
    np.save(os.path.join(output, "dev_labels.npy"), dev_gold)
    np.save(os.path.join(output, "test_logits.npy"), test_logits)
    np.save(os.path.join(output, "test_labels.npy"), test_gold)

    settings = {
        "model_name": model_name,
        "processed_dir": os.path.abspath(processed_dir),
        "seed": seed,
        "max_length": max_length,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "fp16": fp16,
        "lora": lora,
    }
    metrics = {
        "test": test_metrics,
        "best_dev_macro_f1": best_f1,
        **parameter_counts,
        "training_seconds": training_seconds,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "data": {split: frame_fingerprint(frame) for split, frame in frames.items()},
        "settings": settings,
    }
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return LoopResult(
        test_metrics,
        best_f1,
        parameter_counts["trainable_params"],
        parameter_counts["total_params"],
    )


def _build_loaders(frames, tokenizer, max_length, batch_size, eval_batch_size):
    datasets = {
        split: EncodedReviews(
            frame["text"].astype(str).tolist(),
            frame["label"].map(_LABEL2ID).tolist(),
            tokenizer,
            max_length,
        )
        for split, frame in frames.items()
    }
    return {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=pad_collate
        ),
        "dev": DataLoader(
            datasets["dev"], batch_size=eval_batch_size, collate_fn=pad_collate
        ),
        "test": DataLoader(
            datasets["test"], batch_size=eval_batch_size, collate_fn=pad_collate
        ),
    }


def _numeric_labels(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].map(_LABEL2ID)
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), "label"].astype(str).unique())
        raise ValueError(f"unknown labels: {unknown}")
    return labels.to_numpy(dtype=np.int64)


def _stratified_train_cap(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap >= len(frame):
        return frame
    groups = []
    for _label, group in frame.groupby("label", sort=True):
        sample_size = max(1, round(cap * len(group) / len(frame)))
        groups.append(group.sample(n=min(len(group), sample_size), random_state=seed))
    return pd.concat(groups).sort_index().head(cap).reset_index(drop=True)


def _parameter_counts(model, head_modules: tuple[str, ...]) -> dict[str, int]:
    named = list(model.named_parameters())
    return {
        "trainable_params": sum(
            parameter.numel() for _, parameter in named if parameter.requires_grad
        ),
        "adapter_params": sum(
            parameter.numel()
            for name, parameter in named
            if parameter.requires_grad and "lora_" in name
        ),
        "head_params": sum(
            parameter.numel()
            for name, parameter in named
            if parameter.requires_grad and _belongs_to_any(name, head_modules)
        ),
        "total_params": sum(parameter.numel() for _, parameter in named),
    }


def _belongs_to_any(parameter_name: str, module_names: tuple[str, ...]) -> bool:
    return any(
        parameter_name.startswith(f"{module_name}.")
        or f".{module_name}." in f".{parameter_name}."
        for module_name in module_names
    )
