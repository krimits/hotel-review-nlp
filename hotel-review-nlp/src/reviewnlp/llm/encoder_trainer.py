"""Custom PyTorch training loop for encoder sentiment models (DistilBERT).

Deliberately NOT ``transformers.Trainer``: the loop below is hand-written so
the project demonstrates real PyTorch engineering - manual tokenization into
Tensors, AdamW with hand-rolled linear warmup+decay, mixed precision via
``torch.amp``, gradient clipping, dev-based model selection.

It powers two experiments:
  * ``train_distilbert.py``       - full fine-tune of all parameters
  * ``train_distilbert_lora.py``  - freezes the encoder and injects the
    **from-scratch LoRA** from ``reviewnlp.lora`` (paper reproduction,
    comparable to the HF PEFT route used for Qwen).
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
from reviewnlp.lora.lora import inject_lora, mark_only_lora_trainable
from reviewnlp.utils.seed import set_seed

_LABEL2ID = {"negative": 0, "positive": 1}
_ID2LABEL = {v: k for k, v in _LABEL2ID.items()}


@dataclass
class LoopResult:
    test_metrics: dict
    best_dev_f1: float
    trainable_params: int
    total_params: int


class EncodedReviews(Dataset):
    """Tokenized reviews held in memory as int tensors."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in enc["input_ids"]]
        self.attention = [torch.tensor(a, dtype=torch.long) for a in enc["attention_mask"]]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention[idx], self.labels[idx]


def pad_collate(batch):
    """Right-pad (input_ids, attention_mask, labels) to batch max length."""
    max_len = max(len(b[0]) for b in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.stack([b[2] for b in batch])
    for i, (input_ids, attn, _label) in enumerate(batch):
        ids[i, : len(input_ids)] = input_ids
        mask[i, : len(attn)] = attn
    return ids, mask, labels


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
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
    """Shared engine for full-FT and from-scratch-LoRA DistilBERT runs."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    frames = {
        split: pd.read_parquet(os.path.join(processed_dir, f"{split}.parquet"))
        for split in ("train", "dev", "test")
    }
    if train_cap:
        for split in ("train",):
            frames[split] = (
                frames[split].groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), train_cap // 2), random_state=seed))
            )
            print(f"{split} capped to {len(frames[split]):,} rows")

    train_ds = EncodedReviews(
        frames["train"]["text"].tolist(),
        frames["train"]["label"].map(_LABEL2ID).tolist(),
        tokenizer,
        max_length,
    )
    dev_ds = EncodedReviews(
        frames["dev"]["text"].tolist(),
        frames["dev"]["label"].map(_LABEL2ID).tolist(),
        tokenizer,
        max_length,
    )
    test_ds = EncodedReviews(
        frames["test"]["text"].tolist(),
        frames["test"]["label"].map(_LABEL2ID).tolist(),
        tokenizer,
        max_length,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    dev_loader = DataLoader(dev_ds, batch_size=eval_batch_size, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, collate_fn=pad_collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=_ID2LABEL, label2id=_LABEL2ID
    )
    if lora:  # from-scratch LoRA (reviewnlp.lora), not peft
        replaced = inject_lora(
            model, target_modules=lora["target_modules"], r=lora["r"],
            alpha=lora["alpha"], dropout=lora.get("dropout", 0.0),
        )
        mark_only_lora_trainable(model)
        print(f"from-scratch LoRA injected into {len(replaced)} modules (r={lora['r']})")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {total_params:,} total | {trainable_params:,} trainable ({trainable_params / total_params:.2%})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(warmup_ratio * total_steps))
    scaler = torch.amp.GradScaler(enabled=fp16 and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, None
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, t0 = 0.0, time.perf_counter()
        for ids, mask, labels in train_loader:
            lr_now = _warmup_cosine(global_step, total_steps, warmup, lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = criterion(logits, labels.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1

        dev_logits = predict_logits(model, dev_loader, device)
        dev_preds = dev_logits.argmax(-1)
        dev_gold = frames["dev"]["label"].map(_LABEL2ID).values
        m = binary_metrics(dev_gold, dev_preds)
        print(
            f"epoch {epoch} | loss {running / steps_per_epoch:.4f} | "
            f"dev macro-F1 {m['macro_f1']:.4f} | lr {lr_now:.2e} | {time.perf_counter() - t0:.0f}s"
        )
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_logits = predict_logits(model, test_loader, device)
    test_preds = test_logits.argmax(-1)
    test_gold = frames["test"]["label"].map(_LABEL2ID).values
    test_metrics = binary_metrics(test_gold, test_preds)
    print(f"TEST macro-F1 {test_metrics['macro_f1']:.4f} acc {test_metrics['accuracy']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    if lora:
        from reviewnlp.lora.lora import lora_state_dict

        torch.save(lora_state_dict(model), os.path.join(out_dir, "lora_scratch.pt"))
        model.save_pretrained(out_dir)  # base + wrapped modules, reloadable
    else:
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    np.save(os.path.join(out_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(out_dir, "test_labels.npy"), test_gold)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"test": test_metrics, "best_dev_macro_f1": best_f1,
                   "trainable_params": trainable_params, "total_params": total_params}, f, indent=2)
    return LoopResult(test_metrics, best_f1, trainable_params, total_params)
