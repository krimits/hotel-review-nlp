"""Train the student of the triage diagnosis protocol once, and write its logits (PROTOCOL.md version 6).

Meant for Colab (notebook 29), where the raw CSV is fetched and checked by scripts/fetch_booking_515k.py:

    python scripts/train_triage_student.py --csv data/raw/booking_reviews_515k.csv

1. It rebuilds every text it needs from the raw CSV by `raw_row` (the 10,000 teacher reviews, the 300
   development and 500 test reviews, the 100 check reviews) and stops unless each one matches its key's SHA-256.
   No review text is read from anywhere else.
2. It trains DistilBERT at the pinned revision as a multi-label classifier on the teacher's labels only, with the
   settings version 6 fixes (SETTINGS below): no search, no early stopping, the last epoch kept.
3. It writes, into --out (docs/experiments/triage_diagnosis/student), the logits of the development, test and
   check reviews (raw_row, set, one column per catalogue type) and run.json (settings, revision, commit, losses,
   times). The weights are not written there.

It reads no label but the teacher's. Thresholds and scores are later steps.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import triage_diagnosis_sample as sample  # noqa: E402

FOLDER = sample.FOLDER
OUT = FOLDER / "student"
# PROTOCOL.md version 6, "The student".
SETTINGS = {
    "base": "distilbert/distilbert-base-uncased", "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
    "max_length": 256, "epochs": 4, "batch_size": 32, "lr": 2e-5, "warmup_ratio": 0.06, "weight_decay": 0.01,
    "schedule": "linear", "fp16": True, "seed": 42, "eval_batch_size": 64,
}
SUBSETS = (("development", "key.csv", "development"), ("test", "key.csv", "test"), ("check", "check_key.csv", None))


class Stop(RuntimeError):
    """The run cannot go on; nothing is written."""


def rebuild_texts(raw: pd.DataFrame, key: pd.DataFrame) -> list[str]:
    """The whole-review text of each key row, rebuilt from the raw CSV's fields and checked against its hash."""
    rows = raw.loc[key["raw_row"].astype(int)]
    texts = [sample.whole_review(neg, pos) for neg, pos in zip(rows["Negative_Review"], rows["Positive_Review"],
                                                                 strict=True)]
    wrong = [int(r) for r, t, h in zip(key["raw_row"], texts, key["text_sha256"], strict=True)
             if sample.text_hash(t) != h]
    if wrong:
        raise Stop(f"{len(wrong)} rebuilt text(s) differ from the key's hash, e.g. raw_row {wrong[:5]}; nothing trained")
    return texts


def targets(labels: pd.DataFrame, key: pd.DataFrame, types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The teacher's 0/1 labels in the key's order, and a mask of the reviews kept (failed ones are left out)."""
    table = labels.set_index("raw_row")
    if set(table.index) != set(key["raw_row"].astype(int)):
        raise Stop("the teacher's labels are not for the teacher sample's reviews")
    table = table.loc[key["raw_row"].astype(int)]
    return table[types].to_numpy(dtype=np.float32), (table["failed"] == 0).to_numpy()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, tokenizer, texts: list[str], y: np.ndarray, settings: dict, device: str,
          pos_weight: np.ndarray | None = None) -> list[float]:
    """The fixed recipe: AdamW, linear decay after warmup, BCE through the model's multi-label head; mean loss per
    epoch. Batches are reshuffled each epoch from the seed.

    With `pos_weight` (one weight per type), the loss is BCE with those positive weights on the model's logits
    instead: the triage student v2 protocol's variant B. Without it nothing changes."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    model.to(device)
    model.train()
    steps = settings["epochs"] * -(-len(texts) // settings["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    schedule = get_linear_schedule_with_warmup(optimizer, int(settings["warmup_ratio"] * steps), steps)
    fp16 = settings["fp16"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    order = torch.Generator().manual_seed(settings["seed"])
    targets_all = torch.tensor(y)
    weighted = None if pos_weight is None else torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    losses = []
    for _ in range(settings["epochs"]):
        total, batches = 0.0, 0
        for index in torch.randperm(len(texts), generator=order).split(settings["batch_size"]):
            batch = tokenizer([texts[i] for i in index.tolist()], truncation=True, max_length=settings["max_length"],
                              padding=True, return_token_type_ids=False, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                if weighted is None:
                    loss = model(**batch, labels=targets_all[index].to(device)).loss
                else:
                    loss = weighted(model(**batch).logits.float(), targets_all[index].to(device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total, batches = total + loss.item(), batches + 1
        losses.append(total / batches)
    return losses


def predict(model, tokenizer, texts: list[str], settings: dict, device: str) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), settings["eval_batch_size"]):
            batch = tokenizer(texts[start:start + settings["eval_batch_size"]], truncation=True,
                              max_length=settings["max_length"], padding=True, return_token_type_ids=False, return_tensors="pt").to(device)
            out.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config.num_labels), dtype=np.float32)


def load_pretrained(types: list[str]) -> tuple[object, object]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SETTINGS["base"], revision=SETTINGS["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        SETTINGS["base"], revision=SETTINGS["revision"], num_labels=len(types),
        problem_type="multi_label_classification", id2label=dict(enumerate(types)),
        label2id={t: i for i, t in enumerate(types)})
    return model, tokenizer


def run(csv: Path, out: Path = OUT, samples: Path = FOLDER / "samples", labels_dir: Path = FOLDER / "labels",
        settings: dict | None = None, load: Callable[[list[str]], tuple[object, object]] = load_pretrained,
        device: str | None = None) -> dict:
    """Rebuild and check every text, train once, write the logits and run.json."""
    import torch

    settings = settings or SETTINGS
    if (out / "run.json").exists():
        raise Stop(f"{out / 'run.json'} exists; the student is trained once")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    types = sample.catalogue_types()
    raw = pd.read_csv(csv, usecols=["Negative_Review", "Positive_Review"]).fillna("").astype(str)
    teacher_key = pd.read_csv(samples / "teacher_key.csv")
    texts = rebuild_texts(raw, teacher_key)
    y, keep = targets(pd.read_csv(labels_dir / "teacher_haiku_teacher.csv"), teacher_key, types)
    scored = {}
    for name, key_name, part in SUBSETS:
        key = pd.read_csv(samples / key_name)
        key = key[key["set"] == part] if part else key
        scored[name] = (key, rebuild_texts(raw, key))
    seed_everything(settings["seed"])
    model, tokenizer = load(types)
    started = time.time()
    losses = train(model, tokenizer, [t for t, k in zip(texts, keep, strict=True) if k], y[keep], settings, device)
    trained = time.time() - started
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, (key, subset_texts) in scored.items():
        logits = predict(model, tokenizer, subset_texts, settings, device)
        table = pd.DataFrame(logits, columns=types)
        table.insert(0, "set", key["set"].to_numpy())
        table.insert(0, "raw_row", key["raw_row"].astype(int).to_numpy())
        table.sort_values("raw_row").to_csv(out / f"logits_{name}.csv", index=False, float_format="%.6f",
                                            lineterminator="\n")
        counts[name] = len(table)
    import transformers

    record = {
        "protocol": sample.PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sample.trends._sha256(sample.PROTOCOL),
        "code_commit": sample.pilot.git_commit(),
        "settings": settings, "types": types, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "trained_on": int(keep.sum()), "left_out_failed": int((~keep).sum()),
        "loss_per_epoch": [round(loss, 6) for loss in losses], "training_seconds": round(trained, 1),
        "logits": counts,
        "packages": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "teacher_labels_sha256": sample.trends._sha256(labels_dir / "teacher_haiku_teacher.csv"),
        "raw_csv_sha256": sample.trends._sha256(csv),
    }
    (out / "run.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if sample.trends._sha256(args.csv) != sample.RAW_SHA256:
        raise SystemExit(f"{args.csv} is not the Kaggle file the protocol fixes; nothing trained")
    try:
        record = run(args.csv, args.out)
    except Stop as error:
        raise SystemExit(str(error)) from None
    print(f"trained on {record['trained_on']} reviews in {record['training_seconds']} s on {record['gpu'] or 'cpu'};"
          f" loss per epoch {record['loss_per_epoch']}; logits {record['logits']} -> {args.out}")


if __name__ == "__main__":
    main()
