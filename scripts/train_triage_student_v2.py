"""Train the triage student v2 variants A and B once each, and write their logits (PROTOCOL.md version 1, §3).

Meant for Colab (notebook 30), where the raw CSV is fetched and checked by scripts/fetch_booking_515k.py:

    python scripts/train_triage_student_v2.py --csv data/raw/booking_reviews_515k.csv \
        --weights /content/drive/MyDrive/triage_student_v2

1. It rebuilds every text it needs from the raw CSV by `raw_row` (the 30,000 teacher reviews, the 600 development
   and the 500 test reviews) and stops unless each one matches its key's SHA-256. No review text is read from
   anywhere else.
2. It trains DistilBERT at the pinned revision twice, from the same seed, on the teacher's labels of the training
   reviews only (`held_out == 0`; a review the teacher failed on is left out), with the first protocol's version 6
   settings (`train_triage_student.SETTINGS`): no search, no early stopping, the last epoch kept.
   - A: BCE through the model's multi-label head, no class weights, exactly the first protocol's recipe;
   - B: BCE with `pos_weight` per type = min(negatives / positives in the training labels, 20).
   Variant C uses B's model and differs only in its thresholds, a later local step.
3. It writes, into --out (docs/experiments/triage_student_v2/student), each variant's logits for the development,
   test and held-out reviews (raw_row, set, one column per catalogue_v2 type) and run.json (settings, the
   positive weights, revision, commit, losses, times, hashes). With --weights it saves each variant's model and
   tokenizer there, outside the repository (the owner's Google Drive), and records their SHA-256; without it, the
   weights are not kept and run.json says so.

It reads no label but the teacher's. Thresholds, the choice and the score are later steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_triage_student as first  # noqa: E402
import triage_student_v2 as v2  # noqa: E402

sample = first.sample
OUT = v2.FOLDER / "student"
SETTINGS = first.SETTINGS  # PROTOCOL.md §3: "with the first protocol's version 6 settings"
POS_WEIGHT_CAP = 20.0  # PROTOCOL.md §3, variant B: min(negatives / positives in the training labels, 20)
VARIANTS = ("A", "B")
SCORED = ("development", "test", "held_out")
Stop = first.Stop


def positive_weights(y: np.ndarray, cap: float = POS_WEIGHT_CAP) -> np.ndarray:
    """min(negatives / positives, cap) per type; a type with no positive gets the cap."""
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    ratio = np.divide(negatives, positives, out=np.full(y.shape[1], np.inf), where=positives > 0)
    return np.minimum(ratio, cap).astype(np.float32)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(csv: Path, out: Path = OUT, samples: Path = v2.SAMPLES, labels_dir: Path = v2.LABELS,
        weights: Path | None = None, settings: dict | None = None,
        load: Callable[[list[str]], tuple[object, object]] = first.load_pretrained, device: str | None = None) -> dict:
    """Rebuild and check every text, train A and B once each, write their logits and run.json."""
    import torch

    settings = settings or SETTINGS
    if (out / "run.json").exists():
        raise Stop(f"{out / 'run.json'} exists; the variants are trained once")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    types = sample.catalogue_types(v2.CATALOGUE)
    raw = pd.read_csv(csv, usecols=["Negative_Review", "Positive_Review"]).fillna("").astype(str)
    teacher_key = pd.read_csv(samples / "teacher_key.csv")
    teacher_texts = first.rebuild_texts(raw, teacher_key)
    teacher_labels = labels_dir / "teacher_haiku_teacher.csv"
    y, answered = first.targets(pd.read_csv(teacher_labels), teacher_key, types)
    training = answered & (teacher_key["held_out"] == 0).to_numpy()
    held_out = teacher_key["held_out"].to_numpy() == 1
    key = pd.read_csv(samples / "key.csv")
    scored = {name: key[key["set"] == name] for name in ("development", "test")}
    scored["held_out"] = teacher_key[held_out].assign(set="held_out")
    texts = {name: first.rebuild_texts(raw, part) for name, part in scored.items()}
    train_texts = [text for text, keep in zip(teacher_texts, training, strict=True) if keep]
    pos_weight = positive_weights(y[training])

    out.mkdir(parents=True, exist_ok=True)
    variants = {}
    for variant in VARIANTS:
        first.seed_everything(settings["seed"])
        model, tokenizer = load(types)
        started = time.time()
        losses = first.train(model, tokenizer, train_texts, y[training], settings, device,
                             pos_weight=pos_weight if variant == "B" else None)
        seconds = time.time() - started
        counts = {}
        for name, part in scored.items():
            logits = first.predict(model, tokenizer, texts[name], settings, device)
            table = pd.DataFrame(logits, columns=types)
            table.insert(0, "set", part["set"].to_numpy())
            table.insert(0, "raw_row", part["raw_row"].astype(int).to_numpy())
            table.sort_values("raw_row").to_csv(out / f"logits_{variant}_{name}.csv", index=False,
                                                float_format="%.6f", lineterminator="\n")
            counts[name] = len(table)
        saved = None
        if weights is not None:
            folder = weights / variant
            model.save_pretrained(folder)
            tokenizer.save_pretrained(folder)
            saved = {"folder": folder.as_posix(),
                     "files": {p.name: sha256_of(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        variants[variant] = {"loss": "BCE, model head" if variant == "A" else "BCE with pos_weight",
                             "loss_per_epoch": [round(loss, 6) for loss in losses],
                             "training_seconds": round(seconds, 1), "logits": counts, "weights": saved}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    import transformers

    record = {
        "protocol": v2.PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sample.trends._sha256(v2.PROTOCOL),
        "code_commit": sample.pilot.git_commit(),
        "settings": settings, "types": types, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "trained_on": int(training.sum()), "left_out_failed": int((~answered).sum()),
        "held_out": int(held_out.sum()),
        "pos_weight_b": {t: round(float(w), 6) for t, w in zip(types, pos_weight, strict=True)},
        "pos_weight_cap": POS_WEIGHT_CAP,
        "variants": variants,
        "weights_kept": weights is not None,
        "packages": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "teacher_labels_sha256": sample.trends._sha256(teacher_labels),
        "raw_csv_sha256": sample.trends._sha256(csv),
    }
    (out / "run.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--weights", type=Path, default=None,
                        help="a folder outside the repository for both variants' weights (e.g. on Google Drive)")
    args = parser.parse_args()
    if args.weights is not None and ROOT.resolve() in [args.weights.resolve(), *args.weights.resolve().parents]:
        raise SystemExit("--weights must be outside the repository: weights are never committed")
    if sample.trends._sha256(args.csv) != sample.RAW_SHA256:
        raise SystemExit(f"{args.csv} is not the Kaggle file the protocol fixes; nothing trained")
    try:
        record = run(args.csv, args.out, weights=args.weights)
    except Stop as error:
        raise SystemExit(str(error)) from None
    for variant, result in record["variants"].items():
        print(f"{variant}: loss per epoch {result['loss_per_epoch']}, {result['training_seconds']} s, "
              f"logits {result['logits']}")
    print(f"trained on {record['trained_on']} reviews on {record['gpu'] or 'cpu'}; weights kept: "
          f"{record['weights_kept']} -> {args.out}")


if __name__ == "__main__":
    main()
