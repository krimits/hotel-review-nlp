"""DistilBERT full fine-tune CLI (baseline transformer for the benchmark).

Run:  python -m reviewnlp.llm.train_distilbert --config configs/distilbert.yaml
"""

from __future__ import annotations

import argparse

from reviewnlp.llm.encoder_trainer import run_encoder_training
from reviewnlp.utils.seed import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distilbert.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    run_encoder_training(
        model_name=cfg["model"]["name"],
        processed_dir=data_cfg.get("processed_dir", "data/processed"),
        out_dir=cfg["output"]["model_dir"],
        seed=cfg["seed"],
        max_length=cfg["model"]["max_length"],
        batch_size=cfg["train"]["batch_size"],
        eval_batch_size=cfg["train"]["eval_batch_size"],
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        fp16=cfg["train"]["fp16"],
        train_cap=data_cfg.get("train_cap"),
    )


if __name__ == "__main__":
    main()
