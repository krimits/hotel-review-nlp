"""DistilBERT + from-scratch LoRA (reviewnlp.lora) - the paper reproduction.

Same encoder, same loop as the full fine-tune, but only the low-rank
attention adapters are trainable. Comparing this run against
``train_distilbert.py`` quantifies the LoRA paper's core claim on our task:
comparable quality at a fraction of the trainable parameters.

Run:  python -m reviewnlp.llm.train_distilbert_lora --config configs/distilbert.yaml \
        --r 8 --alpha 16
"""

from __future__ import annotations

import argparse

from reviewnlp.llm.encoder_trainer import run_encoder_training
from reviewnlp.utils.seed import load_config

# DistilBERT attention projection names (HF naming, cf. q_proj/k_proj on LLMs)
DISTILBERT_TARGETS = ["q_lin", "v_lin"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()
    cfg = load_config(args.config)

    run_encoder_training(
        model_name=cfg["model"]["name"],
        processed_dir="data/processed",
        out_dir=f"{cfg['output']['model_dir']}_lora_scratch",
        seed=cfg["seed"],
        max_length=cfg["model"]["max_length"],
        batch_size=cfg["train"]["batch_size"],
        eval_batch_size=cfg["train"]["eval_batch_size"],
        epochs=cfg["train"]["epochs"],
        lr=1e-4,  # adapters tolerate ~10x higher LR (LoRA paper, section 5.2)
        weight_decay=0.01,
        warmup_ratio=cfg["train"]["warmup_ratio"],
        fp16=cfg["train"]["fp16"],
        lora={"r": args.r, "alpha": args.alpha, "dropout": args.dropout,
              "target_modules": DISTILBERT_TARGETS},
    )


if __name__ == "__main__":
    main()
