"""Qwen2.5-0.5B-Instruct + QLoRA fine-tuning - the centerpiece LLM run.

Pipeline (Hugging Face stack):
  1. load the base model in **4-bit NF4** via ``bitsandbytes`` (QLoRA paper,
     Dettmers et al. 2023: NF4 quantization + double quantization),
  2. attach LoRA adapters with ``peft`` targeting all attention + MLP
     projections listed in the config,
  3. fine-tune with ``trl.SFTTrainer`` on the instruction-formatted dataset,
     training on the completion (the one-word label) only.

Sized for a free Colab T4 (16 GB): 0.5B params in 4-bit (~0.6 GB) plus
optimizer state for ~2M adapter params fits comfortably; gradient
checkpointing keeps activations low.

Run:  python -m reviewnlp.llm.train_qlora --config configs/qlora_qwen.yaml
      (or open notebooks/02_train_qlora_colab.ipynb on Colab)
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from reviewnlp.llm.prompt_format import LABELS, format_prompt
from reviewnlp.utils.seed import load_config, set_seed

_LABEL_MAP = {"negative": "negative", "positive": "positive"}


def build_sft_dataset(processed_dir: str, train_cap: int, dev_cap: int, seed: int):
    """Return (train_ds, dev_ds) of {'prompt': ..., 'completion': ...} rows."""
    from datasets import Dataset

    frames = {}
    for split, cap in (("train", train_cap), ("dev", dev_cap)):
        df = pd.read_parquet(os.path.join(processed_dir, f"{split}.parquet"))
        if cap:
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda g, c=cap: g.sample(n=min(len(g), c // 2), random_state=seed))
            )
        frames[split] = Dataset.from_dict(
            {
                "prompt": [format_prompt(t) for t in df["text"]],
                "completion": [f"{_LABEL_MAP[lbl]}<|im_end|>" for lbl in df["label"]],
                # keep the raw label for eval-time metrics
                "label": df["label"].tolist(),
            }
        )
        print(f"{split}: {len(frames[split]):,} formatted examples")
    return frames["train"], frames["dev"]


def load_model_and_tokenizer(name: str, quantization: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # correct for causal-LM training

    model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # QLoRA: NF4 > naive int4
            bnb_4bit_use_double_quantum=True,   # double quantization saves ~0.4 bit/param
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlora_qwen.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers.trainer_utils import set_seed as hf_set_seed
    from trl import SFTConfig, SFTTrainer

    hf_set_seed(cfg["seed"])

    model, tokenizer = load_model_and_tokenizer(cfg["model"]["name"], cfg["model"]["quantization"])
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["train"]["grad_checkpointing"])

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds, dev_ds = build_sft_dataset(
        cfg["data"]["processed_dir"],
        cfg["data"]["train_cap"],
        cfg["data"]["dev_cap"],
        cfg["seed"],
    )

    t = cfg["train"]
    sft_cfg = SFTConfig(
        output_dir=cfg["output"]["model_dir"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        num_train_epochs=t["epochs"],
        learning_rate=t["lr"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        optim=t["optimizer"],
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=t["grad_checkpointing"],
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="no",
        report_to="none",
        max_length=cfg["model"]["max_length"],
        # completion-only loss: the prompt is masked out of the loss
        completion_only_loss=True if t.get("train_on_completion_only") else False,
        dataset_text_field=None,  # we supply pre-formatted prompt/completion columns
        packing=False,
        seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=dev_ds.remove_columns([c for c in dev_ds.column_names if c not in ("prompt", "completion")]),
        processing_class=tokenizer,
    )
    trainer.train()

    out_dir = cfg["output"]["model_dir"]
    trainer.save_model(os.path.join(out_dir, "adapter"))  # adapter-only checkpoint
    tokenizer.save_pretrained(os.path.join(out_dir, "adapter"))
    with open(os.path.join(out_dir, "adapter", "lora_config_used.json"), "w") as f:
        json.dump({"lora": cfg["lora"], "model": cfg["model"]["name"],
                   "labels": list(LABELS)}, f, indent=2)
    print(f"adapter saved to {out_dir}/adapter")


if __name__ == "__main__":
    main()
