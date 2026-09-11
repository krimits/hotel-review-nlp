"""Inference helpers used by the benchmark, the API and the notebooks.

Three model families, one predict interface:

  * classical sklearn pipeline      (joblib)
  * BiLSTM                          (torch checkpoint + vocab.json)
  * encoder (DistilBERT / +LoRA)    (HF checkpoint + test_logits.npy cache)
  * Qwen QLoRA adapter              (peft adapter on 4-bit base)

The benchmark (reviewnlp.evaluation.benchmark) calls these functions so every
model is scored under identical conditions on the identical test set.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from reviewnlp.llm.prompt_format import parse_label


def predict_classical(pipe, texts: list[str]) -> np.ndarray:
    """sklearn Pipeline -> array of 'negative'/'positive' strings."""
    return pipe.predict(list(texts))


def predict_encoder(checkpoint_dir: str, texts: list[str], max_length: int = 256, batch_size: int = 64) -> np.ndarray:
    """HF encoder checkpoint -> label strings (batched, GPU if available)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device).eval()

    preds = []
    for start in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[start : start + batch_size]]
        enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        preds += logits.argmax(-1).cpu().tolist()
    id2label = model.config.id2label
    return np.array([id2label[p] for p in preds])


def predict_qwen_qlora(adapter_dir: str, texts: list[str], max_new_tokens: int = 4, batch_size: int = 16) -> np.ndarray:
    """Fine-tuned Qwen adapter (4-bit base) -> label strings via generation."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from reviewnlp.llm.prompt_format import format_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant = None
    if device == "cuda":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    with open(os.path.join(adapter_dir, "lora_config_used.json")) as f:
        base_name = json.load(f)["model"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
  tokenizer.padding_side = "left"  
    model = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=quant, device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    preds = []
    for start in range(0, len(texts), batch_size):
        chunk = [format_prompt(t) for t in texts[start : start + batch_size]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model.generate(
                **{k: v.to(model.device) for k, v in enc.items()},
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        preds += [parse_label(tokenizer.decode(t, skip_special_tokens=True)) for t in new_tokens]
    return np.array(preds)


def load_predict_fn(model_type: str, path: str) -> Callable[[list[str]], np.ndarray]:
    """Uniform predict(texts)->labels factory used by the benchmark/API."""
    if model_type == "classical":
        import joblib

        pipe = joblib.load(path)
        return lambda texts: predict_classical(pipe, texts)
    if model_type == "encoder":
        return lambda texts: predict_encoder(path, texts)
    if model_type == "qwen_qlora":
        return lambda texts: predict_qwen_qlora(path, texts)
    if model_type == "cached_logits":
        # models already evaluated by their training scripts: reload saved logits
        logits = np.load(f"{path}/test_logits.npy")
        _gold = np.load(f"{path}/test_labels.npy")
        id2label = {0: "negative", 1: "positive"}
        cached = np.array([id2label[i] for i in logits.argmax(-1)])
        return lambda texts: cached if len(texts) == len(cached) else _raise(
            f"cached_logits requires the exact benchmark test set ({len(cached)} rows)"
        )
    raise ValueError(f"unknown model_type: {model_type}")


def _raise(msg: str):
    raise ValueError(msg)


def load_test_set(processed_dir: str) -> pd.DataFrame:
    return pd.read_parquet(f"{processed_dir}/test.parquet")
