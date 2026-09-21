"""Batch ABSA over texts — chat-template prompts, optional adapter, left padding.

Changes vs v8: the adapter is OPTIONAL (variant='base' or 'adapter') so the
A/B attribution the review demanded is possible; prompts are built with
tokenizer.apply_chat_template(..., add_generation_prompt=True); generations
that exhaust the token budget are flagged instead of being parsed silently.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reviewnlp.absa.extract import format_absa_messages, parse_absa_output

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 320


def load_absa_model(adapter_dir: str | None = None, device=None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    variant = "base"
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        variant = "adapter"
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, variant


def extract_aspects_batch(texts, adapter_dir=None, batch_size=8, device=None):
    tokenizer, model, variant = load_absa_model(adapter_dir, device)
    records = []
    for start in range(0, len(texts), batch_size):
        chunk = [" ".join(str(t).split())[:4000] for t in texts[start : start + batch_size]]
        prompt_texts = [
            tokenizer.apply_chat_template(
                format_absa_messages(t), tokenize=False, add_generation_prompt=True
            )
            for t in chunk
        ]
        encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts]
        batch = tokenizer.pad(
            [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in encoded],
            padding=True,
            return_tensors="pt",
        )
        batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        budget_hit = new_tokens.shape[1] == MAX_NEW_TOKENS  # EOS never emitted
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for text, raw in zip(chunk, decoded):
            parsed = parse_absa_output(raw, review=text)
            records.append({
                "text": text,
                "raw_generation": raw,
                "generation_hit_token_budget": bool(budget_hit),
                **parsed,
            })
    return records