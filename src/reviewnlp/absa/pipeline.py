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


def generate_aspect_records(tokenizer, model, texts, batch_size: int = 8) -> list[dict]:
    """Generate and parse ABSA records with an already-loaded tokenizer and model.

    Split out of extract_aspects_batch so a long-lived caller - the serving
    model manager - can reuse one loaded model across requests and still get
    real batched generation with left padding, instead of reimplementing the
    loop one review at a time.
    """
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
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, text, raw in zip(new_tokens, chunk, decoded, strict=True):
            parsed = parse_absa_output(raw, review=text)
            records.append({
                "text": text,
                "raw_generation": raw,
                # Per row, not per chunk: generation runs until every row in the
                # batch is done, so a chunk-wide check flagged rows that had
                # emitted EOS long before. A row that never emitted it is the
                # one that was actually cut off.
                "generation_hit_token_budget": _hit_token_budget(row, tokenizer.eos_token_id),
                **parsed,
            })
    return records


def _hit_token_budget(row, eos_token_id) -> bool:
    """True when this row never emitted EOS, so its output was cut off."""
    if eos_token_id is None:
        return True
    return not bool((row == eos_token_id).any())


def extract_aspects_batch(texts, adapter_dir=None, batch_size=8, device=None):
    """Load a model and run generate_aspect_records over texts."""
    tokenizer, model, _variant = load_absa_model(adapter_dir, device)
    return generate_aspect_records(tokenizer, model, texts, batch_size=batch_size)
