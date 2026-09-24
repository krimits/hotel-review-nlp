"""One bounded recovery attempt for demo reviews with no grounded findings."""

from __future__ import annotations

import torch

from reviewnlp.absa.extract import format_absa_messages, parse_absa_output
from reviewnlp.absa.pipeline import MAX_NEW_TOKENS, generate_aspect_records


def analyze_review(tokenizer, model, text: str) -> dict:
    """Keep quoted findings; retry once with clearer formatting when none survive."""
    first = generate_aspect_records(tokenizer, model, [text], batch_size=1)[0]
    if first.get("aspects") or first.get("generation_hit_token_budget"):
        return first

    messages = format_absa_messages(text)
    messages[0]["content"] += (
        " Return only a JSON array. Every element must be an object with the "
        "keys aspect, sentiment, and quote. Use one of the listed aspects and "
        "one of positive, negative, neutral. Copy each quote word for word "
        "from the review. Do not add an explanation or markdown. If the review "
        "has no supported aspects, return []."
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    batch = tokenizer.pad(
        [{"input_ids": ids, "attention_mask": [1] * len(ids)}],
        padding=True,
        return_tensors="pt",
    )
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, batch["input_ids"].shape[1] :]
    raw = tokenizer.batch_decode(continuation.unsqueeze(0), skip_special_tokens=True)[0]
    parsed = parse_absa_output(raw, review=text)
    retried = {
        "text": text,
        "raw_generation": raw,
        "generation_hit_token_budget": (
            tokenizer.eos_token_id is None
            or not bool((continuation == tokenizer.eos_token_id).any())
        ),
        "retry_used": True,
        **parsed,
    }
    # Preserve a valid abstention if the retry only produced malformed output.
    if first.get("json_valid") and not retried["json_valid"]:
        return {**first, "retry_used": True}
    return retried
