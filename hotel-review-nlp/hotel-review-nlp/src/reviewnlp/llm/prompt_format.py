"""Instruction formatting for the Qwen sentiment classifier.

We cast sentiment classification as *constrained generation*: the model must
emit one token-class ("positive" / "negative"). This makes parsing trivial,
keeps eval deterministic (greedy decoding over the first few tokens), and
matches how classification is deployed with LLMs in production.

The chat template below matches the official Qwen2.5-Instruct format:
<|im_start|>system ... <|im_end|><|im_start|>user ... <|im_end|><|im_start|>assistant
"""

from __future__ import annotations

LABELS = ("negative", "positive")

SYSTEM_PROMPT = (
    "You are a hotel review sentiment classifier. "
    "Classify the review as exactly one of: positive, negative. "
    "Answer with a single word, nothing else."
)

USER_TEMPLATE = "Classify the following hotel review:\n\n{review}"

ASSISTANT_PREFIX = "<|im_start|>assistant\n"


def format_prompt(review: str) -> str:
    """Full Qwen2.5 chat-format prompt, ending right before the answer."""
    review = " ".join(str(review).split())[:4000]  # hard cap, Qwen handles long ctx
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{USER_TEMPLATE.format(review=review)}<|im_end|>\n"
        f"{ASSISTANT_PREFIX}"
    )


def parse_label(generated_text: str) -> str:
    """Extract the class from generated text; fall back gracefully.

    Handles raw completions like ' positive', 'positive<|im_end|>', stray
    punctuation, or a whole sentence starting with a label word.
    """
    text = str(generated_text).strip().lower()
    for token in text.replace(".", " ").replace(",", " ").split():
        if token in LABELS:
            return token
        # handle concatenated tokens like 'positive<' or 'positive.'
        for label in LABELS:
            if token.startswith(label):
                return label
    return "positive" if "positive" in text else "negative"


def completion_target(label: str) -> str:
    """The exact completion the model is trained to emit."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    return label
