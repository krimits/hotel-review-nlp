"""Suggested actions for one review, written by the base Qwen2.5-0.5B-Instruct.

No adapter and no training: the model is asked, with its chat template, for a short JSON list of problems the
guest reports, each with the exact words that show it, a measure, a department and the facts the hotel must
check. The model is small, so nothing it says is trusted until it is checked here:

- the output must be JSON in the asked-for form, and a malformed entry is dropped, not repaired;
- the excerpt must be in the review (whitespace and case aside), or the action is dropped;
- the department must be on the closed list, or the action is dropped, not coerced to `other`;
- at most five actions are kept.

What the code cannot check is whether an action is useful. That needs people reading them (docs/TRIAGE.md).
Nothing is loaded until the first review that needs it, and the generator is off unless
REVIEWNLP_TRIAGE_QWEN_ENABLED is set.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from reviewnlp.absa.extract import review_key
from reviewnlp.triage.schemas import DEPARTMENTS, SuggestedAction

PROMPT_VERSION = "actions-v1"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 400
MAX_ACTIONS = 5
MAX_REVIEW_CHARS = 4000
MAX_FIELD_CHARS = 400

SYSTEM_PROMPT = (
    "You help a hotel manager act on one guest review. List the concrete problems the guest reports that the "
    "hotel could fix. For each problem give: problem (what went wrong, in your own words), excerpt (the exact "
    "words from the review that show it, copied character for character, and short), measure (one concrete "
    "action the hotel could take), department (exactly one of: " + ", ".join(DEPARTMENTS) + "), and to_confirm "
    "(facts the review does not state and the hotel must check before acting, such as a room number or a date; "
    "an empty list if there are none). Do not invent facts. List only problems the guest actually reports: "
    "praise is not a problem. If there is nothing the hotel could fix, answer with an empty list. "
    'Answer with JSON only, in this form: {"actions": [{"problem": "...", "excerpt": "...", "measure": "...", '
    '"department": "...", "to_confirm": []}]}'
)


@dataclass(frozen=True)
class ActionSignals:
    """What the earlier stages said. The model is told that they can be wrong and are not evidence."""

    sentiment_label: str
    sentiment_confidence: float | None
    flagged_topics: list[str] = field(default_factory=list)
    hotel_context: str | None = None


def build_messages(review: str, signals: ActionSignals) -> list[dict]:
    review = " ".join(str(review).split())[:MAX_REVIEW_CHARS]
    confidence = "" if signals.sentiment_confidence is None else f" ({signals.sentiment_confidence:.2f})"
    lines = [f"Review:\n\n{review}\n",
             "Signals from other tools. They can be wrong, and they are not evidence:",
             f"- Overall sentiment: {signals.sentiment_label}{confidence}",
             f"- Complaint topics flagged: {', '.join(signals.flagged_topics) or 'none'}"]
    if signals.hotel_context:
        lines.append(f"- {signals.hotel_context}")
    lines.append("\nList the actions as JSON.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


@dataclass(frozen=True)
class ParsedActions:
    actions: list[SuggestedAction]
    dropped: int
    json_valid: bool
    error: str | None = None


def _json_object(raw: str) -> Any | None:
    """The first JSON value in the text that is an object with an `actions` list, or a bare list of objects.

    A JSON value that is not that is skipped whole, so an empty list inside some other object is not mistaken
    for the model saying there is nothing to fix.
    """
    decoder, position = json.JSONDecoder(), 0
    while position < len(raw):
        if raw[position] not in "{[":
            position += 1
            continue
        try:
            value, length = decoder.raw_decode(raw[position:])
        except ValueError:
            position += 1
            continue
        if (isinstance(value, list) and all(isinstance(entry, dict) for entry in value)) or (
                isinstance(value, dict) and isinstance(value.get("actions"), list)):
            return value
        position += length
    return None


def _text(value: object, maximum: int = MAX_FIELD_CHARS) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value if 3 <= len(value) <= maximum else None


def parse_actions(raw: str, review: str) -> ParsedActions:
    """The valid, grounded actions in the generation, and how many entries were rejected."""
    value = _json_object(str(raw))
    if value is None:
        return ParsedActions([], 0, False, "no JSON with an actions list")
    entries = value if isinstance(value, list) else value["actions"]
    haystack, kept, seen, dropped = review_key(review), [], set(), 0
    for entry in entries:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        problem, excerpt, measure = (_text(entry.get(name)) for name in ("problem", "excerpt", "measure"))
        department, to_confirm = entry.get("department"), entry.get("to_confirm", [])
        if (None in (problem, excerpt, measure) or department not in DEPARTMENTS
                or not isinstance(to_confirm, list) or len(to_confirm) > 5
                or any(_text(item) is None for item in to_confirm)
                or review_key(excerpt) not in haystack):
            dropped += 1
            continue
        key = (review_key(excerpt), review_key(problem))
        if key in seen or len(kept) >= MAX_ACTIONS:
            dropped += 1
            continue
        seen.add(key)
        kept.append(SuggestedAction(problem=problem, excerpt=excerpt, measure=measure, department=department,
                                    to_confirm=[_text(item) for item in to_confirm]))
    return ParsedActions(kept, dropped, True)


@dataclass(frozen=True)
class GenerationResult:
    raw: str
    hit_token_budget: bool
    model: str
    prompt_version: str = PROMPT_VERSION


class ActionGenerator(Protocol):
    model_name: str

    def generate(self, review: str, signals: ActionSignals) -> GenerationResult: ...


def _dtype_for(torch_module: Any, device: str) -> Any:
    """bfloat16 on a GPU that has it natively (compute capability 8 or more), float32 everywhere else.

    On a CPU, and on an older GPU such as a Colab T4, bfloat16 is slow or emulated, and float16 can overflow in
    this model family. Half a billion parameters in float32 is 2 GB.
    """
    if str(device).startswith("cuda") and torch_module.cuda.get_device_capability(device)[0] >= 8:
        return torch_module.bfloat16
    return torch_module.float32


def _load_qwen(model_id: str, device: str | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_dtype_for(torch, device)).eval()
    return tokenizer, model.to(device)


class QwenActionGenerator:
    """Loads the model on the first review that needs it; one generation at a time."""

    def __init__(self, model_id: str = BASE_MODEL, device: str | None = None, max_new_tokens: int = MAX_NEW_TOKENS,
                 loader: Callable[[], tuple[Any, Any]] | None = None):
        self.model_name = model_id
        self.max_new_tokens = max_new_tokens
        self._loader = loader or (lambda: _load_qwen(model_id, device))
        self._load_lock, self._run_lock = threading.Lock(), threading.Lock()
        self._bundle: tuple[Any, Any] | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> QwenActionGenerator | None:
        """None unless REVIEWNLP_TRIAGE_QWEN_ENABLED is set; REVIEWNLP_TRIAGE_QWEN_MODEL and _DEVICE are optional."""
        env = os.environ if environ is None else environ
        if env.get("REVIEWNLP_TRIAGE_QWEN_ENABLED", "").strip().lower() not in {"1", "true", "yes"}:
            return None
        return cls(model_id=env.get("REVIEWNLP_TRIAGE_QWEN_MODEL", "").strip() or BASE_MODEL,
                   device=env.get("REVIEWNLP_TRIAGE_QWEN_DEVICE", "").strip() or None)

    def _models(self) -> tuple[Any, Any]:
        if self._bundle is None:
            with self._load_lock:
                if self._bundle is None:
                    self._bundle = self._loader()
        return self._bundle

    def generate(self, review: str, signals: ActionSignals) -> GenerationResult:
        import torch

        tokenizer, model = self._models()
        prompt = tokenizer.apply_chat_template(build_messages(review, signals), tokenize=False,
                                               add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        # Only what generate() uses. A tokenizer that also returns token_type_ids would make it refuse the call.
        inputs = {name: encoded[name].to(model.device) for name in ("input_ids", "attention_mask")}
        with self._run_lock, torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        eos = tokenizer.eos_token_id
        return GenerationResult(raw=tokenizer.decode(new_tokens, skip_special_tokens=True),
                                hit_token_budget=eos is None or not bool((new_tokens == eos).any()),
                                model=self.model_name)
