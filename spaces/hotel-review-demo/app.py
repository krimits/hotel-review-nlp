"""Hotel-review sentiment demo: DistilBERT encoder vs Qwen2.5-0.5B QLoRA vs GreekBERT.

Companion to https://github.com/krimits/hotel-review-nlp.
- DistilBERT: krimits/distilbert-hotel-reviews, pinned to the clean-split model (full
  fine-tune; macro-F1 0.9642 on a test set that shares no review text with training).
- Qwen2.5-0.5B-Instruct + QLoRA adapter from
  krimits/hotel-review-nlp-qwen-qlora-adapter (public copy of run v1, trained
  on 20k reviews). Its legacy-split macro-F1 0.9571 was read from first-step
  label scores, not from generated text like this demo's answers
  (docs/experiments/results/qwen_legacy_v1/ in the repo).
- GreekBERT (nlpaueb/bert-base-greek-uncased-v1) fine-tuned on the pinned
  DGurgurov/greek_sa corpus: krimits/greek-hotel-reviews-sentiment
  (test macro-F1 0.9086). Domain note: the Greek training corpus is
  Twitter political sentiment (Tsakalidis et al. 2018), so Greek-language
  hotel reviews are a cross-domain use — results are indicative, and the
  framing is stated rather than hidden.

v2: rebuild after the Greek checkpoint repo gained its tokenizer files
(vocab.txt / tokenizer_config.json / special_tokens_map.json). Before that,
AutoTokenizer fell back to a default BERT vocab — Greek text tokenized into
UNKs and the Greek path collapsed to the majority class.

v3: every model download names a commit (DistilBERT already did), so a push to
any model repo cannot change what a running demo answers. The two own-repo
commits are written by scripts/pin_review_demo_revisions.py.

The Qwen path replicates the training-time decision rule exactly: greedy generation
of the label word over the Qwen2.5-Instruct chat prompt used in training, with the
robust line/token parser from src/reviewnlp/llm/prompt_format.py.

Each model is wired to its own click event: the encoder's answer does not wait
for the other paths, and a failure surfaces as a note
instead of killing the whole UI.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

DISTILBERT_ID = "krimits/distilbert-hotel-reviews"
DISTILBERT_REVISION = "7306aebcaaebc00d579f5d0a91001ae376f18158"  # the clean-split model, pinned so the texts match the weights
QWEN_ADAPTER_REPO = "krimits/hotel-review-nlp-qwen-qlora-adapter"
QWEN_ADAPTER_REVISION = "1458bdb9058fbb629e72ce83d9b0f44a09271941"
QWEN_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GREEK_ID = "krimits/greek-hotel-reviews-sentiment"
GREEK_REVISION = "9962dad48fe95287991ef54e7a4b6a6254b19843"
LABELS = ("negative", "positive")
GREEK_MAX_LENGTH = 160  # same truncation as the training config

# --- Exact prompt constants from src/reviewnlp/llm/prompt_format.py (commit 530cf9e) ---
SYSTEM_PROMPT = (
    "You are a hotel review sentiment classifier. "
    "Classify the review as exactly one of: positive, negative. "
    "Answer with a single word, nothing else."
)
USER_TEMPLATE = "Classify the following hotel review:\n\n{review}"
ASSISTANT_PREFIX = "assistant\n"


def format_prompt(review: str) -> str:
    review = " ".join(str(review).split())[:4000]
    return (
        f"system\n{SYSTEM_PROMPT}\n"
        f"user\n{USER_TEMPLATE.format(review=review)}\n"
        f"{ASSISTANT_PREFIX}"
    )


def parse_label(generated_text: str) -> str:
    """Robust parser — identical logic to the verified run-v3 decision rule."""
    text = str(generated_text).strip().lower()
    for token in text.replace(".", " ").replace(",", " ").split():
        if token in LABELS:
            return token
        for label in LABELS:
            if token.startswith(label):
                return label
    return "positive" if "positive" in text else "negative"


print("Loading DistilBERT encoder ...")
distilbert_dir = Path(snapshot_download(DISTILBERT_ID, revision=DISTILBERT_REVISION))
dtok = AutoTokenizer.from_pretrained(distilbert_dir)
dmodel = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)
dmodel.eval()
DISTILBERT_WEIGHTS_SHA256 = hashlib.sha256((distilbert_dir / "model.safetensors").read_bytes()).hexdigest()

print("Loading GreekBERT sentiment checkpoint ...")
gtok = AutoTokenizer.from_pretrained(GREEK_ID, revision=GREEK_REVISION)
gmodel = AutoModelForSequenceClassification.from_pretrained(GREEK_ID, revision=GREEK_REVISION)
gmodel.eval()
assert dict(gmodel.config.id2label) == {0: "negative", 1: "positive"}, (
    "Greek checkpoint id2label does not match the demo label convention"
)

print("Loading Qwen2.5-0.5B base + QLoRA adapter (fp32 CPU) ...")
adapter_dir = Path(snapshot_download(QWEN_ADAPTER_REPO, revision=QWEN_ADAPTER_REVISION,
                                     allow_patterns=["adapter/*"])) / "adapter"
qtok = AutoTokenizer.from_pretrained(adapter_dir)
if qtok.pad_token is None:
    qtok.pad_token = qtok.eos_token
qtok.padding_side = "left"  # required for correct batched generation
base = AutoModelForCausalLM.from_pretrained(QWEN_BASE, revision=QWEN_BASE_REVISION, dtype=torch.float32)
qmodel = PeftModel.from_pretrained(base, str(adapter_dir))
qmodel.eval()
print("Models ready.")


def predict_distilbert(text: str):
    """Returns (confidences, note) — exactly two values for two components."""
    if not text.strip():
        return {"negative": 0.0, "positive": 0.0}, "empty input"
    started = time.perf_counter()
    inputs = dtok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = dmodel(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    confidences = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    elapsed_ms = (time.perf_counter() - started) * 1000
    note = f"{elapsed_ms:.0f} ms on CPU"
    return confidences, note


def predict_qwen(text: str):
    """Returns (confidences, note) — exactly two values for two components."""
    if not text.strip():
        return {"negative": 0.0, "positive": 0.0}, "empty input"
    started = time.perf_counter()
    prompt = format_prompt(text)
    inputs = qtok(prompt, return_tensors="pt")
    with torch.inference_mode():
        generated = qmodel.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=qtok.pad_token_id,
            eos_token_id=qtok.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    raw = qtok.decode(new_tokens, skip_special_tokens=True)
    label = parse_label(raw)
    elapsed_ms = (time.perf_counter() - started) * 1000
    # The QLoRA model reports decisions, not calibrated probabilities: the label
    # word it generated IS the prediction; displayed as full-confidence by design.
    confidences = {label: 1.0}
    note = f"{elapsed_ms / 1000:.2f} s on CPU — generated: {raw.strip()[:60]!r}"
    return confidences, note


def predict_greek(text: str):
    """Greek-language path. Returns (confidences, note) — two values, two components."""
    if not text.strip():
        return {"negative": 0.0, "positive": 0.0}, "empty input"
    started = time.perf_counter()
    inputs = gtok(text, return_tensors="pt", truncation=True, max_length=GREEK_MAX_LENGTH)
    with torch.no_grad():
        logits = gmodel(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    confidences = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    elapsed_ms = (time.perf_counter() - started) * 1000
    note = f"{elapsed_ms:.0f} ms on CPU — trained on Tsakalidis et al. (2018), cross-domain here"
    return confidences, note


def classify_qwen_safe(review: str):
    """A decoder failure must surface as a note, never kill the UI."""
    try:
        return predict_qwen(review)
    except Exception as error:  # noqa: BLE001 — deliberate: report any failure inline
        return {"negative": 0.0, "positive": 0.0}, (
            f"Qwen path failed: {type(error).__name__}: {error}"
        )


def model_info():
    """Which weights this running app loaded, for the check after a deploy."""
    return {
        "distilbert_repo": DISTILBERT_ID,
        "distilbert_revision": DISTILBERT_REVISION,
        "loaded_commit": distilbert_dir.name,  # the snapshot folder is named after its commit
        "weights_sha256": DISTILBERT_WEIGHTS_SHA256,
        "revisions": {QWEN_BASE: QWEN_BASE_REVISION, QWEN_ADAPTER_REPO: QWEN_ADAPTER_REVISION,
                      GREEK_ID: GREEK_REVISION},
        "adapter_loaded_commit": adapter_dir.parent.name,
    }


EXAMPLES = [
    "The room was spotless, the bed huge and comfortable, and the staff upgraded us for free.",
    "Filthy bathroom, broken air conditioning, and front desk ignored three complaints.",
    "Great location and breakfast, but the wifi never worked and the walls are paper thin.",
    "Worst hotel I have ever stayed at. Rude staff, stains on the sheets, never again.",
    "Εξαιρετικό δωμάτιο, θέα θάλασσα και άψογη εξυπηρέτηση από όλο το προσωπικό.",
    "Απαράδεκτη καθαριότητα, κρύο νερό και θόρυβος όλη τη νύχτα. Ποτέ ξανά.",
]

with gr.Blocks(title="Hotel Review Sentiment — encoder vs decoder vs Greek") as demo:
    gr.Markdown(
        "# 🏨 Hotel Review Sentiment — DistilBERT vs Qwen QLoRA vs GreekBERT\n"
        "The same review, three model families. "
        "DistilBERT full fine-tune: macro-F1 **0.9642** on a test set that shares no review "
        "text with training. Qwen QLoRA (**0.9571** on the legacy split, read from first-step "
        "label scores rather than generated text) and GreekBERT (**0.9086**, Greek "
        "political-tweet corpus, cross-domain on hotel text) were measured on other test "
        "sets, so the three numbers are not comparable. "
        "[Project repo](https://github.com/krimits/hotel-review-nlp) · "
        "[DistilBERT model card](https://huggingface.co/krimits/distilbert-hotel-reviews)"
    )
    with gr.Row():
        text_in = gr.Textbox(label="Hotel review (English or Greek)", lines=4, placeholder="Paste a review ...")
    with gr.Row():
        btn = gr.Button("Classify", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### DistilBERT full fine-tune (67.0M params)")
            out_d_label = gr.Label(label="Prediction")
            out_d_note = gr.Textbox(label="Server processing time / note", interactive=False)
        with gr.Column():
            gr.Markdown("### Qwen2.5-0.5B + QLoRA (adapters)")
            out_q_label = gr.Label(label="Prediction")
            out_q_note = gr.Textbox(label="Server processing time / raw generation", interactive=False)
        with gr.Column():
            gr.Markdown("### GreekBERT sentiment (ελληνικά)")
            out_g_label = gr.Label(label="Prediction")
            out_g_note = gr.Textbox(label="Server processing time / note", interactive=False)

    # Independent events on the same input: they run concurrently, and each
    # model's result lands even if another is still thinking or fails.
    btn.click(predict_distilbert, inputs=text_in, outputs=[out_d_label, out_d_note])
    btn.click(classify_qwen_safe, inputs=text_in, outputs=[out_q_label, out_q_note])
    btn.click(predict_greek, inputs=text_in, outputs=[out_g_label, out_g_note])
    gr.Examples(examples=EXAMPLES, inputs=[text_in])
    # Not shown: lets a deploy check ask which DistilBERT weights this process loaded.
    info_button = gr.Button(visible=False)
    info_out = gr.JSON(visible=False)
    info_button.click(model_info, outputs=info_out, api_name="model_info")

if __name__ == "__main__":
    demo.launch()
