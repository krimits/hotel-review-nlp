---
title: Hotel Review Sentiment — DistilBERT vs Qwen QLoRA vs GreekBERT
emoji: 🏨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: mit
models:
  - krimits/distilbert-hotel-reviews
  - krimits/hotel-review-nlp-qwen-qlora-adapter
  - krimits/greek-hotel-reviews-sentiment
short_description: Encoder vs decoder-LoRA sentiment demo, English and Greek
---

# Hotel Review Sentiment — encoder vs decoder vs Greek, side by side

Interactive companion to [`krimits/hotel-review-nlp`](https://github.com/krimits/hotel-review-nlp):
the same review is classified by three model families.

- **DistilBERT full fine-tune** (macro-F1 0.9642 on a test set that shares no review text with
  training), pinned to one model commit — an encoder, one forward pass per review.
- **Qwen2.5-0.5B + QLoRA** — a decoder LLM fine-tuned with parameter-efficient adapters on
  20k reviews, generating the label token by token. Its legacy-split macro-F1 of 0.9571 was read
  from the first-step scores of the two label tokens, not from generated text like this demo's
  answers ([record](https://github.com/krimits/hotel-review-nlp/blob/main/docs/experiments/results/qwen_legacy_v1/README.md)).
- **GreekBERT sentiment** (test macro-F1 0.9086) — `nlpaueb/bert-base-greek-uncased-v1`
  fine-tuned on the pinned `DGurgurov/greek_sa` corpus (Tsakalidis et al. 2018). Domain note,
  stated not hidden: that corpus is Twitter political sentiment, so Greek-language hotel
  reviews are a **cross-domain** use — indicative, not an in-domain benchmark.

The Qwen and GreekBERT numbers were measured on other test sets, so they are not comparable
with DistilBERT's — try the three side by side and compare their errors.
Try a Greek review in the third column.

**Response times.** Each answer shows its server processing time on this Space's CPU:
tokenization, inference and output processing, without queueing, network or rendering in the
browser. No benchmark of this Space has been run, so the speeds of the three models here have
not been compared. The project's [benchmark](https://github.com/krimits/hotel-review-nlp/blob/c17c569e7298e197cccc5185fe22595d33493cb8/docs/experiments/results/distilbert_v2/latency.json)
timed a DistilBERT checkpoint of the same architecture and size, trained on the out-of-time
split rather than the one served here: a median of 57 ms per review on one CPU thread, one
review at a time, tokenization included.
