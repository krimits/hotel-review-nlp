---
title: Hotel Review Sentiment — DistilBERT vs Qwen QLoRA vs GreekBERT
emoji: 🏨
colorFrom: blue
colorTo: purple
sdk: gradio
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

- **DistilBERT full fine-tune** (macro-F1 0.9634, ~25 ms/review) — production-speed encoder.
- **Qwen2.5-0.5B + QLoRA** (macro-F1 0.9571 on a 20k subset, ~1–3 s/review) — a decoder LLM
  fine-tuned with parameter-efficient adapters, emitting the label as generated text.
- **GreekBERT sentiment** (test macro-F1 0.9086, ~100–200 ms/review) — `nlpaueb/bert-base-greek-uncased-v1`
  fine-tuned on the pinned `DGurgurov/greek_sa` corpus (Tsakalidis et al. 2018). Domain note,
  stated not hidden: that corpus is Twitter political sentiment, so Greek-language hotel
  reviews are a **cross-domain** use — indicative, not an in-domain benchmark.

In the unified benchmark the two LoRA-based English approaches were **statistically
indistinguishable** (exact McNemar p = 0.525) — try both and compare their errors.
Try a Greek review in the third column.