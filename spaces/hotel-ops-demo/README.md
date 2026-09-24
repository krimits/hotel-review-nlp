---
title: Hotel Review Operations Demo
emoji: 🏨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
python_version: "3.11"
models:
  - yangheng/deberta-v3-base-absa-v1.1
---

# Hotel Review Operations · demo

Paste English hotel reviews, one per paragraph, or upload a `.csv` / `.txt` file.
The app answers in Greek, for the hotel owner:

- **what to fix first**: the aspects that the most reviews complain about, each with
  example quotes and a suggested action;
- **what guests appreciate**;
- **every finding with the clause of the review it comes from**, plus a CSV export.

## How it works

1. Each review is split into clauses with regular expressions. Only whitespace is
   ever inserted, so every quote is verbatim review text.
2. A lexicon names eight hotel aspects in each clause: cleanliness, staff, location,
   room, food, noise, value and facilities.
3. [`yangheng/deberta-v3-base-absa-v1.1`](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1)
   (MIT) reads the clause once for each word that named an aspect. For example,
   "breakfast" in "breakfast was cold" comes out negative. A neutral answer produces
   no finding. A clause like "the room was clean, not much of a view" produces both a
   positive and a negative room finding.

The counts are of reviews, not sentences. When a review both praises and criticises
an aspect, it is counted on both sides, so a complaint is never hidden behind a
compliment. Non-English reviews are skipped and listed.

## How well it works

The measurements used `scripts/eval_space_triage.py` on
`data/eval/space_triage_test2.json`. That set is 30 real English hotel reviews,
labelled by hand before anything ran on them. It stayed untouched until the method
and its settings had been fixed on a separate development set. The earlier
approaches ran on the same 30 reviews, on the same CPU class as this Space:

| | this demo | earlier `main` (Qwen2.5-0.5B → JSON) | earlier deployed Space |
|---|---|---|---|
| complaints found | **82%** (32/39) | 23% (9/39) | 8% (3/39) |
| complaints reported that are real | **80%** (32/40) | 41% (9/22) | 50% (3/6) |
| aspect named correctly | **98%** | 69% | 63% |
| praise found / real | 88% / 95% | 45% / 73% | 9% / 64% |
| time per review (CPU) | 0.5–1 s | about 22 s | about 22 s |

It still misses or invents some complaints. The main causes are:

- sarcasm and negation;
- comparisons with other hotels;
- problems described without an aspect word (for example "a complete fire trap").

That is why every finding shows its quote. Check it before acting on it.

This is a demonstration, not a verified production system. Nothing is stored, but
don't paste personal guest data.

Source: https://github.com/krimits/hotel-review-nlp (`spaces/hotel-ops-demo`).
To update this private Space from a machine authenticated as `krimits` with
`hf auth login`, run `python scripts/deploy_hf_space.py --resume` from the repository
root. It uploads only the demo's own files and never touches the project's datasets.
