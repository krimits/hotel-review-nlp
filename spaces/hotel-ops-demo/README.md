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
The app answers in Greek, for the owner of a hotel or an apartment:

- **what to fix first**: the topics that the most reviews complain about, each with
  example quotes and a suggested action. There are 30 topics an owner can act on,
  in 8 categories: bed, air conditioning, check-in, lift and stairs, storage,
  odours, and so on;
- **what guests appreciate**;
- **every finding with the clauses of the review it rests on**, plus a CSV export.

The counts are of reviews, not sentences. A guest who writes about the small bed
three times is one complaint with three quotes. When a review both praises and
criticises a topic, it is counted on both sides.

Each finding can carry a note:

- a problem the guest reported that was not solved;
- a complaint worded as a suggestion;
- a probable typo read as a word ("cod" for "cot");
- a finding the model was unsure about.

## How it works

1. Each review is split into sentences and clauses with regular expressions. A
   clause after "but" or "and" that names nothing of its own stays with the clause
   before it, so "Airconds provided but all not cold" is read as one. Only whitespace
   is ever inserted, so every quote is verbatim review text.
2. A lexicon names the topics. Rules stop a word from naming the wrong one:
   - "welcoming" about an apartment is not about the staff;
   - "moving furniture" upstairs is noise;
   - in "got the room at 3rd floor", the room is only a place;
   - "service" used as a verb is not about the staff.
3. [`yangheng/deberta-v3-base-absa-v1.1`](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1)
   (MIT) reads the clause once for each topic, through the topic's noun where there
   is one. A neutral answer produces no finding.
4. Rules correct what a general-purpose sentiment model reads the wrong way round.
   Each of these is a complaint:
   - a suggestion ("should be serviced");
   - something missing that a guest expects ("no kettle", "without lift");
   - an air conditioner that is "not cold" or a shower that is "not hot";
   - pests, and hair or stains found.

   A problem that was reported and not solved is a finding of its own.

## How well it works

**Set A: the owner's six reviews.** The project owner ran six apartment reviews
through the previous version and checked every finding. Each error they found is now
a check in `data/eval/space_triage_user6.json`, including the quote the finding must
rest on. All six reviews pass with the real model.

**Set B: the guests' own labels.** These are Booking.com reviews from
[crawlfeeds/Booking-Hotel-Reviews-Dataset](https://huggingface.co/datasets/crawlfeeds/Booking-Hotel-Reviews-Dataset)
(CC BY-NC 4.0; downloaded by the script, not stored here). Each guest wrote what they
liked and what they disliked in separate fields. The rules were tuned on 150 reviews
(DEV) and measured on 300 reviews from 277 other properties (TEST):

| TEST, 300 reviews | previous version | this version |
|---|---|---|
| reviews with a complaint in which one is found | 65% (104/161) | **77%** (124/161; 95% CI 70–83%) |
| complaints found that come from the "disliked" text | 87% (174/200) | **87%** (229/262; 95% CI 83–91%) |
| reviews with praise in which praise is found | 90% (255/283) | **93%** (262/283) |
| praise found that comes from the "liked" text | 97% (626/646) | 96% (694/726) |
| time per review (CPU) | 0.24 s | 0.29 s |

Guests sometimes write a criticism under "liked", so the second row is a lower bound.
TEST was run twice. The first run used the rules as frozen after tuning (77% and 88%).
The second used this version, after two fixes found in set A (77% and 87%).

**Set C: the owner's topic labels.** The project owner is labelling 40 TEST reviews
by topic, without seeing the output. This will measure precision and recall per topic.
The results are pending.

**TripAdvisor sets labelled by Claude.** The labels in
`data/eval/space_triage_{dev,test1,test2}.json` were made by Claude, the AI assistant
that wrote this code. It read each review before anything ran on it; no independent
person labelled them. The sets are kept to catch regressions across the 8 categories.
On TEST2:

| TEST2, 30 reviews | previous version | this version |
|---|---|---|
| complaints found | 82% (32/39) | 85% (33/39) |
| complaints reported that are labelled | 80% (32/40) | 77% (33/43) |

It still misses or invents some complaints. The main causes are:

- sarcasm;
- comparisons with other hotels;
- typos ("Call the hose");
- topics outside the 30.

That is why every finding shows its quotes. Check them before acting on it. The
"unsure" note marks only a few findings, and on TEST they were real about as often as
the rest, so it is a weak signal.

To reproduce the numbers, run `python scripts/eval_space_triage.py user6|booking|topics|labels`.
It needs transformers, torch, sentencepiece and protobuf.

This is a demonstration, not a verified production system. Nothing is stored, but
don't paste personal guest data.

Source: https://github.com/krimits/hotel-review-nlp (`spaces/hotel-ops-demo`).
To update this private Space from a machine authenticated as `krimits` with
`hf auth login`, run `python scripts/deploy_hf_space.py --resume` from the repository
root. It uploads only the demo's own files and never touches the project's datasets.
