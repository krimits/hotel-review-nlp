# Phase A: whole-review operational-topic baseline

Status: prepared, no new human labels collected and no performance claim. Primary is
`complaint-lexicon-v1` in `src/reviewnlp/triage/lexicon.py`; Jev is off. No Microsoft/Jev experiment
or held-out topic-model run is required to collect this ground truth.

This task is **multi-label complaints**: bathroom, cleanliness, air_conditioning, pests,
responsiveness, plus a separate record of other complaints. It is not the five-way primary-topic
task (cleanliness/service/location/value/other). Those 500 annotations and dev F1 scores do not
validate this stage. The old negative-field pilot and ABSA evaluations concern different inputs/rules.

1. Freeze the source commit, lexicon version/SHA, input hashes and sampling procedure before evaluation.
   Keep private review text outside GitHub/Space. Draw whole English reviews independently of lexicon
   matches or sentiment; deduplicate/group repeats before separating any development/evaluation sets.
2. Two people independently label every topic on each whole review using the
   [topic definitions](../../annotation/complaint_topics_guideline.md). Adapt only the input framing:
   review includes praise and complaints; label an actual complaint even in a positive review. Record
   ambiguous labels as `unsure`. Separately record hypothetical/resolved/pending issue status for the
   generator; do not silently redefine the complaint labels to mean only pending issues.
3. Do not show model/rule output while labelling. Preserve original sheets, agreement and SHA-256 before
   adjudication. Report who annotated, independence and unresolved disagreements. Never make the lexicon's
   own output the ground truth. If reviewing an already inspected set, call it development evidence.
4. Score on the entire eligible sample, not just matched reviews. For each topic publish TP, FP, FN,
   support, precision, recall/F1 and uncertainty intervals. A lexical `yes` is a detection; `unsure`
   is an abstention, and a true complaint with `unsure` still counts as missed in overall recall.
   Report abstentions/human referral workload separately; do not hide misses by excluding abstentions.
   Human `unsure` is reported separately with explicit denominators. Zero predicted positives means
   undefined precision, not 100%. No ECE/confidence threshold is valid for these nonprobabilistic rules.
5. Inspect false matches and missed topics on development data only. Track unknown categories and
   referral coverage rather than automatically assigning a department. Keep adversarial/edge probes
   (hypothetical lift, repaired lamp, praise, suggestions, negation) separate from a representative sample.
6. Only after ground truth and a frozen evaluation policy decide whether targeted rule changes,
   supervised models or an opt-in Jev comparison justify their cost. Do not repeatedly tune on the
   reserved evaluation. This topic pilot alone cannot establish correct evidence interpretation,
   proposed-measure usefulness or production readiness of the complete system.

Start with a feasible representative pilot and report its actual size, topic support and intervals.
Rare topics may need a separately reported targeted sample. No sample size, target precision/recall,
decision threshold or final candidate selection has been agreed or measured here.

## Offline tools and human handoff

Prepare a private CSV with exactly `review_id,text,language`. Use whole, deduplicated English
reviews from the intended deployment population, with language `en` and at most 4,000 characters.
Redact personal identifiers before handing texts to reviewers if appropriate. Record where/how the
population was collected; the tool cannot certify language, representativeness or independence.
Do not substitute the five-way primary-topic labels or the already inspected synthetic development cases.

After committing the pilot sources, run from the repository root (50 is an illustrative size, not a
validated sample-size requirement):

```bash
python scripts/triage_lexicon_pilot.py prepare --reviews private_whole_reviews.csv \
  --size 50 --seed 20261013 --population "Describe the actual hotels, dates and collection method" \
  --output runs/lexicon_pilot_RUN
```

The tool samples uniformly before reading any labels and freezes the lexicon/protocol/input hashes.
Give **only `sheet_A.csv` to person A and `sheet_B.csv` to person B**. Keep `key.json`,
`predictions.json` and `manifest.json` closed while annotating. Each person fills every topic with
`1` (actual complaint), `0` (no complaint) or `unsure` (ambiguous), sets `done=yes`, and can use
`note` to explain ambiguity. `other` means an actual complaint outside the five supported topics.
Keep `item`, `text`, `text_sha256` unchanged. A suggestion implying a lack can count as a complaint;
a successfully fixed incident still counts as reported, although it need not receive a corrective
measure. This sheet evaluates topic detection, not pending-issue status or measure quality.

Save completed copies under separate names. Before discussion:

```bash
python scripts/triage_lexicon_pilot.py agreement --run runs/lexicon_pilot_RUN \
  --ratings-a completed_A.csv --ratings-b completed_B.csv \
  --output runs/lexicon_pilot_RUN/agreement.json
```

It rejects incomplete/edited texts, reports agreement and three-label Cohen's kappa per topic,
and lists each disagreement. Keep the original completed sheets and this report unchanged; record
the agreement before adjudication as the protocol requires. Constant labels yield undefined kappa,
not evidence of perfect reliability. Reviewer independence remains explicitly unverified by software.

Resolve disagreements together (preferably with an independent adjudicator) in `adjudicated.csv`,
using the same columns. A correction to a previously agreed label requires an explanation in `note`.
Then score once against that ground truth:

```bash
python scripts/triage_lexicon_pilot.py score --run runs/lexicon_pilot_RUN \
  --ratings-a completed_A.csv --ratings-b completed_B.csv \
  --agreement runs/lexicon_pilot_RUN/agreement.json --gold adjudicated.csv \
  --output runs/lexicon_pilot_RUN/scored.json
```

The report has per-topic counts, precision/recall/F1, per-topic Wilson intervals, human-uncertain
denominators, abstentions, and micro metrics for the five supported topics. `other` is reported
separately as an unsupported topic; lexical abstentions count as misses for true complaints.
All reviews currently require human referral because `other` remains unknown, even when a supported
topic matches. Binomial intervals do not model dependence between reviews from the same hotel.
Outputs contain no quality/promotion claim and make no external model calls.

**Keep the entire run private and out of the Space.** Both reviewer sheets contain review text;
the key can identify the original reviews. Share a verified text-free aggregate only after reviewing
the completed evidence. The human labels and the correct population are the remaining inputs;
the tool does not generate them.
