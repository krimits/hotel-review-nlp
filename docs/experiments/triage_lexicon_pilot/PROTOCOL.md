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
