# Decision note: Jev (System One) vs the complaint-topic lexicon

**Status: pre-registered — commit this file BEFORE the first API call.**
This note fixes the comparison rule in advance, the same way the DistilBERT
vs TF-IDF comparison rule was fixed before training. If the rule changes
later, it changes in a new file, never here.

## Question

Does a TypeSafe System One model (`jev-latest`) label the five complaint
topics (bathroom, cleanliness, air_conditioning, pests, responsiveness)
well enough to replace or supplement the lexicon-based topic detection that
the complaint-trends analysis currently uses?

Known baseline weakness (from the project's own audit): "The lexicon that
names the topics is right in only about half of their sampled quotes."

## Fixed before the first call

- **Model**: `jev-latest` via `POST https://api.typesafe.ai/v1/systemone`.
  One request per review with five independent `choice` questions, one per
  topic, answer space `{1, 0, unsure}`.
- **Prompt source**: the question rubrics are distilled from
  `complaint_topics_guideline.md` only. The gold key (`key.csv`) is never
  opened or used. The scorer reads gold labels from the handed-in,
  adjudicated `sheet_A.csv` itself (item + five topic columns).
- **Sample**: stratified draw from the 300-item pilot gold, seed 0,
  n = 100 (all items with `--limit 0`). Per topic, gold-positive items are
  drawn first to guarantee representation, then the remainder is filled with
  gold-negative items. The sample list is written to the run directory
  before the first API call.
- **Metrics, per topic**:
  - precision, recall, F1 over items where the model gave `1`/`0` and the
    gold is `1`/`0`; `unsure` predictions count as abstentions and are only
    reported as a rate;
  - Cohen's kappa over all sampled items with classes `{0, 1, unsure}`;
  - expected calibration error (10 bins) of the reported confidence against
    correctness on `0`/`1` predictions;
  - p50/p95 latency per review (one request = five questions);
  - cost estimate from reported token usage (input+output) using the price
    filled below.
- **One run.** No prompt iteration between runs. If the first run fails for
  technical reasons (auth, rate limits), that is recorded and the run may be
  repeated once with the identical protocol.

## Decision rule (fill the blanks before committing, then do not edit)

Adopt Jev as a first-stage topic labeler (behind a provider interface, so
the pipeline never hard-depends on the vendor) if **all** of:

- mean kappa across the five topics >= `0.70` (Landis–Koch "substantial";
  the lexicon baseline sits near chance on the sampled quotes)
- per-topic precision >= `0.85` and recall >= `0.70` (the escalate-gate use
  case needs high precision; recall may lag because `unsure` abstentions
  are routed to human review, not dropped)
- calibration: ECE <= `0.10` (needed for the confidence threshold of the
  escalate-gate to mean anything)
- cost <= `0.50` EUR per 1,000 reviews at reported token usage (at the
  full 515k-review corpus this stays under ~260 EUR; above that the
  DistilBERT path dominates on cost)
- p50 latency <= `3000` ms per review (one request = five questions)

Otherwise: record the numbers in `docs/EXPERIMENT_LOG.md`, keep the lexicon,
and treat the result as evidence — a clean negative is also a finding.

## Provenance requirements for the run record

`code_sha256` of this script, the exact sample list with its hash, the model
string returned by the API, the run timestamp, and the full raw responses
(`responses.jsonl`) are stamped into `summary.json`.

## Sign-off

- Filled by: Evgenios Krimitsas  Date: 2026-09-30
- Committed before first API call: yes (dry-run smoke test only; the first
  real call awaits `TYPESAFE_API_KEY`)