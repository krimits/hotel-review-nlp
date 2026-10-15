# Decision: DistilBERT's full fine-tune against the scratch LoRA, across training seeds

**15 October 2026. The rule of the [seed protocol, version 1](distilbert_seed_robustness/PROTOCOL.md),
applied as written to runs made after it was locked. Scope: development evidence on a public
dataset; no selection is frozen and no model is promoted.**

## Question

Does the full fine-tune's lead over the scratch LoRA on the out-of-time test hold across three
training seeds?

## Answer

**Yes. The lead holds across seeds, and its size is clear but small.** In every seed the full
fine-tune is better and its paired interval excludes zero (`all(x > 0 for x in lo)`). Seed 44's
lead is under one point, so the size is "clear but small", not "meaningful".

| Seed | Full fine-tune | Scratch LoRA | d (full − LoRA) | Paired 95% interval |
| ---: | ---: | ---: | ---: | :--- |
| 42 (phase 2 runs) | 0.9491 | 0.9424 | 0.0068 | [0.0041, 0.0094] |
| 43 | 0.9496 | 0.9428 | 0.0067 | [0.0042, 0.0094] |
| 44 | 0.9480 | 0.9436 | 0.0045 | [0.0017, 0.0071] |

Out-of-time test, 13,921 reviews; macro-F1; 2,000 paired resamples, seed 0.

## Secondary results (cannot change the answer)

- **Spread over three seeds.** Full fine-tune mean 0.9489, standard deviation 0.0008; LoRA mean
  0.9429, standard deviation 0.0006.
- **Against TF-IDF + Naive Bayes (the phase 2 plan's extra seed).** Every seed of both methods is
  meaningfully better. The full fine-tune leads by 0.0308 (seed 42), 0.0312 (seed 43) and 0.0297
  (seed 44); every paired interval lies between 0.025 and 0.036. The phase 2 decision stays on
  seed 42; these are reported beside it.
- **Cost.** Full fine-tune about 14.0 minutes and 2,535 MiB peak GPU memory per new seed; LoRA
  9.5 minutes and 1,564 MiB. The seed-42 runs took 13.3 and 8.9 minutes.

## What this decision does not claim

- **Production accuracy.** Development splits of a public Booking.com dataset.
- **An interval over seeds.** Three seeds give a rough spread; a fourth could land elsewhere.
- **Anything about the random split.** Only the out-of-time view was re-run.
- **Bit-for-bit reproducibility.** With fp16 on a T4, a seed fixes initialisation and data
  order, not every floating-point sum.
- **A change to the phase 2 decision.** It stands on seed 42.

## How it was run

Notebook 27, the same session as the [BiLSTM decision](decision_bilstm_clean_splits.md), pinned to
`ed4070c`. Seeds 43 and 44 kept every setting of the seed-42 runs except the seed. The bundle's
78 files matched their manifest, and the comparison was recomputed locally from the logits.
CI recomputes it on every push.

## Artifacts

- [`results/distilbert_v2/time/`](results/distilbert_v2/time/): `distilbert_seed43/`,
  `distilbert_seed44/`, `distilbert_lora_scratch_seed43/`, `distilbert_lora_scratch_seed44/`.
- [`results/distilbert_seed_robustness.json`](results/distilbert_seed_robustness.json).
- [`results/model_comparison_v2.json`](results/model_comparison_v2.json): `primary_other_seeds`
  now holds the extra seeds against Naive Bayes, recomputed by `scripts/compare_split_models.py`.
- [`docs/RUNS.md`](../RUNS.md), entry `20261015-seeds-and-bilstm-results`.
