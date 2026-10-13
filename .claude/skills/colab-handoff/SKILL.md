---
name: colab-handoff
description: Prepare, verify, and consume Colab handoff bundles for hotel-review-nlp (run_config.yaml, metrics.json, hashes) using the existing scripts/verify_distilbert_handoff.py. Use when a run moves between the local repo and a Colab notebook, or when a bundle's integrity is in question.
---

# colab-handoff

## Direction 1: repo → Colab (what the human pastes into the notebook)

- The notebook template asserts the repo commit (full 40-char SHA) in its first cell and
  refuses to run against a different snapshot. Never weaken that assertion.
- The bundle leaving the repo carries: the pinned commit, any data fingerprints the
  protocol locked, and the protocol path. Nothing from `.env` or secrets.

## Direction 2: Colab → repo (the handoff bundle)

The bundle must contain what the existing verifier expects:

- `run_config.yaml` — the exact configuration the run used (seed, hyperparameters).
- `metrics.json` — per-epoch and final metrics as produced by the run.
- `trackio_final_metrics.json` (when the notebook used trackio) — final summary.
- The verifying script is `scripts/verify_distilbert_handoff.py` (CI runs it). Read its
  current checks before constructing a bundle; if the bundle type differs, extend that
  script in the same PR rather than bypassing it.

## Rules

- A bundle that fails verification is returned to the human gate with the verifier's
  exact output — never patched by hand to make it pass.
- Handoff artifacts a decision will cite must be versioned or hash-locked in the run
  folder; the protocol's fingerprint references them by hash.
- The notebook is never the source of truth for numbers: the bundle is. If a notebook
  output and the bundle disagree, the bundle wins and the discrepancy is reported.
