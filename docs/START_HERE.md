# Start here — a ten-minute review path

The README is long because it carries the evidence for every claim. This page is the
short way in: five stops, in order, each with what to look at and what it is meant to
show. Nothing here is a summary you have to trust — every stop points at a file you can
check.

## 1. The claim and the rule that was fixed before it (2 min)

[`docs/experiments/decision_distilbert_vs_nb.md`](experiments/decision_distilbert_vs_nb.md)

Fine-tuned DistilBERT reaches **0.9642 macro-F1** on a random split and **0.9491** on
reviews written after the training period, about 3 points above a TF-IDF + Naive Bayes
baseline chosen on dev. What matters is the order of events: the decision rule was
written down before training, and the note applies it as written.

Comparisons use a paired bootstrap of the macro-F1 difference on the same test reviews,
not two independent scores put side by side.

## 2. The contamination we found in our own data (3 min)

[README, historical results](../README.md#historical-results-legacy-split-text-overlap)

The first version of this benchmark scored five model families on a test set that shares
**170 review texts with training**. We found it, audited it (finding F02), rebuilt the
splits so no review text crosses train/dev/test, and re-ran DistilBERT and the scratch
LoRA on the clean data.

The old numbers were kept rather than deleted, and labelled. The rebuilt splits also
come in three views — random, out-of-time and by-hotel — because an out-of-time split
answers a different question from a random one.

**Closed since:** the BiLSTM was re-run on the clean splits under a
[protocol](experiments/bilstm_clean_splits/PROTOCOL.md) locked before training. DistilBERT is
meaningfully better on later reviews, across three BiLSTM seeds
([decision](experiments/decision_bilstm_clean_splits.md)). The full fine-tune's lead over LoRA
also held across three DistilBERT seeds ([decision](experiments/decision_distilbert_seeds.md)).

**What is still open:** Qwen QLoRA is
[not carried forward](experiments/decision_qwen_binary_scope.md) for binary scoring; its
row stays labelled legacy and its comparisons stay unconfirmed on clean data.

## 3. A paper implemented from its equations, checked against the reference (2 min)

[`src/reviewnlp/lora/lora.py`](../src/reviewnlp/lora/lora.py) ·
[`tests/test_lora.py`](../tests/test_lora.py)

LoRA (Hu et al., 2021) in pure PyTorch: the update rule, the official initialisation,
`merge`/`unmerge`, adapter-only checkpoints. The tests assert each property, check the
scaling against a hand computation, and compare the implementation numerically against
Hugging Face `peft` on a locally built tiny BERT — with non-zero adapter weights, and an
assertion that the adapters actually change the output, so the comparison exercises the
`alpha / r` scaling and the `B @ A` product rather than two copies of the base model.

"I implemented the paper" is a tested claim here, not an assertion.

## 4. The service (2 min)

[`src/reviewnlp/serving/app.py`](../src/reviewnlp/serving/app.py) ·
[live demo](https://huggingface.co/spaces/krimits/hotel-review-demo)

FastAPI with `/predict`, `/absa` and health checks, env-configured model loading, batch
paths, a Docker image and a dynamic INT8 quantisation benchmark. The `/triage` endpoint
is explicitly experimental and unvalidated — [what it does and what has not been
measured](TRIAGE.md).

## 5. How to check anything else (1 min)

[`docs/EVIDENCE_MAP.md`](EVIDENCE_MAP.md)

Every claim mapped to the artifact that backs it. Numbers in the README come from
preserved run artifacts under [`docs/experiments/results/`](experiments/results/), and
`scripts/verify_distilbert_handoff.py` recomputes the DistilBERT pair from saved logits
on every CI run.

---

## If you would rather run it than read it

```bash
pip install -e ".[dev,serving]"
pytest tests -q                 # offline: no dataset, no weights, no GPU
ruff check src tests scripts spaces
```

## What this project is not

- Not production accuracy: these are development splits of a public dataset.
- Not a selected or promoted model: the triage and topic experiments record
  `selection_frozen: false` and `promoted: false` deliberately.
- Not fully reproducible from a clean machine: there is no dependency lockfile and no
  pinned Python version yet.

The experiment process itself — protocols locked before runs, append-only run log, the
agent workflow used to build this — is documented separately in
[`docs/AGENT_HARNESS.md`](AGENT_HARNESS.md).
