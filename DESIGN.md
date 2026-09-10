# Design decisions

The rationale behind every choice an interviewer is likely to probe.

## 1. Label construction: schema over thresholds

The Booking.com dataset carries `Reviewer_Score` (2.5-10) that most tutorials turn into
labels with a cutoff ("7+ = positive"). That invents a boundary the business never approved
and drags every 6.9/10 review into training as a negative. Instead we exploit the *schema*:
each review has two independent free-text fields, `Positive_Review` and `Negative_Review`,
and the platform fills the unused one with markers ("No Positive" / "No Negative").

- positive-only text present + "No Negative" → `positive`
- "No Positive" + negative-only text present → `negative`
- both present (mixed sentiment) → dropped from the binary benchmark

Properties: zero threshold arbitrariness, one written text per row, and the label is the
guest's own framing rather than our score discretization. The cost - dropping mixed rows -
is honest and documented; they are a natural future third class.

## 2. Why keep five model families

The benchmark is the deliverable. Each family answers a different engineering question:

| Family | Question it answers |
|---|---|
| TF-IDF + NB / LR-SGD | How far do bag-of-words + linear models get? (strong on lexical sentiment cues) |
| BiLSTM (pure torch) | Can I write the training loop myself - packing, clipping, AMP, early stop? |
| DistilBERT full FT | What does full transformer fine-tuning buy at 100% trainable params? |
| DistilBERT + scratch LoRA | Does *my* LoRA implementation match PEFT and preserve quality with ~1.1% trainable params? |
| Qwen2.5 QLoRA | Does the instruction-tuned LLM route work under free-Tier GPU constraints? |

Comparing them on one frozen test set with McNemar turns opinions into a table.

## 3. LoRA: fidelity to the paper, verified not assumed

`reviewnlp/lora/lora.py` is intentionally dependency-free. Fidelity points:

- update rule `h = W0x + (α/r)·B(A(x))` with `ΔW = BA` (rank decomposition, paper §4.1/4.2);
- initialization: `A` ~ kaiming-uniform(`a=√5`), `B = 0` (official code) ⇒ the fine-tune
  starts exactly at the pretrained function - the tests assert this bit-for-bit;
- scaling `α/r` applied on the adapter output, dropout on the adapter input only;
- `merge()` folds `ΔW` into `W0`; `unmerge()` reverses it - both tested for round-trip parity;
- compact checkpoint containing the adapters plus the newly initialized task head;
- merged, ordinary Hugging Face checkpoint for wrapper-free inference.

The `peft` equivalence test copies weights between implementations on a locally-constructed
tiny BERT and requires identical outputs (`atol=1e-5`). "I implemented the paper" is thus a
tested claim, not a vibe. Applying it to DistilBERT (`train_distilbert_lora.py`) then
tests the paper's practical finding: adapters plus the task head can reach competitive quality
while training roughly two orders of magnitude fewer parameters than full fine-tuning.

## 4. QLoRA under free-Colab constraints

A T4 has 16 GB. The budget for Qwen2.5-0.5B:

| Component | Cost |
|---|---|
| base weights in NF4 (double quant) | ~0.4-0.6 GB |
| LoRA adapters r=16 on 7 projection types | ~2-3 M trainable params |
| paged AdamW-8bit optimizer state | small, spillover-safe |
| activations (grad checkpointing, len 320, eff. batch 32) | the real constraint |

Choices that matter and why: NF4 + double quantization (QLoRA paper's core), completion-only
loss (label is one token - training on the prompt would waste 95% of the compute on
reinforcing the prompt), right-padding with `pad_token=eos`, `use_cache=False` with
checkpointing, cosine schedule with short warmup. 20k examples / 1 epoch is deliberately
small: sentiment is an easy, low-entropy task and the benchmark exists to *prove* adequacy,
not to bake in overkill.

## 5. Evaluation methodology

- **One test set, never tuned.** Test rows are sampled once by seed and capped per class;
  every family scores the same ids. Dev is used for early stopping only.
- **Exact McNemar**, not the chi-square approximation: with 10k test rows, discordant pairs
  are few and the exact binomial p-value stays valid (Dietterich 1998).
- **Latency is measured, not guessed**: p50/p95 per text, batch of 64, three repeats after
  warmup - same protocol before/after INT8 quantization.
- **Artifacts, not screenshots**: `results.json`, `confusion.png`, `latency.png` are
  generated files; the README table is meant to be filled from them.

## 6. Serving: boring on purpose

FastAPI + Uvicorn + Docker, configured by environment variables. A `stub` model type keeps
the API contract testable without weights (CI), the real models load once at startup, and
the batch endpoint amortizes tokenization. Dynamic INT8 quantization targets CPU inference -
the deployment most teams actually have - and the benchmark script quantifies the trade
(~2× speedup, ≤0.1 F1 delta expected for this task class). Heavier serving (vLLM, Triton)
is the natural next step for the QLoRA model and listed on the roadmap rather than
half-implemented here.

## 7. Reproducibility

- One YAML per experiment; every CLI takes `--config`.
- Single `set_seed` helper covering `random`/`numpy`/`torch`/CUDA.
- Vocab and vectorizers fit on train only (leakage tests in the suite).
- Model selection on dev macro-F1; test is touched once per run.

## 8. What I would do with more compute

1. Full 515k-row training with score-based 3-class labels and calibration curves.
2. QLoRA on a 7B-class model + ONNX/GPTQ export for the serving path.
3. DDP on 2 GPUs, then a scaling curve (throughput vs. batch/world-size).
4. Distillation: teacher = fine-tuned LLM, student = DistilBERT - closing the loop with the
   LLM-as-annotator notebook.
