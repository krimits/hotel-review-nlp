[![CI](https://github.com/krimits/hotel-review-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/krimits/hotel-review-nlp/actions/workflows/ci.yml)

# hotel-review-nlp

End-to-end sentiment classification of hotel reviews: **classical ML baselines → pure-PyTorch BiLSTM → DistilBERT full fine-tune → LoRA implemented from the paper's equations (verified against PEFT with tests)**, all evaluated on one frozen test set, plus FastAPI serving and INT8 quantization.

The headline: **I implemented LoRA from the equations (Hu et al., 2021, §4), proved it numerically equivalent to Hugging Face `peft` with tests, and measured what the 1.1%-trainable-params trade-off actually costs in macro-F1.**

## Results

One frozen test set (13,278 reviews, 10,000 positive / 3,278 negative), never tuned. Model selection on dev macro-F1 only.

| Model | Macro-F1 | Accuracy | Trainable params |
| :--- | :---: | :---: | :---: |
| TF-IDF (word) + Naive Bayes | 0.9345 | 95.09% | 100% |
| TF-IDF (word) + LR-SGD | 0.9173 | 94.11% | 100% |
| TF-IDF (char) + Naive Bayes | 0.8545 | 89.91% | 100% |
| TF-IDF (char) + LR-SGD | 0.5681 | 78.87% | 100% |
| BiLSTM (pure PyTorch), best of 3 seeds | **0.9521** | **96.42%** | 100% |
| BiLSTM, mean ± std over 3 seeds | 0.9507 ± 0.0012 | 96.31 ± 0.09% | 100% |
| DistilBERT full fine-tune (full 119k) | **0.9634** | **97.27%** | 100% (66.9M) |
| DistilBERT + from-scratch LoRA (full 119k) | 0.9573 | 96.80% | **1.10% (739,586)** |
| Qwen2.5-0.5B QLoRA (20k subset) | 0.9571 | 96.71% | ~2.6% adapter |

Unified benchmark on the frozen 13,278-row test set: every model family — classical, BiLSTM, both DistilBERT variants, Qwen QLoRA — scored on the identical ordered rows, with exact McNemar tests for all 10 pairs. Nine of ten differences are significant at α=0.05; the one exception: **scratch-LoRA vs Qwen QLoRA is statistically indistinguishable (p = 0.525)** — 739K trainable params on a 67M encoder matches a 0.5B decoder LLM fine-tuned via QLoRA on this task. Full numbers and p-values: [`runs/benchmark/results.json`](https://huggingface.co/datasets/krimits/hotel-review-nlp-frozen-splits/blob/main/runs/benchmark/results.json).

LoRA vs full fine-tune on the identical 20k subset: LoRA trains 1.10% of parameters, in half the wall-clock (94 s vs 186 s) at 43.5% less peak GPU memory (970 MB vs 1,716 MB), for −0.94 pp macro-F1. Full-data full-FT comparison and McNemar significance tests: see [Methodology](DESIGN.md).

## Labels from the schema, not a score threshold

Most tutorials label this dataset by a `Reviewer_Score` cutoff ("7+ = positive"). That invents a boundary the business never approved. Instead, the project uses the dataset schema itself: each review has independent `Positive_Review` / `Negative_Review` free-text fields, and the platform fills the unused one with "No Positive"/"No Negative" markers — so the label is the guest's own framing, with mixed reviews honestly dropped (see [DESIGN.md §1](DESIGN.md)).

From 515k raw rows: **147,140 unambiguous labeled reviews** → train 118,990 / dev 14,872 / test 13,278.

## Quickstart

```bash
git clone https://github.com/krimits/hotel-review-nlp.git
cd hotel-review-nlp
make install          # -e ".[dev,serving]" (+ .[llm] where a GPU exists)

make data             # Booking.com CSV -> processed splits (see data/raw/README.md)
make baselines        # 2 feature views × 2 models -> runs/classical/
make bilstm           # pure-PyTorch BiLSTM, early stopping on dev macro-F1
make distilbert       # DistilBERT full fine-tune (GPU)
make qlora            # Qwen2.5-0.5B QLoRA — run on Colab GPU (notebooks/02)
make benchmark        # all models, one test set, exact McNemar -> runs/benchmark/

make serve            # MODEL_TYPE=encoder MODEL_PATH=runs/distilbert uvicorn ...
make quantbench       # dynamic INT8 latency/F1 trade-off -> runs/quantization/
make docker           # containerized serving (stub model, CI-safe)
```

Every experiment is config-file driven (`configs/*.yaml`) — no magic numbers in cells. CI runs `ruff` and the full test suite, including the **LoRA ↔ PEFT numerical equivalence test**, on pinned `transformers==4.56.2` / `peft==0.17.1` / `accelerate==1.10.1`.

## Why the LoRA tests matter

`src/reviewnlp/lora/lora.py` is dependency-free (torch only): `h = W0x + (α/r)·B(A(x))`, `A ~ kaiming_uniform(a=√5)`, `B = 0` so ΔW = 0 at step 0, α/r scaling on the adapter path, dropout on the adapter input only, merge/unmerge round-trip. `tests/test_lora.py` asserts all of it — including a 1:1 output comparison against `peft` on a locally constructed tiny BERT, weights copied between the two implementations. "I implemented the paper" is a tested claim, not a vibe.

## Project structure

```
src/reviewnlp/
  data/            preprocess: schema-based labels, splits (leakage-tested)
  baselines/       TF-IDF + NB / LR-SGD; pure-PyTorch BiLSTM training loop
  lora/            LoRA from scratch (paper-faithful, zero deps beyond torch)
  llm/             DistilBERT fine-tune (full + scratch-LoRA), Qwen QLoRA
  evaluation/      unified benchmark: same test ids, exact McNemar, latency
  serving/         FastAPI app: /health, /predict, /predict/batch; stub mode
configs/           one YAML per experiment — no magic numbers
notebooks/         01 EDA · 02 QLoRA on Colab · 03 LLM annotation demo · 04 ablations
tests/             unit + property tests incl. LoRA↔PEFT equivalence
docs/EXPERIMENT_LOG.md    week-by-week experiment narrative (debugging evidence)
DESIGN.md          every decision an interviewer will probe, with rationale
```

## Roadmap

- Full-data DistilBERT comparison (fair vs the BiLSTM's 119k-row training) — in progress
- Score-based 3-class labels + calibration curves
- QLoRA on a 7B-class model; ONNX/GPTQ export for the serving path
- Distillation: fine-tuned LLM teacher → DistilBERT student

See [DESIGN.md §8](DESIGN.md) for the full list.