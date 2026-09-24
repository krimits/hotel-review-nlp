# Notebooks

**These are templates, not records.** They are committed without saved outputs
and are meant to be run top to bottom on Colab. If you want to see results that
were actually produced, read
[`docs/experiments/notebooks/`](../docs/experiments/notebooks/) instead — those
are preserved with their outputs intact.

`tests/test_notebooks.py` keeps these files honest: every pull request checks
that each notebook is valid JSON, that its code cells parse, that the
`reviewnlp` helpers it imports still exist, and that every committed file it
references is actually in the repository. It deliberately does **not** check for
saved outputs.

## What each one needs

| Notebook | Needs | Where to run |
| :--- | :--- | :--- |
| `01_eda.ipynb` | `data/raw/booking_reviews_515k.csv` (Kaggle) | Local CPU |
| `02_train_qlora_colab.ipynb` | processed parquet + GPU | Colab T4 |
| `03_llm_annotation_demo.ipynb` | a fine-tuned adapter | Colab T4 |
| `04_distilbert_lora_colab.ipynb` | processed parquet + GPU | Colab T4 |
| `05_distilbert_full_and_lora_colab.ipynb` | processed parquet + GPU | Colab T4 |
| `06_qwen_qlora_colab.ipynb` | processed parquet + GPU | Colab T4 |
| `07_greek_sentiment_colab.ipynb` | pinned Greek tweets dataset + GPU; separately labeled Greek hotel CSV for domain test | Colab GPU |

None of them can run in CI: the Kaggle CSV is not redistributable, the
checkpoints are not committed, and the runner has no GPU.

`data/raw/booking_reviews_515k.csv` is not in the repository — see
[`data/raw/README.md`](../data/raw/README.md) for how to download it, and note
the Windows filename warning there.

## Saving a run as evidence

Colab keeps outputs in the file it hands back, so the flow is just:

1. Run the notebook top to bottom on Colab.
2. `File → Download → Download .ipynb`.
3. Copy it into `docs/experiments/notebooks/` with an `_executed` suffix, next
   to the runs already preserved there.
4. Commit it, and reference it from the README's results table.

Keeping executed copies separate from the templates is deliberate: a template
stays small and diffable, while an executed notebook carries megabytes of
output that would otherwise churn on every re-run.

To execute one locally once its inputs are present:

```bash
pip install nbconvert ipykernel
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Removed

`04_bilstm_threshold_tuning_colab.ipynb` was deleted. It called
`train_bilstm("configs/bilstm_seed100_threshold.yaml")`, but neither that config
nor the threshold-aware `train_bilstm` it needs was ever merged — both live only
in the stale draft PR #2, which was written against the pre-flattening layout.
The notebook would have failed on its first real cell for anyone who opened it.

The experiment itself did run, and its evidence is kept:
[`docs/experiments/notebooks/bilstm_threshold_tuning_executed.ipynb`](../docs/experiments/notebooks/bilstm_threshold_tuning_executed.ipynb)
holds all eight cells with outputs — a dev-selected threshold of 0.685 moving
test macro-F1 by +0.08 pp over the 0.5 default.
