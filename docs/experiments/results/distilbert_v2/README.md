# DistilBERT on the clean splits (phase 2)

[Notebook 10](../../../../notebooks/10_distilbert_v2_colab.ipynb) produced these files on Colab, on a Tesla T4,
from commit [`994a103`](https://github.com/krimits/hotel-review-nlp/commit/994a103574d07833d466e9553680a9cab46bfe76).
That commit already held the [analysis plan](../../phase2_analysis_plan.md). The files are committed as
they came back: each one matches its size and SHA-256 in [`artifact_manifest.json`](artifact_manifest.json).

## Results

Macro-F1 with a 95% bootstrap interval (2,000 resamples of the test reviews):

| Model | Random test (13,263 reviews) | Out-of-time test (13,921 reviews) |
|---|---|---|
| TF-IDF + Naive Bayes, chosen on dev | 0.9351 (0.9303–0.9398) | 0.9184 (0.9133–0.9230) |
| DistilBERT, full fine-tune | **0.9642** (0.9605–0.9677) | **0.9491** (0.9450–0.9531) |
| DistilBERT, scratch LoRA | 0.9576 (0.9537–0.9613) | 0.9424 (0.9380–0.9465) |

On the out-of-time test, the full fine-tune beats Naive Bayes by 0.0308 macro-F1 (paired 95% interval
0.0264–0.0350). By the rule fixed in the plan, this is a meaningful gain. The
[decision note](../../decision_distilbert_vs_nb.md) gives the other comparisons, the costs and what to use
when.

## What is here

- **`{random,time}/{distilbert,distilbert_lora_scratch}/`**, one folder per run:
  - `metrics.json`: test metrics, the best dev macro-F1, parameters, training time, peak GPU memory,
    the data fingerprints and the settings;
  - dev and test logits and labels, in test order;
  - `request.json` (commit, GPU, module and arguments), `run_config.yaml` and `train_log.txt`;
  - for the LoRA runs, `lora_scratch_config.json`.
- **`latency.json`**: CPU latency, the whole test set, size and hardware, from
  [`benchmark_latency.py`](../../../../scripts/benchmark_latency.py), on the out-of-time test.
- **`environment.json`**: the commit, the GPU and the package versions.
- **`model_comparison_v2.json`**: the comparison as the notebook wrote it. The repository's copy,
  [`../model_comparison_v2.json`](../model_comparison_v2.json), is recomputed from the logits and is identical.
- **Not here: the model weights**, 255 MB per run.

## Checks

[`tests/test_phase2_results.py`](../../../../tests/test_phase2_results.py) runs in CI on every push. It checks that:
- every file matches the manifest, and every run comes from the same commit;
- that commit holds the plan, and the plan's comparisons and rule have not changed since;
- each run used the settings of the plan, on the same data as Naive Bayes;
- the recorded scores come from the saved logits;
- the comparison is reproduced from the logits.

To recompute the comparison from the repository root:

```bash
python scripts/compare_split_models.py
```
