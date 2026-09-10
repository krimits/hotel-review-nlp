# Verified full-data DistilBERT comparison

Notebook 05 produced these Colab artifacts from training commit [`530cf9e`](https://github.com/krimits/hotel-review-nlp/commit/530cf9e14eea7e527a240ea470ae3a8e222c4d3f). Verification on 2026-09-10 recomputed both models' development and test scores from their saved logits, matched their ordered labels to the local frozen parquets, and reproduced the exact McNemar result.

## Results

| Measurement | Full fine-tuning | Scratch LoRA + task head |
|---|---:|---:|
| Train / dev / test rows | 118,990 / 14,872 / 13,278 | 118,990 / 14,872 / 13,278 |
| Best dev macro-F1 | 0.9602 | 0.9522 |
| Test macro-F1 | 0.9634 | 0.9573 |
| Test accuracy | 0.9727 | 0.9680 |
| Trainable / total parameters | 66,955,010 / 66,955,010 | 739,586 / 67,102,466 |
| Training seconds | 872.566 | 567.171 |
| Peak allocated CUDA memory, MiB | 2,534.441 | 1,562.987 |
| Learning rate | 2e-5 | 1e-4 |

Both runs used a Tesla T4, seed 42, two epochs, maximum length 256, batch size 32, evaluation batch size 64, FP16, weight decay 0.01, and warmup ratio 0.06. LoRA used rank 8, alpha 16, zero dropout, and the query/value projections. Its 739,586 trainable parameters comprise 147,456 adapter parameters and 592,130 task-head parameters.

LoRA used 35.0% less training time and 38.3% less peak allocated GPU memory, with 0.61 percentage points lower macro-F1. These are recorded costs for one run of each method; the verification recomputes their ratios, not the original GPU measurements.

Confusion matrices use rows = true labels and columns = predicted labels, ordered `[negative, positive]`:

```text
Full FT: [[3114, 164], [199, 9801]]
LoRA:    [[3106, 172], [253, 9747]]
```

Exact McNemar: **122 full-only correct / 60 LoRA-only correct**, 182 discordant predictions, **p = 5.02808218497859e-06**. Full FT makes 363 errors and LoRA makes 425. This is a paired classification-error test; it is not a test of the macro-F1 statistic or a multi-seed comparison.

## Evidence and verification

- [verification.json](verification.json): recomputed metrics, paired counts, integrity checks, and limitations.
- [distilbert_comparison.csv](distilbert_comparison.csv): the original comparison table.
- [data_manifest.json](data_manifest.json): exact parquet hashes, ordered semantic fingerprints, class counts, and known overlaps.
- [environment.json](environment.json): runtime versions, GPU, training commit, and 29 source-file hashes.
- [artifact_manifest.json](artifact_manifest.json): the original manifest for the full Colab bundle.
- Model subdirectories: unchanged metrics, request/config files, logs, and development/test logits and labels.

All 23 supplied files listed in the original artifact manifest passed byte-size and SHA256 checks. The manifest itself is also preserved. All 29 recorded source hashes matched the training commit. Both models' saved labels match the same ordered local development and test rows, and all three parquet byte hashes and semantic fingerprints match the archived dataset manifest.

Reproduce the checks from the repository root:

```bash
python scripts/verify_distilbert_handoff.py
```

To additionally verify the archived local dataset:

```bash
python scripts/verify_distilbert_handoff.py --processed-dir data/processed
```

The second command produced the committed report. If the training commit is unavailable in a shallow Git clone, source-hash verification is reported as unavailable; fetching that commit enables the source check. The dataset check is optional because raw reviews and parquet files are not published here.

The LoRA `run_config.yaml` contains the shared base settings. Its [request.json](distilbert_lora_scratch/request.json) records the `--lr 0.0001` and adapter overrides; [metrics.json](distilbert_lora_scratch/metrics.json) records the effective settings. The base YAML alone is insufficient to reproduce that run.

## Scope and limitations

This is the small evaluation handoff. Thirteen full-bundle files listed in the original manifest are intentionally absent: the two model weights, model/tokenizer configurations and vocabularies, and the scratch adapter checkpoint. They are not required to recompute saved-prediction metrics, but the full bundle is required for model reload, serving, quantization, and live-inference latency verification. Those checks were not performed during this import.

The legacy dataset contains 180 normalized train/dev text overlaps, 170 train/test overlaps, and 24 dev/test overlaps. Its results support a historical comparison on identical data, not a leakage-free estimate. McNemar does not correct these data limitations. A deduplicated dataset requires fresh training of all compared models.

This comparison covers two encoders. It does not complete the five-family benchmark or validate the pending Qwen run. Historical classical results predate the analyzer/selection fixes, and the reported three-seed BiLSTM summary still needs individual run artifacts.
