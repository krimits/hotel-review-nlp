# Serving the INT8 model

The [INT8 decision note](../../decision_int8_and_batching.md) measured dynamic quantization on
the full test set and recommended it for single-review CPU serving, but the API served FP32.
`MODEL_QUANTIZE=dynamic_int8` closes that gap: the encoder backend quantizes its linear layers
at load, so a deployment needs the published FP32 checkpoint and one environment variable, not
a second artifact. `/health` reports `quantized: true`, and an unknown mode fails at load with
the allowed values named instead of quietly serving FP32.

This directory holds the receipt for the served path.

## What the receipt shows

Same machine, same rows, one thread: does switching to INT8 change the labels?

| On this machine, 1,024 test rows | FP32 | INT8 |
| :--- | ---: | ---: |
| Macro-F1 | 0.9550 | 0.9551 |
| Median time per review, batch 1 | 92.1 ms | **56.5 ms** (1.6×) |
| Agreement with the other model | — | **99.61%** (4 of 1,024 differ) |

- **The quantization changes almost nothing.** Macro-F1 moves by +0.0001 and the two models
  disagree on four reviews. That is the same-machine property the decision note measured
  (99.4% agreement, −0.0005 macro-F1).
- **Against the recorded run**, the served model agrees on 99.51% of these rows. The small
  difference is a different quantized backend and CPU, not a different model.
- **The speedup is smaller here** (1.6× against the recorded 3.0×). This machine takes
  92.1 ms per review for FP32 (recorded: 51.5 ms on an Intel Xeon at 2.00 GHz), and the
  ratio is a property of the hardware as much as of the quantization.

## Why 1,024 rows and not 13,263

On this Windows CPU the quantized model runs at ~330 ms per review single-threaded, so the
full test set would take about 73 minutes. The receipt covers the **first 1,024 test rows** —
the same size the repository already uses for the published-weights agreement check — and
says so in `rows`. The full-set figure remains the decision note's, measured on Colab.

`scripts/verify_int8_serving.py --limit 0` runs the full set where the hardware allows it.

## The gate it enforces

| Check | Rule | This run |
| :--- | :--- | :--- |
| INT8 and FP32 agree | ≥ 99% | 99.61% ✔ |
| Macro-F1 within the one-point margin | \|Δ\| ≤ 0.01 | 0.0001 ✔ |
| Still faster than FP32 | ≥ 1.2× | 1.6× ✔ |
| Agrees with the recorded INT8 run | ≥ 99% | 99.51% ✔ |

## What it does not claim

- **Full-set reproduction on this machine.** The recorded 0.9637 is the Colab measurement; the
  1,024-row INT8 score here (0.9551) is not comparable with it row for row, which is why the
  gate is INT8 against FP32 on the same rows.
- **Other hardware or backends.** One CPU, one thread, dynamic quantization of the linear
  layers only.
- **The Space.** The Hugging Face demo still serves FP32; the receipt covers the API backend.

## Reproduce

```bash
python scripts/fetch_booking_515k.py
python -m reviewnlp.data.preprocess --config configs/baselines.yaml
hf download krimits/distilbert-hotel-reviews \
  --revision 7306aebcaaebc00d579f5d0a91001ae376f18158 --local-dir models/distilbert
python scripts/verify_int8_serving.py --model models/distilbert --limit 1024 --threads 1
```

Receipt: [`receipt.json`](receipt.json) — rows, thresholds, verdict, the macro-F1 of both
models, agreement figures, latencies and the environment (torch 2.13.0+cpu, transformers
4.56.2, Python 3.13.5, one thread).
