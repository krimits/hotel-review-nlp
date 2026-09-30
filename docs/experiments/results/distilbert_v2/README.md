# DistilBERT on the clean splits (phase 2)

[Notebook 10](../../../../notebooks/10_distilbert_v2_colab.ipynb) produced these files on Colab, on a Tesla T4,
from commit [`994a103`](https://github.com/krimits/hotel-review-nlp/commit/994a103574d07833d466e9553680a9cab46bfe76).
That commit already held the [analysis plan](../../phase2_analysis_plan.md). The files from Colab are committed
as they came back: each one matches its size and SHA-256 in [`artifact_manifest.json`](artifact_manifest.json).
Two records of the model's publication were added later (see [Published](#published)).

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
- **`published.json`** and **`model_card.md`**: the records of the publication (see [Published](#published)).
  They are not in the manifest, which describes the files from Colab.
- **Not here: the model weights**, 255 MB per run. Those of the random-split full fine-tune are on the
  Hugging Face Hub.

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

## Published

On **29 September 2026**, [notebook 11](../../../../notebooks/11_publish_model_colab.ipynb) published the
random-split full fine-tune to the Hugging Face Hub and pinned the demo to it. It ran on a Colab CPU, with no
retraining, and used
[`publish_model.py`](https://github.com/krimits/hotel-review-nlp/blob/21d535ceafb6d1b95d4747d806c6153c2471718d/scripts/publish_model.py)
at commit [`21d535ceafb6d1b95d4747d806c6153c2471718d`](https://github.com/krimits/hotel-review-nlp/commit/21d535ceafb6d1b95d4747d806c6153c2471718d).

| What | Where |
|---|---|
| The model | [`krimits/distilbert-hotel-reviews` at `7306aebcaaebc00d579f5d0a91001ae376f18158`](https://huggingface.co/krimits/distilbert-hotel-reviews/tree/7306aebcaaebc00d579f5d0a91001ae376f18158) |
| Its weights | `model.safetensors`, SHA-256 `f2a2bbc9a31c8f3b1d9bee46ec11bb3d8942e7e59c0ef4711c3cb2a6c1cf81c2` |
| The previous model | tag `legacy-split-v1`, at [`9fe2f7f9963c206294200880b2401347804a80d6`](https://huggingface.co/krimits/distilbert-hotel-reviews/tree/9fe2f7f9963c206294200880b2401347804a80d6); weights SHA-256 `f30daa44e795b7015151892ff5ec285e14a916e122902aa0617a15bf96ec54b7` |
| The demo | [`krimits/hotel-review-demo` at `328a3065eb69b34914276330e25f420c7bdede33`](https://huggingface.co/spaces/krimits/hotel-review-demo/tree/328a3065eb69b34914276330e25f420c7bdede33), which loads the model at `7306aebcaaebc00d579f5d0a91001ae376f18158` |

**What was checked, in order.** A failed check stops the script before the next write.
1. **Before any write.**
   - The run's records in Google Drive were byte-identical to this bundle.
   - An agreement check on 1,024 examples: on the first 1,024 test reviews, with the same tokenizer, order
     and maximum length (256), the weights gave the saved test logits. The largest difference was
     3.6 × 10⁻⁶, against a tolerance of 0.001 (absolute; relative 0), and 1,024 of 1,024 predictions were
     equal.
   - The Hub's `main` still held the legacy model.
   - The demo, at `f4362f1351cb85427aae352cdb36666294172d44`, labelled a positive review as positive, ran
     Gradio 6.27.0, and the patch applied to its files.
2. **Upload.**
   - The tag `legacy-split-v1` was set on the legacy commit.
   - Then one commit, with the legacy commit as its parent, replaced the files.
   - Afterwards, `main` was `7306aeb…` and held exactly the 15 files of the manifest. The tag still marked
     the legacy commit, and the weights downloaded from `7306aeb…` had the local SHA-256.
3. **Demo.** One commit, `328a306…` (parent `f4362f1…`), made the demo load the model at `7306aeb…` and
   report what it loaded. Its README pins Gradio 6.27.0.
4. **Demo check.**
   - While the demo rebuilt, its runtime still reported the old commit `f4362f1…`, first building, then
     starting. The check waited.
   - It passed at 09:13 UTC, once the runtime reported `328a306…` and the demo reported the model commit
     `7306aeb…` and the new weights' SHA-256.
   - The demo then labelled one positive and one negative review correctly.

**The records.**
- [`published.json`](published.json): the state file, as the script wrote it in Google Drive.
- [`model_card.md`](model_card.md): the card committed to the Hub, byte for byte. Its SHA-256,
  `42c18c832146c4913385b9c19b5b8adf22588adacec9961cad93589dd50c0dae`, is the one in the manifest of
  `published.json`.
- [The executed notebook](../../notebooks/11_publish_model_colab_executed.ipynb): the output of every step,
  with the card and the waits.

**What the tests show, and what they do not.**
- [`tests/test_publication_record.py`](../../../../tests/test_publication_record.py) runs offline in CI. It
  checks that the three records agree with each other and with the committed results:
  - the card and the run's files match the manifest of the published commit;
  - the legacy weights' hash is the one in the [legacy bundle](../distilbert_legacy_full_v1/);
  - the two checks passed as defined above;
  - the card gives only this checkpoint's score as its result.
- It does **not** check that the Hub and the demo are still in that state today.
- A separate read of the Hub on the same day, 29 September 2026, matched the records:
  - `7306aeb…` held the 15 files and `.gitattributes`, and the card on `main` was 4,789 bytes;
  - the tag resolved;
  - the demo's `app.py` at `328a306…` pinned `7306aeb…`.

**Later: a text-only change to the demo, 30 September 2026.**
- **What changed.** The demo's README and interface gave response times that no recorded measurement
  supports: ~25 ms, ~50 ms, ~1–3 s and ~100–200 ms. They were removed in commit
  [`9265c76a748e0a9b3a65d601c6206549535500aa`](https://huggingface.co/spaces/krimits/hotel-review-demo/tree/9265c76a748e0a9b3a65d601c6206549535500aa),
  after `328a306…`. The time shown with each answer is now called server processing time. The model
  pin and the app's logic did not change.
- **Checks before the change:**
  - Both files were rebuilt from the published patch and matched `328a306…` byte for byte.
  - The new `app.py` was built with stand-in models on the Space's versions: Gradio 6.27.0,
    transformers 5.17.0 and peft 0.20.0.
- **Checks after the change:**
  - The two files on the Space's `main` had the SHA-256 of the files that were prepared.
  - A call to the running demo reported the running commit `9265c76…` and the model commit
    `7306aeb…`, with the weights' SHA-256 `f2a2bbc9…`.
  - The demo answered two calls to `/predict_distilbert`.
