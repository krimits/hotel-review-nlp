# Sentiment Space update — 14 October 2026

Space: [krimits/hotel-review-demo](https://huggingface.co/spaces/krimits/hotel-review-demo), commit
[`71db6ee14c50254574f90137d9554fcddff22e2b`](https://huggingface.co/spaces/krimits/hotel-review-demo/commit/71db6ee14c50254574f90137d9554fcddff22e2b).
The owner uploaded it with `hf upload` from `main` at `9dd1742` (the merge of #49), from
[`spaces/hotel-review-demo/`](../../../../spaces/hotel-review-demo/).

**What changed in the live app.**
- Every model download now names a commit.
- Qwen gets the prompt it was trained on, with its chat markers.
- The Qwen number is described as first-step label scores.

## Checks

| Check | How | Result |
| :--- | :--- | :--- |
| Files on the Hub | At the Space commit: the file listing, two byte ranges of `app.py` read through the Hub API (the revision constants, 529 bytes at offset 2,432; the prompt, 330 bytes at offset 3,419), and `requirements.txt` read whole | All three files have their sizes on `main` (12,252, 2,481 and 52 bytes). Both ranges and `requirements.txt` match `main` byte for byte. |
| Startup | The owner's container log | The adapter snapshot was 10 files, 51.1 MB. The Qwen base weights were 988 MB. Then `Models ready.` and Gradio running on port 7860. |
| Loaded models | The owner called the Space's `/model_info` endpoint with `gradio_client`; the output is transcribed in [model_info.json](model_info.json) | DistilBERT loaded commit `7306aeb`, weights SHA-256 `f2a2bbc9…` (the [published model](../../results/distilbert_v2/README.md#published)). The Qwen adapter's loaded snapshot is commit `1458bdb`. The requested revisions are Qwen base `7ae5576` and GreekBERT `9962dad`. |
| One live request | The owner typed "The room was clean but the wifi kept dropping." | DistilBERT negative (93%). Qwen negative; it generated `negative\nnegative`, and the parser took the first label. GreekBERT positive (99%). |

`from_pretrained` and `snapshot_download` given a full commit SHA can load only that commit. So the
requested revisions for the Qwen base and GreekBERT are also what was loaded, though `model_info`
reports only the request for those two.

## Limits

- **GreekBERT on English text.** It is a Greek model trained on Greek political tweets. Its answer on
  an English review is out of domain and means nothing. The Space README asks visitors to try Greek
  reviews in that column.
- **Times.** The UI showed 429 ms, 0.97 s and 832 ms. That is one observation each, with the three
  models running at once on the Space's CPU. It is not a benchmark of the Space.
- **Qwen quality with the corrected prompt has not been measured.**
- **Dev mode was on during this check.** While the models loaded, the page returned HTTP 500 under a
  "Running" badge, because Dev mode shows the container's state, not the app's. Turn Dev mode off
  before a demo.
