---
license: mit
base_model: distilbert-base-uncased
language:
  - en
tags:
  - sentiment-analysis
  - text-classification
  - hotel-reviews
library_name: transformers
pipeline_tag: text-classification
---

# DistilBERT hotel-review sentiment classifier

A full fine-tune of `distilbert-base-uncased` that labels an English hotel review as negative or
positive. It is part of the [`krimits/hotel-review-nlp`](https://github.com/krimits/hotel-review-nlp) project. It was trained and
tested on splits of Booking.com reviews that share no review text.

## Result of this model

It was scored on its test set of 13,263 reviews (3,263 negative). That set
shares no review text with training. It was scored once, with the epoch that did best on dev.

| Measure | This model | TF-IDF + Naive Bayes, same test |
|---|---:|---:|
| Macro-F1 (95% bootstrap interval) | **0.9642** (0.9605–0.9677) | 0.9351 (0.9303–0.9398) |
| Accuracy | 97.33% | 95.14% |
| Negative reviews: recall / precision | 0.953 / 0.939 | 0.913 / 0.892 |
| Errors | 354 | 644 |

- **Against Naive Bayes.** On the same reviews, the gain is
  +0.0291 macro-F1, with a paired 95% interval of
  0.0246 to 0.0335. Exact McNemar test:
  - 387 reviews only this model gets right;
  - 97 only Naive Bayes gets right;
  - p = 4.9 × 10⁻⁴².
- **Training:** 117,880 reviews. The best dev macro-F1 was
  0.9615.

## Related experiment, with a different checkpoint

A second model, with the same architecture and training recipe, was trained on another split of the
same reviews.
- **Training data:** reviews written up to 9 March 2017.
- **Result:** 0.9491 macro-F1 (0.9450–0.9531)
  on reviews written from 26 May 2017 to 3 August 2017.
- **What it means.** That is a different checkpoint, trained on different data. It is not a
  measurement of this model, nor an estimate of it. It shows only that scores can be lower on
  reviews from another period.

## Cost

- **Size:** 66,955,010 parameters, 255 MiB of weights.
- **Speed.** The timings come from the related checkpoint above. It has the same architecture and
  size as this model.
  - 57 ms per review (median, one review at a time,
    one thread of an Intel(R) Xeon(R) CPU @ 2.00GHz);
  - 254 reviews per second on a Tesla T4.
- **Training time:** 13.6 minutes
  on a Tesla T4.

## Limits

- **Data.** English Booking.com reviews of European hotels, written from 4 August 2015
  to 3 August 2017.
- **Two classes.** Reviews with both positive and negative text were left out of training and
  test.
- **Test set.** Each test set holds at most 10,000 positive reviews.
- **One training seed (42).** The interval covers the sampling of test reviews, not
  the randomness of training.

## Training settings

`distilbert-base-uncased`, seed 42, 2 epochs, batch
32, maximum length 256, learning rate 2e-05,
warm-up 6%, weight decay 0.01, fp16 training. The
checkpoint kept is the epoch with the best dev macro-F1.

## Provenance

- **Training.** Trained by notebook 10 of the project, from commit
  [`994a103`](https://github.com/krimits/hotel-review-nlp/commit/994a103574d07833d466e9553680a9cab46bfe76). Its metrics, logits and logs are
  [committed](https://github.com/krimits/hotel-review-nlp/tree/21d535ceafb6d1b95d4747d806c6153c2471718d/docs/experiments/results/distilbert_v2/random/distilbert). The CI recomputes its scores from those logits. The
  [decision note](https://github.com/krimits/hotel-review-nlp/blob/21d535ceafb6d1b95d4747d806c6153c2471718d/docs/experiments/decision_distilbert_vs_nb.md) compares it with Naive Bayes.
- **Weights:** `model.safetensors` SHA-256 `f2a2bbc9a31c8f3b1d9bee46ec11bb3d8942e7e59c0ef4711c3cb2a6c1cf81c2`.
- **Check before publishing.** An agreement check on 1,024 examples: the weights reproduced the saved test logits
  within an absolute tolerance of 0.001. The largest difference was
  3.6e-06, and 1,024 of 1,024
  predictions were equal.
- **Publishing code:** commit [`21d535c`](https://github.com/krimits/hotel-review-nlp/commit/21d535ceafb6d1b95d4747d806c6153c2471718d) of the project
  (`scripts/publish_model.py`).
- **The previous model** is kept under the tag
  [`legacy-split-v1`](https://huggingface.co/krimits/distilbert-hotel-reviews/tree/legacy-split-v1) (commit
  `9fe2f7f`). It was trained on an earlier split whose test set shared 170 texts
  with training.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("krimits/distilbert-hotel-reviews")
model = AutoModelForSequenceClassification.from_pretrained("krimits/distilbert-hotel-reviews")
inputs = tokenizer("The room was spotless and the staff was wonderful.", return_tensors="pt")
print(model.config.id2label[model(**inputs).logits.argmax(-1).item()])  # positive
```
