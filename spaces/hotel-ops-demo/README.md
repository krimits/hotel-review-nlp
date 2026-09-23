---
title: Hotel Review Operations Demo
emoji: 🏨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
python_version: "3.11"
models:
  - Qwen/Qwen2.5-0.5B-Instruct
---

# Hotel Review Operations · demo

Try English hotel reviews one at a time, inspect quoted aspect findings and
session-scoped complaint recommendations. Uses the base Qwen2.5-0.5B-Instruct
model and the parsing and recommendation code from the pinned
[`hotel-review-nlp` commit](https://github.com/krimits/hotel-review-nlp/tree/b153785776562d323bf7dbd7ffcb77fac35c01a7).

This is a demonstration, not a verified production hotel system. Aspect
accuracy has not been measured against human annotations. Data is held in
temporary Gradio session state and is not durable. Avoid personal guest data.
Greek hotel reviews are not supported by this ABSA demo.

Source and development history: https://github.com/krimits/hotel-review-nlp/pull/16

To create this private Space from a machine authenticated as `krimits` with
`hf auth login`, run `python scripts/deploy_hf_space.py` from the repository
root. If the Space was created but an interrupted upload needs retrying, run
`python scripts/deploy_hf_space.py --resume`. This command never reads or
uploads the project's training/test datasets or hotel guest records.
