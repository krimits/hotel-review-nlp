# Real smoke run: 3 October 2026

The uploaded archive ran the code at
[7da4eb8](https://github.com/krimits/hotel-review-nlp/commit/7da4eb83fff576c33a7a1ba492f29b0c0531fc07)
on a Colab T4. The original result files are preserved byte for byte;
execution.log is stored as execution.txt because logs are ignored by git.
[manifest.json](manifest.json) records their SHA-256 values.

| Arm | Reviews | Jev successful | Qwen wrote text | Accepted actions |
|---|---:|---:|---:|---:|
| without-jev | 14 | disabled | 6 | 0 |
| with-jev | 14 | 14 | 9 | 0 |

The script exited 0 and both arms completed. This confirms execution, not the
quality of suggestions. No action passed the unchanged parser in either arm.
Invalid JSON roots, empty measures, missing departments, and empty action lists
appear in [results.jsonl](results.jsonl). Some output copied sentiment metadata
into a measure; whether removing that metadata helps is an untested hypothesis.

Jev additionally routed mix-1, mix-3, and mix-4, compared with sentiment
alone. These are authored synthetic cases, not independent gold labels and not
evidence of a general recall gain. Jev's topic differences remain observations
requiring human review.

The run used the published DistilBERT at
7306aebcaaebc00d579f5d0a91001ae376f18158, base
Qwen/Qwen2.5-0.5B-Instruct without an adapter, actions-v1, and Jev resolved to
typesafe/jev-1.13-20260917. The Qwen weights revision was not recorded by this
historical script; the new comparison pins it.

No Jev billing cost was recorded. The first Qwen call included loading, so the
smoke timings are not a latency benchmark.

These fourteen texts remain outside prompt/model tuning. The next experiment
uses [new development and reserved cases](../../triage_generator/README.md).
