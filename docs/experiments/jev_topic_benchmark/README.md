# Jev against the complaint lexicon

A comparison, fixed before it is run, of TypeSafe's Jev with the lexicon
that names the complaint topics. It uses the pilot's 200 random texts.

- [DECISION_v2.md](DECISION_v2.md): the design and the decision rule in
  force.
- [DECISION.md](DECISION.md): the first draft, kept as it was committed. It
  was replaced before any request was sent; version 2 says why.
- The script:
  [`scripts/benchmark_jev_topics.py`](../../../scripts/benchmark_jev_topics.py).
  It is tested offline, with a fake API, in
  [`tests/test_benchmark_jev_topics.py`](../../../tests/test_benchmark_jev_topics.py).

**Status: not run.** The project owner first checks the provider's data
terms. Until then, the script is only run without `--allow-external-api`,
and it sends nothing.
