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

**Status: not run.** On 30 September one request was refused at
authentication, because the key belonged to another provider (see the end of
DECISION_v2.md). No answer exists yet. The texts can go to TypeSafe or through
OpenRouter (`--route`). Without `--allow-external-api` the script sends
nothing, and the project owner first checks the data terms of each provider.
