# Jev against the complaint lexicon

A comparison, fixed before it was run, of TypeSafe's Jev with the lexicon
that names the complaint topics. It uses the pilot's 200 random texts.

- [DECISION_v2.md](DECISION_v2.md): the design and the decision rule in
  force.
- [DECISION.md](DECISION.md): the first draft, kept as it was committed. It
  was replaced before any request was sent; version 2 says why.
- [results/README.md](results/README.md): what the run found, with its files.
- [The confirmation protocol](../../annotation/confirmation_protocol.md): how the result was to be checked on
  400 new random texts, labelled for responsiveness alone. It was fixed before the sample was drawn.
- [confirmation/results/README.md](confirmation/results/README.md): what the confirmation found, with its files.
- The script:
  [`scripts/benchmark_jev_topics.py`](../../../scripts/benchmark_jev_topics.py).
  It is tested offline, with a fake API, in
  [`tests/test_benchmark_jev_topics.py`](../../../tests/test_benchmark_jev_topics.py).

**Status: run on 1 October 2026, confirmed on 2 October 2026.** Under the rule
fixed beforehand, Jev finds more responsiveness complaints than the lexicon: 13
of 16 against 2, with 10 of its 23 flags wrong. The rule said to confirm that on
a new sample before any use. The confirmation, on 400 new texts under a protocol
locked before they were drawn, met the same rule: 19 of 33 against 2, with 7 of
Jev's 26 flags wrong. That is not a licence to use Jev. Its recall estimate was
lower than in the pilot, nothing is known about the other topics or about whole
reviews, and any use needs a separate decision on the data, the cost and the
false alarms a hotel owner would accept. The texts went through OpenRouter to
TypeSafe. The script sends nothing without `--allow-external-api`, and the
project owner checks the data terms of each provider first.
