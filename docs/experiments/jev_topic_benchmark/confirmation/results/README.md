# Confirmation of Jev's responsiveness result: results

The run followed the [locked protocol](../../../../annotation/confirmation_protocol.md).
The project owner made it on 2 October 2026, from their own machine, with the
committed [script](../../../../../scripts/benchmark_jev_topics.py), which has not
changed since the protocol was locked.

**Outcome under the locked rule: confirmed: Jev finds more responsiveness
complaints on new texts too, and at least half of its flags are right.**

## What was run

- **Texts:** the 400 new random texts, 200 per period, labelled by the owner
  before any system saw them. Each was sent on its own, with the five questions
  of the first comparison. Only responsiveness is scored, because the other four
  topics have no labels here.
- **Order:** the labels, the key and the manifest were committed at 08:14 UTC on
  2 October. The first request was sent at 08:39 UTC. A
  [test](../../../../../tests/test_jev_confirmation_results.py) checks that
  order from the git history.
- **Route:** OpenRouter's System One endpoint. The model asked for was
  `jev-latest`. It answered as `typesafe/jev-1.13-20260917`, the version that
  answered in the pilot, and each answer names TypeSafe as its provider. So the
  texts went to OpenRouter and to TypeSafe.
- **One run:** 400 requests, from 08:39:05 to 08:51:28 UTC, in item order. Every
  one was answered on the first attempt, and none was retried or unreadable. The
  saved log holds no other request. It has one pause of 534 seconds, after item
  252, and it does not say why. Every other gap is 1 second or less. The script
  resumes a stopped run only if the texts, labels, key, questions, route and
  model are the same as before.
- **Complete:** 400 of 400 answered, and `--limit` was not used. The summary
  records the hashes of the script, the sheet of texts (the sheet the labels came
  back on), the key, the labels and the questions. The
  [test](../../../../../tests/test_jev_confirmation_results.py) checks that each
  matches the committed file or the script locked with the protocol, and that the
  whole summary is reproduced from the saved answers.

## The locked rule

Among the 33 texts labelled 1, Jev found 19 of the 33 and the lexicon 2.

| | Found | Of the 33 responsiveness complaints |
|---|---|---|
| Lexicon | 2 | 0.06 (Wilson 95%: 0.02 to 0.20) |
| Jev | 19 | 0.58 (0.41 to 0.73) |

- **Discordant texts:** 17 found only by Jev, none found only by the lexicon.
  The lexicon's 2 are among Jev's 19. The exact two-sided McNemar p-value is
  1.5 × 10⁻⁵ (2 / 2¹⁷), against the rule's 0.05. The summary shows 0.0 because
  it rounds p-values to four decimals.
- **Guard:** 19 of the 26 texts Jev flagged are labelled 1, so precision is 0.73
  (0.54 to 0.86). The guard asks for at least 0.5. It holds on the estimate, and
  the lower end of the interval is also above it.
- **Left out:** the 5 texts labelled `unsure`, as the protocol says, so 395 are
  scored. Jev answered `1` on two of the five and `0` on three. These are not
  scored.

## Compared with the pilot

The rule judges the new texts alone. The last column pools the two samples, which
the protocol allows as a description only. No rule uses it.

| | Pilot, 200 texts | Confirmation, 400 texts | Both together |
|---|---|---|---|
| Labelled 1 | 16 | 33 | 49 |
| Jev recall | 13 of 16: 0.81 (0.57 to 0.93) | 19 of 33: 0.58 (0.41 to 0.73) | 32 of 49: 0.65 (0.51 to 0.77) |
| Lexicon recall | 2 of 16: 0.125 (0.04 to 0.36) | 2 of 33: 0.06 (0.02 to 0.20) | 4 of 49: 0.08 (0.03 to 0.19) |
| Jev precision | 13 of 23: 0.57 (0.37 to 0.74) | 19 of 26: 0.73 (0.54 to 0.86) | 32 of 49: 0.65 (0.51 to 0.77) |
| Lexicon precision | 2 of 6 | 2 of 8 | 4 of 14 |

The recall estimate is lower than in the pilot and the precision estimate is
higher. The intervals overlap, so this does not show that either changed, and
nothing here says why the estimates differ. Both samples agree that Jev finds
several times as many of these complaints as the lexicon, and that some of its
flags are wrong.

## Other measurements

- **False alarms on the texts labelled 0:** 6 flagged only by Jev, 5 only by the
  lexicon and 1 by both (exact McNemar p = 1.0). So Jev's 7 wrong flags are the 6
  and the 1, and the lexicon's 6 are the 5 and the 1.
- **Per period:**
  - In the base period there are 15 complaints. Jev found 9 and the lexicon 0.
    Jev's flags were right 9 times in 11, and the lexicon's 0 times in 3.
  - In the recent period there are 18. Jev found 10 and the lexicon 2. Jev's flags
    were right 10 times in 15, and the lexicon's 2 times in 5.
  - That is too few flags to say whether Jev's precision differs between the
    periods (9 of 11 against 10 of 15). It matters for trend work, where a change
    in precision looks like a change in complaints.
- **`unsure` was never used:** 0 of the 2,000 answers (five questions on 400
  texts), as in the pilot (0 of 1,000). Jev gives no abstention to route to a
  person. Only the probabilities are left.
- **Probabilities:** the calibration error in responsiveness is 0.017. The 19
  right flags have a probability for `1` between 0.49 and 1.00 (median 0.93), and
  the 7 wrong ones between 0.54 and 0.88 (median 0.62). The label with the highest
  probability is the answer, so a flag can have a probability for `1` below 0.5.
  The probability separates right and wrong flags only partly, as in the pilot.
  This was looked at after the answers came in, and no threshold was chosen on
  these texts.
- **Time:** the median is 0.55 s per text and the 95th percentile 0.65 s, for the
  five questions together. It was measured on one machine and includes the
  network, so it is not the service's processing time.
- **Tokens and cost:** 824,179 input tokens and 78,000 output tokens. OpenRouter
  reports a cost of 0.0346 (US dollars) for the 400 texts, which is 0.087 per
  1,000 texts. In every request, the cost equals the input tokens at 0.042 per
  million, so output tokens add nothing. These are its figures, not checked
  against an invoice.

## What it shows, and what it doesn't

**Shows:** on these 400 new random texts and these labels, Jev finds far more of
the responsiveness complaints the lexicon misses, and chance does not explain the
difference. At least half of its flags are right. The rule fixed beforehand is
met, so the pilot's result repeats in direction.

**Doesn't show:**
- **How much it finds.** Jev found 19 of 33, with an interval of 0.41 to 0.73. It
  missed 14, about two complaints in five. The pilot's 0.81 did not repeat.
- **That its flags can be trusted.** Seven of its 26 flags were wrong, about one
  in four. The bar of 0.5 was this confirmation's. It does not say that a hotel
  owner would accept that many false alarms. That is decided separately, before
  any use.
- **Anything about other topics, whole reviews or `/triage`.** Both runs used the
  negative field and the first comparison's questions, and the confirmation scored
  responsiveness alone. The `/triage` endpoint asks other questions of whole
  reviews, and those have not been measured
  ([TRIAGE.md](../../../../TRIAGE.md)).
- **That the labels are the last word.** There was one annotator, the project
  owner. The only evidence on how reliable the labels are is the pilot's: kappa
  0.83 (0.63 to 0.96) for responsiveness, on the 79 texts both annotators labelled
  1 or 0.
- **A version that stays.** `jev-latest` is an alias. This run got
  `jev-1.13-20260917` again, and a later one may get another.
- **That it is fit for the demo.** The protocol says a pass does not put Jev in
  the demo. The hotel's guest reviews would leave the demo for two companies, the
  demo would depend on a vendor and on a model version, and the false-alarm rate a
  hotel owner would accept has to be decided. That needs a separate decision.

## Files

Three files, as the script wrote them. Only the line endings changed, from CRLF
to LF, because of the repository's `.gitattributes`. The SHA-256 of each as
received is in the test. None holds a review text or a key. The labels, the key,
the manifest and the sheet's hash are one folder up.

- `summary.json`: the numbers above, with the hashes of everything that went
  in, the decision and the timings.
- `predictions.csv`: per text and topic, the label Jev chose, the probability of
  that label, its probability for `1` and its `confidence`. All five topics are
  there, and only responsiveness is scored.
- `responses.jsonl`: the API's answers as received, with the time of each
  request. The answers carry two fields the provider's SDK does not list: an
  OpenRouter generation `id` and the `provider`. `usage` also carries a `cost`.
