# Jev against the lexicon: results

The run followed [DECISION_v2.md](../DECISION_v2.md). The project owner made
it on 1 October 2026, from their own machine, with the committed
[script](../../../../scripts/benchmark_jev_topics.py).

**Outcome under the locked rule: Jev finds more responsiveness complaints than
the lexicon. The rule says to confirm this on a new sample before any use.**

## What was run

- **Texts:** the pilot's 200 random texts, 100 per period, sent one by one,
  five questions each.
- **Route:** OpenRouter's System One endpoint. The model asked for was
  `jev-latest`. It answered as `typesafe/jev-1.13-20260917`, and each answer
  names TypeSafe as its provider. So the texts went to OpenRouter and to
  TypeSafe.
- **Order of events:**
  - On 30 September, one request (item 1) went to TypeSafe's own API with an
    OpenRouter key and was refused at authentication. No answer exists.
  - On 1 October at 08:48 UTC, three texts (items 1, 2 and 4) went through
    OpenRouter as a smoke run.
  - From 08:50 to 08:52 UTC, the other 197 went. The three smoke answers are
    part of the 200, and nothing changed between the two commands.
- **Complete:** 200 requests, no retry, 200 answers in the documented form.
  The summary records the hashes of the script, the sheet of texts (the
  handed-in sheet A), the key, the labels and the questions. A
  [test](../../../../tests/test_jev_results.py) checks that each matches the
  committed file, and that the whole summary is reproduced from the saved
  answers.

## The locked rule

| | Found | Of the 16 responsiveness complaints |
|---|---|---|
| Lexicon | 2 | 0.125 (Wilson 95%: 0.04 to 0.36) |
| Jev | 13 | 0.81 (0.57 to 0.93) |

- **Discordant texts:** 11 found only by Jev, none found only by the lexicon.
  The lexicon's 2 are among Jev's 13. The exact McNemar p-value is 0.001,
  against the rule's 0.05.
- **Guard:** 13 of the 23 texts Jev flagged are labelled 1, so precision is
  0.57 (0.37 to 0.74). The guard asks for at least 0.5. It holds on the
  estimate, and the interval reaches below it.

## All five topics

Reported, not deciding. Each cell is the count, the estimate and the Wilson 95%
interval.

| Topic | Labelled 1 | Lexicon recall | Jev recall | Lexicon precision | Jev precision |
|---|---|---|---|---|---|
| Bathroom | 13 | 12 of 13: 0.92 (0.67 to 0.99) | 13 of 13: 1.00 (0.77 to 1.00) | 12 of 16: 0.75 (0.51 to 0.90) | 13 of 15: 0.87 (0.62 to 0.96) |
| Cleanliness | 11 | 9 of 11: 0.82 (0.52 to 0.95) | 10 of 11: 0.91 (0.62 to 0.98) | 9 of 15: 0.60 (0.36 to 0.80) | 10 of 13: 0.77 (0.50 to 0.92) |
| Air conditioning | 18 | 16 of 18: 0.89 (0.67 to 0.97) | 17 of 18: 0.94 (0.74 to 0.99) | 16 of 17: 0.94 (0.73 to 0.99) | 17 of 19: 0.89 (0.69 to 0.97) |
| Pests | 2 | 2 of 2 | 2 of 2 | 2 of 2 | 2 of 2 |
| Responsiveness | 16 | 2 of 16: 0.125 (0.04 to 0.36) | 13 of 16: 0.81 (0.57 to 0.93) | 2 of 6 | 13 of 23: 0.57 (0.37 to 0.74) |

- **Counts only** where the denominator is below 10, as in the pilot. The
  cleanliness row leaves out the one text whose final label is `unsure`.
- **The differences outside responsiveness are within the noise.** Each topic
  has 11 to 18 positives. For bathroom, cleanliness and air conditioning, the
  texts only one system finds are 1 to 2 against 0 to 1, and the McNemar p-value
  is 1.0 each time.
- **False alarms on the texts labelled 0:** in responsiveness, 8 texts flagged
  only by Jev and 2 only by the lexicon (p = 0.11). In bathroom it is 1 only by
  Jev against 3 only by the lexicon, and in cleanliness 2 against 5.

## Other measurements

- **Per period, in responsiveness:** of 8 complaints in the base period, Jev
  found 6 and the lexicon 0. Of 8 in the recent period, Jev found 7 and the
  lexicon 2. Jev's flags were right 6 times in 14 in the base period and 7 times
  in 9 in the recent one. That is too few to say whether its precision differs
  between periods, and the recent period is below the pilot's threshold of 10.
  It matters for trend work, where a change in precision looks like a change in
  complaints.
- **`unsure` was never used:** 0 of the 1,000 answers. Jev gives no
  abstention to route to a person. Only the probabilities are left.
- **Probabilities:** the calibration error is 0.007 over all answers and
  between 0.002 and 0.035 by topic. Almost every answer is `0` with probability
  near 1, so this says little about doubtful texts. In responsiveness, the 13
  right flags have a probability for `1` between 0.71 and 1.00, and the 10 wrong
  ones between 0.56 and 0.93. The probability separates them only partly. This
  was looked at after the answers came in, and no threshold was chosen on these
  texts.
- **Time:** the median is 0.60 s per text and the 95th percentile 0.92 s, for
  the five questions together. It was measured on one machine and includes the
  network, so it is not the service's processing time.
- **Tokens and cost:** 411,382 input tokens and 39,000 output tokens. OpenRouter
  reports a cost of 0.0173 (US dollars) for the 200 texts, which is 0.086 per
  1,000 texts. In every request, the cost equals the input tokens at 0.042 per
  million, so output tokens add nothing. These are its figures, not checked
  against an invoice.

## What it shows, and what it doesn't

**Shows:** on these 200 random texts and these labels, Jev finds far more of the
responsiveness complaints the lexicon misses, and chance does not explain the
difference. The rule fixed beforehand is met.

**Doesn't show:**
- **How it does on other texts.** There are 16 positives, so the recall
  interval spans 0.57 to 0.93. The 300 texts are now a test set, so nothing is
  tuned on them. Confirmation needs new random texts, labelled without seeing
  either system's answers, under the same rule fixed beforehand.
- **That its flags can be trusted.** Ten of the 23 were wrong, so on these texts
  about four alerts in ten were false.
- **That the labels are the last word.** They are the project owner's, with a
  second annotator on 80 of the 300 texts, and the questions come from the
  owner's guideline.
- **A version that stays.** `jev-latest` is an alias. This run got
  `jev-1.13-20260917`, and a later one may get another.
- **That it is fit for the demo.** The hotel's guest reviews would leave the
  demo for two companies, and the demo would depend on a vendor.

## Files

Three files, as the script wrote them. Only the line endings changed, from CRLF
to LF, because of the repository's `.gitattributes`. The SHA-256 of each as
received is in the test. None holds a review text or a key.

- `summary.json`: the numbers above, with the hashes of everything that went
  in, the decision and the timings.
- `predictions.csv`: per text and topic, the label Jev chose, the probability of
  that label, its probability for `1` and its `confidence`.
- `responses.jsonl`: the API's answers as received, with the time of each
  request. The answers carry two fields the provider's SDK does not list: an
  OpenRouter generation `id` and the `provider`. `usage` also carries a `cost`.

The calibration errors differ from a rerun on Python 3.11 in the 16th digit,
because Python 3.12 sums floats more exactly. Every count, interval and p-value
is the same.
