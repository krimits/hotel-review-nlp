# Decision: do hoteliers accept the urgent flags the triage system raises on their own guests' reviews?

**18 November 2026. The rule of the [hotel pilot protocol, version 1](hotel_pilot/PROTOCOL.md), applied as written,
once, to the records exported when the owner closed the collection. Scope: the judgement of five hotels the owner
recruited, on reviews they chose to bring, not blind; the outcome is a recommendation the owner decides on, and no
model is promoted.**

## Question

When each hotel sets which problem types are urgent for it, do hoteliers accept the urgent flags the system raises
on their own guests' reviews: at least 80% of them?

The system is `anthropic/claude-haiku-5.5-20261007` through OpenRouter with `teacher_prompt_v2.md` and
`catalogue_v2`, run live in the Space `krimits/hotel-triage-pilot`. An urgent flag is a type the system reports when
that type is in the hotel's urgent set; it is accepted when the hotelier marks it "right and urgent".

## Primary comparison

A hotel qualifies with at least 30 rated reviews and 10 rated urgent flags.

- **`acc`**, the primary: the mean, over the qualifying hotels, of each hotel's share of urgent flags accepted.
- **`worst`**: the lowest of those shares.
- **`miss`**: the mean share of a hotel's rated reviews where it reports a missing urgent problem.

Rule, decided only with at least 5 qualifying hotels: `acc >= 0.80 and worst >= 0.60 and miss <= 0.10` Accepted;
`acc < 0.60` Not accepted; otherwise Partly accepted. Computed from the counts as exact fractions.

## Result

| Hotel | Rated reviews | Rated urgent flags | Accepted | Share accepted | Missing urgent reported |
| :--- | ---: | ---: | ---: | ---: | ---: |
| P01 | 71 | 10 | 7 | 0.700 | 4 (0.056) |
| P02 | 45 | 10 | 10 | 1.000 | 0 (0.000) |
| P03 | 52 | 11 | 11 | 1.000 | 1 (0.019) |
| P04 | 31 | 15 | 11 | 0.733 | 0 (0.000) |
| P05 | 36 | 15 | 15 | 1.000 | 2 (0.056) |
| **Five qualifying hotels** | 235 | 61 | 54 | `acc` **0.8867**, `worst` **0.70** | `miss` **0.0262** |

- **The run is decided:** all five hotels qualify, the minimum the owner set.
- **The order held:** the protocol with the owner's fields (`b8295d6`), the Space (`acba649`, with retries towards
  the Hub only from `2530207`), and the record of the export (`6870e9c`) were committed before the rule was applied,
  once. No verdict share was computed before the record; counts of reviews and flags were, to tell the owner when a
  hotel qualified.

Source: `hotel_pilot/results/result.json` (SHA-256 `841670d1…`), from the 479 exported files recorded by their
digest (`ca4f8274…`) in `records/record.json` (`5512834b…`).

## Decision

**Accepted.** `acc` = 0.8867 ≥ 0.80, `worst` = 0.70 ≥ 0.60 and `miss` = 0.0262 ≤ 0.10, so the first branch fires.

The protocol's recommendation for this outcome, which the owner decides on: **develop the service commercially on
this system, starting with a paid pilot.**

**Read it with the first limit below:** the outcome rests on urgent flags from reviews that two hotels added one at
a time, after they were told they lacked flags. On their whole batches alone, those two hotels do not qualify, and
the run would not be decided.

## Reported, not deciding

| Urgent type | Accepted | Present, not urgent | No such problem |
| :--- | ---: | ---: | ---: |
| Cleanliness | 23 | 1 | 0 |
| Serious plumbing | 12 | 0 | 0 |
| Air conditioning and heating | 9 | 1 | 3 |
| Pests | 5 | 0 | 0 |
| Broken locks and doors | 3 | 0 | 1 |
| Injury risks and fire safety | 1 | 0 | 0 |
| Safety incidents | 1 | 0 | 1 |

- **Every hotel kept all seven urgent types on.** The per-hotel urgent set, the main change from the blind
  evaluation, was never used: no hotel switched a type off.
- **By script:** Greek reviews 46 flags, 0.913 accepted; Latin-script reviews 15 flags, 0.800 accepted.
- **Rejections:** 7 of 61 flags, 5 of them "no such problem" (the system seeing a problem that is not there,
  mostly air conditioning) and 2 "present, not urgent".
- **Missing urgent problems reported:** cleanliness 3, air conditioning 3, other 1, in 7 of 235 reviews.
- **Other problems and usefulness:** "right" in 231 of 235 reviews, "partly" in 4, never "no"; mean usefulness from
  4.92 to 5.00 of 5 per hotel. P03 (52 of 52) and P05 (36 of 36) gave "right" and 5 of 5 to every review.
- **Analysed, never rated:** 6 reviews, all P02's.
- **Cost and time:** $0.118 per 1,000 reviews through OpenRouter; median 9.81 seconds per call (up to 25 reviews).

### How the outcome depends on the reviews added one at a time (exploratory, not deciding)

| Hotel | Whole batches only: reviews / flags / accepted | Added one at a time: reviews / flags / accepted |
| :--- | :--- | :--- |
| P02 | 41 / 7 / 7 | 4 / 3 / 3 |
| P03 | 43 / 6 / 6 | 9 / 5 / 5 |

- P01, P04 and P05 brought whole batches only.
- On whole batches alone, P02 and P03 fall below 10 flags, three hotels qualify, and the rule gives "not decided".
- Over those three hotels the mean share accepted is 0.811 and the lowest 0.70: above the thresholds, but on three
  hotels where the owner set five.

## Scope and limits

- **The deciding flags include ones the hotels chose to bring.** The protocol asks for the most recent reviews, all
  of them, without choosing (§2), and cannot check it. P02 and P03 reached 10 flags only with reviews added one at a
  time after the count was short: 3 of P02's 4 and 5 of P03's 9 carried an urgent flag, against 6 of 41 and 4 of
  43 reviews in their batches, and all 8 were accepted. Asked how P02's 4 were picked, the owner declared the hotel complete
  without saying.
- **12 of P03's 52 reviews are identical in text to reviews of P01,** and 1 to a review the owner used in a test
  under code P06. The owner cannot explain it. One of those 12 carried an urgent flag; it was accepted.
- **Who rated is the owner's report, not verified:** each hotel's own hotelier, on its own guests' reviews. The
  Space cannot tell who types a code.
- **Not blind:** the hoteliers knew the system is the owner's product. Two hotels gave every review the highest
  answers. Their acceptance may be kind.
- **Five hotels the owner recruited,** not a sample of hotels; no interval is claimed over hotels. Hotels count
  equally, from 31 to 71 rated reviews each.
- **Missing urgent problems are counted only when a hotelier notices them;** `miss` is a floor, not a recall.
- **The model was checked before only on English Booking.com reviews from 2015–2017.** Here 46 of the 61 flags came
  from Greek reviews; the pilot is the first test on them, by these hotels.
- **No review text is kept,** so a wrong analysis cannot be read again; only its types and the verdict remain.
- **The records are not in this repository.** They stay in the private dataset and the owner's export; the
  repository holds one digest and the aggregates. Only the owner can recompute the result.
- **Not a comparison:** with the student, with another LLM, or with the blind evaluation, where two hoteliers
  accepted the same system's urgent triage on 59% of other hotels' reviews. The two runs differ in reviews,
  blinding, raters and question, so the gap is not a measured improvement.
- **The service depends on OpenRouter and Anthropic.** The model can change or be withdrawn, and each review's text
  is sent to them.
- No model is promoted by this note; the outcome is a recommendation the owner decides on.

## Artifacts

| What | Where | SHA-256 |
| :--- | :--- | :--- |
| Protocol, version 1 | `docs/experiments/hotel_pilot/PROTOCOL.md` (`b8295d6`) | `66f7c727…` |
| Result | `hotel_pilot/results/result.json` | `841670d1…` |
| Record of the export | `hotel_pilot/records/record.json` (`6870e9c`) | `5512834b…` |
| The 479 exported files | private dataset `krimits/hotel-triage-pilot-records` and the owner's export; not committed | digest `ca4f8274…` |
| Prompt, catalogue, Greek labels | `teacher_prompt_v2.md`, `catalogue_v2.yaml`, `labels_el.yaml` | `f731026f…`, `5c592af1…`, `d0f992b9…` |
| The Space | `spaces/hotel-pilot/`, data bundle `56e2e448a86b` | |
| Run log | `docs/RUNS.md`, entries of 11 to 18 November 2026 | |
