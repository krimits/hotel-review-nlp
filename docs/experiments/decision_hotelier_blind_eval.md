# Decision: do hoteliers, judging blind, find the student's urgent triage right nearly as often as the teacher's?

**11 November 2026. The rule of the [hoteliers' blind evaluation protocol, version 1](hotelier_blind_eval/PROTOCOL.md),
applied as written, once, to the ratings recorded before the rule was run. Scope: the judgement of two hoteliers
the owner chose, on 60 public 2015–2017 English reviews of other hotels, with both systems' outputs computed in
advance; no selection is frozen and no model is promoted.**

## Question

On reviews where the two systems flag urgent problems, do hoteliers, judging blind, find the student's urgent
triage correct nearly as often as the teacher's: no more than 10 points less often?

The teacher is `anthropic/claude-haiku-5.5-20261007` with `teacher_prompt_v2.md`; the student is variant B of the
[triage student v2 protocol](triage_student_v2/PROTOCOL.md) with its locked thresholds; both are read through
`catalogue_v2`.

## Primary comparison

Each hotelier saw each review with the two outputs side by side as "Σύστημα 1" and "Σύστημα 2", and answered for
each whether its urgent part is right (Q1, yes or no).

- **`d`**: the IPW mean, over the 146 reviews where either system emits an urgent type (U), of the difference in
  the share of hoteliers answering yes, student minus teacher. It is 0 by construction where the two urgent outputs
  are the same.
- **`lo_d`, `hi_d`**: the 2.5th and 97.5th percentiles of `d` over 2,000 bootstrap resamples (seed 0) of the U
  reviews within strata.
- **`p_teacher`**: the IPW share of U whose teacher output the hoteliers judge right on Q1.

Rule, in order, decided only with at least 2 complete hoteliers: `p_teacher < 0.80` Neither ready;
`lo_d >= -0.10` Student non-inferior; `hi_d < -0.10` Student inferior; otherwise Not established.

## Result

| On U, 146 reviews, two complete hoteliers | Value |
| :--- | ---: |
| `p_teacher`, teacher's urgent triage judged right (IPW) | **0.5873** |
| `d`, student minus teacher (IPW) | +0.0130 |
| `lo_d`, bootstrap 2.5th percentile | −0.0579 |
| `hi_d`, bootstrap 97.5th percentile | +0.0812 |

- **The run is complete:** both hoteliers rated all 60 reviews (the 49 where the urgent outputs differ and 11 of
  the 97 where they are the same); there are no partial raters.
- **The order held:** the protocol with the owner's fields (`ce4b0af`), the rated set and both outputs
  (`e36b023`), and the record of the ratings (`9ca07b2`) were committed before the rule was applied, once
  (`f354eaa`).

Source: `hotelier_blind_eval/results/result.json` (SHA-256 `045d0343…`), from the 120 ratings recorded by their
digest (`01f2ce30…`) in `ratings/record.json` (`443e7027…`).

## Decision

**Neither ready.** `p_teacher` = 0.5873 is below 0.80, so the first branch fires: the hoteliers judge even the
teacher's urgent triage right on fewer than 80% of U.

The protocol's recommendation for this outcome, which the owner decides on: **revisit the catalogue and the triage
before building on either system.**

The first branch fires before the comparison of the two systems is read, so the rule does **not** conclude that
the student is non-inferior. Reported only: the interval of `d` lies wholly above −0.10.

## Reported, not deciding

| Per system, both hoteliers | Teacher | Student |
| :--- | ---: | ---: |
| Urgent part judged right, the 49 reviews where the outputs differ (unweighted) | 0.520 | 0.571 |
| Urgent part judged right, the 11 rated reviews where they are the same | 0.727 | 0.727 |
| "Not right" ratings, of 120 | 53 | 48 |
| … because of a false urgent flag | 35 | 35 |
| … because of a missed urgent problem | 16 | 9 |
| … because of a wrong department | 2 | 4 |
| Non-urgent problems right: yes / partly / no | 45 / 63 / 12 | 47 / 64 / 9 |
| Usefulness, mean of 1 to 5 | 3.41 | 3.49 |
| Preferred, of 120 (the same: 34) | 45 | 41 |

- **The main objection is urgency itself.** Two thirds of the teacher's "not right" ratings and three quarters of
  the student's name a false urgent flag: something marked urgent that the hotelier would not fix before the next
  arrival. Both systems apply `catalogue_v2`, where every problem of an urgent type is urgent. Which types the
  hoteliers objected to is not in this result; the reasons were counted per rating, not per type.
- **The two hoteliers often disagree.** Fleiss' kappa on Q1 over the 120 outputs is 0.33. Alone, on the 49 reviews
  where the outputs differ, H01 puts the student 12 points ahead (+0.122) and H02 puts it 2 points behind (−0.020).
- **They answered consistently.** Where the two urgent outputs of a review were identical, no hotelier judged one
  right and the other not (0 of 22).
- **Agreement with ChatGPT says little about a hotelier's judgement.** A hotelier's Q1 agrees with whether that
  output's urgent departments are the ChatGPT reference's in 0.5875 of the 240 ratings. That reference was the
  yardstick of the two earlier triage protocols.

## Scope and limits

- **Two hoteliers, chosen by the owner,** the minimum the owner set. The result is those two people's judgement,
  and their agreement is low. The intervals cover the sampling of reviews, not of hoteliers.
- **Who rated is the owner's report, not verified:** two different hoteliers, each giving all 60 answers
  themselves, none by the owner, no AI tool as far as the owner knows. The Space cannot tell who types a code.
  From the record: H01 rated from 6 to 9 November, and H02's first answer came five minutes after H01's last.
- **The ratings are not in this repository.** The Space told the hoteliers their answers are kept in a private
  dataset, and the owner decided on 11 November to keep that. The repository holds one digest over the 120 files;
  only the owner can recompute the result. This departs from the repository's rule that every evaluation artifact
  is versioned here.
- **Q1 is strict:** one yes or no on the whole urgent part, "exactly what must be fixed before the next guest, in
  the right department". One flag too many makes the answer no.
- **Other hotels' reviews,** public, 2015–2017, English, written after checkout, judged by hoteliers who did not
  host those guests.
- **The rated set is built where the systems differ.** Figures over U use weights; the part of `p_teacher` that
  comes from the reviews with the same urgent output rests on 11 of them.
- **Problems both systems miss are not assessed:** the 354 reviews where neither flags anything urgent were not
  shown.
- **The outputs were precomputed.** A live service with the teacher behind an API, or the student on a CPU, was
  not evaluated. The blinding can be imperfect if one system's style gives it away.
- **What the result does not say.** It does not say the triage is useless: the hoteliers rated usefulness 3.4 to
  3.5 of 5 and found the non-urgent problems right or partly right in nine ratings of ten. It says that these two
  hoteliers do not accept the urgent flags as they stand, for either system.
- `selection_frozen` and `promoted` stay false. No Space change follows.

## Artifacts

| What | Where | SHA-256 |
| :--- | :--- | :--- |
| Protocol, version 1 | `docs/experiments/hotelier_blind_eval/PROTOCOL.md` (`ce4b0af`) | `6aa74caf…` |
| Result | `hotelier_blind_eval/results/result.json` (`f354eaa`) | `045d0343…` |
| Record of the ratings | `hotelier_blind_eval/ratings/record.json` (`9ca07b2`) | `443e7027…` |
| The 120 ratings | private dataset `krimits/hotelier-blind-eval-ratings` and the owner's export; not committed | digest `01f2ce30…` |
| Rated set and both outputs | `hotelier_blind_eval/samples/rated_key.csv`, `outputs.json` (`e36b023`) | `bdc98b58…`, `7424b85b…` |
| Greek display labels | `hotelier_blind_eval/labels_el.yaml` | `d0f992b9…` |
| The Space | `spaces/hotelier-blind-eval/`, data bundle `4c1e63b9d2c3` | |
| Run log | `docs/RUNS.md`, entries of 5 to 11 November 2026 | |
