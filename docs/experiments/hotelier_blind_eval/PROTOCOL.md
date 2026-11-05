# Hoteliers' blind evaluation of the triage teacher and student: protocol — version 1, DRAFT

**Status: draft, 5 November 2026, awaiting the owner's approval.** It is locked only by the commit that adds its
lock test. Nothing in it has been drawn, shown to anyone or rated.

**Why.** The [triage student v2 protocol](../triage_student_v2/PROTOCOL.md) closed with **Not met**
([decision](../decision_triage_student_v2.md)): against ChatGPT's labels, the student (variant B) caught 0.842 of
the urgent problems on its 500 test reviews and its teacher 0.949. Both figures measure agreement with an LLM.
The owner chose, on 5 November 2026, to let hoteliers judge the two systems instead: **one Space, blind, with the
ratings recorded in the Space**, and to decide from their judgement which system the product goes forward with.

## 1. Question

**On reviews where the two systems flag urgent problems, do hoteliers, judging blind, find the student's urgent
triage correct nearly as often as the teacher's: no more than 10 points less often?**

- **Teacher:** `anthropic/claude-haiku-5.5-20261007` with `teacher_prompt_v2.md`.
- **Student:** variant B of the v2 protocol, with its locked thresholds.
- **Catalogue:** both are read through [`catalogue_v2`](../triage_diagnosis/catalogue_v2.yaml).
- **The judgement asked:** whether the urgent part of the output is right for the review. An urgent problem is
  one that must be fixed before the next guest arrives in the room.

**Scope of any claim:** the judgement of the hoteliers who took part, on public 2015–2017 English reviews of other
hotels. Not production accuracy, not every hotelier's view.

## 2. Data

### Reviews and outputs: already committed, nothing new is run

The reviews are the v2 protocol's 500 test reviews (`../triage_student_v2/samples/key.csv`). Both systems' outputs
on them are already committed, and neither system was tuned on them:

- **The teacher's labels:** `labels/teacher_haiku_test.csv`.
- **The student's emissions:** variant B's logits `student/logits_B_test.csv`, with its locked thresholds
  `student/thresholds_locked.json` (urgent 0.80, routine 0.90, structural 0.45).

The ChatGPT reference labels of the same reviews are used only for a reported item (§4). Hoteliers never see them.

Because the outputs are computed once from committed files:

- every hotelier sees the same outputs;
- the Space needs **no API key and no model weights**;
- no review text is sent anywhere.

### Groups and the rated set (60 reviews)

Among the 500, an **urgent output** is the set of urgent types a system emits.

| Group | Definition | Reviews | Rated |
| :--- | :--- | ---: | ---: |
| **D**, differ | the two urgent outputs differ | 49 (H 33, N 16) | **all 49** |
| **C**, same | the same urgent output, not empty | 97 (H 75, N 22) | **11** (H 8, N 3), seed **20261106** |
| **Z**, none | neither system emits an urgent type | 354 | not shown |

- **U = D ∪ C** (146 reviews) is where at least one system flags something urgent.
- **The draw of the 11:** per stratum, sorted by `raw_row`, `DataFrame.sample(n, random_state=20261106)`.
- **Each rated review carries a weight:** its test weight `N_s / n_s`, times `C_s / S_s` for a review of C (the
  stratum's size in C over the number drawn from it).
- **What is committed, without text:** the rated set's key and both systems' outputs, the latter as the types each
  system shows. Both files are hashed before anything is shown.

### What a hotelier sees and answers

- **The display.** For each of the 60 reviews, in a random order per hotelier, the hotelier sees the review text
  (English) and the two outputs side by side as **"Σύστημα 1"** and **"Σύστημα 2"**.
  - The order of the two systems is random per (hotelier, review). It is fixed by a hash of the hotelier's code
    and the item, and recorded.
  - Each output lists its problems as the catalogue shows them: type, department, urgent or not, and the
    catalogue measure. They are rendered the same way for both systems, with a fixed Greek label table committed
    with the Space.
- **Per system — Q1, the primary (yes or no):** *"Τα επείγοντα είναι σωστά; Σημειώνει ως επείγοντα ακριβώς όσα
  πρέπει να διορθωθούν πριν φτάσει ο επόμενος επισκέπτης, στο σωστό τμήμα;"*
- **Per system — Q2 (secondary):** if Q1 is no, what is wrong: a missed urgent problem, a false urgent flag, or a
  wrong department.
- **Per system — Q3 (secondary):** are the non-urgent problems right (yes, partly, no)?
- **Per system — Q4 (secondary):** usefulness, from 1 to 5.
- **Per review, after both:** which output the hotelier would rather act on (1, 2 or the same).
- **The hoteliers.** Each takes part with a code the owner gives them (`H01`, `H02`, …). The Space accepts only
  those codes, keeps no names, and records one rating per (code, review): the first submission, never replaced.
  The ratings go to a private Hugging Face dataset. The optional profile is the hotel's type and size class.

## 3. One primary comparison

For each review *i* in D and each **complete** hotelier *h* (one who rated all 60 reviews on Q1):

- `s_ih` and `t_ih` are 1 when the hotelier answered yes on Q1 for the student's and the teacher's output.
- The review's difference is `diff_i` = the mean over complete hoteliers of `s_ih − t_ih`.
- A review of C has the same urgent output from both systems, so its difference is 0 by construction.

- **`d`**, the primary: the IPW mean over U of the difference, student minus teacher:
  `d = Σ_{i∈D} w_i diff_i / Σ_{i∈U} w_i`, with the test weights.
- **`lo_d` and `hi_d`:** the 2.5th and 97.5th percentiles of `d` over 2,000 bootstrap resamples (seed 0) of the U
  reviews within strata. The hoteliers are not resampled.
- **`p_teacher`:** the IPW share of U whose teacher output the hoteliers judge right on Q1. D is counted in full;
  C is estimated from its 11 rated reviews with their weights.

## 4. Decision rule

The run is decided only when **at least 3 hoteliers** are complete; otherwise it is "not decided". The rule is
applied once, to the ratings exported when the owner closes the collection.

| Condition, in order | Outcome |
| :--- | :--- |
| `p_teacher < 0.70` | **Neither ready.** The hoteliers judge even the teacher's urgent triage right on fewer than 70% of U. |
| `lo_d >= -0.10` | **Student non-inferior.** Within 10 points of the teacher. |
| `hi_d < -0.10` | **Student inferior.** More than 10 points behind. |
| otherwise | **Not established.** The student is not shown to be within 10 points. |

What each outcome recommends for the product, as a recommendation the owner decides on:

- **Student non-inferior:** build on the student. It is free to run, keeps reviews in-house, and needs no API.
- **Student inferior, or Not established:** build on the teacher. The student has to show it is within 10 points;
  the teacher does not.
- **Neither ready:** revisit the catalogue and the triage before building on either.

### Reported, not deciding

- `p_student` and `p_teacher` on D and on C.
- Q2 reasons, Q3, Q4 and the preference, per system.
- Agreement between hoteliers on Q1: pairwise and Fleiss' kappa.
- `d` leaving out each hotelier in turn.
- **The identical-output check:** how often a hotelier judged the two identical outputs of a C review differently.
- **Hoteliers against ChatGPT:** how often a hotelier's Q1 agrees with whether that output matches the
  `reference_chatgpt_v2` urgent departments. This shows what agreement with the LLM is worth to a hotelier.
- **The partial raters:** the ratings of hoteliers who did not complete, reported apart.

## 5. Limits

- **Few hoteliers, chosen by the owner,** not a sample of hoteliers. The intervals cover the sampling of reviews,
  not of hoteliers.
- **Other hotels' reviews,** public, 2015–2017, English, written after checkout, judged by hoteliers who did not
  host those guests.
- **The rated set is built where the systems differ.** Rates on D are not rates on all reviews; population
  figures are over U, with weights.
- **Problems both systems miss are not assessed:** Z is not shown. On these 500 reviews, the ChatGPT reference
  has an urgent problem in 1 of the 354.
- **The outputs are precomputed.** A live Space with the teacher behind an API (latency, cost, privacy) or the
  student on a CPU is not evaluated here.
- **The blinding can be imperfect** if one system's style gives it away, for example a habit of flagging more
  types.
- No selection is frozen and no model is promoted by this protocol.

## 6. Order of work and human gates

| Step | What | Human gate |
| :--- | :--- | :--- |
| 1 | Lock with its lock test | the owner approves this draft and its owner fields |
| 2 | Draw the 11; write the rated set's key, both outputs and the Greek label table; commit and hash them | — |
| 3 | Build the Space: the blind display, codes, ratings to a private dataset; tests | — |
| 4 | Publish the Space; set the dataset token as a Space secret; give each hotelier a code | publish, secret, recruiting |
| 5 | The hoteliers rate; the owner closes the collection and exports the ratings | the owner |
| 6 | The rule applied once; decision note | — |

## 7. Fields

| Field | Value | Set by |
| :--- | :--- | :--- |
| Systems | teacher `claude-haiku-5.5` labels and student v2 variant B with its locked thresholds, on the v2 test reviews | owner (5 November 2026) |
| Design | one Space, blind, side by side, ratings in the Space | owner (5 November 2026) |
| Rated set | all 49 of D and 11 of C (H 8, N 3, seed 20261106) | builder |
| Margin | 10 points (`-0.10`) | **owner field, proposed** |
| Floor | `p_teacher >= 0.70` | **owner field, proposed** |
| Minimum complete hoteliers | 3 | **owner field, proposed** |
| Bootstrap | 2,000 resamples, seed 0, reviews within strata | builder |
