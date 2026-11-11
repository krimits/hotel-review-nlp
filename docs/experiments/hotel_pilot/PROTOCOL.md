# Hotel pilot of the triage Space: protocol — version 1, DRAFT

**Status: draft, 11 November 2026, awaiting the owner's approval.** It is locked only by the commit that adds its
lock test. Nothing in it has been built, published or rated.

**Why.** The [hoteliers' blind evaluation](../hotelier_blind_eval/PROTOCOL.md) closed with **Neither ready**
([decision](../decision_hotelier_blind_eval.md)): two hoteliers judged the teacher's urgent triage right on 59% of
the reviews with an urgent flag, below the 80% floor, and found no difference between the teacher and the student.
Most objections were to flags they did not consider urgent, and the two often disagreed with each other. The owner
then decided, on 11 November 2026:

- **the model for an experimental pilot is the LLM** (`claude-haiku-5.5`), because the hotels' reviews are in
  several languages and the student was trained on English only;
- **each hotel sets which problem types are urgent for it**, instead of one definition for all;
- **the pilot stores the hoteliers' judgements, not the review texts.**

**This selection is the owner's decision at a human gate, not a result.** No rule in this repository found the LLM
ready, and none found it better than the student in hoteliers' eyes. It is selected for an experimental pilot; it
is not promoted.

## 1. Question

**When each hotel sets which problem types are urgent for it, do hoteliers accept the urgent flags the system
raises on their own guests' reviews: at least 80% of them?**

- **The system:** `anthropic/claude-haiku-5.5-20261007` through OpenRouter, with
  [`teacher_prompt_v2.md`](../triage_student_v2/teacher_prompt_v2.md) and the structured output of the triage
  student v2 protocol's teacher, unchanged; the types, departments and measures of
  [`catalogue_v2`](../triage_diagnosis/catalogue_v2.yaml).
- **An urgent flag:** a type the system reports for a review, when that type is in the hotel's own urgent set at
  that moment.
- **Accepted:** the hotelier marks the flag "right and urgent".

**Scope of any claim:** the judgement of the hotels that took part, on reviews they chose to bring. Not production
accuracy, and not a comparison with any other system.

## 2. Data

### The hotels and their reviews

- **Hotels:** recruited by the owner. Each takes part with a code the owner gives it and appears only as a
  pseudonym (`P01`, `P02`, …). No hotel or person is named anywhere.
- **Reviews:** each hotel pastes or uploads its own guests' reviews, in any language, as they were published. The
  Space asks for the most recent ones, all of them, without choosing.
- **What leaves the Space:** each review's text is sent to OpenRouter and Anthropic for the analysis. The Space
  says so before the first review, and asks the hotel not to add anything a guest did not publish.
- **What is never stored:** the review text, the hotel's name, the access code.

### The urgent set

At its first visit a hotel sees `catalogue_v2`'s seven urgent types (cleanliness, pests, broken locks and doors,
injury risks and fire-safety faults, safety incidents, air conditioning and heating, serious plumbing), all
switched on, and switches off those it would not fix before the next arrival. It can change the set later; every
rating records the set in force.

A type the hotel switched off is shown as a routine problem, not an urgent one.

### What a hotelier sees and answers

The Space analyses the reviews in the order given, at most 25 per call, and shows them one at a time. The next
review appears only after the current one is rated, so a hotel cannot rate only the analyses it likes.

- **Per urgent flag — the primary:** right and urgent / the problem is there but is not urgent / there is no such
  problem.
- **Per review:**
  - is an urgent problem missing (no / yes, and which type);
  - are the other problems right (yes / partly / no);
  - usefulness, from 1 to 5.

### What is stored, in a private dataset

One record per rated review: the pseudonym, a salted hash of the text (to count a repeated review once), its
script (Greek, Latin or other), the types the system reported, the hotel's urgent set, the answers above, the
time, and the hashes of this protocol, the prompt and the app's data.

**The records are not committed to this repository.** As in the blind evaluation, the repository holds one digest
over them and aggregate results.

## 3. One primary comparison

A hotel **qualifies** when it has rated at least `R` reviews and at least `G` urgent flags.

- **`acc`**, the primary: the mean, over the qualifying hotels, of the hotel's share of urgent flags it accepted.
- **`worst`:** the lowest such share among the qualifying hotels.
- **`miss`:** the mean, over the qualifying hotels, of the share of the hotel's rated reviews where it reports a
  missing urgent problem.

Hotels count equally, whatever the number of reviews each brought.

## 4. Decision rule

The run is decided only when **at least `H` hotels** qualify; otherwise it is "not decided". The rule is applied
once, to the records exported when the owner closes the collection.

| Condition, in order | Outcome |
| :--- | :--- |
| `acc >= 0.80 and worst >= 0.60 and miss <= 0.10` | **Accepted.** The hotels accept the urgent flags. |
| `acc < 0.60` | **Not accepted.** |
| otherwise | **Partly accepted.** |

What each outcome recommends, for the owner to decide:

- **Accepted:** develop the service commercially on this system, starting with a paid pilot.
- **Partly accepted:** fix what the reported items point at (the rejected types, the missed ones, a language) and
  pilot again before any commercial step.
- **Not accepted:** do not build on this system as it is.

### Reported, not deciding

- Each hotel's `acc`, its `miss`, and the number of reviews and flags it rated.
- **Why flags were rejected**, per type: "not urgent" against "no such problem". The first is about what counts
  as urgent; the second is the system seeing a problem that is not there.
- The urgent sets the hotels chose: how many hotels kept each type on.
- `acc` by script (Greek, Latin, other). The system was only ever checked on English reviews.
- The types reported as missing.
- The other problems (yes / partly / no) and the usefulness, per hotel.
- Reviews analysed but never rated, per hotel.
- The hotels that did not qualify, apart.
- Cost and time: OpenRouter's cost per 1,000 reviews and the median seconds per call.

## 5. Limits

- **A few hotels the owner recruited,** not a sample of hotels. No interval is claimed over hotels.
- **Hotels choose what to bring.** The Space asks for the most recent reviews without choosing, and cannot check
  it. Rating in the order given removes only the choice of what to rate.
- **Hoteliers know the system is the owner's product** and are not blind; they may judge kindly.
- **Missing urgent problems are counted only when the hotelier notices them.** `miss` is a floor, not a recall.
- **The model was checked on English Booking.com reviews from 2015–2017 only.** Other languages and other review
  formats are new to it; the pilot is their first test.
- **The service depends on OpenRouter and Anthropic.** The model can change or be withdrawn, and each review's
  text is sent to them.
- **No comparison:** with the student, with another LLM, or with a hotel's own triage.
- **No text is kept,** so a wrong analysis cannot be read again later; only its types and the hotelier's verdict
  remain.
- No model is promoted by this protocol; the outcome is a recommendation the owner decides on.

## 6. Order of work and human gates

| Step | What | Human gate |
| :--- | :--- | :--- |
| 1 | Lock with its lock test | the owner approves this draft and its owner fields |
| 2 | Build the Space: codes, the urgent set, analysis through OpenRouter, the rating screen, records to a private dataset; tests with a fake API | — |
| 3 | Publish the Space; set its secrets (OpenRouter key with a spending limit, dataset token, codes); recruit the hotels | publish, secrets, recruiting |
| 4 | The hotels use and rate; the owner closes the collection and exports the records | the owner |
| 5 | The records recorded by their digest; the rule applied once; decision note | — |

## 7. Fields

| Field | Value | Set by |
| :--- | :--- | :--- |
| Model | `anthropic/claude-haiku-5.5-20261007`, `teacher_prompt_v2.md`, at most 25 reviews per call | owner (11 November 2026) |
| Urgent set | per hotel, from `catalogue_v2`'s seven urgent types, all on at first | owner (11 November 2026) |
| Stored | judgements without review texts, in a private dataset | owner (11 November 2026) |
| `H`, hotels that must qualify | 5 | **owner field, proposed** |
| `R`, rated reviews for a hotel to qualify | 30 | **owner field, proposed** |
| `G`, rated urgent flags for a hotel to qualify | 10 | **owner field, proposed** |
| Thresholds | `acc >= 0.80`, `worst >= 0.60`, `miss <= 0.10`; not accepted below `acc` 0.60 | **owner field, proposed** |
| Cap per hotel | 300 reviews analysed (about $0.02 at the measured $0.07 per 1,000 reviews) | **owner field, proposed** |
