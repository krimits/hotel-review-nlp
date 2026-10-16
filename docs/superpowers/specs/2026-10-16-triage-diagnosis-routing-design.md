# Triage: diagnosis, routing and remedial measures — design

**16 October 2026. Status: design, awaiting the owner's approval. Nothing here is built or measured.**

## Intent

A hotel's complaints manager reads every review through the Hugging Face Space and approves,
edits or rejects what the app proposes: a human in the loop, not an automatic system. The app's
job is to make that person fast and safe. For each review it must:

1. **diagnose** the problems the guest reports, one by one;
2. **route** each problem to the department that owns it;
3. **propose a measure** the manager can approve as is.

The owner's own words fix what matters most: the costliest error is **a real complaint judged
wrongly, so that the responsible department never handles it.** A false alarm costs the manager a
few seconds; a missed complaint costs a guest and leaves the fault in place.

Today's triage is unvalidated: hand-picked thresholds, a narrow lexicon, and a small generator
whose measures were never accepted in the real-model smoke run (0 accepted actions).

## The user and the workload

- One complaints manager per hotel, working in the Space.
- About 80–100 reviews a day, mostly English from Booking.com, fewer in Greek and from Google.
- The manager cannot read every review closely every day, so the app must order them:
  urgent first.

## Goals

1. **Urgent complaints are not missed.** Primary success criterion, set by the owner:
   **recall of urgent problems ≥ 0.90**, measured on a locked, never-inspected test set. A problem
   counts as caught only if it is flagged urgent *and* sent to a correct department.
2. Every detected problem is routed to one of the six departments the hotel uses:
   `reception`, `housekeeping`, `maintenance`, `food_and_beverage`, `management`, `other`
   (confirmed by the owner).
3. Every operationally fixable problem gets a measure from a reviewed catalogue: a concrete
   action, an owner and a deadline.
4. The manager's approve / edit / reject decisions are recorded, so real use produces labels.

## Priority levels (owner's urgent list; the rest is my call)

| Level | Problems | What happens |
| :--- | :--- | :--- |
| **Urgent: today** | hygiene and cleanliness (dirt, stains, mould), pests, safety and security (locks, safe, injury risk, fire safety, theft), air conditioning or heating not working | flagged at the top; measure with a same-day deadline, e.g. "inspect the room before the next arrival" |
| Routine | fixable but not pressing: Wi-Fi, noise, breakfast, staff manner, plumbing that works badly, bed comfort, check-in or billing | routed to the department for its weekly review |
| Structural | not fixable by an operational action: room size, location, view, price | no fake measure; sent to management as information and counted for trends |

The owner's example sets the bar for a measure. "Replace the room's air-conditioning filter within
24 hours" is approved as is: a concrete action, an implied owner, a deadline. "The room is too
small" is rejected: it repeats the complaint and nobody can act on it.

Because Booking reviews are written after checkout, "urgent" means the fault probably still
affects the next guest in that room.

## Non-goals

- **Greek reviews** in this phase. The public dataset has no Greek reviews, so Greek cannot be
  evaluated on real data (owner agreed).
- Real reviews from a specific hotel. Only the public Booking.com dataset is available (owner).
- Acting without the manager. Nothing is sent to a department automatically.
- Free-text measures invented by a small model as the main path.
- Any production accuracy claim. The evidence is from a public dataset of European hotels,
  2015–2017.

## Anti-goals: it fails even if it technically works if

- an urgent problem passes unflagged or goes to the wrong department, and the numbers hide it
  (for example by excluding abstentions, or by scoring only reviews a lexicon matched);
- a structural complaint gets a made-up measure;
- the evaluation is tuned on the test set, or its labels are shaped by the model being tested;
- the README claims more than the locked test shows.

## Constraints

- **Data: the public Booking.com dataset only** (owner). Each review has the guest's own
  "liked" and "disliked" fields. Joined, they make a realistic whole review. The "disliked" field
  shows, for hundreds of thousands of reviews and with no labelling, whether a complaint exists and
  where its text is. It is noisy: "nothing", "all good", and complaints written in the "liked"
  field all occur.
- **Labelling: one person, about one hour a day, and no second person** (owner). At about
  40 seconds a review that is roughly 80 reviews a day, so about 600–1,000 in two weeks.
- **The repository's rules:** a protocol locked before any number exists, a decision rule fixed in
  advance, human gates for Colab, API keys and publishing, and everything versioned.
- **Serving:** the Space runs on a CPU (with ZeroGPU only where already used). Diagnosis must run
  there at 80–100 reviews a day without paid external calls.

## The approach

### 1. Taxonomy and measure catalogue (owner reviews once)

About 15–20 **problem types**, each with one department, one priority level and one catalogue
entry. The catalogue is a versioned file (`problem → department → priority → measure template with
deadline → what to confirm`). The owner reviews and edits it once; that review is the main
evidence that the measures are sensible.

- The urgent types follow the owner's list.
- One problem can involve two departments, e.g. "AC broken and reception sent nobody" →
  maintenance (AC) and reception (response). Diagnosis is **per problem**, not one label per review.
- Measures are **selected** from the catalogue by the diagnosed problem type, not generated
  freely. An LLM may later tailor wording to the review, but never chooses the measure.

### 2. Gold labels from the owner, kept honest without a second person

- Whole reviews drawn from the clean splits' **test** partition, never from model training data.
- **Sampling.** Reviews with complaint text are oversampled so that urgent problems are frequent
  enough. Intervals are reported per stratum, and population estimates are reweighted.
- **Labelling.** For each review the owner ticks the problem types present (multi-label), without
  seeing any model output.
- **Sets.**
  - A **development** set of about 300 reviews: used to choose thresholds and to improve the
    taxonomy.
  - A **test** set of about 500 reviews, locked before any model is scored on it.
- **Replacing the missing second person.**
  - (a) A strong LLM labels the same reviews independently, as a second rater. Agreement is
    reported, and only disagreements go back to the owner for a second look, before any model is
    scored.
  - (b) The owner relabels about 10% of reviews days later, blind, to measure self-consistency.
  - Both are stated as limits; neither equals a second human.
- **The second-rater LLM must differ from any LLM used to make training labels.** Otherwise the
  adjudicated gold would lean toward the model being tested.

### 3. Diagnosis model

Recommended path (my call; the protocol compares it against the baselines):

- **Teacher.** A strong LLM labels several thousand reviews from the training partition with
  problem types. This is a human gate: an API key and a cost, run by the owner.
- **Student.** DistilBERT fine-tuned as a multi-label classifier on those labels in Colab. It runs on
  the Space's CPU, needs no external call, and is fast enough for 100 reviews a day (the INT8 result
  applies).
- **Thresholds.** Chosen per problem type on the development set, biased toward recall for the
  urgent types, as the owner's priority requires. When unsure about an urgent type, the review is
  flagged for the manager, not dropped.
- **Baselines on the same locked test.** The current `complaint-lexicon-v1`, and the teacher LLM
  itself, which shows what the student loses by being small.

### 4. Space: the manager's loop

- The review list is ordered by priority, urgent first. For each review the app shows the
  diagnosed problems with the quoted evidence, the department and the catalogue measure.
- Approve, edit or reject buttons for each proposal. The decisions are written to a private Hugging
  Face dataset. That needs a write token in the Space's secrets: a human gate.
- Those decisions become new development labels over time. They never feed the locked test.

## Evaluation (to be locked in a protocol before any number)

- **Primary:** recall of urgent problems on the locked test (caught = flagged urgent and routed to a
  correct department).
  - The rule meets the owner's target when the point estimate is ≥ 0.90. The Wilson 95% interval is
    reported beside it.
  - The protocol must decide in advance whether the lower bound must also clear a value, given the
    test's urgent support.
- **Secondary:**
  - precision of urgent flags;
  - per-type recall and precision;
  - department accuracy on detected problems;
  - share of fixable problems whose catalogue measure matches the gold problem type;
  - the manager's approval rate (from Space use, reported separately as field evidence).
- **Comparisons:** student against lexicon and against teacher, paired on the same reviews.
- **Limits stated in advance:** one annotator, the LLM second rater, the public 2015–2017 European
  dataset, English only, and post-checkout reviews.

## Phases

| Phase | What | Human gate | Produces numbers |
| :--- | :--- | :--- | :--- |
| 0 | Taxonomy and measure catalogue; the owner reviews them | owner review | no |
| 1 | Protocol locked; labelling sheets; development and test drawn | owner labels, about 2 weeks | no (gold labels only) |
| 2 | Second-rater LLM on the gold sets; disagreements re-checked | API key | agreement only |
| 3 | Teacher labels training reviews; student trained in Colab | API key, Colab | development only |
| 4 | One scoring run on the locked test; decision note | — | yes (the decision) |
| 5 | Space: priority list, catalogue measures, approve/edit/reject logging | HF write token, `hf upload` | no |

## Left to the builder

- The exact problem-type list within the owner's frame. It must keep his urgent four.
- Sample sizes within the labelling budget. Power for urgent recall is the driver.
- The teacher and second-rater models, chosen for cost and independence.
- The labelling-sheet format, with the existing pilot tool as the model.
- The Space layout, within "urgent first" and per-proposal approve/edit/reject.

## Calls made by me, not the owner, for the owner to check

1. Measures are selected from a catalogue, not generated (inferred from his approve/reject example).
2. The routine and structural levels, and which problems fall in each.
3. A strong LLM as an imperfect second rater, and the rule that it differs from the teacher.
4. A teacher-student design: a distilled DistilBERT in the Space, not an LLM called per review.
5. The development/test sizes, about 300 and 500.
