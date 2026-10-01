# Decision note, version 2: Jev against the complaint lexicon

**Status: fixed on 30 September 2026, before any request was sent.** This
note replaces the first draft, [DECISION.md](DECISION.md), which stays as it
was committed. No request was sent under the draft: its sign-off says the
first real call was still waiting for an API key. The draft says that a
changed rule goes in a new file, so this is that file. If this rule changes,
it will change in another new file.

## The question

The [pilot](../../case_study/results/annotation/README.md) measured the
lexicon that names the complaint topics. It finds only 2 of the 16
responsiveness complaints in the random texts. Its precision is about a half
or less for cleanliness and responsiveness.

Does TypeSafe's Jev find more responsiveness complaints than the lexicon, on
the same texts, when it is asked the labelling guideline's questions? And
are at least half of the texts it flags real complaints?

This run cannot put Jev in the demo. The pilot's 300 texts are now a test
set, so if Jev passes, the next step is to confirm the result on a new
sample.

## Fixed before any request

- **Texts.** The pilot's 200 random texts, 100 per period (`in_R` in
  [key.csv](../../case_study/results/annotation/key.csv)). The other 100 were
  picked because the lexicon matched them, so they would favour its recall.
  They are not scored, and they are not sent at all. No text is chosen by its
  label.
- **Gold.** The pilot's final labels
  ([labels_final.csv](../../case_study/results/annotation/labels_final.csv)).
  They are the project owner's labels. A second annotator labelled 80 of the
  texts, and all 7 disagreements were settled on the owner's label. A text
  whose final label is `unsure` for a topic is left out for that topic. Among
  the random texts, that is 1 text, for cleanliness.
- **Lexicon.** Its outputs for the same texts: the `lex_*` columns of the
  key.
- **Jev.** `jev-latest`, via `POST https://api.typesafe.ai/v1/systemone`.
  - **One request per text,** sent in item order. Each topic is a `choice`
    question with the labels `1`, `0` and `unsure`.
  - **The rules come from the
    [labelling guideline](../../annotation/complaint_topics_guideline.md)
    only.** None of them was tuned on the pilot's texts, and no Jev answer
    existed when they were written. The short examples inside the
    guideline's rules are sent; its closing table of made-up examples is
    not.
  - **The questions are locked by a hash.** The model name and questions
    together have the SHA-256
    `96462e1e7c03e823d860742124eea948173418bb98336dc0f3a063633d619754`,
    and a test checks the script against it.
  - **The model the API answers with is recorded,** because `jev-latest` is
    an alias.
- **What counts as found.**
  - **Jev** finds a complaint when it answers `1`. An `unsure`, a `0` or a
    missing answer counts as not found.
  - **The lexicon** finds a complaint when it matches.

## The decision rule

**Primary endpoint: recall of responsiveness complaints,** on the 16 random
texts labelled 1.

- **Count the discordant texts.** Let b be the texts only Jev finds, and c
  the texts only the lexicon finds.
- **Jev is better** if b > c and the exact two-sided McNemar test gives
  p < 0.05, that is, a binomial test of b out of b + c with p = 0.5. For
  example, if Jev also finds the lexicon's 2, it needs 6 more texts
  (p = 0.031); 5 more give p = 0.063. So in practice Jev has to find about
  half of the 16.
- **Guard.** At least half of the random texts Jev flags for responsiveness
  must be labelled 1: precision of at least 0.5. Without the guard, a tool
  that flagged every text would find all 16.
- **If both hold,** Jev goes to a confirmation on a new sample, as a second
  detector for responsiveness beside the lexicon, before any use in the
  demo.
- **Otherwise,** the lexicon stays and the numbers are recorded. A clean
  negative is also a finding.
- **The rule applies only to a complete run:**
  - all 200 texts answered;
  - no `--limit`;
  - texts read from the handed-in sheet A, whose SHA-256 is in
    [agreement.json](../../case_study/results/annotation/agreement.json).

  Otherwise, the summary says "not decided" and gives the reason.

## Reported, not deciding

- **Precision and recall with Wilson intervals,** for both systems and all
  five topics. A denominator below 10 is shown as counts only, as in the
  pilot. Cleanliness precision matters most after responsiveness, because it
  is the lexicon's other weak point.
- **Paired false positives:** on the texts labelled 0, the false positives
  only Jev makes, and those only the lexicon makes.
- **How often Jev answers `unsure`,** and its recall if every `unsure` counted
  as found.
- **Calibration:** the expected calibration error (10 bins) of the
  probability Jev gives the label it chose. The API also returns a separate
  `confidence`, but the error is measured on the probability, which is what
  it is defined on.
- **Time per text** (median and 95th percentile), measured on the machine
  that runs the script, network included.
- **Tokens,** and the cost per 1,000 texts at a price given when the script
  is run. The API documents output tokens as free at present.

These numbers are small. A topic has 2 to 18 positives among the random
texts (pests has 2), so an interval spans ±0.15 to ±0.3. That is why only
one endpoint decides.

## Data and safety

- **The review texts never enter the repository.** They come from a local
  sheet. Everything a run writes stays under `runs/`, which git ignores.
- **Nothing is sent without `--allow-external-api`.** Without the flag, the
  script checks the inputs and writes the first request to `runs/` so it can
  be read.
- **The project owner checks the provider's data terms before using the
  flag:** whether inputs are kept and for how long, whether they are used for
  training, and where they are processed. The texts are guest reviews and
  may name people.
- **The API key is read from `TYPESAFE_API_KEY` and never written
  anywhere.**
- **After the run, these may be committed:** `summary.json`,
  `predictions.csv` (ids, labels, probabilities) and `responses.jsonl` (the
  API's answers, timings and token counts). None of them holds a text.

## One run

- **A run that stops resumes.** If the network fails or a rate limit is hit,
  the answers already received are kept. Running the same command again sends
  only the texts not yet answered. The script refuses to mix answers to
  different questions or inputs.
- **An answer in an unexpected form stops the run.** If the API's answer is
  not in the documented form, the script stops after that text. Only the code
  that reads the answer may then change. The change is recorded in a new
  note, and the saved answers are scored again with `--score-only`, without
  new requests.

## What changed from the first draft, and why

| The draft ([DECISION.md](DECISION.md)) | Now | Why |
|---|---|---|
| 100 of the 300 texts, with each topic's positives drawn first | The 200 random texts | Drawing by label raises the share of positives: bathroom was 38% of the draft's sample, and it is 6.5% of the random texts. So precision comes out too high. A third of the 300 texts were picked by the lexicon, which favours its recall. |
| Fixed thresholds: mean kappa ≥ 0.70; precision ≥ 0.85 and recall ≥ 0.70 per topic | Jev against the lexicon on the same texts, with one primary endpoint | The question is whether Jev beats what is in use. With 2 to 18 positives per topic, a point estimate above a threshold shows little. |
| Calibration error ≤ 0.10, from `confidence` | Reported, from the chosen label's probability | The API returns the probabilities apart from `confidence`. |
| Cost ≤ 0.50 EUR per 1,000 reviews, "above that the DistilBERT path dominates" | Reported | DistilBERT labels sentiment, not topics, so it is not an alternative here. |
| Median time ≤ 3,000 ms | Reported | Time matters for a live demo, not for this question. |
| Baseline "near chance" | The pilot's measured numbers | The pilot measured precision of about 0.9 for air conditioning and 0.75 for bathroom. |
| Gold from "the adjudicated sheet_A" | `labels_final.csv` | Sheet A holds the owner's labels. The final labels equal them because every disagreement went the owner's way. The committed file holds ids and labels only. |
| `key.csv` never opened | The key gives the random texts and the lexicon's outputs | The key is committed. Jev sees only the text and the questions. |
| `unsure` described as an abstention but counted as a miss | A miss for the rule, with a bound where it counts as found | One stated rule. |
| Retries on 429 and 529 only, and an error ended the run with nothing saved | Retries on 408, 429 and 5xx, waiting as the server asks; answers saved as they arrive | As the provider's SDK does it. |
| The API was called whenever a key was set, and output could go anywhere | `--allow-external-api` is required, and output goes under `runs/` | The data terms are checked first, and no text goes into the repository. |
| Some general rules of the guideline missing from the questions | Added: mild complaints and wishes count; neutral mentions and settings do not; bathroom is 1 only if something besides dirt is criticised; `unsure` is not for mild complaints | The annotators had these rules. |
| Lint errors, so CI failed on `main` | Fixed | Ruff flagged `zip()` without `strict=` and a missing final newline. |

## Sign-off

Written on 30 September 2026, at the project owner's request, before any
request was sent. The script that carries it out, and the offline tests of
that script, were committed together with this note.

## Changes after locking

**1 October 2026: a second route to the same model.** The project owner's API
key is an OpenRouter key, which TypeSafe's own API refuses. The first attempt,
on 30 September, sent one request (item 1) to `api.typesafe.ai` with that key.
It was refused at authentication, before any answer, so no answer exists and
nothing was scored.

OpenRouter serves the same System One endpoint. The script now takes
`--route typesafe` (as before) or `--route openrouter`
(`https://openrouter.ai/api/v1/systemone`, key from `OPENROUTER_API_KEY`).
Those are the only two places the texts can go.

- **Unchanged:** the texts, the gold, the questions (the hash above), the model
  name `jev-latest`, the decision rule and its guard. The route is not part of
  the rule.
- **Recorded:** the route and the endpoint go into the run's record, next to the
  model that answers. Through OpenRouter, that model is reported under
  OpenRouter's own id. A run cannot be resumed on another route or model.
- **Keys:** a key is sent only to its own provider, and a redirect is not
  followed.
- **A second party:** OpenRouter handles the texts as well as TypeSafe. Its
  data settings are checked, with the provider's terms, before the flag is
  used.
- **Not tested against the real service:** the OpenRouter request was written
  from its published description, and the environment that wrote it cannot
  reach OpenRouter. A first run of 3 texts (`--limit 3`) shows whether the
  answers come back in the documented form. If they do not, the run stops after
  that text, and the saved answer shows what differs.
