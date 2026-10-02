# Triage: sentiment, complaint topics and suggested actions

**Status: experimental and unvalidated.** Every response says `validation_status: "unvalidated"` and
`thresholds_status: "provisional"`. The workflow runs, and its parts are tested with fakes. No stage has been
measured on whole or mixed reviews, and the thresholds were chosen by hand.

## What it does

`POST /triage` takes one review and chains three stages:

1. **Sentiment.** The app's own model (`MODEL_TYPE`), with its full probability distribution. Required.
2. **Complaint topics.** Jev (TypeSafe System One), asked six questions about the whole review: the five topics
   of the [labelling guideline](annotation/complaint_topics_guideline.md) and `other_complaint`. Off by default.
3. **Suggested actions.** The base Qwen2.5-0.5B-Instruct, no adapter. Off by default, and asked only when the
   routing rules below say so.

Only the sentiment stage is required. If Jev or Qwen is off, the response says `disabled` for that stage. If one
fails, the response is partial and names the kind of failure (for example `timeout` or `http_429`), never the
provider's message.

Code: [`src/reviewnlp/triage/`](../src/reviewnlp/triage/) and
[`serving/triage_router.py`](../src/reviewnlp/serving/triage_router.py).

## Switching the stages on

Nothing is called, and nothing leaves the server, unless these are set.

| Variable | Meaning |
|---|---|
| `REVIEWNLP_JEV_ENABLED` | `1`, `true` or `yes` turns the Jev stage on. |
| `REVIEWNLP_JEV_ROUTE` | `typesafe` (default) or `openrouter`. |
| `TYPESAFE_API_KEY` / `OPENROUTER_API_KEY` | The key of the chosen route. Read from the environment only. |
| `REVIEWNLP_JEV_MODEL` | Default `jev-latest`. The version that answers is recorded. |
| `REVIEWNLP_JEV_TIMEOUT_S`, `_DEADLINE_S`, `_MAX_ATTEMPTS` | Per call, in total, and attempts (10, 20, 3). |
| `REVIEWNLP_TRIAGE_QWEN_ENABLED` | `1`, `true` or `yes` turns the Qwen stage on. |
| `REVIEWNLP_TRIAGE_QWEN_MODEL`, `_DEVICE` | Optional. Default `Qwen/Qwen2.5-0.5B-Instruct`. |

The Qwen weights load on the first review that needs them, not at start-up.

## Data handling

- The hotel's API key is checked **before** any model runs and before anything leaves the server.
- With Jev on, **the review text is sent to the chosen provider.** That makes the provider a processor of guest
  text. The owner checks the provider's terms (retention, use for training) before switching it on. The same
  holds for OpenRouter and the provider behind it.
- The key is never logged, never in a response, and never in the repository. A failed call is recorded by its
  kind only.
- Redirects are not followed, so a key is not sent to another host.
- With a database configured, a result is stored **without the review text.** The stored row holds a SHA-256 of
  the text, the stage results, and the short excerpts the suggested actions quote (at most 400 characters
  each), which are checked to be words of the review. Deleting a review through the existing delete endpoint
  also deletes its triage rows.

## Routing

Qwen is asked for actions when any of these holds. The rules are in
[`routing.py`](../src/reviewnlp/triage/routing.py), and every reason is in the response.

| Reason | When |
|---|---|
| `negative_sentiment` | The sentiment label is negative. |
| `uncertain_sentiment` | Its probability is below 0.80. Also flags the review for a person. |
| `complaint_detected` | Jev answered `1` for any topic. A positive review with a complaint is asked for. |
| `uncertain_complaint` | Jev answered `unsure`, or answered `0` with a probability of `1` in 0.20 to 0.50. Also flags the review for a person. |

If the complaint stage failed, a confidently positive review cannot be told from a positive review with a
complaint, so the review is flagged (`complaint_check_failed`).

**The thresholds are provisional.** 0.80 and the band 0.20–0.50 are not the result of any measurement. The
evaluation below is where they get chosen.

## The questions are not the benchmark's

The [benchmark](experiments/jev_topic_benchmark/results/README.md) and its confirmation tell Jev that the text
is what a guest wrote when asked what they did not like. For a whole review that is wrong, since it can mix
praise and complaints. The triage questions (`full-review-v1`, with their own SHA-256 in every response) say
that the text is a whole review and that praise for other things does not cancel a complaint.
The rules for each topic are the guideline's, in the benchmark's words; a test checks that they have not
drifted.

The benchmark's result therefore does **not** carry over. The pilot measured the negative field, reporting five
topics and deciding on one (responsiveness). Its [confirmation](experiments/jev_topic_benchmark/confirmation/results/README.md)
on 400 new texts scored responsiveness alone and met the same rule. Jev on whole reviews, with these questions,
has not been measured.

## What checks the Qwen output, and what cannot

The model is small, so its output is checked in code before it is shown:

- it must be JSON in the asked-for form; a malformed entry is dropped, not repaired;
- the `excerpt` must be words of the review (whitespace and case aside), or the action is dropped;
- the `department` must be on the closed list (`reception`, `housekeeping`, `maintenance`,
  `food_and_beverage`, `management`, `other`), or the action is dropped, not coerced to `other`;
- at most five actions, no duplicates; `to_confirm` lists what the review does not state.

The response also says how many entries were dropped and whether the token budget was hit.

**Code cannot check whether a suggestion is useful or right.** An action can quote the review and still be a bad
measure. Only people reading them can say. Nothing here is a claim about quality.

## Storage and the dashboard

With `REVIEWNLP_DB_PATH` set, `POST /triage` stores the result under the hotel and the review id, and
`GET /hotels/{hotel_id}/triage/summary?days=30` reads it back. The dashboard (`/dashboard`) has four views on it:

- **Συνολικό συναίσθημα**: the number of reviews per sentiment label;
- **Εντοπισμένα παράπονα**: per topic, how many reviews Jev answered yes, no and unsure (and a note when the
  complaint stage did not run);
- **Προτεινόμενα μέτρα**: each suggested action with its problem, the quoted excerpt, the measure, the
  department and what to confirm;
- **Χρειάζεται έλεγχος**: reviews a person should look at, with the reasons.

The dashboard also has a button to run triage on a pasted review. The tables are additive (`triage_runs`,
`triage_complaints`, `triage_actions`), so an existing database keeps working. The section carries a notice that
the flow is unvalidated.

## What has not been done

- **The independent evaluation.** This is the plan's last step, and it is not done. It needs:
  1. a **development** sample, to choose the thresholds and the prompt, and a separate **final** sample that no
     choice was made on, drawn at random and not by any lexicon;
  2. labels from a person on **whole** reviews, including mixed ones (praise and complaint together), since
     those are what the workflow is for;
  3. a rule written and locked **before** the final run, as the
     [pilot](annotation/pilot_protocol.md) and the
     [confirmation](annotation/confirmation_protocol.md) did: the endpoints, the counts, the intervals;
  4. a person's judgement of the suggested actions: is the problem real, is the measure sensible, is the
     department right. That is the only evidence on their value.
- **Real runs.** DistilBERT, Jev and Qwen have not been run together here: this environment has no weights, no
  keys and no access to the model hub. The tests use fakes for all three, and the dashboard was checked in a real
  browser against those fakes.
- **Sentiment on mixed reviews.** The sentiment model was trained to say positive or negative. A review that is
  both has no right answer from it, which is why the complaint stage exists.
- **Calibration.** The probability Jev gives for `1` is not shown to be calibrated.

Until the evaluation is done, the output is for a person to read, not to act on.
