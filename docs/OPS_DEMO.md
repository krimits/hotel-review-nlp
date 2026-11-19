# Operations demo: review → confirmed findings → tickets → reply

An experimental, unvalidated demo of the product the hotel pilot pointed at. It runs inside the existing FastAPI app
and dashboard, for one test hotel, with reviews pasted in by hand and one notification channel (test e-mail).
Code: `src/reviewnlp/ops/`, routes in `src/reviewnlp/serving/ops_router.py`, the worker in `scripts/ops_worker.py`,
tests in `tests/test_ops.py`.

## What it does

1. **Review card.** A pasted review becomes a list of findings. Each finding shows the guest's own words (an
   excerpt that must be found in the review, or the answer is refused), the problem type and department from
   `catalogue_v2`, and a status:
   - **Εκκρεμεί** (`pending`): the guest says it was still there when they left, never fixed, or came back;
   - **Αναφέρεται ως λυμένο** (`resolved`): the guest says the hotel fixed it during the stay;
   - **Χρειάζεται έλεγχο** (`unclear`): the review does not say.

   Every reported problem is a finding, including one fixed during the stay ("the AC broke and staff repaired it"
   still counts as a complaint). A review is identified by `(hotel_id, source, review_id)`: sending it again
   returns the stored card and is not analysed again; the same sentence in two reviews stays two findings.
2. **Tickets after confirmation.** The manager confirms or rejects each finding. A confirmed finding opens a
   ticket only if its type has an operational measure (not structural) **and** the problem is still open: the
   guest says so, or staff confirm it. A complaint reported as fixed opens nothing unless staff say it came back.
   Where the catalogue needs the room (most room problems), reception must give it first. The ticket goes to the
   catalogue's department queue with the states Ανατέθηκε → Σε εξέλιξη → Επιλύθηκε (forward only).
   The ticket and its pending notification are written in **one SQLite transaction**; the notification has a
   stable send id (`ticket-<id>-assigned`, also the e-mail's Message-ID), is sent at once when the channel
   answers, and otherwise retried by the worker with growing waits, up to five tries.
3. **Reply draft.** Once every finding is decided, a reply is drafted in the review's language from the confirmed
   findings and their tickets' real state: "passed to the department" while a ticket is open, "put right" only
   once it is resolved, an apology and no promise when there is no ticket. The drafter never sees room numbers or
   booking details. The manager edits, approves and copies it; nothing is posted anywhere.

The demo's acceptance checks, as the owner set them, are tests marked `ACCEPTANCE` in `tests/test_ops.py`: the
same submission does not double the tickets; a transient notification failure recovers; a complaint already
resolved does not open an urgent ticket.

## Run it

The short way, the same in cmd and PowerShell: `.venv\Scripts\python.exe scriptsun_ops_demo.py` makes a local demo key for `hotel-demo` (kept in `runs/ops/`), sets everything below for that process only, prints the dashboard address and the key, and starts the app. Add `--live` to use the model once `OPENROUTER_API_KEY` is set. The long way:

```bash
python scripts/create_api_key.py --hotel-id hotel-demo        # the raw key once, and the JSON for the server
export MODEL_TYPE=stub
export REVIEWNLP_DB_PATH=runs/ops/ops.db
export REVIEWNLP_API_KEYS_JSON='{"PASTE_GENERATED_DIGEST_HERE":["hotel-demo"]}'
export REVIEWNLP_OPS_DEMO=1                                    # keyword stand-ins, nothing leaves the machine
uvicorn reviewnlp.serving.app:app --host 127.0.0.1 --port 8000 --workers 1
python scripts/ops_worker.py --every 30                        # in a second terminal: retries notifications
```

Open `http://127.0.0.1:8000/dashboard`, enter `hotel-demo` and the raw key, and use the section
"Κριτική → tickets → απάντηση".

| Setting | Default | Meaning |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | unset | When set, findings and replies come from `anthropic/claude-haiku-5.5-20261007` through OpenRouter (the hotel pilot's model) and the review text is sent to OpenRouter and Anthropic. The owner sets it; it is never logged. |
| `REVIEWNLP_OPS_MODEL` | the pilot's model | Another OpenRouter model id. |
| `REVIEWNLP_OPS_DEMO` | unset | `1` allows the keyword stand-ins when there is no key. They are for showing the flow, not the product. |
| `REVIEWNLP_OPS_MAIL_DIR` | `runs/ops_mail` | Test e-mail: one `.eml` file per notification. |
| `REVIEWNLP_OPS_SMTP_HOST`, `_PORT`, `_USER`, `_PASSWORD`, `_FROM` | unset | Real e-mail through SMTP with STARTTLS, set by the owner. |
| `REVIEWNLP_OPS_NOTIFY_JSON` | a test address | Department → recipient, e.g. `{"maintenance": "...", "*": "..."}`. |

## What it is not

- **Not validated.** Nothing here has been measured: not the findings, the statuses, the assignments or the
  replies. The findings prompt (`ops-findings-v1`) is new: it adds the excerpt and the status to the hotel
  pilot's prompt, so the pilot's result (hotels accepted 0.89 of its urgent flags) does not carry over to it.
- **Before any sending without confirmation in a real hotel,** the owner's plan is to measure, on consecutive
  reviews fixed before the model's output is seen: the urgent problems found and missed, the assignments that
  were right, and the replies approved without substantial edits. That is a paid-pilot protocol, to be locked
  before it starts.
- The review text is stored in the hotel's own database (the card and the reply need it). The model sees it only
  when `OPENROUTER_API_KEY` is set.
- One notification channel, one test hotel, no sign-in beyond the hotel API key, no posting to review sites.
