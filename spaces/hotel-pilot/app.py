"""Hotel pilot of the review-triage service (experimental).

A hotel enters the code the owner gave it, sets which problem types are urgent for it, pastes or uploads its own
guests' reviews, and judges each analysis: every urgent flag, anything urgent that is missing, the other problems,
and how useful it was. The next review appears only after the current one is judged.

Review texts are sent to OpenRouter and Anthropic for the analysis and are never stored; only the judgements are, in
a private dataset, under a pseudonym.

Space secrets, set by the owner: OPENROUTER_API_KEY (with a spending limit), RATINGS_REPO (a private dataset),
HF_TOKEN (read and write access to it) and HOTEL_CODES (JSON, {sha256(code): pseudonym}). Unless all four are set
the Space runs in preview: code `demo`, made-up analyses, nothing sent and nothing saved.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import gradio as gr
import logic

HERE = Path(__file__).resolve().parent
SECRETS = ("OPENROUTER_API_KEY", "RATINGS_REPO", "HF_TOKEN", "HOTEL_CODES")
INTRO = """# Ανάλυση κριτικών ξενοδοχείου · πιλοτική λειτουργία

Επικολλήστε τις πρόσφατες κριτικές των επισκεπτών σας. Για κάθε κριτική θα δείτε ποια προβλήματα αναφέρει, σε ποιο
τμήμα πάει το καθένα και ποια είναι **επείγοντα** για το δικό σας ξενοδοχείο, και θα μας πείτε αν η ανάλυση είναι
σωστή. Η υπηρεσία είναι **πειραματική**.

**Τα δεδομένα σας.** Τα κείμενα των κριτικών στέλνονται στο OpenRouter και στην Anthropic για την ανάλυση. Εδώ **δεν
αποθηκεύεται κανένα κείμενο**: κρατάμε μόνο τις κρίσεις σας, με ψευδώνυμο και χωρίς το όνομα του ξενοδοχείου.
Επικολλήστε μόνο ό,τι δημοσίευσε ο επισκέπτης.
"""
PREVIEW = ("**Δοκιμαστική λειτουργία.** Οι αναλύσεις είναι **ψεύτικες** (δεν διαβάζουν την κριτική), τίποτα δεν "
           "στέλνεται και τίποτα δεν αποθηκεύεται. Κωδικός: `demo`.")
STORAGE_DOWN = "Η αποθήκευση δεν είναι διαθέσιμη αυτή τη στιγμή. Δοκιμάστε ξανά σε λίγο ή ενημερώστε τον διοργανωτή."
ANALYSIS_DOWN = ("Η ανάλυση δεν είναι διαθέσιμη αυτή τη στιγμή. Καμία κριτική δεν χάθηκε: δοκιμάστε ξανά σε λίγο ή "
                 "ενημερώστε τον διοργανωτή.")
NOT_SAVED = ("- Η απάντηση **δεν αποθηκεύτηκε**. Οι επιλογές σας έμειναν στη θέση τους· δοκιμάστε ξανά «Υποβολή» σε "
             "λίγο ή ενημερώστε τον διοργανωτή.")
BUSY = "Ολοκληρώστε πρώτα τις κριτικές που έχουν αναλυθεί· μετά μπορείτε να αλλάξετε τα επείγοντα."


def secret(name: str) -> str:
    """A Space secret without the spaces or line breaks a paste can add around it."""
    return os.environ.get(name, "").strip()


def log(message: str) -> None:
    """For the owner, in the Space's logs. Never a key, a code or a review."""
    print(f"[hotel pilot] {message}", file=sys.stderr, flush=True)


def load(folder: Path) -> tuple[dict, str, dict]:
    display = json.loads((folder / "display.json").read_text(encoding="utf-8"))
    prompt = (folder / "prompt.md").read_text(encoding="utf-8")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return display, prompt, meta


def settings_from_env() -> dict:
    """The store, the codes, the key and whether this is the preview."""
    missing = [name for name in SECRETS if not secret(name)]
    if missing:
        if len(missing) < len(SECRETS):
            log(f"preview, because these secrets are not set: {', '.join(missing)}")
        return {"preview": True, "store": logic.MemoryStore(), "api_key": "",
                "codes": {hashlib.sha256(b"demo").hexdigest(): "DEMO"}}
    return {"preview": False, "store": logic.HubStore(secret("RATINGS_REPO"), secret("HF_TOKEN")),
            "api_key": secret("OPENROUTER_API_KEY"), "codes": json.loads(secret("HOTEL_CODES"))}


def check_store(settings: dict) -> bool:
    """At start, one read of the store, so the logs say at once whether the dataset secrets work."""
    if settings["preview"]:
        log("preview: nothing is sent or saved")
        return True
    try:
        settings["store"].keys("P00", "records")
    except Exception as error:  # noqa: BLE001
        log(f"the start-up check of the ratings store failed: {type(error).__name__}: {error}")
        return False
    log(f"ratings store OK: {settings['store'].repo_id} can be read")
    return True


def build_app(display: dict, prompt: str, meta: dict, settings: dict, transport=logic.urllib_transport) -> gr.Blocks:
    store, codes, preview = settings["store"], settings["codes"], settings["preview"]
    salt = logic.salt_from(codes)
    types = display["order"]
    urgent = logic.urgent_types(display)
    label_of = {t: display["types"][t]["label"] for t in urgent}

    def header(state: dict) -> str:
        return (f"**{state['hotel']}** · βαθμολογημένες κριτικές: **{state['rated']}** · "
                f"επείγοντα για εσάς: {', '.join(label_of[t] for t in urgent if t in state['urgent']) or 'κανένα'}")

    with gr.Blocks(title="Ανάλυση κριτικών ξενοδοχείου") as demo:
        state = gr.State({"hotel": None, "urgent": list(urgent), "queue": [], "rated": 0})
        gr.Markdown(INTRO)
        if preview:
            gr.Markdown(PREVIEW)
        with gr.Group(visible=True) as login:
            code = gr.Textbox(label="Κωδικός συμμετοχής", type="password")
            start = gr.Button("Είσοδος", variant="primary")
            login_message = gr.Markdown()
        with gr.Group(visible=False) as main:
            status = gr.Markdown()
            with gr.Accordion("Ποια προβλήματα είναι επείγοντα για το ξενοδοχείο σας", open=True):
                gr.Markdown("Αφήστε επιλεγμένα όσα θα διορθώνατε **πριν φτάσει ο επόμενος επισκέπτης** στο δωμάτιο. "
                            "Όσα αποεπιλέξετε θα εμφανίζονται ως τακτικά.")
                urgent_boxes = gr.CheckboxGroup([label_of[t] for t in urgent], label="Επείγοντα")
                save_urgent = gr.Button("Αποθήκευση επειγόντων")
                urgent_message = gr.Markdown()
            with gr.Group(visible=True) as entry:
                pasted = gr.Textbox(label="Κριτικές (μία ανά παράγραφο, με κενή γραμμή ανάμεσα)", lines=10)
                uploaded = gr.File(label="ή αρχείο .txt / .csv", file_types=[".txt", ".csv"], type="filepath")
                analyse = gr.Button("Ανάλυση", variant="primary")
                entry_message = gr.Markdown()
            with gr.Group(visible=False) as rating:
                progress = gr.Markdown()
                review = gr.Textbox(label="Κριτική", interactive=False, lines=6)
                gr.Markdown("### Επείγουσες σημάνσεις")
                no_flags = gr.Markdown()
                verdicts = [gr.Radio(list(logic.VERDICTS), label=f"ΕΠΕΙΓΟΝ · {label_of[t]} → "
                                     f"{display['types'][t]['department_el']}", visible=False) for t in urgent]
                gr.Markdown("### Άλλα προβλήματα")
                others_shown = gr.Markdown()
                missing = gr.Radio(list(logic.MISSING), label="Λείπει κάποιο επείγον πρόβλημα που αναφέρει η κριτική;")
                missing_type = gr.Dropdown([*label_of.values(), logic.OTHER_TYPE], label="Αν ναι, ποιο;", value=None)
                others = gr.Radio(list(logic.OTHERS), label="Τα άλλα προβλήματα είναι σωστά;")
                useful = gr.Radio(logic.USEFUL, label="Πόσο χρήσιμη είναι η ανάλυση; (1 = καθόλου, 5 = πολύ)")
                submit = gr.Button("Υποβολή και επόμενη", variant="primary")
                errors = gr.Markdown()
        screen = [entry, rating, status, entry_message, progress, review, no_flags, *verdicts, others_shown, missing,
                  missing_type, others, useful, errors]

        def show(current: dict, note: str = "") -> list:
            """The updates of `screen` for the review at the head of the queue, or for the entry form. The note
            goes where the hotel is looking: under the entry form, or under the review being judged."""
            if not current["queue"]:
                return [gr.update(visible=True), gr.update(visible=False), header(current), note, "", "", "",
                        *[gr.update(visible=False, value=None) for _ in urgent], "", None, None, None, None, ""]
            head = current["queue"][0]
            flagged = logic.flags(head["types"], current["urgent"], display)
            return [gr.update(visible=False), gr.update(visible=True), header(current), "",
                    f"Κριτική **{current['batch'] - len(current['queue']) + 1}** από {current['batch']}",
                    head["text"], "" if flagged else "_Καμία επείγουσα σήμανση σε αυτή την κριτική._",
                    *[gr.update(visible=t in flagged, value=None) for t in urgent],
                    logic.render_others(head["types"], current["urgent"], display), None, None, None, None, note]

        def on_start(entered: str, current: dict):
            hotel = logic.hotel_for(entered or "", codes)
            if hotel is None:
                return current, gr.update(), gr.update(), gr.update(), gr.update(), "Ο κωδικός δεν αναγνωρίστηκε."
            try:
                saved = store.urgent_set(hotel)
                rated = len(store.keys(hotel, "records"))
            except Exception as error:  # noqa: BLE001 - shown to the hotel, logged for the owner
                log(f"reading {hotel}'s progress failed: {type(error).__name__}: {error}")
                return current, gr.update(), gr.update(), gr.update(), gr.update(), STORAGE_DOWN
            new = {"hotel": hotel, "urgent": saved if saved is not None else list(urgent), "queue": [],
                   "rated": rated, "batch": 0}
            return (new, gr.update(visible=False), gr.update(visible=True), header(new),
                    [label_of[t] for t in urgent if t in new["urgent"]], "")

        def on_save_urgent(chosen: list[str], current: dict):
            if current["queue"]:
                return current, gr.update(), BUSY
            new_set = [t for t in urgent if label_of[t] in set(chosen or [])]
            try:
                store.save_urgent_set(current["hotel"], new_set)
            except Exception as error:  # noqa: BLE001
                log(f"saving {current['hotel']}'s urgent set failed: {type(error).__name__}: {error}")
                return current, gr.update(), STORAGE_DOWN
            new = {**current, "urgent": new_set}
            return new, header(new), "Αποθηκεύτηκε."

        def on_analyse(text: str, path: str | None, current: dict):
            try:
                if path:
                    content = Path(path).read_text(encoding="utf-8-sig")
                    given = logic.reviews_from_csv(content) if path.lower().endswith(".csv") else \
                        logic.split_reviews(content)
                else:
                    given = logic.split_reviews(text or "")
                if not given:
                    return current, gr.update(), gr.update(), *show(current, "Δεν βρέθηκε καμία κριτική.")
                rated, analysed = store.keys(current["hotel"], "records"), store.keys(current["hotel"], "analysed")
                kept, note = logic.check_reviews(given, rated, analysed, salt)
            except UnicodeDecodeError:
                return current, gr.update(), gr.update(), *show(
                    current, "Το αρχείο δεν διαβάζεται ως κείμενο. Αποθηκεύστε το ως .txt ή .csv σε UTF-8.")
            except ValueError as error:  # a review longer than the limit: the message is logic's, in Greek
                return current, gr.update(), gr.update(), *show(current, str(error))
            except Exception as error:  # noqa: BLE001
                log(f"reading {current['hotel']}'s records failed: {type(error).__name__}: {error}")
                return current, gr.update(), gr.update(), *show(current, STORAGE_DOWN)
            if not kept:
                return current, gr.update(), gr.update(), *show(current, note or "Δεν έμεινε καμία νέα κριτική.")
            try:
                done = logic.fake_analyse(kept, types) if preview else \
                    logic.analyse(kept, prompt, types, settings["api_key"], transport)
                store.save_analysed([logic.analysed_note(current["hotel"], r) for r in done])
            except logic.AnalysisError as error:
                log(f"analysis failed for {current['hotel']} ({len(kept)} reviews): {error.kind}")
                return current, gr.update(), gr.update(), *show(current, ANALYSIS_DOWN)
            except Exception as error:  # noqa: BLE001
                log(f"recording {current['hotel']}'s analysed reviews failed: {type(error).__name__}: {error}")
                return current, gr.update(), gr.update(), *show(current, STORAGE_DOWN)
            new = {**current, "queue": done, "batch": len(done)}
            return new, "", None, *show(new, note)

        def on_submit(*values):
            *given, lacking, lacking_type, rest, worth, current = values
            head = current["queue"][0]
            flagged = logic.flags(head["types"], current["urgent"], display)
            chosen = dict(zip(urgent, given, strict=True))
            found = logic.problems(flagged, chosen, lacking, lacking_type, rest, worth, display)
            if found:
                return current, *[gr.update() for _ in screen[:-1]], "\n".join(f"- {e}" for e in found)
            try:
                store.save_record(logic.record(current["hotel"], head, current["urgent"], chosen, lacking,
                                               lacking_type, rest, worth, meta, display))
            except Exception as error:  # noqa: BLE001 - the answers stay on screen; nothing moves on
                log(f"saving a record of {current['hotel']} failed: {type(error).__name__}: {error}")
                return current, *[gr.update() for _ in screen[:-1]], NOT_SAVED
            new = {**current, "queue": current["queue"][1:], "rated": current["rated"] + 1}
            done = "" if new["queue"] else f"Ευχαριστούμε! Βαθμολογήσατε {new['batch']} κριτικές. Μπορείτε να φέρετε κι άλλες."
            return new, *show(new, done)

        start.click(on_start, [code, state], [state, login, main, status, urgent_boxes, login_message])
        save_urgent.click(on_save_urgent, [urgent_boxes, state], [state, status, urgent_message])
        analyse.click(on_analyse, [pasted, uploaded, state], [state, pasted, uploaded, *screen])
        submit.click(on_submit, [*verdicts, missing, missing_type, others, useful, state], [state, *screen])
    return demo


if (HERE / "display.json").exists():
    DISPLAY, PROMPT, META = load(HERE)
    SETTINGS = settings_from_env()
    check_store(SETTINGS)
    demo = build_app(DISPLAY, PROMPT, META, SETTINGS)

if __name__ == "__main__":
    demo.launch()
