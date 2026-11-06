"""Hoteliers' blind evaluation of two review analyses.

Each hotelier enters the code the owner gave them, then sees 60 public English reviews, one at a time, each with two
analyses side by side as "Σύστημα 1" and "Σύστημα 2", and answers four questions per analysis and one preference.
Progress is kept: the same code resumes where it stopped. Nothing here says which system is which.

Space secrets, set by the owner: RATINGS_REPO (a private dataset), HF_TOKEN (write access to it) and HOTELIER_CODES
(JSON, {sha256(code): pseudonym}). Without them the Space runs in preview: code `demo`, and nothing is saved.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import gradio as gr
import logic

HERE = Path(__file__).resolve().parent
INTRO = """# Αξιολόγηση: δύο αναλύσεις της ίδιας κριτικής

Για κάθε κριτική βλέπετε δύο αναλύσεις, **Σύστημα 1** και **Σύστημα 2**: ποια προβλήματα αναφέρει ο επισκέπτης, σε
ποιο τμήμα πάει το καθένα, τι πρέπει να γίνει, και ποια είναι **επείγοντα**. **Επείγον** είναι ό,τι πρέπει να
διορθωθεί πριν φτάσει ο επόμενος επισκέπτης στο δωμάτιο.

Κρίνετε κάθε ανάλυση όπως θα την κρίνατε στο δικό σας ξενοδοχείο. Οι κριτικές είναι δημόσιες, στα αγγλικά, για
άλλα ξενοδοχεία (2015–2017). Είναι **60**· μπορείτε να σταματήσετε και να συνεχίσετε αργότερα με τον ίδιο κωδικό.
"""
Q1_LABEL = ("Τα επείγοντα είναι σωστά; Σημειώνει ως επείγοντα ακριβώς όσα πρέπει να διορθωθούν πριν τον επόμενο "
            "επισκέπτη, στο σωστό τμήμα;")


def load(folder: Path) -> tuple[dict[int, dict], dict, dict]:
    items = {entry["item"]: entry for entry in json.loads((folder / "items.json").read_text(encoding="utf-8"))}
    display = json.loads((folder / "display.json").read_text(encoding="utf-8"))
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return items, display, meta


def store_from_env() -> tuple[logic.MemoryStore, bool]:
    repo, token = os.environ.get("RATINGS_REPO"), os.environ.get("HF_TOKEN")
    if repo and token:
        return logic.HubStore(repo, token), False
    return logic.MemoryStore(), True


def codes_from_env(preview: bool) -> dict[str, str]:
    if os.environ.get("HOTELIER_CODES"):
        return json.loads(os.environ["HOTELIER_CODES"])
    return {hashlib.sha256(b"demo").hexdigest(): "DEMO"} if preview else {}


def build_app(items: dict[int, dict], display: dict, meta: dict, store: logic.MemoryStore, codes: dict[str, str],
              preview: bool) -> gr.Blocks:
    total = len(items)

    def side_inputs(number: int):
        gr.Markdown(f"### Σύστημα {number}")
        output = gr.Markdown()
        q1 = gr.Radio(list(logic.Q1), label=Q1_LABEL)
        q2 = gr.CheckboxGroup(list(logic.Q2), label="Αν όχι, τι είναι λάθος στα επείγοντα;")
        q3 = gr.Radio(list(logic.Q3), label="Τα υπόλοιπα (μη επείγοντα) προβλήματα είναι σωστά;")
        q4 = gr.Radio(logic.Q4, label="Πόσο χρήσιμη είναι η ανάλυση; (1 = καθόλου, 5 = πολύ)")
        return output, q1, q2, q3, q4

    def screen(state: dict):
        """The updates for the current item, or for the end when nothing is left."""
        queue = state["queue"]
        if not queue:
            return (gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True, value="**Ευχαριστούμε!** Ολοκληρώσατε όλες τις κριτικές."),
                    "", "", "", "", *([None, [], None, None] * 2), None, "")
        item = queue[0]
        first = logic.system_1(state["hotelier"], item)
        second = logic.SYSTEMS[1 - logic.SYSTEMS.index(first)]
        done = total - len(queue)
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                f"Κριτική **{done + 1}** από {total}", items[item]["text"],
                logic.render(items[item][first], display), logic.render(items[item][second], display),
                *([None, [], None, None] * 2), None, "")

    with gr.Blocks(title="Αξιολόγηση αναλύσεων κριτικών") as demo:
        state = gr.State({"hotelier": None, "queue": []})
        gr.Markdown(INTRO)
        if preview:
            gr.Markdown("**Δοκιμαστική λειτουργία:** οι απαντήσεις δεν αποθηκεύονται. Κωδικός: `demo`.")
        with gr.Group(visible=True) as login:
            code = gr.Textbox(label="Κωδικός συμμετοχής", type="password")
            start = gr.Button("Έναρξη", variant="primary")
            login_message = gr.Markdown()
        with gr.Group(visible=False) as rating:
            progress = gr.Markdown()
            review = gr.Textbox(label="Κριτική", interactive=False, lines=6)
            with gr.Row():
                with gr.Column():
                    out1, q1a, q2a, q3a, q4a = side_inputs(1)
                with gr.Column():
                    out2, q1b, q2b, q3b, q4b = side_inputs(2)
            preference = gr.Radio(list(logic.PREFERENCE),
                                  label="Με ποια ανάλυση θα προτιμούσατε να δουλέψετε για αυτή την κριτική;")
            submit = gr.Button("Υποβολή και επόμενη", variant="primary")
            errors = gr.Markdown()
        finished = gr.Markdown(visible=False)
        shown = [login, rating, finished, progress, review, out1, out2, q1a, q2a, q3a, q4a, q1b, q2b, q3b, q4b,
                 preference, errors]

        def on_start(entered: str, current: dict):
            hotelier = logic.hotelier_for(entered or "", codes)
            if hotelier is None:
                return current, *[gr.update() for _ in shown], "Ο κωδικός δεν αναγνωρίστηκε."
            new = {"hotelier": hotelier, "queue": logic.remaining(hotelier, items, store.done(hotelier))}
            return new, *screen(new), ""

        def on_submit(a1, a2, a3, a4, b1, b2, b3, b4, chosen, current: dict):
            found = (logic.problems(a1, a2, a3, a4, "Σύστημα 1") + logic.problems(b1, b2, b3, b4, "Σύστημα 2")
                     + ([] if chosen in logic.PREFERENCE else ["Διαλέξτε ποια ανάλυση προτιμάτε."]))
            if found:
                return current, *[gr.update() for _ in shown[:-1]], "\n".join(f"- {e}" for e in found)
            item = current["queue"][0]
            store.save(logic.record(current["hotelier"], item, logic.answers(a1, a2, a3, a4),
                                    logic.answers(b1, b2, b3, b4), chosen, meta))
            new = {"hotelier": current["hotelier"], "queue": current["queue"][1:]}
            return new, *screen(new)

        start.click(on_start, [code, state], [state, *shown, login_message])
        submit.click(on_submit, [q1a, q2a, q3a, q4a, q1b, q2b, q3b, q4b, preference, state], [state, *shown])
    return demo


if (HERE / "items.json").exists():
    ITEMS, DISPLAY, META = load(HERE)
    STORE, PREVIEW = store_from_env()
    demo = build_app(ITEMS, DISPLAY, META, STORE, codes_from_env(PREVIEW), PREVIEW)

if __name__ == "__main__":
    demo.launch()
