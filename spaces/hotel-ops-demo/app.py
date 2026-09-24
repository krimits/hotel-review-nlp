"""Hotel review triage for owners: paste reviews, see what to fix first, with the quotes."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock, Thread

import gradio as gr
import logic
import triage

# Invented examples, not real guests' reviews.
SAMPLE_BATCH = "\n\n".join([
    "The room was spotless and the staff were kind, but breakfast was cold.",
    "The Wi-Fi kept disconnecting. We loved the sea view, but the bathroom was dirty.",
    "Great location, five minutes from the metro. The room was tiny and the air conditioning "
    "was noisy all night.",
    "Reception staff were rude when we asked for a late check-out. The pool was lovely though.",
    "Good value for money. Breakfast had little choice and the coffee was awful.",
    "Very quiet room, comfortable beds and a friendly doorman. Parking was expensive.",
])
_model_lock = Lock()
_load_lock = Lock()
# Tables hold prose (quotes, advice): use the text font, not the monospace default.
CSS = ".owner-table * { font-family: var(--font) !important; }"


@lru_cache(maxsize=1)
def _cached_model() -> triage.AbsaModel:
    return triage.AbsaModel()


def _load_model() -> triage.AbsaModel:
    with _load_lock:  # the start-up warm-up and a first request must not load it twice
        return _cached_model()


def score_pairs(pairs, progress=None):
    """The model call, at module level so tests can replace it."""
    return _load_model()(pairs, progress=progress)


def analyze(text, upload, progress=gr.Progress()):  # noqa: B008 - Gradio injects the tracker
    try:
        reviews = logic.collect_reviews(text, upload)
    except logic.InputError as exc:
        raise gr.Error(str(exc)) from exc
    skipped = {n for n, review in enumerate(reviews, start=1) if not triage.looks_english(review)}
    english = [review for n, review in enumerate(reviews, start=1) if n not in skipped]

    progress(0, desc="Φόρτωση μοντέλου (μόνο την πρώτη φορά)…")
    try:
        with _model_lock:
            found = triage.analyze(english, lambda pairs: score_pairs(
                pairs, lambda done: progress(done, desc="Ανάλυση φράσεων…")))
    except Exception as exc:
        raise gr.Error("Το μοντέλο δεν ολοκλήρωσε την ανάλυση. Δοκιμάστε ξανά σε λίγο.") from exc

    in_order = iter(found)
    findings = [[] if n in skipped else next(in_order) for n in range(1, len(reviews) + 1)]
    summary = logic.summarize(findings)
    analysed = len(english)
    return (
        logic.summary_markdown(len(reviews), skipped, summary),
        logic.fix_first_rows(summary, analysed) if analysed else [],
        logic.strength_rows(summary, analysed) if analysed else [],
        logic.finding_rows(findings),
        logic.review_rows(reviews, findings, skipped),
        logic.write_csv(reviews, findings),
    )


with gr.Blocks(title="Hotel Review Triage", css=CSS) as demo:
    gr.Markdown("# 🏨 Τι λένε οι επισκέπτες σας")
    gr.Markdown(
        "Επικολλήστε έως 100 κριτικές στα αγγλικά (μία ανά παράγραφο, με κενή γραμμή ανάμεσα) "
        "ή ανεβάστε αρχείο. Θα δείτε **τι να διορθώσετε πρώτα**, **τι εκτιμούν οι επισκέπτες** "
        "και, για κάθε εύρημα, **τη φράση της κριτικής** από την οποία προέκυψε."
    )
    with gr.Row():
        with gr.Column(scale=3):
            reviews = gr.Textbox(
                label="Κριτικές στα αγγλικά, μία ανά παράγραφο", lines=12,
                placeholder="The staff were friendly, but the bathroom was dirty.\n\n"
                            "Great location, although breakfast was expensive.",
            )
        with gr.Column(scale=1):
            upload = gr.File(label="…ή αρχείο .csv (στήλη κριτικών) ή .txt",
                             file_types=[".csv", ".txt"], type="filepath")
            run = gr.Button("Ανάλυση", variant="primary")
            clear = gr.ClearButton(value="Καθαρισμός")
    gr.Examples(
        examples=[[SAMPLE_BATCH]], inputs=reviews,
        label="Δοκιμάστε με 6 ενδεικτικές κριτικές (συνθετικές)",
    )
    summary = gr.Markdown()
    gr.Markdown("### Τι να διορθώσετε πρώτα")
    fix_first = gr.Dataframe(
        headers=["Πτυχή", "Κριτικές", "% κριτικών", "Τι γράφουν", "Τι μπορείτε να κάνετε"],
        column_widths=["13%", "9%", "10%", "38%", "30%"], max_height=1200,
        interactive=False, wrap=True, elem_classes="owner-table",
    )
    gr.Markdown("### Τι εκτιμούν οι επισκέπτες")
    strengths = gr.Dataframe(
        headers=["Πτυχή", "Κριτικές", "% κριτικών", "Τι γράφουν"],
        column_widths=["13%", "9%", "10%", "68%"], max_height=1200,
        interactive=False, wrap=True, elem_classes="owner-table",
    )
    with gr.Accordion("Όλα τα ευρήματα, με τη φράση της κριτικής", open=False):
        findings = gr.Dataframe(
            headers=["Κριτική", "Πτυχή", "Συναίσθημα", "Λέξη", "Φράση της κριτικής"],
            column_widths=["8%", "14%", "11%", "12%", "55%"],
            interactive=False, wrap=True, elem_classes="owner-table",
        )
    with gr.Accordion("Ανά κριτική", open=False):
        per_review = gr.Dataframe(
            headers=["Κριτική", "Παράπονα", "Έπαινοι", "Σημείωση", "Κείμενο"],
            column_widths=["8%", "20%", "20%", "14%", "38%"],
            interactive=False, wrap=True, elem_classes="owner-table",
        )
    download = gr.File(label="Λήψη όλων των ευρημάτων (CSV για Excel)")
    gr.Markdown(
        "**Πόσο αξιόπιστο είναι:** σε 30 πραγματικές αγγλικές κριτικές ξενοδοχείων, "
        "χαρακτηρισμένες με το χέρι πριν τρέξει το σύστημα, εντόπισε το 82% των παραπόνων "
        "και το 80% των παραπόνων που ανέφερε ήταν πραγματικά. Κάνει λάθη σε ειρωνεία, "
        "αρνήσεις και σε θέματα εκτός των 8 πτυχών, γι' αυτό κάθε εύρημα δείχνει τη φράση "
        "της κριτικής: ελέγξτε την πριν αποφασίσετε.  \n"
        "Οι κριτικές δεν αποθηκεύονται. Μην επικολλάτε προσωπικά στοιχεία επισκεπτών."
    )
    outputs = [summary, fix_first, strengths, findings, per_review, download]
    run.click(analyze, inputs=[reviews, upload], outputs=outputs)
    clear.add([reviews, upload, *outputs])

demo.queue(default_concurrency_limit=1)
if __name__ == "__main__":
    # Download and load the model while the page starts, not on the first click.
    Thread(target=_load_model, daemon=True).start()
    demo.launch()
