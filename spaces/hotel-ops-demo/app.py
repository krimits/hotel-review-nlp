"""Try real English hotel-review aspect extraction in a private session."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock

import gradio as gr
import torch
from inference import analyze_review
from logic import SENTIMENT_NAMES, accept_record, render
from transformers import AutoModelForCausalLM, AutoTokenizer

from reviewnlp.absa.extract import overall_from_aspects
from reviewnlp.absa.pipeline import BASE_MODEL

MAX_INPUT_CHARS = 1200
_inference_lock = Lock()


@lru_cache(maxsize=1)
def _load_model():
    # CPU Basic is the default Space hardware. Float32 avoids relying on
    # bfloat16 CPU support; the small Qwen model is loaded only on first use.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model.to(device).eval()
    return tokenizer, model


def analyze(text: str, history: list[dict] | None):
    clean_text = " ".join(str(text or "").split())
    if not clean_text:
        raise gr.Error("Γράψτε μία κριτική στα αγγλικά.")
    if len(clean_text) > MAX_INPUT_CHARS:
        raise gr.Error(f"Η κριτική πρέπει να έχει έως {MAX_INPUT_CHARS} χαρακτήρες.")
    try:
        with _inference_lock:
            tokenizer, model = _load_model()
            result = analyze_review(tokenizer, model, clean_text)
    except Exception as exc:
        raise gr.Error("Το μοντέλο δεν μπόρεσε να ολοκληρώσει την ανάλυση. Δοκιμάστε ξανά.") from exc

    updated = accept_record(history, clean_text, result)
    accepted = len(updated) > len(history or [])
    aspects = updated[-1]["aspects"] if accepted else []
    if accepted:
        label = SENTIMENT_NAMES[overall_from_aspects(aspects)]
        status = f"**Ανάλυση ολοκληρώθηκε.** Συνολική ένδειξη από τις πτυχές: **{label}**."
        if result.get("salvaged"):
            status += " Η απάντηση περιείχε επιπλέον κείμενο· μετρήθηκαν μόνο πτυχές με αυτούσια αποσπάσματα."
        if result.get("entries_dropped"):
            status += " Κάποιες αναφορές απορρίφθηκαν επειδή δεν πληρούσαν τους κανόνες ελέγχου."
    elif result.get("generation_hit_token_budget"):
        status = "Η απάντηση του μοντέλου κόπηκε. Η κριτική δεν προστέθηκε στα σύνολα."
    elif not result.get("json_valid"):
        error = str(result.get("error") or "")
        if error == "empty generation":
            detail = "Δεν παρήχθη απάντηση."
        elif error == "no JSON array found":
            detail = "Δεν βρέθηκε λίστα πτυχών."
        else:
            detail = "Η λίστα πτυχών δεν μπορούσε να διαβαστεί."
        status = f"{detail} Η κριτική δεν προστέθηκε στα σύνολα."
    elif result.get("quote_absent") or result.get("quote_not_in_review"):
        status = "Δεν βρέθηκαν αυτούσια αποσπάσματα που να στηρίζουν τις πτυχές. Η κριτική δεν προστέθηκε στα σύνολα."
    else:
        status = "Δεν εντοπίστηκε πτυχή ξενοδοχείου στην κριτική. Η κριτική δεν προστέθηκε στα σύνολα."
    aspect_rows, review_rows, complaints, evidence = render(
        updated, {"aspects": aspects} if accepted else None
    )
    return updated, status, aspect_rows, review_rows, complaints, evidence


def clear():
    return [], "Η συνεδρία καθαρίστηκε.", [], [], [], []


with gr.Blocks(title="Hotel Review Operations · Demo") as demo:
    gr.Markdown("# 🏨 Hotel Review Operations")
    gr.Markdown(
        "**Δοκιμαστική λειτουργία για ξενοδόχους:** εντοπίστε συγκεκριμένα παράπονα "
        "σε αγγλικές κριτικές και δείτε τα αποσπάσματα που στηρίζουν τις προτάσεις. "
        "Τα αποτελέσματα προέρχονται από πραγματική εκτέλεση του Qwen2.5-0.5B-Instruct "
        "χωρίς εκπαίδευση για πτυχές ξενοδοχείων. Ελέγξτε τα πριν πάρετε αποφάσεις."
    )
    gr.Markdown(
        "Οι κριτικές αυτής της δοκιμής διατηρούνται προσωρινά στη συνεδρία και χάνονται "
        "με επανεκκίνηση της εφαρμογής. Μην καταχωρείτε προσωπικά στοιχεία επισκεπτών. "
        "Η ανάλυση ελληνικών κριτικών δεν έχει ακόμη επαληθευτεί."
    )
    state = gr.State([])
    with gr.Row():
        review = gr.Textbox(
            label="Αγγλική κριτική", lines=4,
            placeholder="The staff were friendly, but the bathroom was dirty and breakfast was cold.",
            max_length=MAX_INPUT_CHARS,
        )
    with gr.Row():
        submit = gr.Button("Ανάλυση και προσθήκη", variant="primary")
        reset = gr.Button("Καθαρισμός συνεδρίας")
    gr.Examples(
        examples=[
            ["The room was spotless and the staff were kind, but breakfast was cold."],
            ["The Wi-Fi kept disconnecting. We loved the sea view, but the bathroom was dirty."],
        ], inputs=review, label="Ενδεικτικές κριτικές (συνθετικές)",
    )
    status = gr.Markdown("Προσθέστε μία κριτική για να δείτε πραγματική ανάλυση.")
    aspects = gr.Dataframe(
        headers=["Πτυχή", "Συναίσθημα", "Αυτούσιο απόσπασμα"],
        label="Πτυχές της τελευταίας κριτικής", interactive=False,
    )
    gr.Markdown("### Παράπονα και ενέργειες για τη συνεδρία")
    complaints = gr.Dataframe(
        headers=["Πτυχή", "Αρνητικές κριτικές", "Σύνολο αναφορών", "Αρνητικές %", "Προτεινόμενη ενέργεια"],
        label="Ταξινομημένα παράπονα — πραγματικά πλήθη από τις κριτικές σας", interactive=False,
    )
    evidence = gr.Dataframe(
        headers=["Αριθμός κριτικής", "Πτυχή", "Αυτούσιο αρνητικό απόσπασμα"],
        label="Στοιχεία πίσω από τα παράπονα", interactive=False,
    )
    reviews = gr.Dataframe(
        headers=["Αριθμός κριτικής", "Ένδειξη από πτυχές", "Πτυχές"],
        label="Κριτικές της συνεδρίας", interactive=False,
    )
    submit.click(analyze, inputs=[review, state], outputs=[state, status, aspects, reviews, complaints, evidence])
    reset.click(clear, outputs=[state, status, aspects, reviews, complaints, evidence])

demo.queue(default_concurrency_limit=1)
if __name__ == "__main__":
    demo.launch()
