"""Experimental DistilBERT / optional Jev / evidence-first Qwen Space."""
# ruff: noqa: I001 -- spaces must initialize CUDA emulation before other third-party imports.

from __future__ import annotations

import json
import os
from pathlib import Path

import spaces  # Import before model libraries: enables ZeroGPU's CUDA emulation.
import gradio as gr

from reviewnlp.triage.demo_service import space_environment
from reviewnlp.triage.space_runtime import SpaceRuntime

RUNTIME = None
MANIFEST_PATH = Path(__file__).resolve().parent / "source_manifest.json"
SOURCE_SNAPSHOT = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else None


@spaces.GPU(duration=60)
def run_qwen(review, signals):
    # Only plain arguments and returned data cross the ZeroGPU worker boundary.
    return RUNTIME.run_qwen(review, signals)


def initialize():
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = SpaceRuntime(run_qwen)
        RUNTIME.preload()
    return RUNTIME


def model_info():
    return {**space_environment(), "source_snapshot": SOURCE_SNAPSHOT,
            "runtime": RUNTIME.information() if RUNTIME else {"ready": False}}

STATUS_LABELS = {"REAL_PENDING": "Αναφέρθηκε πρόβλημα — επιβεβαιώστε την τρέχουσα κατάσταση",
                 "REAL_RESOLVED": "Δηλώνεται επίλυση στην κριτική — ελέγξτε το απόσπασμα",
                 "HYPOTHETICAL": "Χαρακτηρίστηκε υποθετικό — ελέγξτε το απόσπασμα",
                 "POSITIVE_COMMENT": "Έπαινος", "UNCERTAIN": "Αβέβαιο — χρειάζεται ανθρώπινο έλεγχο"}
DEPARTMENT_LABELS = {"maintenance": "Συντήρηση", "housekeeping": "Καθαριότητα",
                     "food_and_beverage": "Εστίαση", "reception": "Υποδοχή",
                     "management": "Διοίκηση", "other": "Προς επιβεβαίωση"}


def service():
    runtime = initialize()
    if runtime.startup_failure:
        stage = runtime.startup_failure["stage"]
        name = "φόρτωση DistilBERT" if stage == "sentiment_load" else "φόρτωση Qwen"
        raise gr.Error(f"Αποτυχία σταδίου: {name}. Το μοντέλο δεν είναι διαθέσιμο.")
    return runtime.service


def present(result):
    sentiment = result["sentiment"]
    confidence = sentiment["confidence"]
    score = f"{confidence:.1%}" if confidence is not None else "χωρίς πιθανότητα"
    sentiment_text = "θετικό" if sentiment["label"] == "positive" else "αρνητικό"
    title = (f"**Συναίσθημα:** {sentiment_text} ({score}). Η θετική ετικέτα δεν αποκλείει παράπονο.\n\n"
             f"**Χρόνος αυτού του αιτήματος:** {result['timings']['total_ms'] / 1000:.1f} s.")
    for stage in result.get("stage_reports", []):
        if stage["status"] == "error":
            label = {"issues": "εξαγωγή ζητημάτων", "measures": "παραγωγή μέτρων",
                     "jev": "έλεγχος Jev", "qwen_runtime": "διαθεσιμότητα GPU / Qwen"}.get(stage["stage"], "ανάλυση")
            title += f"\n\n**Αποτυχία σταδίου: {label}** (`{stage['error']}`). Ελέγξτε την αρχική κριτική."
    cost = result["api_cost"]
    amount = cost["reported_cost_sum"]
    title += ("\n\n**Κόστος Jev:** δεν έγινε αίτημα." if not cost["attempts"] else
              "\n\n**Κόστος Jev:** άγνωστο — ο πάροχος δεν επέστρεψε πλήρη στοιχεία." if amount is None else
              f"\n\n**Provider usage.cost:** {amount:g} · αναφέρθηκε για {cost['attempts']} αιτήματα· χωρίς συμπέρασμα για το νόμισμα.")
    if amount is not None and not cost["complete_cost_available"]:
        title += " **Ελλιπές άθροισμα: λείπει κόστος για ορισμένα αιτήματα.**"
    title += "\n\nΤο κόστος GPU δεν καταγράφεται εδώ."
    complaints = result["complaints"]
    if complaints["status"] == "disabled":
        title += "\n\nΟ Jev δεν χρησιμοποιήθηκε σε αυτή την ανάλυση."
    elif complaints["status"] == "error":
        title += "\n\n**Ο έλεγχος Jev απέτυχε.** Δείτε τα ζητήματα και ελέγξτε την κριτική."
    else:
        topics = [*complaints["topics"], *([complaints["other_complaint"]] if complaints["other_complaint"] else [])]
        flagged = [item["topic"] for item in topics if item["answer"] == "yes"]
        title += "\n\n**Θέματα που επισήμανε ο Jev:** " + (", ".join(flagged) or "κανένα") + "."
    actions = result["actions"]
    if actions["status"] in {"error", "no_grounded_actions"}:
        title += "\n\n**Η παραγωγή μέτρων δεν ολοκληρώθηκε έγκυρα.** Αυτό δεν σημαίνει ότι δεν υπάρχει πρόβλημα."
    elif not actions["actions"]:
        title += "\n\n**Δεν προέκυψε αποδεκτό μέτρο.** Ελέγξτε τα ζητήματα και την αρχική κριτική."
    else:
        title += "\n\n**Προτάσεις προς επιβεβαίωση.** Ελέγξτε τα αποσπάσματα, την ανάθεση τμήματος και τα στοιχεία πριν ενεργήσετε."
    if result["routing"]["needs_review"]:
        title += "\n\n**Χρειάζεται ανθρώπινος έλεγχος** για αβεβαιότητα, ελλιπή πρόταση ή αποτυχία σταδίου."
    issue_rows = []
    for issue in result["issue_assessments"]:
        evidence = issue.get("evidence", {})
        issue_rows.append([issue["problem"], STATUS_LABELS[issue["status"]], issue["excerpt"],
                           evidence.get("reported") or "—", evidence.get("hypothetical") or "—",
                           evidence.get("resolved") or "—", DEPARTMENT_LABELS[issue["department"]]])
    action_rows = [[item["problem"], item["excerpt"], item["measure"], DEPARTMENT_LABELS[item["department"]],
                    " · ".join(item["to_confirm"]) or "—"] for item in actions["actions"]]
    return title, issue_rows, action_rows, result


def analyze(text, use_jev):
    try:
        result = service().analyze(text, bool(use_jev))
        result["deployment"] = model_info()
        return present(result)
    except gr.Error:
        raise
    except ValueError as error:
        message = {"one_review_requires_3_to_4000_characters": "Δώστε μία αγγλική κριτική 3–4.000 χαρακτήρων.",
                   "jev_not_configured": "Ο Jev δεν έχει ενεργοποιηθεί από τον διαχειριστή."}.get(str(error))
        raise gr.Error(message or "Η ανάλυση δεν ολοκληρώθηκε. Δοκιμάστε ξανά.") from None
    except Exception:
        raise gr.Error("Αποτυχία σταδίου: ανάλυση. Το μοντέλο δεν μπόρεσε να ολοκληρώσει το αίτημα.") from None


def build_demo():
    with gr.Blocks(title="Hotel Review Actions", analytics_enabled=False,
                   css=".owner-table * { font-family: var(--font) !important; }") as demo:
        gr.Markdown("# Από την κριτική σε προτάσεις για το ξενοδοχείο")
        gr.Markdown("Δοκιμαστική έκδοση για **μία κριτική στα αγγλικά**. Παρουσιάζει τα αναφερθέντα ζητήματα, "
                    "τα τεκμήρια και προτεινόμενα μέτρα. **Πειραματική έκδοση με επιλογή αυθεντικών αποσπασμάτων "
                    "(G-source-spans) — δεν έχει επιλεγεί "
                    "ούτε εγκριθεί για παραγωγική χρήση.** Ανεξάρτητη αξιολόγηση εκκρεμεί.")
        text = gr.Textbox(label="Κριτική στα αγγλικά", lines=7, max_lines=14)
        use_jev = gr.Checkbox(label="Χρήση Jev για πρόσθετο έλεγχο παραπόνων", value=False)
        gr.Markdown("Με ενεργό Jev, το κείμενο αποστέλλεται στον πάροχό του. Η εφαρμογή δεν κρατά ιστορικό κριτικών.")
        run = gr.Button("Ανάλυση κριτικής", variant="primary")
        gr.Examples(examples=[["The reading lamp flickered all evening, although the bed was comfortable."],
                              ["The lobby tea was delicious and our suite was comfortable."]], inputs=text,
                    label="Επινοημένα παραδείγματα")
        summary = gr.Markdown()
        gr.Markdown("### Ζητήματα και τεκμήρια — ελέγξτε και όσα εξαιρέθηκαν")
        issues = gr.Dataframe(headers=["Ζήτημα", "Κατάσταση", "Απόσπασμα", "Αναφορά προβλήματος",
                                      "Υποθετική διατύπωση", "Δηλωμένη επίλυση", "Τμήμα"],
                              interactive=False, wrap=True, elem_classes="owner-table")
        gr.Markdown("### Προτεινόμενα μέτρα προς επιβεβαίωση")
        actions = gr.Dataframe(headers=["Πρόβλημα", "Απόσπασμα", "Προτεινόμενο μέτρο", "Τμήμα", "Τι να επιβεβαιώσετε"],
                               interactive=False, wrap=True, elem_classes="owner-table")
        with gr.Accordion("Λεπτομέρειες αυτής της ανάλυσης", open=False):
            diagnostic = gr.JSON(label="Στάδια, χρόνοι, εκδόσεις και αναφερόμενο κόστος API")
        info = gr.Button(visible=False)
        info.click(model_info, inputs=[], outputs=diagnostic, api_name="model_info", queue=False)
        run.click(analyze, inputs=[text, use_jev], outputs=[summary, issues, actions, diagnostic], api_name="analyze")
        gr.Markdown("Τα αποσπάσματα ελέγχονται ως προς την παρουσία τους στο κείμενο. Αυτό δεν αποδεικνύει "
                    "σωστή ερμηνεία ή κατάλληλο μέτρο. Επιβεβαιώστε κάθε πρόταση πριν την εφαρμογή.")
    return demo.queue(default_concurrency_limit=1, max_size=10)


if os.environ.get("SPACE_ID") or os.environ.get("REVIEWNLP_SPACE_EAGER") == "1":
    initialize()  # Register Qwen's CUDA tensors at module scope, before accepting requests.
demo = build_demo()
if __name__ == "__main__":
    demo.launch()
