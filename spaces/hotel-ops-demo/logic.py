"""What the owner sees: read the reviews, rank complaints and strengths, keep the evidence.

Everything here is plain Python over the findings produced by `triage.analyze`,
so it is tested without a model.
"""

from __future__ import annotations

import csv
import re
import tempfile
from pathlib import Path

ASPECT_NAMES = {
    "cleanliness": "Καθαριότητα",
    "staff": "Προσωπικό",
    "location": "Τοποθεσία",
    "room": "Δωμάτιο",
    "food": "Φαγητό / πρωινό",
    "noise": "Θόρυβος",
    "value": "Σχέση τιμής-αξίας",
    "facilities": "Παροχές",
}
SENTIMENT_NAMES = {"positive": "Θετικό", "negative": "Αρνητικό"}
# Same text as reviewnlp.analytics.recommendations.RECOMMENDATION_TEXT; a test
# keeps the two in step, and the Space stays free of a pinned package install.
RECOMMENDATIONS = {
    "cleanliness": (
        "Ενισχύστε τον τελικό έλεγχο καθαριότητας στα δωμάτια, "
        "στα μπάνια και στους κοινόχρηστους χώρους."
    ),
    "staff": (
        "Εξετάστε την εκπαίδευση προσωπικού και τη διαδικασία "
        "διαχείρισης αιτημάτων κατά το check-in και τη διαμονή."
    ),
    "location": (
        "Βελτιώστε την ενημέρωση για πρόσβαση, μετακινήσεις, "
        "θόρυβο και πραγματικές αποστάσεις από τα σημεία ενδιαφέροντος."
    ),
    "room": (
        "Δώστε προτεραιότητα σε συντήρηση δωματίων, κρεβάτια, "
        "μπάνια και εξοπλισμό."
    ),
    "food": (
        "Εξετάστε ποιότητα πρωινού, ποικιλία, θερμοκρασία φαγητού "
        "και συνέπεια στο service."
    ),
    "noise": (
        "Εντοπίστε πηγές θορύβου και εφαρμόστε μέτρα ηχομόνωσης "
        "ή καλύτερη κατανομή δωματίων."
    ),
    "value": (
        "Συγκρίνετε τιμή και παροχές με ανταγωνιστικά ξενοδοχεία "
        "και κάντε σαφέστερη την αξία της διαμονής."
    ),
    "facilities": (
        "Βελτιώστε τη διαθεσιμότητα, συντήρηση και ενημέρωση "
        "για Wi-Fi, πισίνα, parking και λοιπές παροχές."
    ),
}

# About a minute of CPU time at the measured ~0.5-1 s per review.
MAX_REVIEWS = 100
MAX_TOTAL_CHARS = 150_000
QUOTE_PREVIEW = 160
_REVIEW_COLUMN = re.compile(r"review|text|comment|content|feedback|κριτικ|σχόλι", re.I)


class InputError(ValueError):
    """A problem with what the owner pasted or uploaded, worded for the owner."""


def split_pasted(text: str | None) -> list[str]:
    """One review per block; blocks are separated by an empty line."""
    return [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]


def read_upload(path: str | Path) -> list[str]:
    """Reviews from a .csv (the review column) or a .txt (blank-line separated) file."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1")
    if path.suffix.lower() != ".csv":
        return split_pasted(raw)
    rows = list(csv.reader(raw.splitlines()))
    if not rows:
        return []
    header = [cell.strip() for cell in rows[0]]
    named = [i for i, cell in enumerate(header) if _REVIEW_COLUMN.search(cell)]
    if named:
        column, body = named[0], rows[1:]
    else:
        # No recognisable header: take the column with the longest text on average.
        width = max(len(row) for row in rows)
        column = max(range(width), key=lambda i: sum(len(r[i]) for r in rows if len(r) > i))
        body = rows
    return [row[column].strip() for row in body if len(row) > column and row[column].strip()]


def collect_reviews(text: str | None, upload: str | Path | None) -> list[str]:
    reviews = split_pasted(text)
    if upload:
        try:
            reviews += read_upload(upload)
        except (OSError, csv.Error) as exc:
            raise InputError("Το αρχείο δεν διαβάζεται. Ανεβάστε .csv ή .txt σε UTF-8.") from exc
    if not reviews:
        raise InputError("Επικολλήστε τουλάχιστον μία κριτική ή ανεβάστε αρχείο .csv / .txt.")
    if len(reviews) > MAX_REVIEWS:
        raise InputError(f"Έως {MAX_REVIEWS} κριτικές τη φορά· δώσατε {len(reviews)}.")
    if sum(len(r) for r in reviews) > MAX_TOTAL_CHARS:
        raise InputError(f"Έως {MAX_TOTAL_CHARS:,} χαρακτήρες τη φορά. Χωρίστε τις κριτικές σε μικρότερες ομάδες.")
    return reviews


def _preview(quote: str) -> str:
    return quote if len(quote) <= QUOTE_PREVIEW else quote[:QUOTE_PREVIEW - 1].rstrip() + "…"


def summarize(findings: list[list[dict]]) -> dict[str, dict]:
    """Per aspect: which reviews complain or praise it, with their quotes.

    Counts are reviews, not sentences: one guest writing about the breakfast
    three times is one guest. An aspect a review both praises and criticises
    counts on both sides, so a complaint is never hidden behind a compliment.
    """
    summary: dict[str, dict] = {}
    for review, items in enumerate(findings, start=1):
        for item in items:
            side = "complaints" if item["sentiment"] == "negative" else "praise"
            entry = summary.setdefault(item["aspect"], {"complaints": {}, "praise": {}})
            entry[side].setdefault(review, item["quote"])
    return summary


def _ranked(summary: dict, side: str) -> list[tuple[str, dict]]:
    rows = [(aspect, entry[side]) for aspect, entry in summary.items() if entry[side]]
    return sorted(rows, key=lambda row: (-len(row[1]), row[0]))


def _examples(quotes: dict[int, str], limit: int = 2) -> str:
    shortest = sorted(quotes.items(), key=lambda item: len(item[1]))[:limit]
    return " · ".join(f"#{review}: «{_preview(quote)}»" for review, quote in sorted(shortest))


def fix_first_rows(summary: dict, analysed: int) -> list[list]:
    return [
        [ASPECT_NAMES[aspect], len(reviews), f"{len(reviews) / analysed:.0%}",
         _examples(reviews), RECOMMENDATIONS[aspect]]
        for aspect, reviews in _ranked(summary, "complaints")
    ]


def strength_rows(summary: dict, analysed: int) -> list[list]:
    return [
        [ASPECT_NAMES[aspect], len(reviews), f"{len(reviews) / analysed:.0%}", _examples(reviews)]
        for aspect, reviews in _ranked(summary, "praise")
    ]


def finding_rows(findings: list[list[dict]]) -> list[list]:
    """Every finding, complaints first, with the verbatim clause it rests on."""
    rows = [
        [review, ASPECT_NAMES[item["aspect"]], SENTIMENT_NAMES[item["sentiment"]], item["term"],
         item["quote"]]
        for review, items in enumerate(findings, start=1) for item in items
    ]
    return sorted(rows, key=lambda row: (row[2] != SENTIMENT_NAMES["negative"], row[0]))


def review_rows(reviews: list[str], findings: list[list[dict]], skipped: set[int]) -> list[list]:
    rows = []
    for review, (text, items) in enumerate(zip(reviews, findings, strict=True), start=1):
        if review in skipped:
            complaints = praise = "—"
            status = "Δεν αναλύθηκε: δεν φαίνεται αγγλική"
        else:
            complaints = ", ".join(sorted({ASPECT_NAMES[i["aspect"]] for i in items
                                           if i["sentiment"] == "negative"})) or "—"
            praise = ", ".join(sorted({ASPECT_NAMES[i["aspect"]] for i in items
                                       if i["sentiment"] == "positive"})) or "—"
            status = "" if items else "Καμία πτυχή με σαφή γνώμη"
        rows.append([review, complaints, praise, status, _preview(" ".join(text.split()))])
    return rows


def summary_markdown(total: int, skipped: set[int], summary: dict) -> str:
    analysed = total - len(skipped)
    complaints = _ranked(summary, "complaints")
    praise = _ranked(summary, "praise")
    lines = [f"**Αναλύθηκαν {analysed} από {total} κριτικές.**"]
    if skipped:
        numbers = ", ".join(f"#{n}" for n in sorted(skipped))
        lines.append(f"Δεν αναλύθηκαν ({numbers}): υποστηρίζονται μόνο αγγλικές κριτικές.")
    if complaints:
        aspect, reviews = complaints[0]
        lines.append(f"Συχνότερο παράπονο: **{ASPECT_NAMES[aspect]}** "
                     f"σε {len(reviews)} από {analysed} κριτικές.")
    elif analysed:
        lines.append("Δεν εντοπίστηκε κανένα παράπονο.")
    if praise:
        aspect, reviews = praise[0]
        lines.append(f"Ό,τι επαινείται περισσότερο: **{ASPECT_NAMES[aspect]}** "
                     f"σε {len(reviews)} από {analysed} κριτικές.")
    return "  \n".join(lines)


def write_csv(reviews: list[str], findings: list[list[dict]]) -> str:
    """All findings as a UTF-8 CSV the owner can open in Excel."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", prefix="review-findings-", delete=False, encoding="utf-8-sig", newline=""
    )
    with handle:
        writer = csv.writer(handle)
        writer.writerow(["review", "aspect", "sentiment", "word", "quote", "review_text"])
        for review, (text, items) in enumerate(zip(reviews, findings, strict=True), start=1):
            for item in items:
                writer.writerow([review, ASPECT_NAMES[item["aspect"]],
                                 SENTIMENT_NAMES[item["sentiment"]], item["term"], item["quote"], text])
    return handle.name
