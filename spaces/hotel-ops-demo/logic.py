"""What the owner sees: read the reviews, rank the topics to fix and to keep, show the evidence.

Everything here is plain Python over the mentions produced by `triage.analyze`,
so it is tested without a model.
"""

from __future__ import annotations

import csv
import re
import tempfile
from pathlib import Path

ASPECT_NAMES = {
    "cleanliness": "Καθαριότητα",
    "staff": "Εξυπηρέτηση",
    "location": "Τοποθεσία",
    "room": "Δωμάτιο",
    "food": "Φαγητό",
    "noise": "Θόρυβος",
    "value": "Τιμή",
    "facilities": "Παροχές",
}
TOPIC_NAMES = {
    "cleanliness.general": "Καθαριότητα › Γενική",
    "cleanliness.linen": "Καθαριότητα › Ιματισμός & πετσέτες",
    "cleanliness.odour": "Καθαριότητα › Οσμές",
    "cleanliness.pests": "Καθαριότητα › Έντομα",
    "staff.people": "Εξυπηρέτηση › Προσωπικό / οικοδεσπότης",
    "staff.response": "Εξυπηρέτηση › Ανταπόκριση & επικοινωνία",
    "staff.resolution": "Εξυπηρέτηση › Επίλυση προβλημάτων",
    "staff.checkin": "Εξυπηρέτηση › Check-in / check-out",
    "location.general": "Τοποθεσία",
    "room.general": "Δωμάτιο › Γενικά",
    "room.comfort": "Δωμάτιο › Άνεση",
    "room.size": "Δωμάτιο › Μέγεθος",
    "room.bed": "Δωμάτιο › Κρεβάτι",
    "room.climate": "Δωμάτιο › Κλιματισμός & αερισμός",
    "room.bathroom": "Δωμάτιο › Μπάνιο & ντους",
    "room.storage": "Δωμάτιο › Ντουλάπα & αποθήκευση",
    "room.equipment": "Δωμάτιο › Εξοπλισμός & έπιπλα",
    "room.view": "Δωμάτιο › Θέα & μπαλκόνι",
    "food.breakfast": "Φαγητό › Πρωινό",
    "food.dining": "Φαγητό › Εστιατόριο, μπαρ & καφές",
    "noise.general": "Θόρυβος",
    "value.price": "Τιμή › Τιμή & αξία",
    "value.charges": "Τιμή › Επιπλέον χρεώσεις & εγγύηση",
    "facilities.access": "Παροχές › Ανελκυστήρας & σκάλες",
    "facilities.wifi": "Παροχές › Wi-Fi",
    "facilities.parking": "Παροχές › Στάθμευση",
    "facilities.kitchen": "Παροχές › Κουζίνα & πλυντήριο",
    "facilities.family": "Παροχές › Βρέφη & παιδιά",
    "facilities.leisure": "Παροχές › Πισίνα, σπα & γυμναστήριο",
    "facilities.common": "Παροχές › Κοινόχρηστοι χώροι",
}
SENTIMENT_NAMES = {"positive": "Θετικό", "negative": "Αρνητικό"}
FLAG_NAMES = {
    "unresolved": "ανεπίλυτο",
    "suggestion": "πρόταση βελτίωσης",
    "typo": "πιθανό τυπογραφικό",
    "check": "ελέγξτε",
}
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
# Topics whose category text above is too general to act on.
TOPIC_RECOMMENDATIONS = {
    "cleanliness.linen": "Ελέγξτε τη διαχείριση και την τελική επιθεώρηση του ιματισμού (πετσέτες, σεντόνια) "
                         "πριν από κάθε άφιξη.",
    "cleanliness.odour": "Διερευνήστε από πού έρχεται η οσμή (αποχέτευση, υγρασία, αερισμός) και ζητήστε τεχνικό "
                         "έλεγχο όπου χρειάζεται.",
    "cleanliness.pests": "Κάντε άμεσο έλεγχο και απεντόμωση και καταγράψτε τι έγινε.",
    "staff.response": "Ορίστε σαφές κανάλι επικοινωνίας και χρόνο απάντησης σε μηνύματα και αιτήματα.",
    "staff.resolution": "Επανελέγξτε ό,τι έμεινε ανεπίλυτο και επιβεβαιώστε με τον επισκέπτη ότι λύθηκε.",
    "staff.checkin": "Ελέγξτε πόσο σαφείς είναι οι οδηγίες check-in και αν υπάρχει υποστήριξη κατά την άφιξη.",
    "room.comfort": "Ελέγξτε την άνεση του χώρου: φωτισμό, καθίσματα, θερμοκρασία, μικρές λεπτομέρειες.",
    "room.size": "Περιγράψτε με ακρίβεια το μέγεθος του δωματίου στην αγγελία και αξιοποιήστε καλύτερα τον χώρο.",
    "room.bed": "Ελέγξτε μέγεθος και κατάσταση κρεβατιών και στρωμάτων και γράψτε τις διαστάσεις στην αγγελία.",
    "room.climate": "Συντηρήστε και ελέγξτε κλιματισμό και αερισμό σε κάθε δωμάτιο, ιδίως πριν από την περίοδο "
                    "αιχμής.",
    "room.bathroom": "Ελέγξτε ντους, ζεστό νερό, πίεση νερού και αποχέτευση στο μπάνιο.",
    "room.storage": "Προσθέστε κρεμάστρες, ντουλάπα ή ράφια για ρούχα και βρεγμένες πετσέτες.",
    "room.equipment": "Ελέγξτε και αντικαταστήστε ό,τι λείπει ή έχει φθαρεί στον εξοπλισμό και στα έπιπλα.",
    "room.view": "Περιγράψτε με ακρίβεια θέα και μπαλκόνι στην αγγελία.",
    "food.dining": "Εξετάστε ποιότητα, τιμές και εξυπηρέτηση σε εστιατόριο, μπαρ και καφέ.",
    "value.charges": "Αναφέρετε από πριν όλες τις χρεώσεις, την εγγύηση και τον φόρο διαμονής.",
    "facilities.access": "Ενημερώστε σαφώς για όροφο, σκάλες και ανελκυστήρα πριν από την κράτηση και εξετάστε "
                         "βοήθεια με τις αποσκευές.",
    "facilities.wifi": "Ελέγξτε την κάλυψη και την ταχύτητα του Wi-Fi σε όλους τους χώρους.",
    "facilities.parking": "Δώστε σαφείς πληροφορίες για τη στάθμευση: θέση, κόστος και διαθεσιμότητα.",
    "facilities.kitchen": "Ελέγξτε εξοπλισμό κουζίνας και συσκευές (κουζινικά, πλυντήριο) πριν από κάθε άφιξη.",
    "facilities.family": "Αναφέρετε αν υπάρχουν βρεφική κούνια και καρεκλάκι φαγητού και ετοιμάστε τα όταν η "
                         "κράτηση περιλαμβάνει παιδιά.",
    "facilities.leisure": "Ελέγξτε διαθεσιμότητα, καθαριότητα και ωράριο πισίνας, σπα και γυμναστηρίου.",
    "facilities.common": "Ελέγξτε είσοδο, διαδρόμους, φωτισμό και ασφάλεια των κοινόχρηστων χώρων.",
}

# About a minute of CPU time at the measured ~0.5-1 s per review.
MAX_REVIEWS = 100
MAX_TOTAL_CHARS = 150_000
QUOTE_PREVIEW = 160
_REVIEW_COLUMN = re.compile(r"review|text|comment|content|feedback|κριτικ|σχόλι", re.I)


class InputError(ValueError):
    """A problem with what the owner pasted or uploaded, worded for the owner."""


def recommendation(topic: str) -> str:
    return TOPIC_RECOMMENDATIONS.get(topic) or RECOMMENDATIONS[topic.split(".", 1)[0]]


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
    quote = " ".join(quote.split())
    return quote if len(quote) <= QUOTE_PREVIEW else quote[:QUOTE_PREVIEW - 1].rstrip() + "…"


def group(mentions: list[list[dict]]) -> list[dict]:
    """One finding per (review, topic, sentiment), with every quote it rests on.

    A guest who writes about the small bed three times is one complaint with
    three quotes, not three complaints. "check" stays only if no quote of the
    finding was read with confidence; the other flags carry over from any quote.
    """
    findings: dict[tuple, dict] = {}
    for review, items in enumerate(mentions, start=1):
        for item in items:
            key = (review, item["topic"], item["sentiment"])
            finding = findings.setdefault(key, {
                "review": review, "aspect": item["aspect"], "topic": item["topic"],
                "sentiment": item["sentiment"], "quotes": [], "flags": set(), "sure": False,
            })
            if item["quote"] not in finding["quotes"]:
                finding["quotes"].append(item["quote"])
            finding["flags"] |= set(item["flags"]) - {"check"}
            finding["sure"] |= "check" not in item["flags"]
    for finding in findings.values():
        if not finding.pop("sure"):
            finding["flags"].add("check")
    return list(findings.values())


def summarize(findings: list[dict]) -> dict[str, dict]:
    """Per topic, the reviews that complain about it and the reviews that praise it.

    Counts are reviews: a review that both praises and criticises a topic counts
    on both sides, so a complaint is never hidden behind a compliment.
    """
    summary: dict[str, dict] = {}
    for finding in findings:
        side = "complaints" if finding["sentiment"] == "negative" else "praise"
        summary.setdefault(finding["topic"], {"complaints": {}, "praise": {}})[side][finding["review"]] = finding
    return summary


def _ranked(summary: dict, side: str) -> list[tuple[str, dict]]:
    rows = [(topic, entry[side]) for topic, entry in summary.items() if entry[side]]
    return sorted(rows, key=lambda row: (-len(row[1]), -_mentions(row[1]), row[0]))


def _mentions(by_review: dict) -> int:
    return sum(len(finding["quotes"]) for finding in by_review.values())


def _examples(by_review: dict, limit: int = 2) -> str:
    shortest = {review: min(finding["quotes"], key=len) for review, finding in by_review.items()}
    chosen = sorted(shortest.items(), key=lambda item: len(item[1]))[:limit]
    return " · ".join(f"#{review}: «{_preview(quote)}»" for review, quote in sorted(chosen))


def _notes(by_review: dict, other_side: dict) -> str:
    notes = []
    for flag in ("unresolved", "suggestion", "typo", "check"):
        count = sum(flag in finding["flags"] for finding in by_review.values())
        if count:
            notes.append(f"{FLAG_NAMES[flag]}: {count}")
    mixed = len(by_review.keys() & other_side.keys())
    if mixed:
        notes.append(f"και θετικά στην ίδια κριτική: {mixed}")
    return " · ".join(notes)


def fix_first_rows(summary: dict, analysed: int) -> list[list]:
    return [
        [TOPIC_NAMES[topic], len(reviews), f"{len(reviews) / analysed:.0%}", _mentions(reviews),
         _notes(reviews, summary[topic]["praise"]), _examples(reviews), recommendation(topic)]
        for topic, reviews in _ranked(summary, "complaints")
    ]


def strength_rows(summary: dict, analysed: int) -> list[list]:
    return [
        [TOPIC_NAMES[topic], len(reviews), f"{len(reviews) / analysed:.0%}", _mentions(reviews),
         _examples(reviews)]
        for topic, reviews in _ranked(summary, "praise")
    ]


def _flag_text(finding: dict) -> str:
    return ", ".join(FLAG_NAMES[flag] for flag in FLAG_NAMES if flag in finding["flags"])


def finding_rows(findings: list[dict]) -> list[list]:
    """Every finding, complaints first, with all the clauses it rests on."""
    ordered = sorted(findings, key=lambda f: (f["sentiment"] != "negative", f["review"],
                                              list(TOPIC_NAMES).index(f["topic"])))
    return [
        [f["review"], TOPIC_NAMES[f["topic"]], SENTIMENT_NAMES[f["sentiment"]], len(f["quotes"]), _flag_text(f),
         " | ".join(_preview(quote) for quote in f["quotes"])]
        for f in ordered
    ]


def review_rows(reviews: list[str], findings: list[dict], skipped: set[int]) -> list[list]:
    by_review: dict[int, list[dict]] = {}
    for finding in findings:
        by_review.setdefault(finding["review"], []).append(finding)
    rows = []
    for review, text in enumerate(reviews, start=1):
        items = by_review.get(review, [])
        if review in skipped:
            complaints = praise = "—"
            status = "Δεν αναλύθηκε: δεν φαίνεται αγγλική"
        else:
            complaints = ", ".join(TOPIC_NAMES[f["topic"]] for f in items if f["sentiment"] == "negative") or "—"
            praise = ", ".join(TOPIC_NAMES[f["topic"]] for f in items if f["sentiment"] == "positive") or "—"
            status = "" if items else "Καμία πτυχή με σαφή γνώμη"
        rows.append([review, complaints, praise, status, _preview(text)])
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
        topic, reviews = complaints[0]
        lines.append(f"Συχνότερο παράπονο: **{TOPIC_NAMES[topic]}** σε {len(reviews)} από {analysed} κριτικές.")
    elif analysed:
        lines.append("Δεν εντοπίστηκε κανένα παράπονο.")
    if praise:
        topic, reviews = praise[0]
        lines.append(f"Ό,τι επαινείται περισσότερο: **{TOPIC_NAMES[topic]}** σε {len(reviews)} από {analysed} "
                     "κριτικές.")
    unresolved = {review for entry in summary.values() for review, finding in entry["complaints"].items()
                  if "unresolved" in finding["flags"]}
    if unresolved:
        numbers = ", ".join(f"#{n}" for n in sorted(unresolved))
        lines.append(f"**Πρόβλημα που αναφέρθηκε και δεν λύθηκε:** κριτική {numbers}.")
    if analysed:
        lines.append("Οι αριθμοί μετρούν κριτικές: κάθε κριτική μετρά μία φορά ανά θέμα, όσες φορές κι αν το "
                     "αναφέρει.")
    return "  \n".join(lines)


def write_csv(reviews: list[str], findings: list[dict]) -> str:
    """All findings as a UTF-8 CSV the owner can open in Excel, one row per review and topic."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", prefix="review-findings-", delete=False, encoding="utf-8-sig", newline=""
    )
    ordered = sorted(findings, key=lambda f: (f["review"], list(TOPIC_NAMES).index(f["topic"]), f["sentiment"]))
    with handle:
        writer = csv.writer(handle)
        writer.writerow(["review", "category", "topic", "sentiment", "mentions", "notes", "quotes", "review_text"])
        for f in ordered:
            writer.writerow([f["review"], ASPECT_NAMES[f["aspect"]], TOPIC_NAMES[f["topic"]],
                             SENTIMENT_NAMES[f["sentiment"]], len(f["quotes"]), _flag_text(f),
                             " | ".join(" ".join(quote.split()) for quote in f["quotes"]),
                             reviews[f["review"] - 1]])
    return handle.name
