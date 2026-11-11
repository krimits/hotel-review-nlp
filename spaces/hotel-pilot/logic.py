"""The logic of the hotel pilot Space: pure functions and a small store, so the app stays a thin Gradio layer.

- the call to the model: the request, the checks on its answer and the waits, as the system was tested;
- a hotel's urgent set, and which reported types are urgent flags for it;
- what counts as a complete answer, and the record that is saved, which holds no review text.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import http.client
import io
import json
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable

MODEL = "anthropic/claude-haiku-5.5-20261007"
URL = "https://openrouter.ai/api/v1/chat/completions"
BATCH = 25  # reviews per call, as tested
CAP = 300  # reviews analysed per hotel
SUBMISSION = 100  # reviews accepted at once, so a hotel waits a minute, not five
MAX_CHARS = 8000  # a longer review is refused, never cut
TRIES = 3  # the first call and at most two more when the answer cannot be used
TRANSIENT = {408, 429, 500, 502, 503, 504}
WAITS = (1.0, 3.0, 6.0)  # seconds between calls that gave no answer; then the batch fails
TIMEOUT = 120.0
VERDICTS = {"Σωστό και επείγον": "accepted", "Υπάρχει, αλλά δεν είναι επείγον": "not_urgent",
            "Δεν υπάρχει τέτοιο πρόβλημα": "absent"}
MISSING = {"Όχι": False, "Ναι": True}
OTHERS = {"Ναι": "yes", "Εν μέρει": "partly", "Όχι": "no"}
USEFUL = ["1", "2", "3", "4", "5"]
OTHER_TYPE = "Άλλο"
Transport = Callable[[str, dict, str, float], tuple[int, dict, bytes]]


class OutputError(ValueError):
    """An answer whose labels cannot be used; the call is tried again."""


class AnalysisError(RuntimeError):
    """The analysis failed; nothing is shown or recorded. `kind` is for the log, never for the hotel."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


# --- the hotel and its reviews ----------------------------------------------------------------------------------------

def hotel_for(code: str, codes: dict[str, str]) -> str | None:
    """The pseudonym (P01, ...) of an access code, from {sha256(code): pseudonym}; None if it is not one."""
    return codes.get(hashlib.sha256(code.strip().encode("utf-8")).hexdigest())


def split_reviews(text: str) -> list[str]:
    """One review per paragraph: blank lines separate them; white space inside a review is collapsed."""
    reviews, current = [], []
    for line in [*text.replace("\r\n", "\n").split("\n"), ""]:
        if line.strip():
            current.append(line.strip())
        elif current:
            reviews.append(" ".join(current))
            current = []
    return reviews


def reviews_from_csv(text: str) -> list[str]:
    """The reviews of a CSV: its `review`, `text` or `κριτική` column, else its first column."""
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    column = next((header.index(name) for name in ("review", "text", "κριτική") if name in header), None)
    body = rows[1:] if column is not None else rows
    column = column or 0
    return [" ".join(row[column].split()) for row in body if len(row) > column and row[column].strip()]


def script_of(text: str) -> str:
    """greek, latin or other: the script most of the review's letters are in."""
    counts = {"greek": 0, "latin": 0, "other": 0}
    for char in text:
        if char.isalpha():
            name = unicodedata.name(char, "")
            counts["greek" if name.startswith("GREEK") else "latin" if name.startswith("LATIN") else "other"] += 1
    return max(counts, key=lambda k: counts[k]) if any(counts.values()) else "other"


def text_key(text: str, salt: str) -> str:
    """A salted hash of the review, to count a repeated review once without keeping its text."""
    return hashlib.sha256((salt + "|" + " ".join(text.split())).encode("utf-8")).hexdigest()[:24]


def salt_from(codes: dict[str, str]) -> str:
    """The salt: a hash of the access codes' own hashes, so no further secret is needed. It changes with the codes."""
    return hashlib.sha256(json.dumps(codes, sort_keys=True).encode("utf-8")).hexdigest()


def check_reviews(reviews: list[str], rated: set[str], analysed: set[str], salt: str) -> tuple[list[dict], str]:
    """The reviews to analyse now, and a Greek note on what was left out. A review already rated is skipped; a
    review longer than MAX_CHARS refuses the whole submission, because a review is never cut."""
    too_long = [i for i, text in enumerate(reviews, start=1) if len(text) > MAX_CHARS]
    if too_long:
        raise ValueError(f"Οι κριτικές {too_long[:10]} ξεπερνούν τους {MAX_CHARS:,} χαρακτήρες. Δεν αναλύθηκε καμία· "
                         "χωρίστε τις ή αφαιρέστε τις και δοκιμάστε ξανά.")
    seen, fresh, repeated = set(), [], 0
    for text in reviews:
        key = text_key(text, salt)
        if key in rated or key in seen:
            repeated += 1
            continue
        seen.add(key)
        fresh.append({"key": key, "text": text, "script": script_of(text)})
    room = max(CAP - len(analysed | rated), 0)
    new = [r for r in fresh if r["key"] not in analysed]
    again = [r for r in fresh if r["key"] in analysed]  # analysed before, never rated: they cost no room
    kept = (again + new[:room])[:SUBMISSION]
    notes = []
    if repeated:
        notes.append(f"{repeated} κριτικές είχαν ήδη βαθμολογηθεί ή επαναλαμβάνονταν και παραλείφθηκαν.")
    if len(kept) < len(fresh):
        notes.append(f"Αναλύονται {len(kept)} από τις {len(fresh)}: έως {SUBMISSION} τη φορά και έως {CAP} συνολικά.")
    return kept, " ".join(notes)


# --- the call to the model, as tested ---------------------------------------------------------------------------------

def schema(types: list[str]) -> dict:
    entry = {"type": "object", "additionalProperties": False, "required": ["item", "types"],
             "properties": {"item": {"type": "integer"},
                            "types": {"type": "array", "items": {"type": "string", "enum": types}}}}
    return {"type": "object", "additionalProperties": False, "required": ["labels"],
            "properties": {"labels": {"type": "array", "items": entry}}}


def request_body(prompt: str, reviews: list[dict], types: list[str]) -> dict:
    """One call: the prompt as system message, the reviews as a JSON list, the catalogue's schema; nothing else."""
    return {"model": MODEL,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(reviews, ensure_ascii=False)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "labels", "strict": True, "schema": schema(types)}},
            "usage": {"include": True}}


def parse(raw: bytes, items: list[int], types: list[str]) -> dict[int, list[str]]:
    """The labels of one answer, or OutputError if they do not cover exactly the items sent."""
    try:
        content = json.loads(raw)["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        entries = json.loads(content)["labels"]
        pairs = [(int(entry["item"]), list(entry["types"])) for entry in entries]
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise OutputError(f"unparsable output: {type(error).__name__}") from None
    got = [item for item, _ in pairs]
    if sorted(got) != sorted(items) or len(set(got)) != len(got):
        raise OutputError("the items differ from those sent")
    unknown = sorted({t for _, ts in pairs for t in ts} - set(types))
    if unknown:
        raise OutputError(f"unknown types {unknown[:5]}")
    return {item: sorted(set(ts)) for item, ts in pairs}


def cost_of(raw: bytes) -> float | None:
    """The cost OpenRouter reports for a call, in USD; None when it reports none."""
    try:
        cost = (json.loads(raw).get("usage") or {}).get("cost")
    except (ValueError, AttributeError):
        return None
    return float(cost) if isinstance(cost, (int, float)) else None


def urllib_transport(url: str, body: dict, api_key: str, timeout: float) -> tuple[int, dict, bytes]:
    """One POST: (status, headers, body). Timeouts and network failures raise."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - a fixed https URL
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


def send(transport: Transport, body: dict, api_key: str, sleep: Callable[[float], None]) -> bytes:
    """One answer. A rate limit, a server error or a network failure is waited out up to len(WAITS) times; any other
    status fails at once: a wrong key or an empty balance does not get better by asking again."""
    for wait in (*WAITS, None):
        try:
            status, _, raw = transport(URL, body, api_key, TIMEOUT)
        except (OSError, http.client.HTTPException) as error:
            status, raw, kind = None, b"", f"network: {type(error).__name__}"
        else:
            if status == 200:
                return raw
            kind = f"status {status}"
            if status not in TRANSIENT:
                raise AnalysisError(kind)
        if wait is None:
            raise AnalysisError(kind)
        sleep(wait)
    raise AnalysisError("unreachable")


def analyse(reviews: list[dict], prompt: str, types: list[str], api_key: str, transport: Transport = urllib_transport,
            sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> list[dict]:
    """The reported types of each review, BATCH per call; each review also carries its call's seconds, cost and size.
    One batch that cannot be analysed fails the whole submission, so a hotel never rates half of what it brought."""
    out = []
    for start in range(0, len(reviews), BATCH):
        part = reviews[start:start + BATCH]
        items = list(range(1, len(part) + 1))
        body = request_body(prompt, [{"item": i, "text": r["text"]} for i, r in zip(items, part, strict=True)], types)
        began, found, raw = clock(), None, b""
        for _ in range(TRIES):
            raw = send(transport, body, api_key, sleep)
            try:
                found = parse(raw, items, types)
                break
            except OutputError:
                continue
        if found is None:
            raise AnalysisError("the answer could not be used after every try")
        call = {"seconds": round(clock() - began, 2), "cost_usd": cost_of(raw), "reviews": len(part)}
        out += [{**review, "types": found[i], "call": call} for i, review in zip(items, part, strict=True)]
    return out


def fake_analyse(reviews: list[dict], types: list[str]) -> list[dict]:
    """The preview's stand-in: types picked from a hash of the text. It reads nothing and means nothing."""
    out = []
    for review in reviews:
        digest = hashlib.sha256(review["text"].encode("utf-8")).digest()
        picked = sorted({types[digest[i] % len(types)] for i in range(digest[0] % 3 + 1)})
        out.append({**review, "types": picked, "call": {"seconds": 0.0, "cost_usd": 0.0, "reviews": len(reviews)}})
    return out


# --- the urgent set and what is shown ---------------------------------------------------------------------------------

def urgent_types(display: dict) -> list[str]:
    """catalogue_v2's urgent types, in its order: the ones a hotel can switch on and off."""
    return [t for t in display["order"] if display["types"][t]["priority"] == "urgent"]


def flags(types: Iterable[str], urgent_set: Iterable[str], display: dict) -> list[str]:
    """The urgent flags of one review for this hotel: its reported types that are in the hotel's urgent set."""
    chosen = set(urgent_set)
    return [t for t in display["order"] if t in set(types) and t in chosen]


def render_others(types: Iterable[str], urgent_set: Iterable[str], display: dict) -> str:
    """The problems that are not urgent flags, in Greek. An urgent type the hotel switched off shows as routine."""
    flagged = set(flags(types, urgent_set, display))
    rest = [t for t in display["order"] if t in set(types) and t not in flagged]
    if not rest:
        return "_Κανένα άλλο πρόβλημα._"
    lines = []
    for t in rest:
        entry = display["types"][t]
        label = display["routine_el"] if entry["priority"] == "urgent" else entry["priority_el"]
        lines.append(f"- {label} · **{entry['label']}** → {entry['department_el']}<br>Ενέργεια: {entry['measure']}")
    return "\n".join(lines)


# --- the answers and the record ---------------------------------------------------------------------------------------

def problems(flagged: list[str], verdicts: dict[str, str | None], missing: str | None, missing_type: str | None,
             others: str | None, useful: str | None, display: dict) -> list[str]:
    """What is missing from the answers to one review, in Greek; empty when they are complete."""
    errors = [f"Απαντήστε για την επείγουσα σήμανση «{display['types'][t]['label']}»."
              for t in flagged if verdicts.get(t) not in VERDICTS]
    if missing not in MISSING:
        errors.append("Απαντήστε αν λείπει κάποιο επείγον πρόβλημα.")
    elif MISSING[missing] and not missing_type:
        errors.append("Διαλέξτε ποιο επείγον πρόβλημα λείπει.")
    if others not in OTHERS:
        errors.append("Απαντήστε για τα υπόλοιπα προβλήματα.")
    if useful not in USEFUL:
        errors.append("Βάλτε χρησιμότητα από 1 έως 5.")
    return errors


def record(hotel: str, review: dict, urgent_set: list[str], verdicts: dict[str, str], missing: str,
           missing_type: str | None, others: str, useful: str, meta: dict, display: dict,
           now: dt.datetime | None = None) -> dict:
    """The saved judgement of one review. It holds the review's hash and script, never its text."""
    flagged = flags(review["types"], urgent_set, display)
    labels = {display["types"][t]["label"]: t for t in urgent_types(display)}
    return {"hotel": hotel, "key": review["key"], "script": review["script"], "types": list(review["types"]),
            "urgent_set": [t for t in display["order"] if t in set(urgent_set)],
            "flags": {t: VERDICTS[verdicts[t]] for t in flagged},
            "missing_urgent": MISSING[missing],
            "missing_type": (labels.get(missing_type or "", "other") if MISSING[missing] else None),
            "others": OTHERS[others], "useful": int(useful), "call": review["call"],
            "model": meta.get("model"), "prompt_sha256": meta.get("prompt_sha256"),
            "protocol_sha256": meta.get("protocol_sha256"), "bundle": meta.get("bundle"),
            "submitted_at": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")}


def analysed_note(hotel: str, review: dict, now: dt.datetime | None = None) -> dict:
    """What is kept when a review is analysed, before any judgement: enough to count it, nothing of its text."""
    return {"hotel": hotel, "key": review["key"], "script": review["script"],
            "analysed_at": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")}


class MemoryStore:
    """Records kept in memory: the preview mode and the tests. Paths are the private dataset's."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def _names(self) -> list[str]:
        return list(self.files)

    def _write(self, path: str, content: dict, message: str) -> None:
        self.files[path] = json.dumps(content, ensure_ascii=False, indent=1)

    def _read(self, path: str) -> dict | None:
        return json.loads(self.files[path]) if path in self.files else None

    def keys(self, hotel: str, kind: str) -> set[str]:
        """The review keys of a hotel under `records` (rated) or `analysed`."""
        prefix = f"{kind}/{hotel}/"
        return {name[len(prefix):-5] for name in self._names() if name.startswith(prefix) and name.endswith(".json")}

    def save_record(self, rating: dict) -> bool:
        path = f"records/{rating['hotel']}/{rating['key']}.json"
        if path in self._names():
            return False  # the first judgement of a review is kept
        self._write(path, rating, f"{rating['hotel']} rated a review")
        return True

    def save_analysed(self, notes: list[dict]) -> None:
        known = set(self._names())
        for note in notes:
            path = f"analysed/{note['hotel']}/{note['key']}.json"
            if path not in known:
                self._write(path, note, f"{note['hotel']} analysed a review")

    def urgent_set(self, hotel: str) -> list[str] | None:
        saved = self._read(f"settings/{hotel}.json")
        return saved["urgent_set"] if saved else None

    def save_urgent_set(self, hotel: str, urgent_set: list[str], now: dt.datetime | None = None) -> None:
        stamp = (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")
        self._write(f"settings/{hotel}.json", {"hotel": hotel, "urgent_set": urgent_set, "set_at": stamp},
                    f"{hotel} set its urgent types")


class HubStore(MemoryStore):
    """Records in a private Hugging Face dataset."""

    def __init__(self, repo_id: str, token: str) -> None:
        from huggingface_hub import HfApi

        super().__init__()
        self.api, self.repo_id = HfApi(token=token), repo_id

    def _names(self) -> list[str]:
        return self.api.list_repo_files(self.repo_id, repo_type="dataset")

    def _write(self, path: str, content: dict, message: str) -> None:
        self.api.upload_file(path_or_fileobj=json.dumps(content, ensure_ascii=False, indent=1).encode("utf-8"),
                             path_in_repo=path, repo_id=self.repo_id, repo_type="dataset", commit_message=message)

    def _read(self, path: str) -> dict | None:
        from huggingface_hub.errors import EntryNotFoundError

        try:
            local = self.api.hf_hub_download(self.repo_id, path, repo_type="dataset", force_download=True)
        except EntryNotFoundError:
            return None
        with open(local, encoding="utf-8") as handle:
            return json.load(handle)

    def save_analysed(self, notes: list[dict]) -> None:
        """One commit for a whole submission, not one per review."""
        from huggingface_hub import CommitOperationAdd

        known = set(self._names())
        operations = [CommitOperationAdd(path_in_repo=f"analysed/{n['hotel']}/{n['key']}.json",
                                         path_or_fileobj=json.dumps(n, ensure_ascii=False, indent=1).encode("utf-8"))
                      for n in notes if f"analysed/{n['hotel']}/{n['key']}.json" not in known]
        if operations:
            self.api.create_commit(self.repo_id, operations, repo_type="dataset",
                                   commit_message=f"{notes[0]['hotel']} analysed {len(operations)} review(s)")
