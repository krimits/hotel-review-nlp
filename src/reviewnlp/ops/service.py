"""The rules of the operations demo, between the API and the store.

1. A review becomes a card of findings. The same review sent again is not analysed again.
2. The manager confirms or rejects each finding. A confirmed finding opens a ticket only when its type has an
   operational measure (not structural) and the problem is still open: the guest says so (`pending`), or staff
   confirm it (`still_open`). A finding the guest reports as fixed (`resolved`) is counted, and opens nothing,
   unless staff say it is open again. An `unclear` one waits for staff to say. Where the catalogue needs the room,
   reception must give it before the ticket opens.
3. The ticket goes to the catalogue's department, with its notification queued in the same transaction.
4. The reply is drafted once every finding is decided, from the confirmed ones and their tickets' real state.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from reviewnlp.ops.catalogue import Catalogue, load_catalogue
from reviewnlp.ops.findings import Analyser, analyser_from_env
from reviewnlp.ops.llm import OpenRouterClient
from reviewnlp.ops.replies import Drafter, Fact, LLMDrafter, TemplateDrafter, state_of
from reviewnlp.ops.store import Conflict, SqliteOpsStore

STATUS_EL = {"pending": "Εκκρεμεί", "resolved": "Αναφέρεται ως λυμένο", "unclear": "Χρειάζεται έλεγχο"}
TICKET_EL = {"assigned": "Ανατέθηκε", "in_progress": "Σε εξέλιξη", "resolved": "Επιλύθηκε"}
DEFAULT_RECIPIENT = "ops-demo@example.invalid"


class NeedsInput(ValueError):
    """The decision cannot be recorded yet; the message says, in Greek, what is missing."""


def recipients_from_env() -> dict[str, str]:
    """REVIEWNLP_OPS_NOTIFY_JSON: {"maintenance": "...", "*": "..."}; a test address when unset."""
    import json

    raw = os.getenv("REVIEWNLP_OPS_NOTIFY_JSON", "").strip()
    mapping = json.loads(raw) if raw else {}
    if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
        raise ValueError("REVIEWNLP_OPS_NOTIFY_JSON must map departments to e-mail addresses")
    return {"*": DEFAULT_RECIPIENT, **mapping}


@dataclass
class OpsService:
    store: SqliteOpsStore
    analyser: Analyser
    drafter: Drafter
    catalogue: Catalogue = field(default_factory=load_catalogue)
    recipients: dict[str, str] = field(default_factory=lambda: {"*": DEFAULT_RECIPIENT})
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))

    # --- 1. the card ---------------------------------------------------------------------------------------------

    def submit(self, hotel_id: str, source: str, review_id: str, text: str) -> dict:
        text = text.strip()
        if not text:
            raise NeedsInput("Η κριτική είναι κενή.")
        if self.store.find_review(hotel_id, source, review_id, text) is None:
            analysis = self.analyser.analyse(text)
            departments = {t.id: (t.department, t.priority) for t in self.catalogue.types.values()}
            self.store.save_review(hotel_id, source, review_id, text, analysis, departments, self.clock())
        return self.card(hotel_id, source, review_id)

    def card(self, hotel_id: str, source: str, review_id: str) -> dict:
        stored = self.store.review(hotel_id, source, review_id)
        if stored is None:
            raise KeyError(review_id)
        return {
            "hotel_id": hotel_id, "source": source, "review_id": review_id, "text": stored["text"],
            "language": stored["language"], "analyser": stored["analyser"],
            "findings": [self._finding_view(f) for f in stored["findings"]],
            "reply": stored["reply"], "reply_ready": all(f["decision"] != "proposed" for f in stored["findings"])}

    def _finding_view(self, finding: dict) -> dict:
        kind = self.catalogue.types[finding["type"]]
        return {
            "id": finding["id"], "type": kind.id, "label": kind.label_el, "excerpt": finding["excerpt"],
            "department": kind.department, "department_label": self.catalogue.departments_el[kind.department],
            "priority": kind.priority, "priority_label": self.catalogue.priorities_el[kind.priority],
            "status": finding["status"], "status_label": STATUS_EL[finding["status"]],
            "measure": kind.measure_el, "deadline": self.catalogue.deadlines_el.get(kind.deadline or "", None),
            "needs_room": kind.needs_room, "opens_ticket": kind.opens_ticket,
            "decision": finding["decision"], "still_open": None if finding["still_open"] is None
            else bool(finding["still_open"]), "room": finding["room"],
            "ticket_id": finding.get("ticket_id"), "ticket_state": finding.get("ticket_state"),
            "ticket_state_label": TICKET_EL.get(finding.get("ticket_state") or "")}

    # --- 2. and 3. decisions, tickets, notifications --------------------------------------------------------------

    def _own_finding(self, hotel_id: str, finding_id: int) -> dict:
        finding = self.store.finding(finding_id)
        if finding is None or finding["hotel_id"] != hotel_id:
            raise KeyError(finding_id)
        return finding

    def confirm(self, hotel_id: str, finding_id: int, still_open: bool | None = None,
                room: str | None = None) -> dict:
        finding = self._own_finding(hotel_id, finding_id)
        kind = self.catalogue.types[finding["type"]]
        room = (room or "").strip() or None
        if kind.opens_ticket and finding["status"] == "unclear" and still_open is None:
            raise NeedsInput("Η κριτική δεν λέει αν το πρόβλημα λύθηκε. Ρωτήστε το προσωπικό αν παραμένει.")
        opens = kind.opens_ticket and (still_open if still_open is not None else finding["status"] == "pending")
        if opens and kind.needs_room and room is None:
            raise NeedsInput("Ζητήστε τον αριθμό δωματίου από τη ρεσεψιόν πριν ανοίξει το ticket.")
        ticket = notification = None
        if opens:
            ticket = {"hotel_id": hotel_id, "department": kind.department, "priority": kind.priority}
            notification = self._notification(finding, kind, room)
        self.store.decide(finding_id, "confirmed", still_open, room, self.clock(), ticket, notification)
        return self.card(hotel_id, finding["source"], finding["review_id"])

    def reject(self, hotel_id: str, finding_id: int) -> dict:
        finding = self._own_finding(hotel_id, finding_id)
        self.store.decide(finding_id, "rejected", None, None, self.clock())
        return self.card(hotel_id, finding["source"], finding["review_id"])

    def _notification(self, finding: dict, kind, room: str | None) -> dict:
        department = self.catalogue.departments_el[kind.department]
        priority = self.catalogue.priorities_el[kind.priority]
        deadline = self.catalogue.deadlines_el.get(kind.deadline or "", kind.deadline or "")
        where = f" · δωμάτιο {room}" if room else ""
        lines = [f"Νέο ticket #{{ticket_id}} για το τμήμα: {department}", "",
                 f"Πρόβλημα: {kind.label_el} ({priority}){where}",
                 f"Τι λέει ο επισκέπτης: «{finding['excerpt']}»",
                 f"Ενέργεια: {kind.measure_el}"]
        if deadline:
            lines.append(f"Προθεσμία: {deadline}")
        lines += ["", f"Κριτική: {finding['source']} / {finding['review_id']}"]
        recipient = self.recipients.get(kind.department) or self.recipients["*"]
        return {"recipient": recipient, "subject": f"[{priority}] {department} · {kind.label_el}{where}",
                "body": "\n".join(lines)}

    def tickets(self, hotel_id: str, department: str | None = None, state: str | None = None) -> list[dict]:
        out = []
        for ticket in self.store.tickets(hotel_id, department, state):
            kind = self.catalogue.types[ticket["type"]]
            out.append({**ticket, "label": kind.label_el, "state_label": TICKET_EL[ticket["state"]],
                        "department_label": self.catalogue.departments_el[ticket["department"]],
                        "priority_label": self.catalogue.priorities_el[ticket["priority"]]})
        return out

    def move_ticket(self, hotel_id: str, ticket_id: int, state: str) -> dict:
        self.store.set_ticket_state(hotel_id, ticket_id, state, self.clock())
        return next(t for t in self.tickets(hotel_id) if t["id"] == ticket_id)

    # --- 4. the reply ---------------------------------------------------------------------------------------------

    def draft_reply(self, hotel_id: str, source: str, review_id: str) -> dict:
        stored = self.store.review(hotel_id, source, review_id)
        if stored is None:
            raise KeyError(review_id)
        if any(f["decision"] == "proposed" for f in stored["findings"]):
            raise NeedsInput("Επιβεβαιώστε ή απορρίψτε πρώτα όλα τα ευρήματα.")
        facts = []
        for finding in stored["findings"]:
            state = state_of(finding)
            if state is not None:
                kind = self.catalogue.types[finding["type"]]
                facts.append(Fact(problem=f"{kind.id}: {kind.definition}", state=state))
        draft = self.drafter.draft(stored["text"], stored["language"], facts)
        self.store.save_draft(stored["id"], draft, self.drafter.name, self.clock())
        return self.card(hotel_id, source, review_id)

    def approve_reply(self, hotel_id: str, source: str, review_id: str, text: str) -> dict:
        stored = self.store.review(hotel_id, source, review_id)
        if stored is None:
            raise KeyError(review_id)
        if not text.strip():
            raise NeedsInput("Η απάντηση είναι κενή.")
        self.store.approve_reply(stored["id"], text.strip(), self.clock())
        return self.card(hotel_id, source, review_id)


def service_from_env(store: SqliteOpsStore) -> OpsService | None:
    """The service with OpenRouter when OPENROUTER_API_KEY is set; with the demo's stand-ins when
    REVIEWNLP_OPS_DEMO=1; otherwise None (the routes then say analysis is not configured)."""
    demo = os.getenv("REVIEWNLP_OPS_DEMO", "").strip() == "1"
    analyser = analyser_from_env(fake_allowed=demo)
    if analyser is None:
        return None
    client = OpenRouterClient.from_env()
    catalogue = load_catalogue()
    drafter: Drafter = LLMDrafter(client) if client is not None else TemplateDrafter(
        {t.id: t.label_el for t in catalogue.types.values()})
    return OpsService(store, analyser, drafter, catalogue, recipients_from_env())


__all__ = ["Conflict", "NeedsInput", "OpsService", "service_from_env"]
