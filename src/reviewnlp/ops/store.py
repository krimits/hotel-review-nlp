"""SQLite tables of the operations demo: reviews, findings, tickets, the notification outbox and reply drafts.

Identity and idempotency:
- a review is one per (hotel_id, source, review_id). Sending the same text again returns what is stored, so it never
  makes new findings or tickets; the same sentence in two different reviews stays two findings;
- a finding has at most one ticket (UNIQUE finding_id), and a ticket at most one assignment notification (UNIQUE
  send_id). The ticket and its pending notification are written in one transaction, so a crash leaves both or
  neither, and the worker can always find what is left to send.

The review text is kept here: the card shows it and the reply draft needs it. This is the hotel's own database;
the hotel pilot's promise of no stored text was about the pilot's Space, not this product.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from reviewnlp.ops.findings import Analysis

SCHEMA = """
CREATE TABLE IF NOT EXISTS ops_reviews (
  id INTEGER PRIMARY KEY,
  hotel_id TEXT NOT NULL, source TEXT NOT NULL, review_id TEXT NOT NULL,
  text TEXT NOT NULL, text_sha256 TEXT NOT NULL, language TEXT NOT NULL, analyser TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(hotel_id, source, review_id));
CREATE TABLE IF NOT EXISTS ops_findings (
  id INTEGER PRIMARY KEY,
  review_pk INTEGER NOT NULL REFERENCES ops_reviews(id) ON DELETE CASCADE,
  position INTEGER NOT NULL, type TEXT NOT NULL, department TEXT NOT NULL, priority TEXT NOT NULL,
  excerpt TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending', 'resolved', 'unclear')),
  decision TEXT NOT NULL DEFAULT 'proposed' CHECK(decision IN ('proposed', 'confirmed', 'rejected')),
  still_open INTEGER, room TEXT, decided_at TEXT,
  UNIQUE(review_pk, position));
CREATE TABLE IF NOT EXISTS ops_tickets (
  id INTEGER PRIMARY KEY,
  finding_id INTEGER NOT NULL UNIQUE REFERENCES ops_findings(id) ON DELETE CASCADE,
  hotel_id TEXT NOT NULL, department TEXT NOT NULL, priority TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('assigned', 'in_progress', 'resolved')),
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ops_outbox (
  id INTEGER PRIMARY KEY,
  send_id TEXT NOT NULL UNIQUE,
  ticket_id INTEGER NOT NULL REFERENCES ops_tickets(id) ON DELETE CASCADE,
  hotel_id TEXT NOT NULL, recipient TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending', 'sent', 'failed')),
  attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT NOT NULL, last_error TEXT,
  created_at TEXT NOT NULL, sent_at TEXT);
CREATE TABLE IF NOT EXISTS ops_replies (
  review_pk INTEGER PRIMARY KEY REFERENCES ops_reviews(id) ON DELETE CASCADE,
  draft TEXT NOT NULL, approved_text TEXT, state TEXT NOT NULL CHECK(state IN ('draft', 'approved')),
  drafter TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ops_tickets_queue ON ops_tickets(hotel_id, department, state);
CREATE INDEX IF NOT EXISTS ops_outbox_due ON ops_outbox(state, next_attempt_at);
"""
TICKET_STATES = ("assigned", "in_progress", "resolved")


class Conflict(ValueError):
    """The request contradicts what is stored; nothing was changed."""


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SqliteOpsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")  # outside any transaction, where SQLite allows it
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One write transaction (BEGIN IMMEDIATE): all of it is kept, or none."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()

    # --- reviews and findings ------------------------------------------------------------------------------------

    def review_pk(self, connection: sqlite3.Connection, hotel_id: str, source: str, review_id: str) -> int | None:
        row = connection.execute("SELECT id FROM ops_reviews WHERE hotel_id = ? AND source = ? AND review_id = ?",
                                 (hotel_id, source, review_id)).fetchone()
        return int(row["id"]) if row else None

    def find_review(self, hotel_id: str, source: str, review_id: str, text: str) -> int | None:
        """The stored review with this identity and this exact text, or None. Conflict if the text differs and
        a finding of it was already decided."""
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT id, text_sha256 FROM ops_reviews WHERE hotel_id = ? AND source = ? AND review_id = ?",
                (hotel_id, source, review_id)).fetchone()
            if row is None:
                return None
            if row["text_sha256"] == text_sha256(text):
                return int(row["id"])
            decided = connection.execute(
                "SELECT COUNT(*) FROM ops_findings WHERE review_pk = ? AND decision != 'proposed'",
                (row["id"],)).fetchone()[0]
            if decided:
                raise Conflict("this review was already analysed with another text and has decided findings")
            return None

    def save_review(self, hotel_id: str, source: str, review_id: str, text: str, analysis: Analysis,
                    departments: dict[str, tuple[str, str]], now: datetime) -> int:
        """Store the review and its findings; a stored review with another, undecided text is replaced.
        `departments` maps a type to (department, priority), from the catalogue."""
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT id, text_sha256 FROM ops_reviews WHERE hotel_id = ? AND source = ? AND review_id = ?",
                (hotel_id, source, review_id)).fetchone()
            if existing is not None and existing["text_sha256"] == text_sha256(text):
                return int(existing["id"])  # the same review again: what is stored stands
            if existing is not None:
                decided = connection.execute(
                    "SELECT COUNT(*) FROM ops_findings WHERE review_pk = ? AND decision != 'proposed'",
                    (existing["id"],)).fetchone()[0]
                if decided:
                    raise Conflict("this review was already analysed with another text and has decided findings")
                connection.execute("DELETE FROM ops_reviews WHERE id = ?", (existing["id"],))
            inserted = connection.execute(
                "INSERT INTO ops_reviews (hotel_id, source, review_id, text, text_sha256, language, analyser, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (hotel_id, source, review_id, text, text_sha256(text), analysis.language, analysis.analyser,
                 _iso(now))).lastrowid
            if inserted is None:
                raise RuntimeError("SQLite gave no id for the new review")
            pk = int(inserted)
            for position, finding in enumerate(analysis.findings):
                department, priority = departments[finding.type]
                connection.execute(
                    "INSERT INTO ops_findings (review_pk, position, type, department, priority, excerpt, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (pk, position, finding.type, department, priority, finding.excerpt, finding.status))
            return pk

    def review(self, hotel_id: str, source: str, review_id: str) -> dict | None:
        """The review with its findings, their tickets and the reply, or None."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM ops_reviews WHERE hotel_id = ? AND source = ? AND review_id = ?",
                (hotel_id, source, review_id)).fetchone()
            if row is None:
                return None
            findings = [dict(f) for f in connection.execute(
                "SELECT f.*, t.id AS ticket_id, t.state AS ticket_state FROM ops_findings f "
                "LEFT JOIN ops_tickets t ON t.finding_id = f.id WHERE f.review_pk = ? ORDER BY f.position",
                (row["id"],))]
            reply = connection.execute("SELECT * FROM ops_replies WHERE review_pk = ?", (row["id"],)).fetchone()
            return {**dict(row), "findings": findings, "reply": dict(reply) if reply else None}
        finally:
            connection.close()

    def finding(self, finding_id: int) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT f.*, r.hotel_id, r.source, r.review_id, t.id AS ticket_id FROM ops_findings f "
                "JOIN ops_reviews r ON r.id = f.review_pk LEFT JOIN ops_tickets t ON t.finding_id = f.id "
                "WHERE f.id = ?", (finding_id,)).fetchone()
            return dict(row) if row else None
        finally:
            connection.close()

    def decide(self, finding_id: int, decision: str, still_open: bool | None, room: str | None, now: datetime,
               ticket: dict | None = None, notification: dict | None = None) -> int | None:
        """Record the manager's decision and, when given, open its ticket and queue its notification, all in one
        transaction. The same decision again changes nothing and returns the same ticket; another decision on an
        already decided finding is a Conflict."""
        with self.transaction() as connection:
            row = connection.execute("SELECT decision FROM ops_findings WHERE id = ?", (finding_id,)).fetchone()
            if row is None:
                raise KeyError(finding_id)
            if row["decision"] != "proposed":
                if row["decision"] != decision:
                    raise Conflict(f"the finding is already {row['decision']}")
                existing = connection.execute("SELECT id FROM ops_tickets WHERE finding_id = ?",
                                              (finding_id,)).fetchone()
                return int(existing["id"]) if existing else None
            connection.execute(
                "UPDATE ops_findings SET decision = ?, still_open = ?, room = ?, decided_at = ? WHERE id = ?",
                (decision, None if still_open is None else int(still_open), room, _iso(now), finding_id))
            if ticket is None:
                return None
            connection.execute(
                "INSERT INTO ops_tickets (finding_id, hotel_id, department, priority, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'assigned', ?, ?) ON CONFLICT(finding_id) DO NOTHING",
                (finding_id, ticket["hotel_id"], ticket["department"], ticket["priority"], _iso(now), _iso(now)))
            ticket_id = int(connection.execute("SELECT id FROM ops_tickets WHERE finding_id = ?",
                                               (finding_id,)).fetchone()["id"])
            if notification is not None:
                connection.execute(
                    "INSERT INTO ops_outbox (send_id, ticket_id, hotel_id, recipient, subject, body, state, "
                    "next_attempt_at, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?) "
                    "ON CONFLICT(send_id) DO NOTHING",
                    (f"ticket-{ticket_id}-assigned", ticket_id, ticket["hotel_id"], notification["recipient"],
                     notification["subject"], notification["body"].replace("{ticket_id}", str(ticket_id)),
                     _iso(now), _iso(now)))
            return ticket_id

    # --- tickets -------------------------------------------------------------------------------------------------

    def tickets(self, hotel_id: str, department: str | None = None, state: str | None = None) -> list[dict]:
        query = ("SELECT t.*, f.type, f.excerpt, f.room, r.source, r.review_id FROM ops_tickets t "
                 "JOIN ops_findings f ON f.id = t.finding_id JOIN ops_reviews r ON r.id = f.review_pk "
                 "WHERE t.hotel_id = ?")
        args: list[str] = [hotel_id]
        if department:
            query, args = query + " AND t.department = ?", [*args, department]
        if state:
            query, args = query + " AND t.state = ?", [*args, state]
        connection = self._connect()
        try:
            rows = connection.execute(
                query + " ORDER BY CASE t.priority WHEN 'urgent' THEN 0 ELSE 1 END, t.created_at, t.id", args)
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def set_ticket_state(self, hotel_id: str, ticket_id: int, state: str, now: datetime) -> dict:
        if state not in TICKET_STATES:
            raise ValueError(f"unknown ticket state {state!r}")
        with self.transaction() as connection:
            row = connection.execute("SELECT state FROM ops_tickets WHERE id = ? AND hotel_id = ?",
                                     (ticket_id, hotel_id)).fetchone()
            if row is None:
                raise KeyError(ticket_id)
            if TICKET_STATES.index(state) < TICKET_STATES.index(row["state"]):
                raise Conflict(f"a ticket does not go back from {row['state']} to {state}")
            connection.execute("UPDATE ops_tickets SET state = ?, updated_at = ? WHERE id = ?",
                               (state, _iso(now), ticket_id))
        return next(t for t in self.tickets(hotel_id) if t["id"] == ticket_id)

    # --- the outbox ----------------------------------------------------------------------------------------------

    def claim_due(self, now: datetime, lease_seconds: int, limit: int = 20, hotel_id: str | None = None) -> list[dict]:
        """Pending notifications that are due, each claimed for `lease_seconds` so a second worker skips it."""
        claimed = []
        lease_until = _iso(datetime.fromtimestamp(now.timestamp() + lease_seconds, tz=timezone.utc))
        with self.transaction() as connection:
            query = "SELECT * FROM ops_outbox WHERE state = 'pending' AND next_attempt_at <= ?"
            args: list[str | int] = [_iso(now)]
            if hotel_id:
                query, args = query + " AND hotel_id = ?", [*args, hotel_id]
            for row in connection.execute(query + " ORDER BY id LIMIT ?", [*args, limit]).fetchall():
                connection.execute("UPDATE ops_outbox SET next_attempt_at = ? WHERE id = ?", (lease_until, row["id"]))
                claimed.append(dict(row))
        return claimed

    def mark_sent(self, outbox_id: int, now: datetime) -> None:
        with self.transaction() as connection:
            connection.execute("UPDATE ops_outbox SET state = 'sent', attempts = attempts + 1, sent_at = ?, "
                               "last_error = NULL WHERE id = ?", (_iso(now), outbox_id))

    def mark_unsent(self, outbox_id: int, error: str, next_attempt_at: datetime | None) -> None:
        """A failed try: due again at `next_attempt_at`, or failed for good when it is None."""
        with self.transaction() as connection:
            if next_attempt_at is None:
                connection.execute("UPDATE ops_outbox SET state = 'failed', attempts = attempts + 1, last_error = ? "
                                   "WHERE id = ?", (error[:300], outbox_id))
            else:
                connection.execute("UPDATE ops_outbox SET attempts = attempts + 1, last_error = ?, "
                                   "next_attempt_at = ? WHERE id = ?", (error[:300], _iso(next_attempt_at), outbox_id))

    def outbox(self, hotel_id: str) -> list[dict]:
        connection = self._connect()
        try:
            return [dict(row) for row in connection.execute(
                "SELECT id, send_id, ticket_id, recipient, subject, state, attempts, last_error, sent_at "
                "FROM ops_outbox WHERE hotel_id = ? ORDER BY id", (hotel_id,))]
        finally:
            connection.close()

    # --- replies -------------------------------------------------------------------------------------------------

    def save_draft(self, review_pk: int, draft: str, drafter: str, now: datetime) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO ops_replies (review_pk, draft, approved_text, state, drafter, updated_at) "
                "VALUES (?, ?, NULL, 'draft', ?, ?) ON CONFLICT(review_pk) DO UPDATE SET draft = excluded.draft, "
                "approved_text = NULL, state = 'draft', drafter = excluded.drafter, updated_at = excluded.updated_at",
                (review_pk, draft, drafter, _iso(now)))

    def approve_reply(self, review_pk: int, text: str, now: datetime) -> None:
        with self.transaction() as connection:
            updated = connection.execute(
                "UPDATE ops_replies SET approved_text = ?, state = 'approved', updated_at = ? WHERE review_pk = ?",
                (text, _iso(now), review_pk)).rowcount
            if not updated:
                raise Conflict("there is no draft to approve")
