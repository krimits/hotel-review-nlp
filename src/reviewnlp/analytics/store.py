"""Tenant-scoped SQLite aspect storage without retaining the original reviews."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from reviewnlp.absa.extract import review_key


class AspectStore(Protocol):
    def aspect_rows(self, hotel_id: str, days: int) -> list[dict]: ...

    def previous_period_rows(self, hotel_id: str, days: int) -> list[dict]: ...

    def aspect_evidence(self, hotel_id: str, aspect: str, days: int, limit: int) -> list[dict]: ...

    def save_review(
        self, *, hotel_id: str, review_id: str, source: str, text: str,
        language: str, review_date: datetime | None, record: dict,
    ) -> None: ...

    def delete_review(self, hotel_id: str, source: str, review_id: str) -> bool: ...

    def save_triage(
        self, *, hotel_id: str, review_id: str, source: str, text: str,
        language: str, review_date: datetime | None, result: dict,
    ) -> None: ...

    def triage_summary(self, hotel_id: str, days: int, limit_actions: int, limit_review: int) -> dict: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS hotel_reviews (
    id INTEGER PRIMARY KEY,
    hotel_id TEXT NOT NULL,
    source TEXT NOT NULL,
    external_review_id TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    language TEXT NOT NULL,
    review_date TEXT NOT NULL,
    json_valid INTEGER NOT NULL,
    salvaged INTEGER NOT NULL,
    entries_dropped INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(hotel_id, source, external_review_id)
);
CREATE INDEX IF NOT EXISTS idx_review_hotel_date ON hotel_reviews(hotel_id, review_date);
CREATE TABLE IF NOT EXISTS review_aspects (
    review_pk INTEGER NOT NULL REFERENCES hotel_reviews(id) ON DELETE CASCADE,
    aspect TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    quote TEXT NOT NULL,
    PRIMARY KEY(review_pk, aspect)
);
-- Triage (sentiment -> complaint topics -> suggested actions). Its own tables, not the ABSA ones: the five
-- complaint topics are not the eight aspects, and a triaged review has no ABSA extraction to describe. Like
-- the tables above, nothing here holds the review: only its hash, the answers, and spans quoted from it.
CREATE TABLE IF NOT EXISTS triage_runs (
    id INTEGER PRIMARY KEY,
    hotel_id TEXT NOT NULL,
    source TEXT NOT NULL,
    external_review_id TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    language TEXT NOT NULL,
    review_date TEXT NOT NULL,
    status TEXT NOT NULL,
    sentiment_label TEXT NOT NULL,
    sentiment_confidence REAL,
    sentiment_probabilities TEXT,
    sentiment_model_type TEXT NOT NULL,
    complaints_status TEXT NOT NULL,
    complaints_error TEXT,
    jev_route TEXT,
    jev_model TEXT,
    questions_version TEXT NOT NULL,
    questions_sha256 TEXT NOT NULL,
    qwen_triggered INTEGER NOT NULL,
    routing_reasons TEXT NOT NULL,
    needs_review INTEGER NOT NULL,
    review_reasons TEXT NOT NULL,
    thresholds TEXT NOT NULL,
    actions_status TEXT NOT NULL,
    actions_error TEXT,
    actions_model TEXT,
    prompt_version TEXT,
    actions_dropped INTEGER NOT NULL,
    timings TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(hotel_id, source, external_review_id)
);
CREATE INDEX IF NOT EXISTS idx_triage_hotel_date ON triage_runs(hotel_id, review_date);
CREATE TABLE IF NOT EXISTS triage_complaints (
    triage_pk INTEGER NOT NULL REFERENCES triage_runs(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    answer TEXT NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY(triage_pk, topic)
);
CREATE TABLE IF NOT EXISTS triage_actions (
    id INTEGER PRIMARY KEY,
    triage_pk INTEGER NOT NULL REFERENCES triage_runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    problem TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    measure TEXT NOT NULL,
    department TEXT NOT NULL,
    to_confirm TEXT NOT NULL
);
"""


class SqliteAspectStore:
    """One transaction per review; SQLite WAL is suitable for a small deployment."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save_review(
        self, *, hotel_id: str, review_id: str, source: str, text: str,
        language: str, review_date: datetime | None, record: dict,
    ) -> None:
        date = review_date or datetime.now(timezone.utc)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        date = date.astimezone(timezone.utc).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        # One aspect counts once per review. Conflicting polarities abstain.
        grouped: dict[str, list[dict]] = {}
        if record["aspects"] and not record["json_valid"]:
            raise ValueError("cannot store aspects from invalid JSON")
        for item in record["aspects"]:
            if not item.get("quote") or review_key(item["quote"]) not in review_key(text):
                raise ValueError("cannot store an aspect with an unsupported quote")
            grouped.setdefault(item["aspect"], []).append(item)
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO hotel_reviews
                   (hotel_id, source, external_review_id, text_sha256, language,
                    review_date, json_valid, salvaged, entries_dropped, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(hotel_id, source, external_review_id) DO UPDATE SET
                     text_sha256=excluded.text_sha256, language=excluded.language,
                     review_date=excluded.review_date, json_valid=excluded.json_valid,
                     salvaged=excluded.salvaged, entries_dropped=excluded.entries_dropped,
                     updated_at=excluded.updated_at""",
                (hotel_id, source, review_id, hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 language, date, int(record["json_valid"]), int(record["salvaged"]),
                 record["entries_dropped"], now),
            )
            key = connection.execute(
                "SELECT id FROM hotel_reviews WHERE hotel_id=? AND source=? AND external_review_id=?",
                (hotel_id, source, review_id),
            ).fetchone()["id"]
            connection.execute("DELETE FROM review_aspects WHERE review_pk=?", (key,))
            for aspect, entries in grouped.items():
                counts = {s: sum(item["sentiment"] == s for item in entries)
                          for s in ("positive", "negative", "neutral")}
                winner = max(counts, key=counts.get)
                if list(counts.values()).count(counts[winner]) > 1:
                    winner = "neutral"
                quote = next((i["quote"] for i in entries if i["sentiment"] == winner), entries[0]["quote"])
                connection.execute(
                    "INSERT INTO review_aspects(review_pk, aspect, sentiment, quote) VALUES (?, ?, ?, ?)",
                    (key, aspect, winner, quote),
                )

    def _rows(self, hotel_id: str, start: datetime, end: datetime) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT h.id AS review_id, a.aspect, a.sentiment
                   FROM hotel_reviews AS h JOIN review_aspects AS a ON a.review_pk=h.id
                   WHERE h.hotel_id=? AND h.review_date>=? AND h.review_date<?
                   ORDER BY h.id, a.aspect""",
                (hotel_id, start.isoformat(), end.isoformat()),
            ).fetchall()
            return [dict(row) for row in rows]

    def aspect_rows(self, hotel_id: str, days: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        return self._rows(hotel_id, now - timedelta(days=days), now)

    def previous_period_rows(self, hotel_id: str, days: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        return self._rows(hotel_id, now - timedelta(days=days * 2), now - timedelta(days=days))

    def aspect_evidence(self, hotel_id: str, aspect: str, days: int, limit: int) -> list[dict]:
        """Recent negative quote spans, with original source IDs for manual review."""
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT h.source, h.external_review_id AS review_id,
                          h.review_date, a.quote
                   FROM hotel_reviews AS h JOIN review_aspects AS a ON a.review_pk=h.id
                   WHERE h.hotel_id=? AND a.aspect=? AND a.sentiment='negative'
                     AND h.review_date>=? AND h.review_date<?
                   ORDER BY h.review_date DESC, h.id DESC LIMIT ?""",
                (hotel_id, aspect, (now - timedelta(days=days)).isoformat(),
                 now.isoformat(), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_triage(
        self, *, hotel_id: str, review_id: str, source: str, text: str,
        language: str, review_date: datetime | None, result: dict,
    ) -> None:
        """Keep one triage result per review, replacing an earlier one. Excerpts must be spans of the review."""
        date = review_date or datetime.now(timezone.utc)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        date = date.astimezone(timezone.utc).isoformat()
        sentiment, complaints = result["sentiment"], result["complaints"]
        routing, actions, dump = result["routing"], result["actions"], json.dumps
        for action in actions["actions"]:
            if review_key(action["excerpt"]) not in review_key(text):
                raise ValueError("cannot store an action with an unsupported excerpt")
        answers = [*complaints["topics"], *([complaints["other_complaint"]] if complaints["other_complaint"] else [])]
        values = (
            hotel_id, source, review_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), language, date,
            result["status"], sentiment["label"], sentiment["confidence"],
            None if sentiment["probabilities"] is None else dump(sentiment["probabilities"], sort_keys=True),
            sentiment["model_type"], complaints["status"], complaints["error"], complaints["route"],
            complaints["model"], complaints["questions_version"], complaints["questions_sha256"],
            int(routing["qwen_triggered"]), dump(routing["reasons"]), int(routing["needs_review"]),
            dump(routing["review_reasons"]), dump(routing["thresholds"], sort_keys=True),
            actions["status"], actions["error"], actions["model"], actions["prompt_version"], actions["dropped"],
            dump(result["timings"], sort_keys=True), result["validation_status"],
            datetime.now(timezone.utc).isoformat(),
        )
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO triage_runs
                   (hotel_id, source, external_review_id, text_sha256, language, review_date, status,
                    sentiment_label, sentiment_confidence, sentiment_probabilities, sentiment_model_type,
                    complaints_status, complaints_error, jev_route, jev_model, questions_version, questions_sha256,
                    qwen_triggered, routing_reasons, needs_review, review_reasons, thresholds,
                    actions_status, actions_error, actions_model, prompt_version, actions_dropped,
                    timings, validation_status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(hotel_id, source, external_review_id) DO UPDATE SET
                     text_sha256=excluded.text_sha256, language=excluded.language, review_date=excluded.review_date,
                     status=excluded.status, sentiment_label=excluded.sentiment_label,
                     sentiment_confidence=excluded.sentiment_confidence,
                     sentiment_probabilities=excluded.sentiment_probabilities,
                     sentiment_model_type=excluded.sentiment_model_type, complaints_status=excluded.complaints_status,
                     complaints_error=excluded.complaints_error, jev_route=excluded.jev_route,
                     jev_model=excluded.jev_model, questions_version=excluded.questions_version,
                     questions_sha256=excluded.questions_sha256, qwen_triggered=excluded.qwen_triggered,
                     routing_reasons=excluded.routing_reasons, needs_review=excluded.needs_review,
                     review_reasons=excluded.review_reasons, thresholds=excluded.thresholds,
                     actions_status=excluded.actions_status, actions_error=excluded.actions_error,
                     actions_model=excluded.actions_model, prompt_version=excluded.prompt_version,
                     actions_dropped=excluded.actions_dropped, timings=excluded.timings,
                     validation_status=excluded.validation_status, updated_at=excluded.updated_at""",
                values,
            )
            key = connection.execute(
                "SELECT id FROM triage_runs WHERE hotel_id=? AND source=? AND external_review_id=?",
                (hotel_id, source, review_id),
            ).fetchone()["id"]
            connection.execute("DELETE FROM triage_complaints WHERE triage_pk=?", (key,))
            connection.execute("DELETE FROM triage_actions WHERE triage_pk=?", (key,))
            connection.executemany(
                "INSERT INTO triage_complaints(triage_pk, topic, answer, probability) VALUES (?, ?, ?, ?)",
                [(key, item["topic"], item["answer"], item["probability"]) for item in answers],
            )
            connection.executemany(
                """INSERT INTO triage_actions(triage_pk, position, problem, excerpt, measure, department, to_confirm)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(key, position, a["problem"], a["excerpt"], a["measure"], a["department"], dump(a["to_confirm"]))
                 for position, a in enumerate(actions["actions"])],
            )

    def triage_summary(self, hotel_id: str, days: int, limit_actions: int = 20, limit_review: int = 20) -> dict:
        """What triage found in one hotel's reviews of the last `days`: counts, recent actions, reviews to check."""
        now = datetime.now(timezone.utc)
        window = (hotel_id, (now - timedelta(days=days)).isoformat(), now.isoformat())
        where = "r.hotel_id=? AND r.review_date>=? AND r.review_date<?"
        with self._connection() as connection:
            runs = connection.execute(
                f"""SELECT external_review_id, source, review_date, sentiment_label, complaints_status,
                           needs_review, review_reasons FROM triage_runs AS r WHERE {where}
                    ORDER BY review_date DESC, id DESC""", window).fetchall()
            answers = connection.execute(
                f"""SELECT c.topic, c.answer, COUNT(*) AS n FROM triage_complaints AS c
                    JOIN triage_runs AS r ON r.id=c.triage_pk
                    WHERE {where} AND r.complaints_status='ok' GROUP BY c.topic, c.answer""", window).fetchall()
            actions = connection.execute(
                f"""SELECT r.external_review_id, r.source, r.review_date, a.problem, a.excerpt, a.measure,
                           a.department, a.to_confirm
                    FROM triage_actions AS a JOIN triage_runs AS r ON r.id=a.triage_pk WHERE {where}
                    ORDER BY r.review_date DESC, r.id DESC, a.position LIMIT ?""", (*window, limit_actions)).fetchall()
        sentiment: dict[str, int] = {}
        for run in runs:
            sentiment[run["sentiment_label"]] = sentiment.get(run["sentiment_label"], 0) + 1
        complaints: dict[str, dict[str, int]] = {}
        for row in answers:
            complaints.setdefault(row["topic"], {"yes": 0, "no": 0, "unsure": 0})[row["answer"]] = row["n"]
        return {
            "hotel_id": hotel_id, "period_days": days, "reviews": len(runs), "sentiment": sentiment,
            "complaints_evaluated": sum(run["complaints_status"] == "ok" for run in runs),
            "complaints": complaints,
            "actions": [{"review_id": a["external_review_id"], "source": a["source"], "review_date": a["review_date"],
                         "problem": a["problem"], "excerpt": a["excerpt"], "measure": a["measure"],
                         "department": a["department"], "to_confirm": json.loads(a["to_confirm"])} for a in actions],
            "to_check": [{"review_id": run["external_review_id"], "source": run["source"],
                          "review_date": run["review_date"], "reasons": json.loads(run["review_reasons"])}
                         for run in runs if run["needs_review"]][:limit_review],
        }

    def delete_review(self, hotel_id: str, source: str, review_id: str) -> bool:
        """Remove a review's aspects and its triage result, so a deletion leaves nothing derived from it."""
        with self._connection() as connection:
            deleted = connection.execute(
                "DELETE FROM hotel_reviews WHERE hotel_id=? AND source=? AND external_review_id=?",
                (hotel_id, source, review_id),
            ).rowcount
            deleted += connection.execute(
                "DELETE FROM triage_runs WHERE hotel_id=? AND source=? AND external_review_id=?",
                (hotel_id, source, review_id),
            ).rowcount
        return bool(deleted)


@lru_cache(maxsize=8)
def _store_for_path(path: str) -> SqliteAspectStore:
    return SqliteAspectStore(path)


def get_aspect_store() -> AspectStore | None:
    """Explicit database path enables persistence; absent means demo mode."""
    path = os.getenv("REVIEWNLP_DB_PATH")
    return _store_for_path(path) if path else None
