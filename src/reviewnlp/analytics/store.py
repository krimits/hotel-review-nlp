"""Tenant-scoped SQLite aspect storage without retaining the original reviews."""

from __future__ import annotations

import hashlib
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

    def save_review(
        self, *, hotel_id: str, review_id: str, source: str, text: str,
        language: str, review_date: datetime | None, record: dict,
    ) -> None: ...

    def delete_review(self, hotel_id: str, source: str, review_id: str) -> bool: ...


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

    def delete_review(self, hotel_id: str, source: str, review_id: str) -> bool:
        with self._connection() as connection:
            deleted = connection.execute(
                "DELETE FROM hotel_reviews WHERE hotel_id=? AND source=? AND external_review_id=?",
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
