"""Delivery of the outbox: test email to a folder by default, SMTP when the owner configures it.

Each notification has a stable send id (`ticket-<id>-assigned`), used as the e-mail's Message-ID and as the file
name, so a retry after a crash between sending and recording sends the same message, not a second different one.
A failed try is due again after a growing wait; after MAX_ATTEMPTS it is marked failed and shown on the dashboard.
SMTP credentials come from the environment only, set by the owner; nothing here logs them.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from reviewnlp.ops.store import SqliteOpsStore

MAX_ATTEMPTS = 5
FIRST_WAIT_SECONDS = 30
LEASE_SECONDS = 120
SENDER = "reviewnlp-ops@localhost"


class Transport(Protocol):
    name: str

    def send(self, send_id: str, recipient: str, subject: str, body: str) -> None: ...


def message(send_id: str, sender: str, recipient: str, subject: str, body: str) -> EmailMessage:
    mail = EmailMessage()
    mail["From"], mail["To"], mail["Subject"] = sender, recipient, subject
    mail["Message-ID"] = f"<{send_id}@reviewnlp.ops>"
    mail.set_content(body)
    return mail


@dataclass
class FileTransport:
    """Test email: each message is written as `<send_id>.eml`; sending it again overwrites the same file."""

    folder: Path
    name: str = "file"

    def send(self, send_id: str, recipient: str, subject: str, body: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / f"{send_id}.eml").write_bytes(bytes(message(send_id, SENDER, recipient, subject, body)))


@dataclass
class SmtpTransport:
    host: str
    port: int
    user: str
    password: str
    sender: str
    name: str = "smtp"

    @classmethod
    def from_env(cls) -> SmtpTransport | None:
        host = os.getenv("REVIEWNLP_OPS_SMTP_HOST", "").strip()
        if not host:
            return None
        return cls(host=host, port=int(os.getenv("REVIEWNLP_OPS_SMTP_PORT", "587")),
                   user=os.getenv("REVIEWNLP_OPS_SMTP_USER", "").strip(),
                   password=os.getenv("REVIEWNLP_OPS_SMTP_PASSWORD", ""),
                   sender=os.getenv("REVIEWNLP_OPS_SMTP_FROM", SENDER).strip())

    def send(self, send_id: str, recipient: str, subject: str, body: str) -> None:
        with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
            # A verifying context: without it STARTTLS checks no certificate and the login could go to anyone.
            smtp.starttls(context=ssl.create_default_context())
            if self.user:
                smtp.login(self.user, self.password)
            smtp.send_message(message(send_id, self.sender, recipient, subject, body))


def transport_from_env() -> Transport:
    """SMTP when REVIEWNLP_OPS_SMTP_HOST is set; otherwise test email under REVIEWNLP_OPS_MAIL_DIR."""
    smtp = SmtpTransport.from_env()
    if smtp is not None:
        return smtp
    return FileTransport(Path(os.getenv("REVIEWNLP_OPS_MAIL_DIR", "runs/ops_mail")))


def next_try(attempts_before: int, now: datetime) -> datetime | None:
    """When a notification that has now failed `attempts_before + 1` times is due again, or None to give up."""
    if attempts_before + 1 >= MAX_ATTEMPTS:
        return None
    return now + timedelta(seconds=FIRST_WAIT_SECONDS * 2 ** attempts_before)


def deliver_due(store: SqliteOpsStore, transport: Transport, now: datetime | None = None,
                hotel_id: str | None = None, limit: int = 20) -> dict[str, int]:
    """Send what is due. Every error is retried later: a mail server that is down for a while must not lose a
    ticket's notice. Returns how many were sent, put back for later, and given up."""
    now = now or datetime.now(timezone.utc)
    counts = {"sent": 0, "retry": 0, "failed": 0}
    for item in store.claim_due(now, LEASE_SECONDS, limit=limit, hotel_id=hotel_id):
        try:
            transport.send(item["send_id"], item["recipient"], item["subject"], item["body"])
        except Exception as error:  # noqa: BLE001 - recorded on the row and retried; the worker keeps going
            due = next_try(item["attempts"], now)
            store.mark_unsent(item["id"], f"{type(error).__name__}: {error}", due)
            counts["retry" if due else "failed"] += 1
        else:
            store.mark_sent(item["id"], now)
            counts["sent"] += 1
    return counts
