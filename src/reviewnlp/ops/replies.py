"""Reply drafts to a guest, in the review's language, from what the hotel has actually done.

The drafter sees only facts the manager confirmed, each with the state of what was done about it:
- `forwarded`: a ticket exists and is not resolved: the reply may say the matter was passed to the department;
- `fixed`: its ticket is resolved: the reply may say it was put right;
- `noted`: confirmed without a ticket (fixed during the stay, structural, or not open any more): the reply may
  acknowledge and apologise, and promise nothing.
It never sees a room number, a booking reference or a name, so the draft cannot repeat them. The manager edits,
approves and copies the reply; nothing is posted from here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from reviewnlp.ops.llm import LLMError, OpenRouterClient

STATES = ("forwarded", "fixed", "noted")
PROMPT_VERSION = "ops-reply-v1"


@dataclass(frozen=True)
class Fact:
    problem: str  # the catalogue's English definition label, e.g. "hvac_failure: air conditioning or heating"
    state: str


class Drafter(Protocol):
    name: str

    def draft(self, review: str, language: str, facts: list[Fact]) -> str: ...


REPLY_PROMPT = """# A hotel's reply to a guest review

Write the hotel manager's public reply to the guest review you are given, for the manager to edit and approve.

- Write in the language of the review (its ISO 639-1 code is given), in a warm, plain, professional tone, in 3 to 6
  sentences. Thank the guest, and name something they praised if they praised anything.
- Address each listed problem, and only those, according to its state:
  - `forwarded`: say it has been passed to the department responsible to be looked at. Do not say it is fixed.
  - `fixed`: you may say it has been put right.
  - `noted`: acknowledge it and apologise; promise nothing.
- Never invent facts, compensation, discounts, names, dates, room numbers or booking details. Do not mention
  tickets, systems or departments by internal names.
- Return only a JSON object: {"reply": "..."}.
"""
REPLY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reply"],
                "properties": {"reply": {"type": "string"}}}


@dataclass
class LLMDrafter:
    client: OpenRouterClient
    name: str = "openrouter"

    def draft(self, review: str, language: str, facts: list[Fact]) -> str:
        user = json.dumps({"language": language, "review": review,
                           "problems": [{"problem": f.problem, "state": f.state} for f in facts]}, ensure_ascii=False)
        answer = self.client.json_call(REPLY_PROMPT, user, "reply", REPLY_SCHEMA)
        reply = str(answer.get("reply", "")).strip()
        if not reply:
            raise LLMError("an empty reply")
        return reply


_EL = {"thanks": "Σας ευχαριστούμε θερμά για την κριτική σας.",
       "sorry": "Λυπούμαστε για τα προβλήματα που αναφέρετε: {problems}.",
       "forwarded": "Έχουμε διαβιβάσει {what} στα αρμόδια τμήματα για έλεγχο.",
       "fixed": "{what} έχει ήδη αποκατασταθεί.",
       "close": "Ελπίζουμε να σας φιλοξενήσουμε ξανά.",
       "one": "το θέμα", "many": "τα θέματα", "fixed_one": "Το θέμα", "fixed_many": "Τα θέματα"}
_EN = {"thanks": "Thank you very much for your review.",
       "sorry": "We are sorry about the problems you mention: {problems}.",
       "forwarded": "We have passed {what} to the departments responsible to be looked at.",
       "fixed": "{what} has now been put right.",
       "close": "We hope to welcome you again.",
       "one": "this", "many": "these matters", "fixed_one": "This", "fixed_many": "These matters"}


@dataclass
class TemplateDrafter:
    """The demo's stand-in without a key: fixed sentences in Greek or English. It is not the product."""

    labels_el: dict[str, str]  # type id -> Greek label
    name: str = "demo-template"

    def draft(self, review: str, language: str, facts: list[Fact]) -> str:
        greek = language == "el" or bool(re.search(r"[Ͱ-Ͽ]", review))
        words = _EL if greek else _EN
        parts = [words["thanks"]]
        if facts:
            ids = [f.problem.split(":")[0] for f in facts]
            names = [self.labels_el.get(i, i).lower() if greek else i.replace("_", " ") for i in ids]
            parts.append(words["sorry"].format(problems=", ".join(names)))
            forwarded = [f for f in facts if f.state == "forwarded"]
            fixed = [f for f in facts if f.state == "fixed"]
            if forwarded:
                parts.append(words["forwarded"].format(what=words["one" if len(forwarded) == 1 else "many"]))
            if fixed:
                parts.append(words["fixed"].format(what=words["fixed_one" if len(fixed) == 1 else "fixed_many"]))
        parts.append(words["close"])
        return " ".join(parts)


def state_of(finding: dict) -> str | None:
    """The reply state of a finding, or None when it does not go into the reply (not confirmed)."""
    if finding["decision"] != "confirmed":
        return None
    if finding.get("ticket_id") is None:
        return "noted"
    return "fixed" if finding.get("ticket_state") == "resolved" else "forwarded"
