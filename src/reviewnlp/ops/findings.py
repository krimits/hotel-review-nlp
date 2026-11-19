"""Findings of one review: each problem the guest reports, with the words that show it and whether it is still open.

Detecting a complaint and deciding on an urgent ticket need different information. Every problem the guest reports
is a finding, also one the hotel fixed during the stay ("the AC broke and staff repaired it" is still a complaint).
Whether a ticket may open depends on the finding's status:

- `pending`: the guest says the problem was still there when they left, was never fixed, or came back;
- `resolved`: the guest says the hotel fixed or replaced it during the stay;
- `unclear`: the review does not say; staff must confirm it is still open before a ticket opens.

Each excerpt must be the guest's own words, found in the review; an answer with an excerpt that is not in the
review is asked for again, and then refused. The type comes from catalogue_v2; the department, priority and what
to confirm come from the catalogue, never from the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from reviewnlp.absa.extract import review_key
from reviewnlp.ops.catalogue import Catalogue, load_catalogue
from reviewnlp.ops.llm import LLMError, OpenRouterClient

STATUSES = ("pending", "resolved", "unclear")
PROMPT_VERSION = "ops-findings-v1"
TRIES = 3
MAX_CHARS = 8000


@dataclass(frozen=True)
class Finding:
    type: str
    excerpt: str
    status: str


@dataclass(frozen=True)
class Analysis:
    language: str
    findings: tuple[Finding, ...]
    analyser: str


class Analyser(Protocol):
    name: str

    def analyse(self, text: str) -> Analysis: ...


def findings_prompt(catalogue: Catalogue) -> str:
    types = "\n".join(
        f"- `{t.id}`: {t.definition}" + (f"\n  Not this type: {t.excludes}" if t.excludes else "")
        for t in catalogue.types.values())
    rules = "\n".join(f"- {rule}" for rule in catalogue.rules)
    return f"""# Findings in one hotel guest review

You read one guest review and list every problem the guest reports, so the hotel can act on it.

## What to return

Only a JSON object: {{"language": "<ISO 639-1 code of the review>", "findings": [{{"type": "...", "excerpt": "...",
"status": "..."}}]}}. One finding per distinct problem; an empty list when the review reports no problem.

- `type`: the id of one problem type below.
- `excerpt`: the shortest passage that shows the problem, copied exactly from the review, in its language. Do not
  translate, summarise or join separate passages.
- `status`:
  - `pending` when the guest says the problem was still there when they left, was never fixed, or came back;
  - `resolved` when the guest says the hotel fixed or replaced it during the stay;
  - `unclear` when the review does not say whether it was fixed. Most reviews are `unclear`; do not guess.

## Rules

{rules}

## Problem types

{types}

({catalogue.version}, {len(catalogue.types)} types.)
"""


def findings_schema(types: list[str]) -> dict:
    finding = {"type": "object", "additionalProperties": False, "required": ["type", "excerpt", "status"],
               "properties": {"type": {"type": "string", "enum": types}, "excerpt": {"type": "string"},
                              "status": {"type": "string", "enum": list(STATUSES)}}}
    return {"type": "object", "additionalProperties": False, "required": ["language", "findings"],
            "properties": {"language": {"type": "string"}, "findings": {"type": "array", "items": finding}}}


class FindingsError(ValueError):
    """An answer that cannot be used as it stands."""


def check(answer: dict, text: str, catalogue: Catalogue) -> tuple[str, tuple[Finding, ...]]:
    """The language and findings of an answer, or FindingsError. One finding per type: the first one kept."""
    try:
        language = str(answer["language"]).strip().lower()[:8] or "und"
        entries = list(answer["findings"])
    except (KeyError, TypeError) as error:
        raise FindingsError(f"missing field: {error}") from None
    found: dict[str, Finding] = {}
    whole = review_key(text)
    for entry in entries:
        kind, excerpt, status = entry.get("type"), str(entry.get("excerpt", "")).strip(), entry.get("status")
        if kind not in catalogue.types or status not in STATUSES:
            raise FindingsError(f"unknown type or status: {kind!r}, {status!r}")
        if not excerpt or review_key(excerpt) not in whole:
            raise FindingsError("an excerpt is not in the review")
        found.setdefault(kind, Finding(kind, excerpt, status))
    return language, tuple(found.values())


@dataclass
class LLMAnalyser:
    client: OpenRouterClient
    catalogue: Catalogue
    name: str = "openrouter"

    def analyse(self, text: str) -> Analysis:
        if len(text) > MAX_CHARS:
            raise ValueError(f"the review is longer than {MAX_CHARS} characters; it is never cut")
        system, schema = findings_prompt(self.catalogue), findings_schema(self.catalogue.ids)
        problem = "no try made"
        for _ in range(TRIES):
            answer = self.client.json_call(system, json.dumps({"review": text}, ensure_ascii=False), "findings", schema)
            try:
                language, findings = check(answer, text, self.catalogue)
            except FindingsError as error:
                problem = str(error)
                continue
            return Analysis(language, findings, f"{self.name}:{self.client.model}:{PROMPT_VERSION}")
        raise LLMError(f"no usable findings after {TRIES} tries: {problem}")


# --- the demo's stand-in: keyword rules, so the flow can be shown without a key. It is not the product. -------------

_RULES: tuple[tuple[str, str], ...] = (
    ("hvac_failure", r"air[- ]?con\w*|\ba/?c\b|κλιματισ\w*|θέρμανσ\w*|heating"),
    ("cleanliness", r"dirty|stain\w*|filthy|mou?ld\w*|λερωμ\w*|βρώμικ\w*|άπλυτ\w*|μουχλ\w*"),
    ("pests", r"cockroach\w*|bed ?bugs?|\bmice\b|κατσαρίδ\w*|κοριο\w*|ποντίκ\w*"),
    ("plumbing_major", r"no hot water|leak\w*|toilet\w* (?:was )?(?:blocked|broken)|ζεστό νερό|διαρρο\w*|τουαλέτα"),
    ("security_fault", r"(?:lock|door|safe) (?:was )?broken|broken (?:lock|door|safe)|κλειδαρι\w*|χρηματοκιβώτι\w*"),
    ("wifi", r"wi-?fi"),
    ("noise_internal", r"noisy neighbou?rs?|corridor noise|θόρυβ\w*"),
    ("staff_service", r"\brude\b|αγεν\w*"),
    ("food_beverage", r"breakfast was (?:poor|bad|cold)|πρωινό ήταν"),
)
_RESOLVED = r"fixed|repaired|replaced|επισκεύασ\w*|διόρθωσ\w*|αντικατέστησ\w*|φτιάξ\w*"
_PENDING = r"until (?:we|i) left|whole stay|never (?:fixed|worked)|μέχρι την αναχώρηση|όλη (?:τη|την) διαμονή|ποτέ δεν"


@dataclass
class FakeAnalyser:
    catalogue: Catalogue
    name: str = "demo-keywords"

    def analyse(self, text: str) -> Analysis:
        sentences = [s.strip() for s in re.split(r"(?<=[.!;?])\s+|\n+", text) if s.strip()]
        found = []
        for kind, pattern in _RULES:
            for sentence in sentences:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    status = ("resolved" if re.search(_RESOLVED, sentence, re.IGNORECASE)
                              else "pending" if re.search(_PENDING, sentence, re.IGNORECASE) else "unclear")
                    found.append(Finding(kind, sentence, status))
                    break
        language = "el" if re.search(r"[Ͱ-Ͽ]", text) else "en"
        return Analysis(language, tuple(found), self.name)


def analyser_from_env(fake_allowed: bool) -> Analyser | None:
    """OpenRouter when a key is set; the keyword stand-in only when the demo asks for it; otherwise None."""
    client = OpenRouterClient.from_env()
    if client is not None:
        return LLMAnalyser(client, load_catalogue())
    return FakeAnalyser(load_catalogue()) if fake_allowed else None
