"""Local complaint-topic hints, with abstention rather than a claim of no complaint.

This deliberately small rule set is new: it is not the historical 30-topic ABSA
lexicon or its evaluated sentiment model. Rules supply hints, never issue status,
department assignments, confidence scores or measures. Whole-review pilot pending.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping

from reviewnlp.triage.schemas import OTHER, TOPICS, ComplaintsResult, ComplaintTopic, LexiconMatch

VERSION = "complaint-lexicon-v1"
# Recognize short, explicit descriptions, not a noun anywhere near negative sentiment.
LINK = r"\s+(?:(?:was|were|is|are|had|has|very|really|still|always|completely|too|a|the)\s+){0,3}"
RULES = (
    ("bathroom", "bathroom_fault", r"\b(?:shower|toilet|sink|tap|bath|bathtub)" + LINK
     + r"(?:cold|lukewarm|leaking|blocked|broken|clogged|tiny)\b"),
    ("cleanliness", "dirty_surface", r"\b(?:room|bathroom|desk|cupboard|carpet|floor|wall|walls)" + LINK
     + r"(?:dirty|dusty|filthy|stained|mouldy|moldy)\b"),
    ("cleanliness", "dirty_surface_adjective", r"\b(?:dirty|dusty|filthy|stained|mouldy|moldy)\s+"
     + r"(?:room|bathroom|desk|cupboard|carpet|floor|wall|walls)\b"),
    ("air_conditioning", "hvac_fault", r"\b(?:air[ -]?(?:conditioning|conditioner|con)|a/c|ac|heating|heater)"
     + LINK + r"(?:broken|leaking|noisy|ineffective|faulty)\b"),
    ("pests", "observed_pests", r"\b(?:saw|found|noticed)\s+(?:(?:a|an|some|many|several|live)\s+){0,2}"
     + r"(?:ants|cockroaches|cockroach|bed[ -]?bugs|mice|rats)\b"),
    ("responsiveness", "staff_ignored_request", r"\b(?:staff|reception|receptionist|hotel)\s+"
     + r"(?:ignored|ignores)\s+(?:(?:my|our|the|repeated)\s+){0,2}(?:request|requests|complaint|complaints)\b"),
)
# Ambiguous, conditional, denied, historical or repaired reports are left to a
# person/Qwen. A guard anywhere in the review blocks lexical assertions: we cannot
# reliably resolve cross-sentence references or decide which issue was fixed.
GUARD = (r"\b(?:if|unless|would|could|might|not|never|no|without|fixed|repaired|resolved|replaced|"
         r"used to|previously|earlier|formerly)\b|n['’]t\b")
SHA256 = hashlib.sha256(json.dumps({"version": VERSION, "rules": RULES, "guard": GUARD,
    "absence_policy": "unsure_never_no", "coverage": "limited", "language": "en"},
    sort_keys=True).encode()).hexdigest()
_RULES = [(topic, name, re.compile(pattern, re.I)) for topic, name, pattern in RULES]
_GUARD = re.compile(GUARD, re.I)


def enabled_from_env(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get("REVIEWNLP_TRIAGE_LEXICON_FALLBACK", "1").strip().lower() in {"1", "true", "yes"}


class LexiconClassifier:
    def classify(self, text: str) -> ComplaintsResult:
        matches = []
        if not _GUARD.search(text):
            for topic, name, pattern in _RULES:
                for match in pattern.finditer(text):
                    matches.append(LexiconMatch(topic=topic, excerpt=text[match.start():match.end()],
                        start=match.start(), end=match.end(), rule=name))
        matched = {item.topic for item in matches}
        # Absence is abstention, not a negative prediction. Other topics remain
        # unassessed even when we recognized one: a mixed review can contain both.
        return ComplaintsResult(status="ok", source="lexicon", coverage_status="limited",
            questions_version=VERSION, questions_sha256=SHA256,
            topics=[ComplaintTopic(topic=topic, answer="yes" if topic in matched else "unsure", probability=None)
                    for topic in TOPICS],
            other_complaint=ComplaintTopic(topic=OTHER, answer="unsure", probability=None),
            evidence=sorted(matches, key=lambda item: (item.start, item.end, item.topic)))
