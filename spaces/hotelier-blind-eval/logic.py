"""The logic of the hoteliers' blind evaluation Space.

Pure functions and a small store, so the app stays a thin Gradio layer and every rule here is tested:

- which reviews a hotelier still has, in that hotelier's own fixed random order;
- which system is shown as "Σύστημα 1" for each (hotelier, review), from a hash, so the hotelier cannot tell;
- how an output is shown: the same Greek rendering for both systems;
- what counts as a complete answer, and the record that is saved: once per (hotelier, review), never replaced.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable

SYSTEMS = ("x", "y")  # neutral names: which system is which is recorded in the repository, not in the Space
Q1 = {"Ναι": True, "Όχι": False}
Q2 = {"Λείπει επείγον πρόβλημα": "missed_urgent", "Σημειώνει ως επείγον κάτι που δεν είναι": "false_urgent",
      "Λάθος τμήμα σε επείγον πρόβλημα": "wrong_department"}
Q3 = {"Ναι": "yes", "Εν μέρει": "partly", "Όχι": "no"}
Q4 = ["1", "2", "3", "4", "5"]
PREFERENCE = {"Σύστημα 1": "1", "Σύστημα 2": "2", "Το ίδιο": "same"}
PRIORITY_ORDER = {"urgent": 0, "routine": 1, "structural": 2}
NO_PROBLEM = "_Κανένα πρόβλημα._"


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def hotelier_for(code: str, codes: dict[str, str]) -> str | None:
    """The pseudonym (H01, ...) of an access code, from {sha256(code): pseudonym}; None if it is not one."""
    return codes.get(hashlib.sha256(code.strip().encode("utf-8")).hexdigest())


def item_order(hotelier: str, items: Iterable[int]) -> list[int]:
    """The hotelier's own fixed random order of the items."""
    return sorted(items, key=lambda item: _digest(hotelier, "order", item))


def system_1(hotelier: str, item: int) -> str:
    """Which system is shown as "Σύστημα 1" for this hotelier and review."""
    return SYSTEMS[int(_digest(hotelier, "side", item)[:8], 16) % 2]


def remaining(hotelier: str, items: Iterable[int], done: set[int]) -> list[int]:
    return [item for item in item_order(hotelier, items) if item not in done]


def render(types: list[str], display: dict) -> str:
    """One output in Greek: urgent problems first, each with its department, measure and deadline.

    One line per problem, the measure after a line break: Gradio's Markdown strips the common indentation of the
    lines after the first, so a nested measure line would show as a second problem when there is only one."""
    if not types:
        return NO_PROBLEM
    entries = sorted(types, key=lambda t: (PRIORITY_ORDER[display["types"][t]["priority"]],
                                           display["order"].index(t)))
    lines = []
    for t in entries:
        entry = display["types"][t]
        badge = f"**{entry['priority_el']}**" if entry["priority"] == "urgent" else entry["priority_el"]
        when = f" ({entry['deadline_el']})" if entry.get("deadline_el") else ""
        lines.append(f"- {badge} · **{entry['label']}** → {entry['department_el']}<br>"
                     f"Ενέργεια: {entry['measure']}{when}")
    return "\n".join(lines)


def problems(q1: str | None, q2: list[str] | None, q3: str | None, q4: str | None, side: str) -> list[str]:
    """What is missing from one system's answers, in Greek; empty when they are complete and consistent."""
    errors = []
    if q1 not in Q1:
        errors.append(f"{side}: απάντησε αν τα επείγοντα είναι σωστά.")
    elif q1 == "Όχι" and not q2:
        errors.append(f"{side}: σημείωσε τι είναι λάθος στα επείγοντα.")
    elif q1 == "Ναι" and q2:
        errors.append(f"{side}: τα επείγοντα είναι σωστά· άφησε κενό το «τι είναι λάθος».")
    if q3 not in Q3:
        errors.append(f"{side}: απάντησε για τα υπόλοιπα προβλήματα.")
    if q4 not in Q4:
        errors.append(f"{side}: βάλε χρησιμότητα από 1 έως 5.")
    return errors


def answers(q1: str, q2: list[str] | None, q3: str, q4: str) -> dict:
    return {"q1_urgent_right": Q1[q1], "q2_what_is_wrong": [Q2[v] for v in (q2 or [])], "q3_others": Q3[q3],
            "q4_useful": int(q4)}


def record(hotelier: str, item: int, shown_1: dict, shown_2: dict, preference: str, meta: dict,
           now: dt.datetime | None = None) -> dict:
    """The saved rating: what the hotelier saw as 1 and 2, which system each was, and the answers."""
    first = system_1(hotelier, item)
    second = SYSTEMS[1 - SYSTEMS.index(first)]
    return {"hotelier": hotelier, "item": item, "system_1": first, "system_2": second,
            "shown": {"1": shown_1, "2": shown_2}, "by_system": {first: shown_1, second: shown_2},
            "preference": PREFERENCE[preference],
            "preferred_system": {"1": first, "2": second, "same": "same"}[PREFERENCE[preference]],
            "protocol_sha256": meta.get("protocol_sha256"), "bundle": meta.get("bundle"),
            "submitted_at": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")}


def path_of(hotelier: str, item: int) -> str:
    return f"ratings/{hotelier}/item_{item:02d}.json"


class MemoryStore:
    """Ratings kept in memory: the preview mode and the tests."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def done(self, hotelier: str) -> set[int]:
        prefix = f"ratings/{hotelier}/item_"
        return {int(name[len(prefix):-5]) for name in self.files if name.startswith(prefix)}

    def save(self, rating: dict) -> bool:
        path = path_of(rating["hotelier"], rating["item"])
        if path in self.files:
            return False  # the first submission is kept
        self.files[path] = json.dumps(rating, ensure_ascii=False, indent=1)
        return True


class HubStore(MemoryStore):
    """Ratings in a private Hugging Face dataset, one file per (hotelier, review)."""

    def __init__(self, repo_id: str, token: str) -> None:
        from huggingface_hub import HfApi

        super().__init__()
        self.api, self.repo_id = HfApi(token=token), repo_id

    def _names(self) -> list[str]:
        return self.api.list_repo_files(self.repo_id, repo_type="dataset")

    def done(self, hotelier: str) -> set[int]:
        prefix = f"ratings/{hotelier}/item_"
        return {int(name[len(prefix):-5]) for name in self._names() if name.startswith(prefix)}

    def save(self, rating: dict) -> bool:
        path = path_of(rating["hotelier"], rating["item"])
        if path in self._names():
            return False
        self.api.upload_file(path_or_fileobj=json.dumps(rating, ensure_ascii=False, indent=1).encode("utf-8"),
                             path_in_repo=path, repo_id=self.repo_id, repo_type="dataset",
                             commit_message=f"{rating['hotelier']} item {rating['item']}")
        return True
