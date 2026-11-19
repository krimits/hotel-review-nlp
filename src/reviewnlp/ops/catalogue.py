"""The problem catalogue the operations demo assigns by: catalogue_v2 and its Greek display labels.

Both files are byte-for-byte copies of the approved ones under docs/experiments (a test keeps them equal), shipped
with the package so a served app does not depend on the docs tree. Assignment is the catalogue's, never the
model's: the model names a type, the catalogue gives its department, priority, deadline and what to confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import yaml

PRIORITIES = ("urgent", "routine", "structural")


@dataclass(frozen=True)
class ProblemType:
    id: str
    priority: str
    department: str
    definition: str
    excludes: str | None
    measure: str | None
    deadline: str | None
    to_confirm: tuple[str, ...]
    label_el: str
    measure_el: str | None

    @property
    def needs_room(self) -> bool:
        """Whether reception must give the room before the department can act on it."""
        return bool(self.to_confirm) and self.to_confirm[0].startswith("room number")

    @property
    def opens_ticket(self) -> bool:
        """Structural types have no operational measure: management is informed, no department gets a ticket."""
        return self.priority != "structural"


@dataclass(frozen=True)
class Catalogue:
    version: str
    types: dict[str, ProblemType]
    departments_el: dict[str, str]
    priorities_el: dict[str, str]
    deadlines_el: dict[str, str]
    rules: tuple[str, ...]

    @property
    def ids(self) -> list[str]:
        return list(self.types)


def _read(name: str) -> dict:
    return yaml.safe_load(resources.files("reviewnlp.ops").joinpath(name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_catalogue() -> Catalogue:
    raw, labels = _read("catalogue_v2.yaml"), _read("labels_el.yaml")
    if labels.get("catalogue") != raw["version"]:
        raise ValueError("the Greek labels are not for this catalogue")
    types = {}
    for entry in raw["types"]:
        shown = labels["types"][entry["id"]]
        if entry["priority"] not in PRIORITIES or entry["department"] not in raw["departments"]:
            raise ValueError(f"catalogue type {entry['id']} has an unknown priority or department")
        types[entry["id"]] = ProblemType(
            id=entry["id"], priority=entry["priority"], department=entry["department"],
            definition=entry["definition"], excludes=entry.get("excludes"), measure=entry.get("measure"),
            deadline=entry.get("deadline"), to_confirm=tuple(entry.get("to_confirm") or ()),
            label_el=shown["label"], measure_el=shown.get("measure") or labels["structural_measure"])
    return Catalogue(version=raw["version"], types=types, departments_el=labels["departments"],
                     priorities_el=labels["priorities"], deadlines_el=labels["deadlines"],
                     rules=tuple(raw["labelling_rules"]))
