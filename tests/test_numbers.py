"""Check `docs/NUMBERS.md` against the artifacts it points at.

Standard library only. Every row of that table declares a value, the artifact
that holds it and the JSON key inside that artifact. This module parses the
table, resolves each path and key, and fails when a value differs from the
artifact to the precision printed in the table.

Nothing here derives a metric: it re-reads the committed JSON, which is the
evidence `README.md`, `docs/RESULTS.md` and the decision notes point at. A row
that names a file or key that does not exist fails the same way a wrong number
does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "docs" / "NUMBERS.md"

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
KEY_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
COLUMNS = 6


def _rows(doc: Path) -> list[list[str]]:
    """Return the data rows of every six-column markdown table in the file."""
    rows: list[list[str]] = []
    for line in doc.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != COLUMNS:
            continue
        if cells[0].lower() == "value" or set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def _resolve_key(data: object, key: str) -> object:
    """Walk a dotted key path, allowing `[n]` list indices."""
    current = data
    for name, index in KEY_TOKEN_RE.findall(key):
        if name:
            if not isinstance(current, dict) or name not in current:
                raise AssertionError(f"key {name!r} of {key!r} not found in artifact")
            current = current[name]
        else:
            if not isinstance(current, list):
                raise AssertionError(f"{key!r} indexes a non-list")
            current = current[int(index)]
    return current


def check_numbers_document(doc: Path) -> int:
    """Assert every declared value matches its artifact. Returns the row count."""
    rows = _rows(doc)
    assert rows, f"{doc} holds no six-column table rows"

    for value, metric, test_set, scope, artifact, key in rows:
        where = f"{metric} ({test_set})"
        match = LINK_RE.search(artifact)
        assert match, f"{where}: artifact cell is not a link: {artifact!r}"
        target = ROOT / "docs" / match.group(1).split("#")[0]
        assert target.is_file(), f"{where}: artifact not found: {match.group(1)}"
        data = json.loads(target.read_text(encoding="utf-8"))
        raw = _resolve_key(data, key.strip("`"))
        assert isinstance(raw, (int, float)) and not isinstance(raw, bool), (
            f"{where}: artifact value is not a number: {raw!r}"
        )
        decimals = len(value.split(".")[1]) if "." in value else 0
        got = round(float(raw), decimals)
        assert got == float(value), (
            f"{where}: NUMBERS.md says {value}, but {target.name}:{key} holds {raw!r}"
            f" (rounds to {got})"
        )
        assert scope.strip(), f"{where}: the scope cell is empty; every number needs one"

    return len(rows)


def test_numbers_match_their_artifacts() -> None:
    assert check_numbers_document(NUMBERS) >= 27


def test_every_functional_result_has_a_row() -> None:
    rows = _rows(NUMBERS)
    macro_f1 = [r for r in rows if "macro-f1" in r[1].lower()]
    legacy = [r for r in rows if "not comparable" in r[3]]
    triage = [r for r in rows if "urgent recall" in r[1].lower() or r[1] == "`lo`"]
    assert len(macro_f1) >= 18, f"only {len(macro_f1)} macro-F1 rows"
    assert len(legacy) >= 5, f"only {len(legacy)} rows flagged as not comparable"
    assert len(triage) >= 2, f"only {len(triage)} triage rows"
    for row in legacy:
        assert "legacy" in row[3], f"{row[1]} is flagged but does not say why: {row[3]!r}"


def test_a_wrong_number_is_rejected(tmp_path: Path) -> None:
    """The checker must fail on a single wrong digit, not pass on a missing file."""
    text = NUMBERS.read_text(encoding="utf-8")
    assert "0.9642" in text
    mutated = tmp_path / "NUMBERS.md"
    mutated.write_text(text.replace("0.9642", "0.9643", 1), encoding="utf-8")
    try:
        check_numbers_document(mutated)
    except AssertionError as exc:
        assert "0.9643" in str(exc) and "0.9642" in str(exc), str(exc)
    else:
        raise AssertionError("a wrong value was accepted")
