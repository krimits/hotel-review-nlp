"""The hotel pilot's record and its one rule, on exports made with the Space's own record function. No network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pilot = _load("hotel_pilot", ROOT / "scripts" / "hotel_pilot.py")
logic = pilot.space_logic()
DISPLAY = pilot.display_table()
URGENT = logic.urgent_types(DISPLAY)
ACCEPT, NOT_URGENT, ABSENT = list(logic.VERDICTS)


def make_export(folder: Path, hotels: dict[str, dict], meta: dict | None = None) -> Path:
    """An export as `hf download` leaves it. Each hotel: reviews, accepted (of its flags, one per review), missing
    (reviews with a missing urgent problem), and optionally urgent (its set) and extra analysed-only reviews."""
    meta = meta or pilot.meta()
    for hotel, spec in hotels.items():
        urgent_set = spec.get("urgent", URGENT)
        for i in range(spec["reviews"]):
            text = f"{hotel} review {i} dirty"
            item = {"key": logic.text_key(text, "salt"), "text": text,
                    "script": "greek" if i % 5 == 0 else "latin", "types": ["cleanliness", "wifi"],
                    "call": {"seconds": 8.0, "cost_usd": 0.0005, "reviews": 5}}
            verdict = ACCEPT if i < spec["accepted"] else (NOT_URGENT if i % 2 else ABSENT)
            missing = "Ναι" if i < spec.get("missing", 0) else "Όχι"
            rating = logic.record(hotel, item, urgent_set, {"cleanliness": verdict}, missing,
                                  DISPLAY["types"]["pests"]["label"], "Εν μέρει", "4", meta, DISPLAY)
            for kind, content in (("records", rating), ("analysed", logic.analysed_note(hotel, item))):
                path = folder / kind / hotel / f"{item['key']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        for j in range(spec.get("never_rated", 0)):
            path = folder / "analysed" / hotel / f"lost{j:020d}.json"
            path.write_text(json.dumps({"hotel": hotel, "key": path.stem, "script": "latin"}), encoding="utf-8")
        settings = folder / "settings" / f"{hotel}.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps({"hotel": hotel, "urgent_set": urgent_set}), encoding="utf-8")
    (folder / ".gitattributes").write_text("*.json text\n", encoding="utf-8")
    return folder


def five(accepted: int, **more) -> dict[str, dict]:
    return {f"P{n:02d}": {"reviews": 30, "accepted": accepted, **more} for n in range(1, 6)}


def run(tmp_path, hotels, **kwargs):
    export = make_export(tmp_path / "export", hotels)
    return pilot.analyse(export, out=tmp_path / "result.json", require_commits=False, record=None, **kwargs)


def test_the_rule_its_boundaries_and_the_owners_fields():
    assert pilot.outcome(0.80, 0.60, 0.10) == "Accepted"
    assert pilot.outcome(0.80, 0.59, 0.10) == pilot.outcome(0.80, 0.60, 0.11) == "Partly accepted"
    assert pilot.outcome(0.79, 0.79, 0.0) == pilot.outcome(0.60, 0.60, 0.0) == "Partly accepted"
    assert pilot.outcome(0.59, 0.59, 0.0) == "Not accepted"
    assert [c for c, _ in pilot.RULE] == ["acc >= 0.80 and worst >= 0.60 and miss <= 0.10", "acc < 0.60", "otherwise"]
    assert (pilot.H, pilot.R, pilot.G) == (5, 30, 10) and logic.CAP == 300


def test_five_hotels_that_accept_the_flags_give_accepted_and_the_reported_items(tmp_path):
    hotels = five(27, missing=2)
    hotels["P05"] = {"reviews": 30, "accepted": 21, "never_rated": 4, "urgent": [t for t in URGENT if t != "pests"]}
    result = run(tmp_path, hotels)
    decision = result["decision"]
    assert decision["acc"] == round((4 * 0.9 + 0.7) / 5, 6) and decision["worst"] == 0.7
    assert decision["miss"] == round(4 * (2 / 30) / 5, 6) and decision["outcome"] == "Accepted"
    reported = result["reported"]
    assert reported["hotels"]["P05"]["analysed_never_rated"] == 4 and reported["hotels"]["P01"]["qualifies"]
    assert reported["hotels_keeping_each_urgent_type_on"]["pests"] == 4
    assert reported["hotels_keeping_each_urgent_type_on"]["cleanliness"] == 5
    verdicts = reported["verdicts_by_urgent_type"]["cleanliness"]
    assert verdicts["accepted"] == 4 * 27 + 21 and verdicts["not_urgent"] + verdicts["absent"] == 4 * 3 + 9
    assert reported["acc_by_script"]["greek"]["rated_urgent_flags"] == 5 * 6
    assert reported["missing_types"] == {"pests": 8}
    assert reported["cost_usd_per_1000_reviews"] == pytest.approx(0.1) and reported["median_seconds_per_call"] == 8.0
    written = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert "dirty" not in written and '"key"' not in written  # aggregates, no single judgement
    with pytest.raises(SystemExit, match="applied once"):
        pilot.analyse(tmp_path / "export", out=tmp_path / "result.json", require_commits=False, record=None)


def test_the_rule_sees_exact_shares_not_the_rounded_ones_shown(tmp_path):
    # 24 of 30 accepted is exactly 0.80, and 3 of 30 missing exactly 0.10: on the boundary, so Accepted.
    decision = run(tmp_path, five(24, missing=3))["decision"]
    assert (decision["acc"], decision["worst"], decision["miss"]) == (0.8, 0.8, 0.1)
    assert decision["outcome"] == "Accepted"
    # A mean that is exactly 0.80 from unequal hotels: 21, 27, 24, 24 and 24 of 30.
    hotels = five(24)
    hotels["P01"], hotels["P02"] = {"reviews": 30, "accepted": 21}, {"reviews": 30, "accepted": 27}
    decision = run(tmp_path / "unequal", hotels)["decision"]
    assert decision["acc"] == 0.8 and decision["outcome"] == "Accepted"


def test_low_acceptance_is_not_accepted_and_one_poor_hotel_makes_it_partly(tmp_path):
    assert run(tmp_path / "a", five(15))["decision"]["outcome"] == "Not accepted"
    hotels = five(29)
    hotels["P03"] = {"reviews": 30, "accepted": 15}
    decision = run(tmp_path / "b", hotels)["decision"]
    assert decision["acc"] >= 0.80 and decision["worst"] == 0.5 and decision["outcome"] == "Partly accepted"
    assert run(tmp_path / "c", five(29, missing=6))["decision"]["outcome"] == "Partly accepted"


def test_a_hotel_with_too_few_reviews_or_flags_does_not_qualify_and_four_hotels_decide_nothing(tmp_path):
    hotels = five(27)
    hotels["P05"] = {"reviews": 29, "accepted": 29}
    result = run(tmp_path / "a", hotels)
    assert result["decision"]["outcome"].startswith("not decided: 4 qualifying")
    assert result["reported"]["hotels_not_qualifying"] == ["P05"] and not result["reported"]["hotels"]["P05"]["qualifies"]
    few_flags = five(27)
    few_flags["P05"] = {"reviews": 30, "accepted": 0, "urgent": []}  # 30 reviews, no urgent flag at all
    result = run(tmp_path / "b", few_flags)
    assert result["reported"]["hotels"]["P05"]["rated_urgent_flags"] == 0
    assert result["decision"]["outcome"].startswith("not decided")


def test_the_export_is_recorded_by_one_digest_and_analysed_only_while_it_is_that_export(tmp_path):
    export, out = make_export(tmp_path / "export", five(27)), tmp_path / "records" / "record.json"
    with pytest.raises(SystemExit, match="--source is required"):
        pilot.record_export(" ", export, out)
    content = pilot.record_export("five hotels, themselves", export, out)
    assert content["hotels"]["P01"] == {"rated_reviews": 30, "rated_urgent_flags": 30, "analysed_reviews": 30}
    assert (content["export_sha256"], content["files"]) == pilot.export_digest(export) and content["files"] == 305
    assert [p.name for p in out.parent.iterdir()] == ["record.json"] and "accepted" not in out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="never replaced"):
        pilot.record_export("again", export, out)
    victim = next((export / "records" / "P02").iterdir())
    original = victim.read_text(encoding="utf-8")
    victim.write_text(original.replace('"useful": 4', '"useful": 5'), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not hold the recorded export"):
        pilot.analyse(export, out=tmp_path / "result.json", require_commits=False, record=out)
    victim.write_text(original, encoding="utf-8", newline="\n")
    assert pilot.analyse(export, out=tmp_path / "result.json", require_commits=False,
                         record=out)["decision"]["outcome"] == "Accepted"


@pytest.mark.parametrize("change, message", [
    ({"protocol_sha256": "0" * 64}, "another protocol_sha256"),
    ({"prompt_sha256": "0" * 64}, "another prompt_sha256"),
    ({"model": "another/model"}, "another model"),
    ({"hotel": "P09"}, "differs from its path"),
    ({"flags": {}}, "not one per urgent flag"),
    ({"flags": {"cleanliness": "maybe"}}, "not one per urgent flag"),
    ({"missing_urgent": True, "missing_type": None}, "missing-urgent answer"),
    ({"useful": 9}, "usefulness"),
    ({"text": "The room was dirty"}, "holds a review text"),
])
def test_one_wrong_record_refuses_the_whole_export(tmp_path, change, message):
    export = make_export(tmp_path / "export", {"P01": {"reviews": 3, "accepted": 3}})
    victim = sorted((export / "records" / "P01").iterdir())[1]
    victim.write_text(json.dumps({**json.loads(victim.read_text(encoding="utf-8")), **change}), encoding="utf-8")
    with pytest.raises(SystemExit, match=message):
        pilot.record_export("one hotel", export, tmp_path / "records" / "record.json")
    assert not (tmp_path / "records").exists()


def test_a_folder_that_is_not_a_hotels_refuses_the_export(tmp_path):
    export = make_export(tmp_path / "export", {"P01": {"reviews": 3, "accepted": 3}})
    (export / "records" / "DEMO").mkdir()
    with pytest.raises(SystemExit, match="not a hotel's folder"):
        pilot.record_export("one hotel", export, tmp_path / "records" / "record.json")
