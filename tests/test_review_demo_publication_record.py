"""The sentiment Space's 14 October publication: what it loaded is what the repository pins."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "experiments" / "review_demo_publication" / "71db6ee"
APP = ROOT / "spaces" / "hotel-review-demo" / "app.py"
SPACE_COMMIT = "71db6ee14c50254574f90137d9554fcddff22e2b"


def pins() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("pin_review_demo_revisions",
                                                  ROOT / "scripts" / "pin_review_demo_revisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.revisions(APP.read_text(encoding="utf-8"))


def test_the_live_space_loaded_exactly_the_pinned_models():
    info = json.loads((RECORD / "model_info.json").read_text(encoding="utf-8"))
    pinned = pins()
    assert info["distilbert_revision"] == info["loaded_commit"] == pinned["DISTILBERT_REVISION"]
    assert info["revisions"] == {
        "Qwen/Qwen2.5-0.5B-Instruct": pinned["QWEN_BASE_REVISION"],
        "krimits/hotel-review-nlp-qwen-qlora-adapter": pinned["QWEN_ADAPTER_REVISION"],
        "krimits/greek-hotel-reviews-sentiment": pinned["GREEK_REVISION"],
    }
    assert info["adapter_loaded_commit"] == pinned["QWEN_ADAPTER_REVISION"]


def test_the_served_distilbert_weights_are_the_published_model():
    info = json.loads((RECORD / "model_info.json").read_text(encoding="utf-8"))
    published = json.loads((ROOT / "docs/experiments/results/distilbert_v2/published.json").read_text(encoding="utf-8"))
    assert info["weights_sha256"] == published["new_model_sha256"]


def test_the_record_names_the_space_commit_and_its_limits():
    readme = (RECORD / "README.md").read_text(encoding="utf-8")
    assert SPACE_COMMIT in readme
    for limit in ("GreekBERT on English text", "not a benchmark", "has not been measured", "Dev mode"):
        assert limit in readme
