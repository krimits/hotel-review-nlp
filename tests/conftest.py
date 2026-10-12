import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Test UI builds must not launch Gradio version checks or telemetry threads.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Make package sources and repository scripts importable for both pytest entry points.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def frozen_triage_root(tmp_path_factory):
    """Historical format replay uses its actual pinned backend, not today's workflow."""
    manifest = json.loads((ROOT / "docs/experiments/triage_space_publication/e1e37e5/review/source_manifest.json").read_bytes())
    root = tmp_path_factory.mktemp("frozen-triage")
    for name, expected in manifest["files"].items():
        if not name.startswith("reviewnlp/"):
            continue
        path = "src/" + name
        content = subprocess.check_output(["git", "show", manifest["source_commit"] + ":" + path], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return root


@pytest.fixture
def tiny_reviews() -> pd.DataFrame:
    """Mini version of the Booking.com schema for preprocess tests."""
    return pd.DataFrame(
        {
            "Positive_Review": [
                "Great hotel, loved the location",
                "No Positive",
                "Fantastic staff and breakfast",
                "No Positive",
                "Amazing view from the room",
                "No Positive",
            ],
            "Negative_Review": [
                "No Negative",
                "Room was dirty and noisy",
                "No Negative",
                "Awful bed, terrible smell",
                "No Negative",
                "Bad service at the desk",
            ],
        }
    )
