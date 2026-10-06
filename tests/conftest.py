import os
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
