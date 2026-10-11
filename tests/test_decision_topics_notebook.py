"""Guard the Colab sample handoff against unpinned sources and unintended inference."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/22_decision_topics_sample_colab.ipynb"


def test_sample_notebook_has_clean_outputs_and_pins_its_actual_sources():
    notebook = json.loads(NOTEBOOK.read_bytes())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    pin = re.search(r"SOURCE_COMMIT\s*=\s*['\"]([0-9a-f]{40})['\"]", code).group(1)
    ast.parse(code)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
    for path in (
        "scripts/benchmark_decision_topics.py",
        "src/reviewnlp/experiments/topic_benchmark.py",
        "src/reviewnlp/experiments/topic_models.py",
        "configs/decision_topic_sample_requirements.txt",
        "docs/experiments/decision_topic_benchmark/PROTOCOL.md",
        "data/raw/hotel_reviews.csv",
    ):
        frozen = subprocess.check_output(["git", "show", f"{pin}:{path}"], cwd=ROOT)
        assert frozen == (ROOT / path).read_bytes()
    # Never include a provider runner or credential lookup in the labeling notebook.
    assert '"--allow-external-api"' not in code and '"--partition"' not in code
    assert "userdata.get" not in code and "notebook_login" not in code
    assert '"sample", "--output"' in code
    assert "system_site_packages=False" in code
    assert '"--isolated", "install", "--no-user"' in code
    assert '"-I"' in code and "module.__version__" in code
    assert "is_relative_to" in code
