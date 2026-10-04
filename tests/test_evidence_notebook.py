"""Notebook 17 wiring with cached upstream, no API key, complete export even on failure."""

from __future__ import annotations

import ast
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reviewnlp.triage.generator_experiment import (
    CANDIDATES,
    digest_bytes,
    digest_json,
    export_annotation,
    structural_summary,
)
from reviewnlp.triage.workflow_review import export_coverage_review

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/17_triage_evidence_comparison_colab.ipynb"


def cells():
    nb = json.loads(NOTEBOOK.read_text())
    return nb, {cell["id"]: "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"}


def test_notebook_pins_current_modules_and_archived_reference_without_secrets_or_final_data():
    nb, code = cells()
    namespace = {}
    exec(code["parameters"], namespace)
    assert namespace["EXPECTED_CANDIDATES"] == ("F", "G")
    assert namespace["SCRIPT_SHA256"] == digest_bytes((ROOT / "scripts/compare_triage_generators.py").read_bytes())
    for name, expected in namespace["MODULE_HASHES"].items():
        assert digest_bytes((ROOT / name).read_bytes()) == expected
    assert namespace["REFERENCE_RUN_SHA256"] == digest_bytes((ROOT / namespace["REFERENCE_RELATIVE"] / "run.json").read_bytes())
    source = "\n".join(code.values())
    assert "files.upload" not in source and "userdata.get" not in source and "holdout.json" not in source
    assert '"--candidates", "F", "G"' in code["run"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("exit_code,second_stage", [(0, True), (1, True), (0, False)])
def test_top_to_bottom_wiring_checks_budgets_and_downloads_failed_runs(tmp_path, exit_code, second_stage):
    _, code = cells()
    namespace, downloads = {}, []
    exec(code["parameters"], namespace)
    checkout = tmp_path / "checkout"
    for name in ["scripts/compare_triage_generators.py", "docs/experiments/triage_generator/dev.json",
                 *namespace["MODULE_HASHES"], *(namespace["REFERENCE_RELATIVE"] + "/" + name
                                               for name in ("run.json", "upstream.json"))]:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    namespace["REPO_DIR"] = checkout

    def fake_run(arguments, **kwargs):
        assert "OPENROUTER_API_KEY" not in kwargs["env"]
        value = namespace["SOURCE_COMMIT"] if arguments[:3] == ["git", "rev-parse", "HEAD"] else ""
        if "-c" in arguments:
            value = '{"gpu":"fake GPU","torch":"fake"}'
        return SimpleNamespace(stdout=value)

    class Process:
        def __init__(self, arguments, **kwargs):
            assert "OPENROUTER_API_KEY" not in kwargs["env"] and "--allow-external-api" not in arguments
            assert arguments[arguments.index("--candidates") + 1:arguments.index("--candidates") + 3] == ["F", "G"]
            output = Path(arguments[arguments.index("--output") + 1])
            output.mkdir()
            reference = namespace["REFERENCE_DIR"]
            for source_name, target_name in (("run.json", "reference_run.json"), ("upstream.json", "upstream.json")):
                shutil.copyfile(reference / source_name, output / target_name)
            rows = json.loads((checkout / "docs/experiments/triage_generator/dev.json").read_text())["reviews"]
            upstream = json.loads((output / "upstream.json").read_text())["records"]
            records = []
            for candidate in ("F", "G"):
                for row, previous in zip(rows, upstream, strict=True):
                    stages = [{"stage": "issues", "raw": '{"issues":[]}'}]
                    if second_stage or candidate == "F":
                        stages.append({"stage": "measures", "raw": '{"actions":[]}'})
                    records.append({"candidate": candidate, "id": row["id"], "raw": '{"actions":[]}',
                                    "stages": stages, "extracted_issues": [], "upstream_sha256": digest_json(previous),
                                    "error": None, "workflow_error": None, "hit_token_budget": False,
                                    "actions": [], "dropped": 0, "json_valid": True, "full_json_valid": True})
            (output / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            export_annotation(output, rows, records)
            export_coverage_review(output, rows, records)
            files = {name: digest_bytes((output / name).read_bytes()) for name in
                     ("upstream.json", "results.jsonl", "reference_run.json", "human_review.csv",
                      "coverage_review.csv", "annotation_key.json")}
            info = {"split": "dev", "execution_complete": exit_code == 0, "quality_evaluated": False,
                    "source_commit": namespace["SOURCE_COMMIT"], "dataset_sha256": namespace["REVIEWS_SHA256"],
                    "candidates": {name: {**CANDIDATES[name], "revision": namespace["MODEL_REVISION"]} for name in ("F", "G")},
                    "generation_budget": {name: {"max_new_tokens_per_call": 400, "max_calls_per_review": 2} for name in ("F", "G")},
                    "upstream_reused": True, "api_cost": {"attempts": 0},
                    "reference_run": {"run_sha256": namespace["REFERENCE_RUN_SHA256"]}, "files": files,
                    "summary": {name: structural_summary([row for row in records if row["candidate"] == name]) for name in ("F", "G")}}
            (output / "run.json").write_text(json.dumps(info))
            (output / "summary.md").write_text("Offline notebook wiring; human quality pending.")
            self.stdout = io.StringIO("FAKE-NOTEBOOK-SECRET\n")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return exit_code

    def noop(*args, **kwargs):
        return None
    axis = SimpleNamespace(bar=noop, set_ylim=noop, set_title=noop, set_ylabel=noop, text=noop)
    fig = SimpleNamespace(suptitle=noop, tight_layout=noop, savefig=lambda path, **kwargs: Path(path).write_bytes(b"plot stub"))
    plt = SimpleNamespace(subplots=lambda *args, **kwargs: (fig, [axis, axis]), show=noop)
    modules = {"google": SimpleNamespace(), "google.colab": SimpleNamespace(files=SimpleNamespace(download=downloads.append)),
               "IPython": SimpleNamespace(), "IPython.display": SimpleNamespace(Markdown=lambda x: x, display=noop),
               "matplotlib": SimpleNamespace(pyplot=plt), "matplotlib.pyplot": plt}
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {"OPENROUTER_API_KEY": "FAKE-NOTEBOOK-SECRET"}), \
            patch("subprocess.run", side_effect=fake_run), patch("subprocess.Popen", Process), \
            patch("importlib.metadata.version", return_value="fake"):
        for cell_id in ("setup", "reference", "environment", "run", "audit", "structure-plot"):
            exec(code[cell_id], namespace)
        if exit_code or not second_stage:
            with pytest.raises(RuntimeError):
                exec(code["download"], namespace)
        else:
            exec(code["download"], namespace)
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        assert {"coverage_review.csv", "human_review.csv", "notebook_checks.json"} <= set(archive.namelist())
        assert all(b"FAKE-NOTEBOOK-SECRET" not in archive.read(name) for name in archive.namelist())
