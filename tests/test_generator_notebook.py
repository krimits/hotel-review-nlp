"""Exercise notebook wiring without Colab, installation, model downloads, or API calls."""

from __future__ import annotations

import ast
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reviewnlp.triage.generator_experiment import (
    CANDIDATES,
    digest_bytes,
    digest_json,
    export_annotation,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/14_triage_generator_comparison_colab.ipynb"
V3_NOTEBOOK = ROOT / "notebooks/15_triage_prompt_v3_comparison_colab.ipynb"
STAGED_NOTEBOOK = ROOT / "notebooks/16_triage_staged_comparison_colab.ipynb"


def cells(path=NOTEBOOK):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return notebook, {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}


class NotebookTests(unittest.TestCase):
    def test_clean_notebook_pins_the_script_and_dev_cases_and_never_opens_holdout(self):
        notebook, code = cells()
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(code["parameters"], namespace)
        self.assertRegex(namespace["SOURCE_COMMIT"], r"^[a-f0-9]{40}$")
        self.assertEqual(namespace["SCRIPT_SHA256"], digest_bytes((ROOT / "scripts/compare_triage_generators.py").read_bytes()))
        self.assertEqual(namespace["REVIEWS_SHA256"], digest_bytes((ROOT / "docs/experiments/triage_generator/dev.json").read_bytes()))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))
        self.assertNotIn("holdout.json", "\n".join(code.values()))
        self.assertNotIn('"holdout"', code["run"])
        self.assertIn('"dev"', code["run"])

    def test_missing_or_empty_secret_stops_before_installation(self):
        _, code = cells()
        for value in (None, "", "   "):
            namespace = {}
            google = SimpleNamespace()
            colab = SimpleNamespace(userdata=SimpleNamespace(get=lambda name, value=value: value))
            with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "STALE"}), redirect_stdout(io.StringIO()):
                exec(code["parameters"], namespace)
                with self.assertRaises(RuntimeError):
                    exec(code["secret"], namespace)
                self.assertFalse(namespace["SECRET_READY"])
                self.assertNotIn("OPENROUTER_API_KEY", os.environ)

    def test_top_to_bottom_export_and_failed_run_download_before_error(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code):
                self.exercise_notebook(exit_code)

    def exercise_notebook(self, exit_code, path=NOTEBOOK, reference=False, second_stage=True):
        _, code = cells(path)
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            checkout.mkdir()
            for name in ("scripts/compare_triage_generators.py", "docs/experiments/triage_generator/dev.json",
                         "src/reviewnlp/triage/qwen_generator.py", "src/reviewnlp/triage/generator_experiment.py",
                         "src/reviewnlp/triage/staged_generator.py"):
                target = checkout / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            namespace, downloaded, commands = {}, [], []
            secret = "FAKE-NOTEBOOK-SECRET-DO-NOT-EXPORT"

            def fake_run(arguments, **kwargs):
                commands.append(arguments)
                self.assertNotIn("OPENROUTER_API_KEY", kwargs["env"])
                if arguments[:3] == ["git", "rev-parse", "HEAD"]:
                    result = namespace["SOURCE_COMMIT"]
                elif "-c" in arguments:
                    result = json.dumps({"gpu": "fake GPU", "torch": "fake"})
                else:
                    result = ""
                return SimpleNamespace(stdout=result)

            class Process:
                def __init__(self, arguments, **kwargs):
                    self.stdout = io.StringIO("diagnostic " + secret + "\n")
                    output = Path(arguments[arguments.index("--output") + 1])
                    output.mkdir()
                    rows = json.loads((checkout / "docs/experiments/triage_generator/dev.json").read_text())["reviews"]
                    stages = [{"id": row["id"], "complaints": {"status": "ok"}} for row in rows]
                    records = [{"candidate": name, "id": row["id"], "raw": '{"actions":[]}',
                                "upstream_sha256": digest_json(stages[index]), "error": None,
                                "hit_token_budget": False, "actions": []}
                               for name in namespace["EXPECTED_CANDIDATES"] for index, row in enumerate(rows)]
                    for record in records:
                        if record["candidate"] == "F":
                            record["stages"] = [{"stage": "issues", "raw": '{"issues":[]}'}]
                            if second_stage:
                                record["stages"].append({"stage": "measures", "raw": '{"actions":[]}'})
                    (output / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
                    (output / "upstream.json").write_text(json.dumps({"records": stages}))
                    if reference:
                        self_test.assertNotIn("OPENROUTER_API_KEY", kwargs["env"])
                        self_test.assertIn("--reference-run", arguments)
                        self_test.assertNotIn("--allow-external-api", arguments)
                        reference_dir = namespace["REFERENCE_DIR"]
                        shutil.copyfile(reference_dir / "upstream.json", output / "upstream.json")
                        shutil.copyfile(reference_dir / "run.json", output / "reference_run.json")
                    export_annotation(output, rows, records)
                    hashes = {name: digest_bytes((output / name).read_bytes()) for name in
                              ("results.jsonl", "upstream.json", "human_review.csv", "annotation_key.json")}
                    info = {"execution_complete": exit_code == 0, "quality_evaluated": False,
                            "source_commit": namespace["SOURCE_COMMIT"], "dataset_sha256": namespace["REVIEWS_SHA256"],
                            "split": "dev", "generation_budget": {
                                name: {"max_new_tokens_per_call": 400, "max_calls_per_review": 2 if name == "F" else 1}
                                for name in namespace["EXPECTED_CANDIDATES"]}, "candidates": {
                                name: {**CANDIDATES[name], "revision": namespace.get("MODEL_REVISION", "a" * 40)}
                                for name in namespace["EXPECTED_CANDIDATES"]}, "files": hashes}
                    if reference:
                        info.update(upstream_reused=True, api_cost={"attempts": 0}, reference_run={
                            "run_sha256": namespace["REFERENCE_RUN_SHA256"]})
                        info["files"]["reference_run.json"] = digest_bytes((output / "reference_run.json").read_bytes())
                    (output / "run.json").write_text(json.dumps(info))
                    (output / "summary.md").write_text("Synthetic stub run; human review pending.")

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def wait(self):
                    return exit_code

            self_test = self
            reference_zip = Path(directory) / "reference.zip"
            colab = SimpleNamespace(userdata=SimpleNamespace(get=lambda name: secret),
                                    files=SimpleNamespace(download=downloaded.append,
                                                          upload=lambda: {str(reference_zip): reference_zip.read_bytes()}))
            display = SimpleNamespace(Markdown=lambda text: text, display=lambda *args: None)
            modules = {"google": SimpleNamespace(), "google.colab": colab,
                       "IPython": SimpleNamespace(), "IPython.display": display}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {}), \
                    patch("subprocess.run", side_effect=fake_run), patch("subprocess.Popen", Process), \
                    patch("importlib.metadata.version", return_value="fake"), redirect_stdout(io.StringIO()) as stdout:
                exec(code["parameters"], namespace)
                namespace["REPO_DIR"] = checkout
                if reference:
                    os.environ["OPENROUTER_API_KEY"] = secret  # existing Colab secret must not reach the subprocess
                    rows = json.loads((checkout / "docs/experiments/triage_generator/dev.json").read_text())["reviews"]
                    cache = json.dumps({"records": [{"id": row["id"], "complaints": {"status": "ok"}} for row in rows]}).encode()
                    info = json.dumps({"split": "dev", "execution_complete": True,
                                       "dataset_sha256": namespace["REVIEWS_SHA256"], "candidates": {
                                           "C": {"revision": namespace["MODEL_REVISION"]}}}).encode()
                    namespace["REFERENCE_RUN_SHA256"] = digest_bytes(info)
                    namespace["REFERENCE_UPSTREAM_SHA256"] = digest_bytes(cache)
                    with zipfile.ZipFile(reference_zip, "w") as archive:
                        archive.writestr("run.json", info)
                        archive.writestr("upstream.json", cache)
                    exec(code["reference"], namespace)
                else:
                    exec(code["secret"], namespace)
                for name in ("setup", "environment", "run", "audit"):
                    exec(code[name], namespace)
                if exit_code or (path == STAGED_NOTEBOOK and not second_stage):
                    with self.assertRaises(RuntimeError):
                        exec(code["download"], namespace)
                else:
                    exec(code["download"], namespace)
                self.assertEqual(len(downloaded), 1)
                self.assertNotIn(secret, stdout.getvalue())
                with zipfile.ZipFile(downloaded[0]) as archive:
                    self.assertIn("human_review.csv", archive.namelist())
                    self.assertIn("notebook_checks.json", archive.namelist())
                    for name in archive.namelist():
                        self.assertNotIn(secret.encode(), archive.read(name))
                self.assertTrue(any("uninstall" in command for command in commands))
                self.assertFalse(any("torch" in command and "install" in command for command in commands))

    def test_v3_notebook_reuses_reference_without_secret_and_downloads_failed_runs(self):
        notebook, code = cells(V3_NOTEBOOK)
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(code["parameters"], namespace)
        self.assertEqual(namespace["EXPECTED_CANDIDATES"], ("C", "E"))
        self.assertEqual(namespace["SCRIPT_SHA256"], digest_bytes((ROOT / "scripts/compare_triage_generators.py").read_bytes()))
        self.assertNotIn("holdout.json", "\n".join(code.values()))
        self.assertNotIn("userdata.get", "\n".join(code.values()))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code):
                self.exercise_notebook(exit_code, V3_NOTEBOOK, reference=True)

    def test_staged_notebook_pins_the_latest_reference_and_all_changed_modules(self):
        notebook, code = cells(STAGED_NOTEBOOK)
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(code["parameters"], namespace)
        self.assertEqual(namespace["EXPECTED_CANDIDATES"], ("C", "F"))
        self.assertEqual(namespace["SCRIPT_SHA256"], digest_bytes((ROOT / "scripts/compare_triage_generators.py").read_bytes()))
        recorded = ROOT / "docs/experiments/triage_generator/runs/20261003T121429Z_876f0781/run.json"
        self.assertEqual(namespace["REFERENCE_RUN_SHA256"], digest_bytes(recorded.read_bytes()))
        for relative, expected in namespace["MODULE_HASHES"].items():
            self.assertEqual(digest_bytes((ROOT / relative).read_bytes()), expected)
        self.assertNotIn("holdout.json", "\n".join(code.values()))
        self.assertNotIn("userdata.get", "\n".join(code.values()))
        self.assertIn('"--candidates", "C", "F"', code["run"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))

    def test_staged_notebook_runs_top_to_bottom_and_downloads_every_diagnostic_outcome(self):
        for exit_code, second_stage in ((0, True), (1, True), (0, False)):
            with self.subTest(exit_code=exit_code, second_stage=second_stage):
                self.exercise_notebook(exit_code, STAGED_NOTEBOOK, reference=True, second_stage=second_stage)

    def test_wrong_reference_stops_before_setup_or_gpu_calls(self):
        _, code = cells(V3_NOTEBOOK)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "wrong.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("run.json", "{}")
                archive.writestr("upstream.json", "{}")
            namespace = {}
            colab = SimpleNamespace(files=SimpleNamespace(upload=lambda: {}))
            with patch.dict(sys.modules, {"google": SimpleNamespace(), "google.colab": colab}), \
                    redirect_stdout(io.StringIO()):
                exec(code["parameters"], namespace)
                namespace["REFERENCE_ZIP"] = archive_path
                with self.assertRaises(RuntimeError):
                    exec(code["reference"], namespace)
                self.assertFalse(namespace["REFERENCE_READY"])
                self.assertFalse(namespace["SETUP_READY"])

if __name__ == "__main__":
    unittest.main()
