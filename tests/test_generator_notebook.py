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


def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
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
        _, code = cells()
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                checkout = Path(directory) / "checkout"
                checkout.mkdir()
                for name in ("scripts/compare_triage_generators.py", "docs/experiments/triage_generator/dev.json"):
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
                                   for name in CANDIDATES for index, row in enumerate(rows)]
                        (output / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
                        (output / "upstream.json").write_text(json.dumps({"records": stages}))
                        export_annotation(output, rows, records)
                        hashes = {name: digest_bytes((output / name).read_bytes()) for name in
                                  ("results.jsonl", "upstream.json", "human_review.csv", "annotation_key.json")}
                        info = {"execution_complete": exit_code == 0, "quality_evaluated": False,
                                "source_commit": namespace["SOURCE_COMMIT"], "dataset_sha256": namespace["REVIEWS_SHA256"],
                                "split": "dev", "candidates": CANDIDATES, "files": hashes}
                        (output / "run.json").write_text(json.dumps(info))
                        (output / "summary.md").write_text("Synthetic stub run; human review pending.")

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def wait(self):
                        return exit_code

                colab = SimpleNamespace(userdata=SimpleNamespace(get=lambda name: secret),
                                        files=SimpleNamespace(download=downloaded.append))
                display = SimpleNamespace(Markdown=lambda text: text, display=lambda *args: None)
                modules = {"google": SimpleNamespace(), "google.colab": colab,
                           "IPython": SimpleNamespace(), "IPython.display": display}
                with patch.dict(sys.modules, modules), patch.dict(os.environ, {}), \
                        patch("subprocess.run", side_effect=fake_run), patch("subprocess.Popen", Process), \
                        patch("importlib.metadata.version", return_value="fake"), redirect_stdout(io.StringIO()) as stdout:
                    exec(code["parameters"], namespace)
                    namespace["REPO_DIR"] = checkout
                    for name in ("secret", "setup", "environment", "run", "audit"):
                        exec(code[name], namespace)
                    if exit_code:
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


if __name__ == "__main__":
    unittest.main()
