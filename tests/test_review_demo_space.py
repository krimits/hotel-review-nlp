"""The sentiment Space names a commit for every model it downloads; checked offline, no models loaded."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "spaces" / "hotel-review-demo" / "app.py"
LOCAL_SNAPSHOTS = {"distilbert_dir", "adapter_dir"}  # folders a pinned snapshot_download returned


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script("pin_review_demo_revisions", ROOT / "scripts/pin_review_demo_revisions.py")


def model_loads(source: str) -> list[ast.Call]:
    loaders = {"from_pretrained", "snapshot_download"}
    return [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) in loaders]


def loads_from_local_snapshot(call: ast.Call) -> bool:
    """True when the path argument is a folder a pinned snapshot_download returned (PeftModel's is second)."""
    for argument in call.args:
        if isinstance(argument, ast.Call) and getattr(argument.func, "id", None) == "str":
            argument = argument.args[0]
        if isinstance(argument, ast.Name) and argument.id in LOCAL_SNAPSHOTS:
            return True
    return False


def test_every_hub_download_names_a_revision_constant():
    source = APP.read_text(encoding="utf-8")
    calls = model_loads(source)
    assert len(calls) == 9  # 2 snapshots, 3 tokenizers, 3 models, 1 adapter wrap
    for call in calls:
        if loads_from_local_snapshot(call):
            continue
        revision = {keyword.arg: keyword.value for keyword in call.keywords}.get("revision")
        assert isinstance(revision, ast.Name) and revision.id.endswith("_REVISION"), ast.unparse(call)
    assert set(script.revisions(source)) == {"DISTILBERT_REVISION", "QWEN_BASE_REVISION", *script.OWN_REPOS}


def test_every_revision_in_the_app_is_a_commit():
    assert script.unpinned(APP.read_text(encoding="utf-8")) == []


def test_pin_writes_the_resolved_commits_and_check_then_passes(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(APP.read_text(encoding="utf-8"), encoding="utf-8")
    commits = {repo: f"{index:x}" * 40 for index, repo in enumerate(script.OWN_REPOS.values(), start=1)}
    pinned = script.pin(app.read_text(encoding="utf-8"), commits.__getitem__)
    assert script.unpinned(pinned) == []
    assert script.revisions(pinned)["GREEK_REVISION"] == commits["krimits/greek-hotel-reviews-sentiment"]
    assert script.revisions(pinned)["QWEN_BASE_REVISION"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert len(pinned.splitlines()) == len(app.read_text(encoding="utf-8").splitlines())
    app.write_text(pinned, encoding="utf-8")
    assert script.main(["--app", str(app), "--check"]) == 0


def test_pin_refuses_anything_but_a_commit_and_check_reports_what_is_left(tmp_path, capsys):
    source = APP.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="40-hex"):
        script.pin(source, lambda repo: "main")
    with pytest.raises(ValueError, match="not found"):
        script.pin(source.replace("GREEK_REVISION =", "GREEK_REV ="), lambda repo: "a" * 40)
    app = tmp_path / "app.py"
    app.write_text(source.replace(script.revisions(source)["QWEN_BASE_REVISION"], "main"), encoding="utf-8")
    assert script.main(["--app", str(app), "--check"]) == 1
    assert "QWEN_BASE_REVISION" in capsys.readouterr().err
