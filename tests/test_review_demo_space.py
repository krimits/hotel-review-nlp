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


def demo_prompt_code() -> dict:
    """The demo's prompt constants and functions, run on their own: the app loads models on import."""
    names = {"LABELS", "SYSTEM_PROMPT", "USER_TEMPLATE", "ASSISTANT_PREFIX", "format_prompt", "parse_label"}
    body = [node for node in ast.parse(APP.read_text(encoding="utf-8")).body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names)]
    namespace: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), namespace)
    assert names <= set(namespace)
    return namespace


def test_greekbert_gives_no_label_for_text_without_greek():
    """GreekBERT answered English reviews confidently and meaninglessly; it now says why it does not."""
    names = {"GREEK_LETTERS", "GREEK_ONLY_NOTE", "has_greek", "predict_greek"}
    body = [node for node in ast.parse(APP.read_text(encoding="utf-8")).body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names)]
    namespace: dict = {"re": __import__("re")}  # no tokenizer or model: a non-Greek review must not reach them
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), namespace)
    assert names <= set(namespace)
    for english in ("The room was dirty and the staff were rude.", "Great stay, 10/10!"):
        assert namespace["predict_greek"](english) == ({"negative": 0.0, "positive": 0.0}, namespace["GREEK_ONLY_NOTE"])
    for greek in ("Εξαιρετικό δωμάτιο.", "ἄριστον", "Room was ok αλλά θορυβώδες"):
        assert namespace["has_greek"](greek)
    assert namespace["predict_greek"]("   ")[1] == "empty input"


def test_the_demo_prompts_qwen_exactly_as_training_did():
    from reviewnlp.llm import prompt_format

    demo = demo_prompt_code()
    for review in ("Great stay.", "  The   bed\nwas fine,\tthe shower cold. ", "x" * 5000, "Εξαιρετικό δωμάτιο."):
        assert demo["format_prompt"](review) == prompt_format.format_prompt(review)
    assert demo["format_prompt"]("ok").count("<|im_start|>") == 3
    for generated in ("positive\npositive\n", " Negative.", "positive<|im_end|>", "unsure", "positively great"):
        assert demo["parse_label"](generated) == prompt_format.parse_label(generated)
