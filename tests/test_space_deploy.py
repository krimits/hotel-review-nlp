"""Guard the account and privacy of the one-shot HF deployment helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deploy_hf_space.py"
_SPEC = importlib.util.spec_from_file_location("hotel_space_deploy", _PATH)
deploy_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy_module)


class RecordingApi:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def whoami(self):
        return {"name": self.name}

    def create_repo(self, **kwargs):
        self.calls.append(("create", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs))


def test_wrong_account_never_creates_or_uploads():
    api = RecordingApi("another-user")
    with pytest.raises(RuntimeError, match="Nothing was uploaded"):
        deploy_module.deploy(api)
    assert api.calls == []


def test_private_space_and_only_source_files_uploaded():
    api = RecordingApi("krimits")
    url = deploy_module.deploy(api)
    assert url == "https://huggingface.co/spaces/krimits/hotel-review-operations-demo"
    assert api.calls[0] == ("create", {
        "repo_id": "krimits/hotel-review-operations-demo", "repo_type": "space",
        "space_sdk": "gradio", "private": True, "exist_ok": False,
    })
    assert api.calls[1][0] == "upload"
    assert api.calls[1][1]["allow_patterns"] == [
        "README.md", "app.py", "logic.py", "requirements.txt", "triage.py",
    ]
    # Files of an older version, such as inference.py, must not linger in the Space.
    assert api.calls[1][1]["delete_patterns"] == ["*"]
