"""Build a self-contained experimental Space package. No login, Hub writes or model downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ("app.py", "README.md", "requirements.txt")
MODULES = (
    "__init__.py", "absa/__init__.py", "absa/aspects.py", "absa/extract.py",
    "triage/__init__.py", "triage/schemas.py", "triage/questions.py", "triage/jev_client.py",
    "triage/routing.py", "triage/qwen_generator.py", "triage/staged_generator.py",
    "triage/evidence_generator.py", "triage/generator_experiment.py", "triage/pipeline.py",
    "triage/demo_service.py",
)


def build_package(root: Path, output: Path, *, source_commit: str | None = None) -> dict:
    """Explicit sources only: no review datasets, checkpoints, environment files or generated results."""
    sources = {name: root / "spaces/hotel-triage-demo" / name for name in SPACE_FILES}
    sources.update({"reviewnlp/" + name: root / "src/reviewnlp" / name for name in MODULES})
    missing = [str(path.relative_to(root)) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package sources: " + ", ".join(missing))
    if source_commit is None:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip():
            raise ValueError("tracked checkout changes must be committed before packaging")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("source_commit must be a full commit hash")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, path in sources.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {"source_commit": source_commit, "files": hashes,
                "validation_status": "unvalidated", "hub_writes": False}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT / "runs").resolve()) or args.output.resolve() == (ROOT / "runs").resolve():
        parser.error("--output must be a new directory inside runs/")
    manifest = build_package(ROOT, args.output)
    print("Prepared", len(manifest["files"]), "files from", manifest["source_commit"], "without Hub writes.")


if __name__ == "__main__":
    main()
