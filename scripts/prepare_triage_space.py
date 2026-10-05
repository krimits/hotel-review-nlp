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
    "triage/space_runtime.py",
    "triage/span_evidence_generator.py",
)


def verify_package(package: Path) -> dict:
    """Refuse extra files, symlinks and changes since the allowlisted package was built."""
    manifest_path = package / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {*SPACE_FILES, *("reviewnlp/" + name for name in MODULES)}
    actual = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()}
    if actual != expected | {"source_manifest.json"} or set(manifest["files"]) != expected:
        raise ValueError("package_file_allowlist_mismatch")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise ValueError("package_symlink_rejected")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != digest:
            raise ValueError("package_hash_mismatch: " + name)
    if manifest.get("selection_status") != "experimental_not_selected_not_promoted":
        raise ValueError("experimental_selection_status_required")
    commit = manifest["source_commit"]
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("invalid_source_commit")
    return manifest


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
    from reviewnlp.triage.demo_service import DISTILBERT_REVISION, QWEN_REVISION
    from reviewnlp.triage.evidence_generator import CATEGORY_DEPARTMENTS
    from reviewnlp.triage.questions import QUESTIONS_SHA256, QUESTIONS_VERSION
    from reviewnlp.triage.span_evidence_generator import (
        CANDIDATE,
        SPAN_POLICY,
        VERSION,
        prompt_fingerprint,
    )

    manifest = {"source_commit": source_commit, "files": hashes,
                "validation_status": "unvalidated", "hub_writes": False,
                "selection_status": "experimental_not_selected_not_promoted",
                "runtime_snapshot": {"candidate": CANDIDATE, "prompt_version": VERSION,
                    "prompt_sha256": prompt_fingerprint(), "source_span_policy": SPAN_POLICY,
                    "distilbert_revision": DISTILBERT_REVISION,
                    "qwen_revision": QWEN_REVISION, "category_departments": CATEGORY_DEPARTMENTS,
                    "jev_questions_version": QUESTIONS_VERSION, "jev_questions_sha256": QUESTIONS_SHA256,
                    "jev_default": False, "jev_model_policy": "mutable route; resolved model reported per request",
                    "routing": "all_reviews_diagnostic", "max_review_characters": 4000,
                    "distilbert_max_tokens": 256, "max_calls_per_review": 2, "max_new_tokens_per_call": 400,
                    "do_sample": False, "zero_gpu_precision": "bfloat16", "gpu_duration_seconds": 60,
                    "gpu_billing": "unknown", "requirements_sha256": hashes["requirements.txt"]}}
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
