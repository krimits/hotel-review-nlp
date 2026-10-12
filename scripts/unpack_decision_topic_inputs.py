"""Validate the locked human-label ZIP before any model or provider work in Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp.experiments import topic_benchmark as task  # noqa: E402

LOCK = ROOT / "docs/experiments/decision_topic_benchmark/annotation_record.json"
PROTOCOL = ROOT / "docs/experiments/decision_topic_benchmark/PROTOCOL.md"
MAX_UNCOMPRESSED_BYTES = 10 * 1024 * 1024


def unpack_inputs(
    package: Path, output: Path, *, lock_path: Path = LOCK, protocol: Path = PROTOCOL
) -> dict:
    """No extractall, path traversal, substitutions, overwrite or trust in a ZIP's own manifest."""
    lock = json.loads(lock_path.read_bytes())
    expected = lock["input_files_sha256"]
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected) | {"input_manifest.json"}:
            raise ValueError("exact_locked_zip_files_required")
        if sum(item.file_size for item in archive.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("input_zip_too_large")
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
                raise ValueError("unsafe_zip_path")
        content = {name: archive.read(name) for name in names}
    manifest = json.loads(content["input_manifest.json"])
    if (
        manifest.get("version") != lock["version"]
        or manifest.get("files") != expected
        or manifest.get("labels_sha256") != lock["labels_sha256"]
        or manifest.get("sample_manifest_sha256") != lock["sample_manifest_sha256"]
    ):
        raise ValueError("input_manifest_does_not_match_locked_submission")
    for name, sha in expected.items():
        if hashlib.sha256(content[name]).hexdigest() != sha:
            raise ValueError("input_artifact_changed")
    if output.exists():
        raise ValueError("new_input_directory_required")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Re-run the semantic input guards: hashes alone cannot replace complete human judgments.
    regenerated = output / "labels_rechecked.json"
    record = task.finish_labels(
        output / "sample",
        output / "completed_A.csv",
        output / "completed_B.csv",
        output / "completed_final.csv",
        regenerated,
        protocol=protocol,
    )
    if (
        record["sample_size"] != lock["sample_size"]
        or task.digest(regenerated) != lock["labels_sha256"]
        or task.digest(output / "sample/sample_manifest.json") != lock["sample_manifest_sha256"]
    ):
        raise ValueError("locked_annotations_differ_after_revalidation")
    return {
        "sample_size": record["sample_size"],
        "labels_sha256": task.digest(regenerated),
        "independence": record["independence"],
        "all_labels_valid": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(unpack_inputs(args.package, args.output), indent=2))


if __name__ == "__main__":
    main()
