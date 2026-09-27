"""Fetch the Booking.com 515K reviews CSV and check it is the file the project was built on.

The dataset is published on Kaggle as "515K Hotel Reviews Data in Europe". A
byte-identical copy is on the Hugging Face Hub. The size and SHA-256 below are
those of the Kaggle download the original splits came from, as recorded in
PROJECT_COMPLETION_AUDIT_2026-09-09.md. A file that does not match is refused,
so every split built from it traces back to the same source.

    python scripts/fetch_booking_515k.py          # download, check, place in data/raw
    python scripts/fetch_booking_515k.py --check  # only check the file already there
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

REPO = "Dricz/515k-Hotel-Reviews-In-Europe"
REVISION = "cdaea79962c5d667dd5177956ef0353001f6ade3"
FILENAME = "Hotel_Reviews.csv"
SIZE = 238_154_765
SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "booking_reviews_515k.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, size: int = SIZE, expected: str = SHA256) -> None:
    """Refuse a file that is not byte-identical to the project's source."""
    actual_size = path.stat().st_size
    if actual_size != size:
        raise SystemExit(f"{path}: {actual_size} bytes, expected {size}; it is not the project's source file")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{path}: SHA-256 {actual}, expected {expected}")


def fetch(destination: Path = DESTINATION) -> Path:
    if destination.exists():
        verify(destination)
        return destination
    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(REPO, FILENAME, repo_type="dataset", revision=REVISION))
    verify(downloaded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copyfile(downloaded, partial)
    partial.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check the file already in place")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.check:
        verify(args.destination)
    else:
        fetch(args.destination)
    print(f"{args.destination}: {SIZE:,} bytes, SHA-256 {SHA256} ({REPO} at {REVISION[:7]})")


if __name__ == "__main__":
    main()
