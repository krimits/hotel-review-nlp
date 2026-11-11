"""Publish the hotel pilot Space, and make the hotels' access codes. The owner runs both.

    python scripts/hotel_pilot.py bundle                       # writes runs/hotel_pilot/space/ (git-ignored)
    hf auth login                                              # on your own machine, with your own account
    python scripts/deploy_hotel_pilot_space.py publish         # the Space (public, code-gated) and its private dataset
    python scripts/deploy_hotel_pilot_space.py codes --count 5

`publish` refuses another account, a bundle built for another version of the protocol, or a missing file. It
creates what is missing: the private records dataset and the Space (public: the hotels have no Hugging Face
accounts, and nothing is analysed or recorded without a code). If either already exists it stops, unless `--resume`
is given: then it uses what exists, refuses a records dataset that is public, and uploads the bundle again.

Then, in the Space's settings, set four secrets: OPENROUTER_API_KEY (a key with a spending limit set on OpenRouter),
RATINGS_REPO, HF_TOKEN (a fine-grained token with read and write access to the records dataset only) and
HOTEL_CODES. Type the names by hand; paste only the values.

`codes` prints, on your terminal only, one access code per hotel with its pseudonym (P01, ...), and the HOTEL_CODES
value, which holds only their SHA-256. Keep one code for your own test and delete what it records before the
collection. Changing HOTEL_CODES later also changes the salt of the review hashes, so add hotels by extending the
same list only before the collection starts. Never paste a key, a token or a code into source code, a commit, or a
shared chat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deploy_hotelier_space import ensure_repo  # noqa: E402

OWNER = "krimits"
SPACE_ID = f"{OWNER}/hotel-triage-pilot"
DATASET_ID = f"{OWNER}/hotel-triage-pilot-records"
FILES = ["README.md", "app.py", "logic.py", "requirements.txt", "prompt.md", "display.json", "meta.json"]
BUNDLE = ROOT / "runs" / "hotel_pilot" / "space"
PROTOCOL = ROOT / "docs" / "experiments" / "hotel_pilot" / "PROTOCOL.md"
PROMPT = ROOT / "docs" / "experiments" / "triage_student_v2" / "teacher_prompt_v2.md"


def deploy(api: HfApi, *, resume: bool = False, bundle: Path = BUNDLE, protocol: Path = PROTOCOL,
           prompt: Path = PROMPT) -> dict[str, str]:
    signed_in = api.whoami()["name"]
    if signed_in != OWNER:
        raise RuntimeError(f"Connected to '{signed_in}', expected '{OWNER}'. Nothing was uploaded.")
    missing = [name for name in FILES if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bundle files: {', '.join(missing)}; run `hotel_pilot.py bundle`")
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    if meta["protocol_sha256"] != hashlib.sha256(protocol.read_bytes()).hexdigest():
        raise RuntimeError("The bundle was built for another version of the protocol. Nothing was uploaded.")
    if (bundle / "prompt.md").read_bytes() != prompt.read_bytes():
        raise RuntimeError("The bundle's prompt is not the tested teacher prompt. Nothing was uploaded.")
    dataset = ensure_repo(api, DATASET_ID, "dataset", resume, private=True)
    if dataset is not None and not dataset.private:
        raise RuntimeError(f"{DATASET_ID} is public; the records must be private. Nothing was uploaded.")
    ensure_repo(api, SPACE_ID, "space", resume, space_sdk="gradio", private=False)
    api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=str(bundle), allow_patterns=FILES,
                      delete_patterns=["*"], commit_message=f"Deploy the hotel pilot, bundle {meta['bundle']}")
    return {"space": f"https://huggingface.co/spaces/{SPACE_ID}",
            "records": f"https://huggingface.co/datasets/{DATASET_ID}", "bundle": meta["bundle"]}


def make_codes(count: int, token=secrets.token_urlsafe) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """(pseudonym, code) per hotel, and the HOTEL_CODES value: {sha256(code): pseudonym}."""
    pairs = [(f"P{n:02d}", token(9)) for n in range(1, count + 1)]
    return pairs, {hashlib.sha256(code.encode("utf-8")).hexdigest(): name for name, code in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("publish", help="create what is missing and upload the bundle")
    first.add_argument("--resume", action="store_true", help="use the Space and dataset that already exist")
    second = steps.add_parser("codes", help="print access codes and the HOTEL_CODES secret value")
    second.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.step == "publish":
        for name, value in deploy(HfApi(), resume=args.resume).items():
            print(f"{name}: {value}")
    else:
        pairs, value = make_codes(args.count)
        for name, code in pairs:
            print(f"{name}  {code}")
        print("\nHOTEL_CODES =", json.dumps(value))


if __name__ == "__main__":
    main()
