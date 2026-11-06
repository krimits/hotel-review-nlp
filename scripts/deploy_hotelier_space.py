"""Publish the hoteliers' blind evaluation Space, and make the hoteliers' access codes. The owner runs both.

    python scripts/hotelier_blind_eval.py bundle          # writes runs/hotelier_blind_eval/space/ (git-ignored)
    hf auth login                                          # on your own machine, with your own account
    python scripts/deploy_hotelier_space.py publish        # the Space (public, code-gated) and its private dataset
    python scripts/deploy_hotelier_space.py codes --count 2

`publish` refuses another account, a bundle built for another version of the protocol, or a missing file. It
creates the private ratings dataset and the Space (public: the hoteliers have no Hugging Face accounts, and nothing
is recorded without a code), and uploads exactly the bundle's files. If either already exists it stops, unless
`--resume` is given: then it uses what exists, creates what is missing, refuses a ratings dataset that is public,
and uploads the bundle again. Then, in the Space's settings, set the secrets
RATINGS_REPO, HF_TOKEN (a fine-grained token with write access to the ratings dataset only) and HOTELIER_CODES.

`codes` prints, on your terminal only, one access code per hotelier with its pseudonym, and the HOTELIER_CODES
value, which holds only their SHA-256. Give each hotelier their own code; keep the list yourself. Never paste a
token or a code into source code, a commit, or a shared chat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
OWNER = "krimits"
SPACE_ID = f"{OWNER}/hotelier-blind-eval"
DATASET_ID = f"{OWNER}/hotelier-blind-eval-ratings"
FILES = ["README.md", "app.py", "logic.py", "requirements.txt", "items.json", "display.json", "meta.json"]
BUNDLE = ROOT / "runs" / "hotelier_blind_eval" / "space"
PROTOCOL = ROOT / "docs" / "experiments" / "hotelier_blind_eval" / "PROTOCOL.md"


def ensure_repo(api: HfApi, repo_id: str, repo_type: str, resume: bool, **create):
    """Create the repo if it is missing. One that exists is used only with --resume, so an earlier run that stopped
    halfway can be finished; its info is returned. Without --resume an existing repo stops the deployment."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=False, **create)
        return None
    if not resume:
        raise RuntimeError(f"{repo_id} already exists. Rerun with --resume to use it. Nothing was uploaded.")
    return info


def deploy(api: HfApi, *, resume: bool = False, bundle: Path = BUNDLE, protocol: Path = PROTOCOL) -> dict[str, str]:
    signed_in = api.whoami()["name"]
    if signed_in != OWNER:
        raise RuntimeError(f"Connected to '{signed_in}', expected '{OWNER}'. Nothing was uploaded.")
    missing = [name for name in FILES if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bundle files: {', '.join(missing)}; run `hotelier_blind_eval.py bundle`")
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    if meta["protocol_sha256"] != hashlib.sha256(protocol.read_bytes()).hexdigest():
        raise RuntimeError("The bundle was built for another version of the protocol. Nothing was uploaded.")
    dataset = ensure_repo(api, DATASET_ID, "dataset", resume, private=True)
    if dataset is not None and not dataset.private:
        raise RuntimeError(f"{DATASET_ID} is public; the ratings must be private. Nothing was uploaded.")
    ensure_repo(api, SPACE_ID, "space", resume, space_sdk="gradio", private=False)
    api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=str(bundle), allow_patterns=FILES,
                      delete_patterns=["*"], commit_message=f"Deploy the blind evaluation, bundle {meta['bundle']}")
    return {"space": f"https://huggingface.co/spaces/{SPACE_ID}",
            "ratings": f"https://huggingface.co/datasets/{DATASET_ID}", "bundle": meta["bundle"]}


def make_codes(count: int, token=secrets.token_urlsafe) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """(pseudonym, code) per hotelier, and the HOTELIER_CODES value: {sha256(code): pseudonym}."""
    pairs = [(f"H{n:02d}", token(9)) for n in range(1, count + 1)]
    return pairs, {hashlib.sha256(code.encode("utf-8")).hexdigest(): name for name, code in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    first = steps.add_parser("publish", help="create the private ratings dataset and the Space, and upload the bundle")
    first.add_argument("--resume", action="store_true", help="upload again to the existing Space and dataset")
    second = steps.add_parser("codes", help="print access codes and the HOTELIER_CODES secret value")
    second.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.step == "publish":
        for name, value in deploy(HfApi(), resume=args.resume).items():
            print(f"{name}: {value}")
    else:
        pairs, secret = make_codes(args.count)
        for name, code in pairs:
            print(f"{name}  {code}")
        print("\nHOTELIER_CODES =", json.dumps(secret), file=sys.stdout)


if __name__ == "__main__":
    main()
