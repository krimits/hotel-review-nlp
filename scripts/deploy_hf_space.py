"""Create the private hotel-operations demo with the current HF CLI login.

Run from the repository root after `hf auth login` on your own machine. Never
paste an access token into source code, a shell command, or a support ticket.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

OWNER = "krimits"
REPO_ID = f"{OWNER}/hotel-review-operations-demo"
FILES = ["README.md", "app.py", "logic.py", "requirements.txt", "triage.py"]
DEMO_DIR = Path(__file__).resolve().parents[1] / "spaces" / "hotel-ops-demo"


def deploy(api: HfApi, *, resume: bool = False, demo_dir: Path = DEMO_DIR) -> str:
    """Publish exactly the audited demo files to the intended HF account.

    Any other file already in the Space (such as a module from an older
    version) is removed in the same commit; `.gitattributes` is always kept.
    """
    signed_in = api.whoami()["name"]
    if signed_in != OWNER:
        raise RuntimeError(
            f"Connected to '{signed_in}', expected '{OWNER}'. Nothing was uploaded."
        )
    missing = [name for name in FILES if not (demo_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Space files: {', '.join(missing)}")

    if resume:
        api.repo_info(repo_id=REPO_ID, repo_type="space")
    else:
        # Refuse to silently replace another Space with this name.
        api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio",
                        private=True, exist_ok=False)

    api.upload_folder(repo_id=REPO_ID, repo_type="space", folder_path=str(demo_dir),
                      allow_patterns=FILES, delete_patterns=["*"],
                      commit_message="Deploy hotel review operations demo")
    return f"https://huggingface.co/spaces/{REPO_ID}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="Upload again to this Space if it already exists (e.g. after a failed build)")
    args = parser.parse_args()
    print(deploy(HfApi(), resume=args.resume))


if __name__ == "__main__":
    main()
