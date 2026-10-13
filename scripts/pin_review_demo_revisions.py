"""Pin the hotel-review-demo Space's own model repos to the commits they serve today.

Reads the current commit of each repo from the Hub (public repos, read-only, no token) and writes
it into spaces/hotel-review-demo/app.py, so a later push to a model repo cannot change what the
demo answers. ``--check`` only reports whether every revision in the app is a 40-hex commit.
No Hub write, no model download, no Space upload: publishing the Space stays with the operator.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "spaces" / "hotel-review-demo" / "app.py"
OWN_REPOS = {
    "QWEN_ADAPTER_REVISION": "krimits/hotel-review-nlp-qwen-qlora-adapter",
    "GREEK_REVISION": "krimits/greek-hotel-reviews-sentiment",
}
REVISION_LINE = re.compile(r'^(?P<name>[A-Z_]+_REVISION) = "(?P<value>[^"]*)".*$', re.MULTILINE)
COMMIT = re.compile(r"[0-9a-f]{40}")


def revisions(source: str) -> dict[str, str]:
    return {match["name"]: match["value"] for match in REVISION_LINE.finditer(source)}


def unpinned(source: str) -> list[str]:
    return sorted(name for name, value in revisions(source).items() if not COMMIT.fullmatch(value))


def pin(source: str, resolve: Callable[[str], str]) -> str:
    """Replace each own-repo revision line with the commit ``resolve`` returns for that repo."""
    found = revisions(source)
    missing = sorted(set(OWN_REPOS) - set(found))
    if missing:
        raise ValueError("revision constants not found: " + ", ".join(missing))
    commits = {}
    for name, repo in OWN_REPOS.items():
        commit = resolve(repo)
        if not COMMIT.fullmatch(commit or ""):
            raise ValueError(f"{repo} did not resolve to a 40-hex commit: {commit!r}")
        commits[name] = commit

    def replace(match: re.Match) -> str:
        name = match["name"]
        return f'{name} = "{commits[name]}"' if name in commits else match[0]

    return REVISION_LINE.sub(replace, source)


def hub_commit(repo: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo)
    print(f"{repo}: {info.sha} (last modified {info.last_modified})")
    return info.sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app", type=Path, default=APP)
    parser.add_argument("--check", action="store_true", help="only report unpinned revisions")
    args = parser.parse_args(argv)
    source = args.app.read_text(encoding="utf-8")
    if not args.check:
        source = pin(source, hub_commit)
        args.app.write_text(source, encoding="utf-8")
    left = unpinned(source)
    if left:
        print("not pinned to a commit: " + ", ".join(left), file=sys.stderr)
        return 1
    print("every model revision in the app is a commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
