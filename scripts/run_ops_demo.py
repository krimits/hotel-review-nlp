"""Start the operations demo locally, with no environment variables to type: the same in cmd and PowerShell.

    .venv\\Scripts\\python.exe scripts\\run_ops_demo.py                 # keyword stand-ins, nothing leaves the machine
    .venv\\Scripts\\python.exe scripts\\run_ops_demo.py --live          # the model, if OPENROUTER_API_KEY is set

It keeps one local demo key for the test hotel in runs/ops/ (git-ignored), makes it on the first run and prints it
on every run, sets the database, the hotel key and the demo switch for this process only, and starts the app on
127.0.0.1. The key opens only this local demo; it is not a key for a real hotel. With --live the review text goes
to OpenRouter and Anthropic, and the analyses go to a separate database, because a review is never analysed twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "runs" / "ops"


def demo_key(hotel_id: str) -> str:
    """The local demo key for this hotel: made once, then read back."""
    path = FOLDER / f"demo-key-{hotel_id}.txt"
    if not path.exists():
        FOLDER.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hotel-id", default="hotel-demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--live", action="store_true", help="use the model through OpenRouter (needs its key)")
    args = parser.parse_args()
    if args.live and not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("--live needs OPENROUTER_API_KEY set in this terminal first")
    key = demo_key(args.hotel_id)
    os.environ.update({
        "MODEL_TYPE": os.getenv("MODEL_TYPE", "stub"),
        "REVIEWNLP_DB_PATH": str(FOLDER / ("ops-live.db" if args.live else "ops.db")),
        "REVIEWNLP_API_KEYS_JSON": json.dumps({hashlib.sha256(key.encode("utf-8")).hexdigest(): [args.hotel_id]}),
        "REVIEWNLP_OPS_MAIL_DIR": str(FOLDER / "mail"),
    })
    if args.live:
        os.environ.pop("REVIEWNLP_OPS_DEMO", None)
    else:
        os.environ["REVIEWNLP_OPS_DEMO"] = "1"
        os.environ.pop("OPENROUTER_API_KEY", None)  # the stand-ins must not call out by accident
    print(f"Dashboard:  http://127.0.0.1:{args.port}/dashboard")
    print(f"Hotel id:   {args.hotel_id}")
    print(f"API key:    {key}   (local demo key, kept in runs/ops/)")
    print(f"Analysis:   {'the model through OpenRouter' if args.live else 'keyword stand-ins (nothing is sent out)'}", flush=True)
    print(f"Test email: {FOLDER / 'mail'}")
    sys.path.insert(0, str(ROOT / "src"))
    import uvicorn

    uvicorn.run("reviewnlp.serving.app:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
