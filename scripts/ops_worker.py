"""Send the operations demo's due notifications: once, or every few seconds until stopped.

    REVIEWNLP_DB_PATH=runs/ops/ops.db python scripts/ops_worker.py --once
    REVIEWNLP_DB_PATH=runs/ops/ops.db python scripts/ops_worker.py --every 30

Test e-mail goes to REVIEWNLP_OPS_MAIL_DIR (default runs/ops_mail) as one .eml file per notification; SMTP is used
only when the owner sets REVIEWNLP_OPS_SMTP_HOST and its credentials in the environment. A failed try is retried
with growing waits and given up after five; no model is called and nothing is posted to any review site.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewnlp.ops.notify import deliver_due, transport_from_env  # noqa: E402
from reviewnlp.ops.store import SqliteOpsStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="send what is due and stop")
    group.add_argument("--every", type=float, help="seconds between rounds")
    args = parser.parse_args()
    path = os.getenv("REVIEWNLP_DB_PATH")
    if not path:
        raise SystemExit("set REVIEWNLP_DB_PATH to the operations database")
    store, transport = SqliteOpsStore(path), transport_from_env()
    while True:
        counts = deliver_due(store, transport)
        if any(counts.values()):
            print(f"[{transport.name}] sent {counts['sent']}, retry later {counts['retry']}, gave up {counts['failed']}",
                  flush=True)
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
