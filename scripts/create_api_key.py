"""Generate an independent, high-entropy API key scoped to one hotel."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotel-id", required=True)
    args = parser.parse_args()
    hotel_id = args.hotel_id.strip()
    if not hotel_id:
        parser.error("--hotel-id cannot be blank")
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    print("Show the raw key to the hotel administrator only once:")
    print(token)
    print("Set REVIEWNLP_API_KEYS_JSON on the server to:")
    print(json.dumps({digest: [hotel_id]}))
    print("Use HTTPS for all requests carrying X-API-Key. Never commit the key.")


if __name__ == "__main__":
    main()
