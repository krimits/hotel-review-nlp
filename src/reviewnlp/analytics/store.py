"""Where aspect rows come from.

scripts/schema.sql describes the tables that would hold them, but nothing in
this repository implements that schema and no driver is declared in
pyproject.toml. So `get_aspect_store` returns None and the recommendations
endpoint reports 501 rather than inventing numbers.

This is the single seam to fill in: implement `AspectStore` against whatever
storage arrives and return it from `get_aspect_store`. Everything downstream —
aggregation, trend, priority, the response shape — is already implemented and
tested against it.
"""

from __future__ import annotations

from typing import Protocol


class AspectStore(Protocol):
    """Read access to extracted aspect rows for one hotel.

    A row is one aspect mention from one review:
    `{"review_id": str, "aspect": str, "sentiment": str}`. A review that
    mentions the same aspect twice should still be counted once, so `review_id`
    lets the aggregation deduplicate — the same rule `run_absa.py` uses for its
    per-review aspect frequencies.
    """

    def aspect_rows(self, hotel_id: str, days: int) -> list[dict]:
        """Rows from the last `days` days."""
        ...

    def previous_period_rows(self, hotel_id: str, days: int) -> list[dict]:
        """Rows from the `days` window immediately before that, for the trend."""
        ...


def get_aspect_store() -> AspectStore | None:
    """The configured store, or None when there is no persistence layer.

    Returns None today. Wiring a real store in is the only change needed here.
    """
    return None
