"""Draw the pilot sample for labelling the complaint lexicon.

The design is fixed in docs/annotation/pilot_protocol.md, and this script
follows it. The periods are February-July 2016 and February-July 2017, and the
hotels are those of the second run of scripts/complaint_trends.py. In each
period it draws:

    R    100 random texts that pass has_complaint
    M_t  for each of five topics, random texts in which the lexicon names the
         topic, outside R and drawn independently per topic (cleanliness 14,
         responsiveness 14, bathroom 8, air conditioning 8, pests 6)
    B    20 texts from R and 20 from the M strata, for the second annotator

    python scripts/annotation_sample.py --expect-hotels 863

Writes into --out (default runs/annotation, git-ignored because the sheets hold
review text):

    sheet_A.csv, sheet_B.csv  item, text and empty label columns
    key.csv                   item -> raw_row, period, strata and lexicon topics;
                              no text
    manifest.json             population sizes, the filter's blind spot, seed,
                              code commit and the protocol's SHA-256
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import complaint_trends as trends  # noqa: E402

PROTOCOL = ROOT / "docs" / "annotation" / "pilot_protocol.md"
PERIODS = ("base", "recent")
TOPICS = {"bathroom": "room.bathroom", "cleanliness": "cleanliness.general", "air_conditioning": "room.climate",
          "pests": "cleanliness.pests", "responsiveness": "staff.response"}
R_SIZE = 100
M_SIZES = {"bathroom": 8, "cleanliness": 14, "air_conditioning": 8, "pests": 6, "responsiveness": 14}
B_FROM_R = 20
B_FROM_M = 20
SINCE = "2016-02"
MIN_HOTEL_REVIEWS = 30
SHEET_COLUMNS = ["item", "text", *TOPICS, "done", "note"]


def period_of(frame: pd.DataFrame, params: dict[str, str]) -> pd.Series:
    """'base', 'recent' or '' for each review, by calendar month."""
    month = frame["review_date"].dt.strftime("%Y-%m")
    period = pd.Series("", index=frame.index)
    for name in PERIODS:
        period[month.between(params[f"{name}_start"], params[f"{name}_end"])] = name
    return period


def compared_hotels(frame: pd.DataFrame, period: pd.Series, min_reviews: int) -> list[str]:
    """Addresses with at least min_reviews reviews of any kind in each period, as in query 03."""
    inside = period != ""
    counts = (frame.loc[inside].groupby([frame.loc[inside, "address"], period[inside]]).size()
              .unstack(fill_value=0).reindex(columns=list(PERIODS), fill_value=0))
    return sorted(counts.index[(counts["base"] >= min_reviews) & (counts["recent"] >= min_reviews)])


def population(frame: pd.DataFrame, period: pd.Series, hotels: list[str], workers: int = 1) -> pd.DataFrame:
    """The texts of P_p, indexed by review_id, with the lexicon's verdict for each of the five topics."""
    rows = frame[(period != "") & frame["address"].isin(hotels) & frame["has_complaint_text"]]
    found = pd.DataFrame(trends.tag_complaints(rows, workers=workers), columns=["review_id", "topic", "quote"])
    pop = pd.DataFrame({"raw_row": rows["raw_row"].to_numpy(), "period": period[rows.index].to_numpy(),
                        "text": rows["negative"].to_numpy()}, index=pd.Index(rows.index, name="review_id"))
    for column, topic in TOPICS.items():
        pop[f"lex_{column}"] = pop.index.isin(found.loc[found["topic"] == topic, "review_id"])
    return pop


def blind_spot(frame: pd.DataFrame, period: pd.Series, hotels: list[str]) -> dict[str, dict[str, int]]:
    """What has_complaint set aside in each period: Booking's placeholder or an empty field, and the rest."""
    counts = {}
    for name in PERIODS:
        rows = frame[(period == name) & frame["address"].isin(hotels) & ~frame["has_complaint_text"]]
        placeholder = rows["negative"].str.strip().str.lower().isin(["", "no negative"])
        counts[name] = {"placeholder_or_empty": int(placeholder.sum()), "excluded_by_filter": int((~placeholder).sum())}
    return counts


def draw(pop: pd.DataFrame, seed: int = 0, r_size: int = R_SIZE, m_sizes: dict[str, int] | None = None,
         b_from_r: int = B_FROM_R, b_from_m: int = B_FROM_M) -> tuple[pd.DataFrame, dict]:
    """The key, one row per sampled text with its strata, and the realised sizes per period."""
    m_sizes = dict(M_SIZES if m_sizes is None else m_sizes)
    rng = np.random.default_rng(seed)
    rows, sizes = [], {}
    for name in PERIODS:
        texts = pop[pop["period"] == name]
        ids = np.sort(texts.index.to_numpy())
        if len(ids) < r_size:
            raise ValueError(f"{name}: {len(ids)} texts, fewer than R = {r_size}")
        in_r = set(rng.choice(ids, size=r_size, replace=False).tolist())
        drawn_for: dict[int, list[str]] = {}
        realised, shortfall = {}, {}
        for column in TOPICS:
            eligible = np.sort(texts.index[texts[f"lex_{column}"] & ~texts.index.isin(list(in_r))].to_numpy())
            size = min(m_sizes[column], len(eligible))
            for review_id in (rng.choice(eligible, size=size, replace=False).tolist() if size else []):
                drawn_for.setdefault(review_id, []).append(column)
            realised[column], shortfall[column] = size, m_sizes[column] - size
        from_r = rng.choice(np.sort(list(in_r)), size=min(b_from_r, len(in_r)), replace=False).tolist()
        m_texts = np.sort(list(drawn_for))
        from_m = rng.choice(m_texts, size=min(b_from_m, len(m_texts)), replace=False).tolist() if len(m_texts) else []
        in_b = set(from_r) | set(from_m)
        for review_id in sorted(in_r | set(drawn_for)):
            rows.append({"review_id": review_id, "period": name, "in_R": int(review_id in in_r),
                         "drawn_for": ";".join(drawn_for.get(review_id, [])), "in_B": int(review_id in in_b)})
        sizes[name] = {"population": len(ids),
                       "lexicon_matches": {column: int(texts[f"lex_{column}"].sum()) for column in TOPICS},
                       "R": r_size, "M": realised, "M_shortfall": shortfall, "M_texts": len(drawn_for),
                       "texts_drawn_for_two_topics": sum(realised.values()) - len(drawn_for),
                       "B_from_R": len(from_r), "B_from_M": len(from_m)}
    key = pd.DataFrame(rows)
    lexicon = pop[[f"lex_{column}" for column in TOPICS]].astype(int)
    key = key.join(pop["raw_row"], on="review_id").join(lexicon, on="review_id")
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    key.insert(0, "item", np.arange(1, len(key) + 1))
    columns = ["item", "raw_row", "review_id", "period", "in_R", "drawn_for", *lexicon.columns, "in_B"]
    return key[columns], sizes


def sheets(key: pd.DataFrame, pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two annotators' sheets: item and text, with empty label columns and no hint of the answer."""
    sheet = pd.DataFrame({"item": key["item"].to_numpy(), "text": pop.loc[key["review_id"], "text"].to_numpy()})
    for column in SHEET_COLUMNS[2:]:
        sheet[column] = ""
    return sheet, sheet[key["in_B"].to_numpy() == 1].reset_index(drop=True)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_sample(csv: Path, out: Path, seed: int = 0, workers: int = 1, expect_hotels: int | None = None,
                 overwrite: bool = False, r_size: int = R_SIZE, m_sizes: dict[str, int] | None = None,
                 b_from_r: int = B_FROM_R, b_from_m: int = B_FROM_M,
                 min_hotel_reviews: int = MIN_HOTEL_REVIEWS) -> dict:
    """Draw the sample, write the sheets, the key and the manifest, and return the manifest."""
    if (out / "sheet_A.csv").exists() and not overwrite:
        raise SystemExit(f"{out / 'sheet_A.csv'} exists; a drawn sample is never redrawn by accident "
                         "(pass --overwrite to replace it)")
    frame, source = trends.load_reviews(csv)
    params = trends.analysis_windows(frame["review_date"], min_hotel_reviews, SINCE)
    period = period_of(frame, params)
    hotels = compared_hotels(frame, period, min_hotel_reviews)
    if expect_hotels is not None and len(hotels) != expect_hotels:
        raise SystemExit(f"{len(hotels)} hotels compared, expected {expect_hotels}")
    pop = population(frame, period, hotels, workers)
    key, sizes = draw(pop, seed, r_size, m_sizes, b_from_r, b_from_m)
    sheet_a, sheet_b = sheets(key, pop)
    blind = blind_spot(frame, period, hotels)

    out.mkdir(parents=True, exist_ok=True)
    sheet_a.to_csv(out / "sheet_A.csv", index=False, encoding="utf-8-sig")
    sheet_b.to_csv(out / "sheet_B.csv", index=False, encoding="utf-8-sig")
    key.to_csv(out / "key.csv", index=False)
    manifest = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": trends._sha256(PROTOCOL),
        "code_commit": git_commit(),
        "raw_csv_sha256": trends._sha256(csv),
        "reviews": source["reviews"],
        "seed": seed,
        "windows": {name: params[name] for name in ("base_start", "base_end", "recent_start", "recent_end",
                                                    "min_hotel_reviews")},
        "hotels_compared": len(hotels),
        "design": {"R": r_size, "M": dict(M_SIZES if m_sizes is None else m_sizes), "B_from_R": b_from_r,
                   "B_from_M": b_from_m},
        "periods": {name: {**sizes[name], **blind[name]} for name in PERIODS},
        "items": len(key),
        "second_annotator_items": int(key["in_B"].sum()),
        "topics": TOPICS,
        "sheet_columns": SHEET_COLUMNS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "annotation")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="the protocol fixes 0")
    parser.add_argument("--expect-hotels", type=int, help="stop unless this many hotels are compared (863)")
    parser.add_argument("--overwrite", action="store_true", help="replace a sample already drawn in --out")
    args = parser.parse_args()
    manifest = build_sample(args.csv, args.out, args.seed, args.workers, args.expect_hotels, args.overwrite)
    for name in PERIODS:
        numbers = manifest["periods"][name]
        print(f"{name}: {numbers['population']:,} texts; lexicon matches {numbers['lexicon_matches']}; "
              f"M drawn {numbers['M']}; set aside by the filter {numbers['excluded_by_filter']:,}")
    print(f"{manifest['items']} texts in sheet_A.csv, {manifest['second_annotator_items']} in sheet_B.csv "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
