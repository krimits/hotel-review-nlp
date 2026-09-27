"""Which complaints are rising and need a look? A SQL analysis of the Booking 515K reviews.

A complaint is a topic named in the guest's own "negative" field: the guest
gave the polarity, and the lexicon of the hotel-ops Space
(spaces/hotel-ops-demo/triage.py) names the topic without a model, so no model
drift can fake a trend. A review counts once per topic.

    python scripts/complaint_trends.py                  # the full file
    python scripts/complaint_trends.py --limit 30000    # a quick look
    python scripts/complaint_trends.py --since 2016-02  # compare only months after February 2016

1. Builds a SQLite database (analysis/complaint_trends/sql/00_schema.sql):
   hotels, reviews and complaints, plus the analysis windows in `params`.
2. Runs the queries 01-07 in that folder and saves each result as a CSV.
3. In Python: resamples hotels to put a 95% interval (and a stricter one, for
   all the topics tested at once) on each topic's change within the same hotels,
   and flags a topic as rising or falling only when the strict interval
   excludes zero and the change is at least 10% of the base rate. It also
   gives each topic's change as a share of the reviews that complain at all,
   which tells a topic that rises faster than complaints from one that rises
   with them.
4. Draws two charts and samples 20 quotes per period for every flagged topic,
   to check by reading that the trend is in the reviews, not in the lexicon.

Outputs go to --out (default runs/complaint_trends): results.json, csv/,
monthly_rates.png, within_hotel_change.png and quotes_for_review.json.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "analysis" / "complaint_trends" / "sql"
QUERIES = ("01_monthly_rates", "02_year_over_year", "03_hotel_counts", "04_within_hotel",
           "05_guest_mix", "06_checks", "07_complaint_text_share")
CITIES = ("Amsterdam", "Barcelona", "London", "Milan", "Paris", "Vienna")
TRAVELLER_TYPES = ("Couple", "Solo traveler", "Family with young children", "Family with older children",
                   "Group", "Travelers with friends")
MIN_RELATIVE_CHANGE = 0.10
# A negative field that opens like this says there was nothing to complain about,
# unless the guest goes on with "but", "except" or "however".
NO_COMPLAINT = re.compile(
    r"^\W*(?:nothing|none|n\s?a\b|nil\b|no complaints?|not much|"
    r"all (?:was )?(?:good|great|perfect|fine)|"
    r"everything (?:was )?(?:perfect|great|fine|good|excellent|wonderful|amazing|fantastic))\b",
    re.I,
)
CONTINUES = re.compile(r"\b(?:but|except|however)\b", re.I)
TOPIC_LABELS = {
    "cleanliness.general": "Cleanliness", "cleanliness.linen": "Linen & towels",
    "cleanliness.odour": "Odours", "cleanliness.pests": "Pests",
    "staff.people": "Staff & host", "staff.response": "Responsiveness",
    "staff.resolution": "Problem resolution", "staff.checkin": "Check-in & check-out",
    "location.general": "Location", "room.general": "Room (general)", "room.comfort": "Comfort",
    "room.size": "Room size", "room.bed": "Bed", "room.climate": "Air conditioning & ventilation",
    "room.bathroom": "Bathroom & shower", "room.storage": "Wardrobe & storage",
    "room.equipment": "Equipment & furniture", "room.view": "View & balcony",
    "food.breakfast": "Breakfast", "food.dining": "Restaurant, bar & coffee", "noise.general": "Noise",
    "value.price": "Price & value", "value.charges": "Extra charges & deposit",
    "facilities.access": "Lift & stairs", "facilities.wifi": "Wi-Fi", "facilities.parking": "Parking",
    "facilities.kitchen": "Kitchen & laundry", "facilities.family": "Babies & children",
    "facilities.leisure": "Pool, spa & gym", "facilities.common": "Common areas",
}


def _load_triage():
    path = ROOT / "spaces" / "hotel-ops-demo" / "triage.py"
    spec = importlib.util.spec_from_file_location("space_triage", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRIAGE = _load_triage()


# --- Reviews ------------------------------------------------------------------

def city_of(address: str) -> str:
    return next((city for city in CITIES if city in address), "Other")


def guest_type(tags: str) -> tuple[str, str]:
    """Trip type and traveller type from Booking's tag list, e.g. "[' Leisure trip ', ' Couple ']"."""
    try:
        values = [str(tag).strip() for tag in ast.literal_eval(tags)]
    except (ValueError, SyntaxError):
        values = []
    trip = next((tag for tag in values if tag in ("Leisure trip", "Business trip")), "Unknown")
    traveller = next((tag for tag in values if tag in TRAVELLER_TYPES), "Unknown")
    return trip, traveller


def has_complaint(text: str) -> bool:
    """False for Booking's "No Negative" and for "nothing"-style answers, unless the guest goes on with "but"."""
    text = str(text).strip()
    if not text or text.lower() == "no negative":
        return False
    return not NO_COMPLAINT.match(text) or bool(CONTINUES.search(text))


def load_reviews(csv_path: Path, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per distinct review, with the fields the analysis needs."""
    raw = pd.read_csv(csv_path, nrows=limit)
    before = len(raw)
    raw = raw.drop_duplicates(subset=["Hotel_Address", "Review_Date", "Reviewer_Nationality",
                                      "Negative_Review", "Positive_Review", "Reviewer_Score"])
    frame = pd.DataFrame({
        "hotel": raw["Hotel_Name"].astype(str).str.strip(),
        "address": raw["Hotel_Address"].astype(str).str.strip(),
        "review_date": pd.to_datetime(raw["Review_Date"], format="%m/%d/%Y"),
        "nationality": raw["Reviewer_Nationality"].fillna("").astype(str).str.strip().replace("", "Unknown"),
        "score": raw["Reviewer_Score"].astype(float),
        "negative": raw["Negative_Review"].fillna("").astype(str).str.strip(),
    })
    frame["city"] = frame["address"].map(city_of)
    types = raw["Tags"].fillna("[]").map(guest_type)
    frame["trip_type"] = [trip for trip, _ in types]
    frame["traveller_type"] = [traveller for _, traveller in types]
    frame["has_complaint_text"] = frame["negative"].map(has_complaint)
    frame = frame.reset_index(drop=True)
    return frame, {"rows_read": before, "duplicate_rows_dropped": before - len(frame), "reviews": len(frame)}


# --- Complaint topics ---------------------------------------------------------

def complaint_topics(text: str) -> list[tuple[str, str]]:
    """(topic, first clause naming it) for a complaint text, each topic once."""
    found: dict[str, str] = {}
    for clause in TRIAGE.split_clauses(text):
        for topic, _ in TRIAGE.clause_topics(clause):
            found.setdefault(topic, clause)
    return list(found.items())


def _topics_of_chunk(chunk: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    return [(review_id, topic, quote) for review_id, text in chunk for topic, quote in complaint_topics(text)]


def tag_complaints(frame: pd.DataFrame, workers: int = 1, chunk_size: int = 2000) -> list[tuple[int, str, str]]:
    items = [(int(i), text) for i, text, complains in
             zip(frame.index, frame["negative"], frame["has_complaint_text"], strict=True) if complains]
    chunks = [items[start:start + chunk_size] for start in range(0, len(items), chunk_size)]
    if workers > 1:
        with Pool(workers) as pool:
            parts = pool.map(_topics_of_chunk, chunks)
    else:
        parts = [_topics_of_chunk(chunk) for chunk in chunks]
    return [row for part in parts for row in part]


# --- Database -----------------------------------------------------------------

def analysis_windows(dates: pd.Series, min_hotel_reviews: int = 30, since: str | None = None) -> dict[str, str]:
    """Full calendar months, and the same months one year apart (up to 12).

    `since` (yyyy-mm) starts the comparison at that month, for instance after
    a change in the review form, so that both periods come after it. The
    monthly series still covers every full month.
    """
    first, last = dates.min(), dates.max()
    first_month = first.to_period("M") + (0 if first.day == 1 else 1)
    last_month = last.to_period("M") - (0 if last.is_month_end else 1)
    start = max(first_month, pd.Period(since, freq="M")) if since else first_month
    months = pd.period_range(start, last_month, freq="M")
    span = min(12, len(months) - 12)
    if span < 1:
        raise ValueError("the comparison needs more than a year of full months")
    recent = months[-span:]
    base = recent - 12
    return {"first_month": str(first_month), "last_month": str(last_month), "comparison_start": str(start),
            "base_start": str(base[0]), "base_end": str(base[-1]),
            "recent_start": str(recent[0]), "recent_end": str(recent[-1]),
            "min_hotel_reviews": str(min_hotel_reviews)}


def build_database(frame: pd.DataFrame, complaints: list[tuple[int, str, str]], params: dict[str, str],
                   path: Path | str = ":memory:") -> sqlite3.Connection:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).unlink(missing_ok=True)
    connection = sqlite3.connect(str(path))
    connection.executescript((SQL_DIR / "00_schema.sql").read_text(encoding="utf-8"))
    hotels = frame.drop_duplicates("address")[["hotel", "address", "city"]].reset_index(drop=True)
    hotel_id = {address: index + 1 for index, address in enumerate(hotels["address"])}
    with connection:
        connection.executemany("INSERT INTO hotels VALUES (?, ?, ?, ?)",
                               [(hotel_id[row.address], row.hotel, row.address, row.city)
                                for row in hotels.itertuples(index=False)])
        connection.executemany(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(int(index), hotel_id[row.address], row.review_date.strftime("%Y-%m-%d"),
              row.review_date.strftime("%Y-%m"), row.nationality, row.trip_type, row.traveller_type,
              float(row.score), int(row.has_complaint_text))
             for index, row in zip(frame.index, frame.itertuples(index=False), strict=True)])
        connection.executemany("INSERT INTO complaints VALUES (?, ?, ?)", complaints)
        connection.executemany("INSERT INTO params VALUES (?, ?)", sorted(params.items()))
    return connection


def run_queries(connection: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {name: pd.read_sql_query((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8"), connection)
            for name in QUERIES}


# --- Statistics ----------------------------------------------------------------

def within_hotel_intervals(hotel_counts: pd.DataFrame, n_resamples: int = 2000, seed: int = 0,
                           levels: tuple[float, ...] = (0.95,)) -> pd.DataFrame:
    """Change in each topic's rate within the same hotels, with hotel-resampling intervals.

    The recent rate is weighted by each hotel's share of base-period reviews,
    as in 04_within_hotel.sql; resampling whole hotels keeps the reviews of a
    hotel together, since they are not independent of each other.
    """
    hotels = np.sort(hotel_counts["hotel_id"].unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hotels), size=(n_resamples, len(hotels)))
    multiplicity = np.stack([np.bincount(draw, minlength=len(hotels)) for draw in draws]).astype(float)
    rows = []
    for topic, group in hotel_counts.groupby("topic"):
        wide = group.pivot(index="hotel_id", columns="period", values=["reviews", "complaining"]).reindex(hotels)
        base_n, recent_n = wide[("reviews", "base")].to_numpy(float), wide[("reviews", "recent")].to_numpy(float)
        base_c, recent_c = wide[("complaining", "base")].to_numpy(float), wide[("complaining", "recent")].to_numpy(float)
        recent_rate = recent_c / recent_n

        def change(weights, base_n=base_n, base_c=base_c, recent_rate=recent_rate):
            base_rate = (weights @ base_c) / (weights @ base_n)
            return (weights @ (base_n * recent_rate)) / (weights @ base_n) - base_rate, base_rate

        point, base_rate = change(np.ones(len(hotels)))
        boot, _ = change(multiplicity)
        row = {"topic": topic, "hotels": len(hotels), "base_rate": base_rate, "within_hotel_change": point,
               "relative_change": point / base_rate if base_rate else float("nan")}
        for level in levels:
            low, high = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
            row[f"ci{level * 100:g}"] = [float(low), float(high)]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("within_hotel_change", ascending=False).reset_index(drop=True)


def flag(row: pd.Series, interval: str) -> str:
    low, high = row[interval]
    if low > 0 and row["relative_change"] >= MIN_RELATIVE_CHANGE:
        return "rising"
    if high < 0 and row["relative_change"] <= -MIN_RELATIVE_CHANGE:
        return "falling"
    return ""


def sample_quotes(connection: sqlite3.Connection, params: dict[str, str], topics: list[str],
                  per_period: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    samples = {}
    for topic in topics:
        samples[topic] = {}
        for period in ("base", "recent"):
            quotes = [quote for (quote,) in connection.execute(
                "SELECT c.quote FROM complaints AS c JOIN reviews AS r ON r.review_id = c.review_id "
                "WHERE c.topic = ? AND r.month BETWEEN ? AND ? ORDER BY c.review_id",
                (topic, params[f"{period}_start"], params[f"{period}_end"]))]
            picked = rng.choice(len(quotes), size=min(per_period, len(quotes)), replace=False) if quotes else []
            samples[topic][period] = [quotes[index] for index in sorted(picked)]
    return samples


# --- Charts --------------------------------------------------------------------

SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # categorical slots 1-3, validated all-pairs
RISING, FALLING, STEADY = "#e34948", "#2a78d6", "#83827d"   # diverging poles and a neutral, >= 3:1
PERIOD = "#f0efec"


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SOFT, labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def draw_monthly(monthly: pd.DataFrame, topics: list[str], params: dict[str, str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    _style(axis)
    for period in ("base", "recent"):
        start = pd.Period(params[f"{period}_start"], freq="M").to_timestamp()
        end = pd.Period(params[f"{period}_end"], freq="M").to_timestamp(how="end")
        axis.axvspan(start, end, color=PERIOD, zorder=0)
        axis.annotate(f"{period} period", (start, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                      textcoords="offset points", fontsize=8, color=INK_SOFT)
    for colour, topic in zip(SERIES, topics[:len(SERIES)], strict=False):
        series = monthly[monthly["topic"] == topic]
        months = pd.PeriodIndex(series["month"], freq="M").to_timestamp()
        values = series["rate_3m"].to_numpy() * 100
        axis.plot(months, values, color=colour, linewidth=2, label=TOPIC_LABELS.get(topic, topic))
        axis.annotate(TOPIC_LABELS.get(topic, topic), (months[-1], values[-1]), xytext=(6, 0),
                      textcoords="offset points", va="center", fontsize=9, color=INK)
    axis.set_ylabel("% of reviews complaining (3-month mean)", color=INK_SOFT, fontsize=9)
    # Headroom above the highest line keeps the period labels clear of the data.
    highest = monthly.loc[monthly["topic"].isin(topics[:len(SERIES)]), "rate_3m"].max() * 100
    axis.set_ylim(0, highest * 1.15)
    if len(topics[:len(SERIES)]) > 1:
        axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    axis.set_title("Complaint topics, share of all reviews per month", loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def draw_within_hotel(summary: pd.DataFrame, interval: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary.sort_values("within_hotel_change").reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8, 0.28 * len(rows) + 1.4), dpi=150, facecolor=SURFACE)
    _style(axis)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    colours = {"rising": RISING, "falling": FALLING, "": STEADY}
    for position, row in rows.iterrows():
        low, high = (value * 100 for value in row[interval])
        colour = colours[row["flag"]]
        axis.plot([low, high], [position, position], color=colour, linewidth=2, solid_capstyle="round")
        axis.plot(row["within_hotel_change"] * 100, position, "o", color=colour, markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.5)
    axis.axvline(0, color=INK_SOFT, linewidth=1)
    axis.set_yticks(range(len(rows)), [TOPIC_LABELS.get(topic, topic) for topic in rows["topic"]], fontsize=8.5)
    axis.set_xlabel("Change within the same hotels, percentage points of reviews", color=INK_SOFT, fontsize=9)
    handles = [plt.Line2D([], [], color=colours[name], marker="o", linewidth=2, label=label)
               for name, label in (("rising", "rising"), ("falling", "falling"), ("", "no clear change"))]
    axis.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    axis.set_title("Which complaints changed, recent year against the same months a year earlier",
                   loc="left", color=INK, fontsize=10.5)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


# --- Run -------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "booking_reviews_515k.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "complaint_trends")
    parser.add_argument("--limit", type=int, help="read only the first N rows (a quick look)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-hotel-reviews", type=int, default=30)
    parser.add_argument("--since", help="start the comparison at this month (yyyy-mm), e.g. after a form change")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "csv").mkdir(exist_ok=True)

    frame, source = load_reviews(args.csv, args.limit)
    source.update(csv_sha256=_sha256(args.csv), limit=args.limit)
    print(f"{source['reviews']:,} reviews; tagging complaint topics ...", flush=True)
    complaints = tag_complaints(frame, workers=args.workers)
    params = analysis_windows(frame["review_date"], args.min_hotel_reviews, args.since)
    connection = build_database(frame, complaints, params, args.out / "reviews.sqlite")
    results = run_queries(connection)
    for name, table in results.items():
        table.to_csv(args.out / "csv" / f"{name}.csv", index=False)

    checks = results["06_checks"]
    failed = checks[checks["value"] != 0]
    if not failed.empty:
        raise SystemExit(f"checks failed, nothing reported:\n{failed.to_string(index=False)}")

    n_topics = results["04_within_hotel"]["topic"].nunique()
    strict = 1 - 0.05 / n_topics
    summary = within_hotel_intervals(results["03_hotel_counts"], args.resamples, args.seed, levels=(0.95, strict))
    strict_name = f"ci{strict * 100:g}"
    sql = results["04_within_hotel"].set_index("topic")
    summary = summary.join(sql[["recent_rate", "recent_rate_base_mix", "raw_change", "mix_effect"]], on="topic")
    shares = results["02_year_over_year"].set_index("topic")
    summary["share_of_complaints_change"] = (summary["topic"].map(shares["recent_share_of_complaints"])
                                             / summary["topic"].map(shares["base_share_of_complaints"]) - 1)
    if not np.allclose(summary["within_hotel_change"].to_numpy(),
                       sql.loc[summary["topic"], "within_hotel_change"].to_numpy()):
        raise SystemExit("the Python and SQL within-hotel changes disagree")
    summary["flag"] = summary.apply(flag, axis=1, interval=strict_name)
    flagged = {name: summary.loc[summary["flag"] == name, "topic"].tolist() for name in ("rising", "falling")}

    to_draw = flagged["rising"] or summary["topic"].head(3).tolist()
    draw_monthly(results["01_monthly_rates"], to_draw, params, args.out / "monthly_rates.png")
    draw_within_hotel(summary, "ci95", args.out / "within_hotel_change.png")
    quotes = sample_quotes(connection, params, flagged["rising"] + flagged["falling"], seed=args.seed)
    (args.out / "quotes_for_review.json").write_text(json.dumps(quotes, indent=1, ensure_ascii=False),
                                                      encoding="utf-8")
    report = {
        "source": source,
        "params": params,
        "hotels": int(connection.execute("SELECT COUNT(*) FROM hotels").fetchone()[0]),
        "complaints": len(complaints),
        "checks": checks.to_dict(orient="records"),
        "complaint_text_share": results["07_complaint_text_share"].to_dict(orient="records"),
        "year_over_year": results["02_year_over_year"].to_dict(orient="records"),
        "within_hotel": summary.to_dict(orient="records"),
        "strict_interval": strict_name,
        "guest_mix": results["05_guest_mix"].to_dict(orient="records"),
        "flagged": flagged,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(summary[["topic", "base_rate", "within_hotel_change", "ci95", strict_name, "share_of_complaints_change",
                   "flag"]].to_string(index=False))
    print(f"rising: {flagged['rising']}  falling: {flagged['falling']}  -> {args.out}")


if __name__ == "__main__":
    main()
