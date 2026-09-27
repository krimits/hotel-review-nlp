# Datasets

This project uses two datasets. Only `hotel_reviews.csv` is committed.

## 1. `hotel_reviews.csv` (committed, 658 rows)

Unlabeled TripAdvisor-style reviews for two NYC hotels (Hudson Hotel,
Westin New York at Times Square), 2008-2009. Columns: `Hotel`, `Date`,
`Review Title`, `Review`.

It plays the role of **real, unlabeled production traffic**: the annotation
demo (`notebooks/03_llm_annotation_demo.ipynb`) shows how a fine-tuned LLM
turns it into a labeled corpus, and the FastAPI service uses it in the
`/predict` examples.

## 2. Booking.com 515K Hotel Reviews (not committed - download it)

Source: Kaggle dataset **"515k Hotel Reviews Data in Europe"**
(https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe),
~515k reviews of 1,493 hotels, each with a positive and a negative free-text
field plus a 2.5-10 reviewer score.

**One command:** `make data` runs `scripts/fetch_booking_515k.py`, which downloads a
byte-identical copy from the Hugging Face Hub
([`Dricz/515k-Hotel-Reviews-In-Europe`](https://huggingface.co/datasets/Dricz/515k-Hotel-Reviews-In-Europe),
pinned revision `cdaea79`). It checks the size (238,154,765 bytes) and the SHA-256
(`a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9`) against the Kaggle
download the original splits were built from, and refuses any other file. Then it builds
the three v2 splits (random, out-of-time, unseen hotels); each split directory gets a
`data_manifest.json` with the raw file's hash.

To use your own Kaggle download instead, download and extract the Kaggle archive into a **separate temporary directory**,
then copy its `Hotel_Reviews.csv` into this project as
`data/raw/booking_reviews_515k.csv`. The baseline config and EDA notebook use that name.

**Windows:** do not extract `Hotel_Reviews.csv` directly into `data/raw/`.
Windows treats it as the same filename as the committed `hotel_reviews.csv`,
which contains a different schema and only 658 unlabeled reviews.

Example after downloading and extracting the archive elsewhere:

```powershell
Copy-Item -LiteralPath 'C:\path\to\extracted\Hotel_Reviews.csv' -Destination 'data\raw\booking_reviews_515k.csv'
```

Only the small unlabeled `hotel_reviews.csv` is tracked. The 515K Booking CSV,
processed parquet files and model weights remain local and are ignored by Git.

### Label construction (no score-threshold ambiguity)

Each row has two independent free-text fields:

- `Positive_Review` - "No Positive" when the guest wrote none
- `Negative_Review` - "No Negative" when the guest wrote none

We build a **clean binary sentiment dataset** from unambiguous rows only:

| Positive_Review | Negative_Review | label    | text used              |
|-----------------|-----------------|----------|------------------------|
| written         | "No Negative"   | positive | Positive_Review        |
| "No Positive"   | written         | negative | Negative_Review        |
| written         | written         | *dropped from binary task* | - |

Mixed reviews are dropped from the binary benchmark (they could be added as a
3rd class later). See `reviewnlp/data/preprocess.py` for the implementation
and `DESIGN.md` for the rationale.
