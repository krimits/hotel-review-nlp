"""TF-IDF + logistic regression baseline for the Greek splits.

The `baseline:` section of configs/greek_bert.yaml described this model in
full — analyzer limits, regularisation, class weighting — but nothing read
it, so GreekBERT had no floor to clear. A fine-tune that beats nothing is not
evidence. This trains on the same frames the encoder sees, selects on the
validation split and reports test metrics in the shared schema, so the two
numbers sit in one table.

Character n-grams are deliberately not used here: the corpus has known
mojibake (lone surrogates from an upstream encoding fault), which inflates
character-level features with artefacts of the corruption rather than signal.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.greek.config import BaselineConfig
from reviewnlp.greek.data import LABEL_NAMES


def build_pipeline(config: BaselineConfig) -> Pipeline:
    """Word n-gram TF-IDF into a linear classifier, exactly as configured."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.max_features,
                    ngram_range=tuple(config.ngram_range),
                    min_df=config.min_df,
                    max_df=config.max_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    random_state=config.seed,
                ),
            ),
        ]
    )


def train_baseline(frames: dict[str, pd.DataFrame], config: BaselineConfig) -> dict:
    """Fit on train, report on validation and test. Returns a metrics dict."""
    pipeline = build_pipeline(config)
    pipeline.fit(frames["train"]["text"], frames["train"]["label"])

    scores = {}
    for split in ("validation", "test"):
        frame = frames[split]
        scores[split] = binary_metrics(
            frame["label"].to_numpy(),
            pipeline.predict(frame["text"]),
            label_names=LABEL_NAMES,
            label_values=tuple(range(len(LABEL_NAMES))),
        )

    return {
        "model_family": "TF-IDF word n-grams + logistic regression (Greek baseline)",
        "settings": {
            "max_features": config.max_features,
            "ngram_range": list(config.ngram_range),
            "min_df": config.min_df,
            "max_df": config.max_df,
            "C": config.C,
            "max_iter": config.max_iter,
            "class_weight": config.class_weight,
            "seed": config.seed,
        },
        "validation": scores["validation"],
        "test": scores["test"],
        "vocabulary_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
    }
