"""Classical ML baselines: TF-IDF features + MultinomialNB / LR-SGD.

Two feature views are compared (word 1-2 grams and char 3-5 grams) so the
benchmark table also shows *why* char n-grams are strong on informal review
text (typos, ALL CAPS, elongated words like "gooooood").
"""

from __future__ import annotations

import argparse
import json
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from reviewnlp.data.preprocess import load_processed
from reviewnlp.evaluation.metrics import binary_metrics
from reviewnlp.utils.seed import load_config, set_seed

_MODELS = {"MultinomialNB": MultinomialNB, "SGDClassifier": SGDClassifier}


def _feature_views(cfg: dict) -> dict[str, object]:
    w = cfg["baselines"]["tfidf_word"]
    c = cfg["baselines"]["tfidf_char"]
    return {
        "word": TfidfVectorizer(
            max_features=w["max_features"],
            ngram_range=tuple(w["ngram_range"]),
            min_df=w["min_df"],
            sublinear_tf=w["sublinear_tf"],
        ),
        "char": TfidfVectorizer(
            max_features=c["max_features"],
            ngram_range=tuple(c["ngram_range"]),
            min_df=c["min_df"],
            sublinear_tf=c["sublinear_tf"],
        ),
    }


def train_classical_baselines(config_path: str, df_override: dict | None = None) -> dict:
    """Train NB and LR-SGD on both feature views; return a metrics dict.

    ``df_override`` lets tests inject a tiny DataFrame instead of the
    full processed dataset.
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    data = df_override or load_processed(cfg["data"]["processed_dir"])
    train_df, _dev_df, test_df = data["train"], data["dev"], data["test"]
    views = _feature_views(cfg)

    results: dict = {}
    for view_name, vectorizer in views.items():
        X_train = vectorizer.fit_transform(train_df["text"])
        X_test = vectorizer.transform(test_df["text"])
        y_train, y_test = train_df["label"].values, test_df["label"].values

        for model_cfg in cfg["baselines"]["models"]:
            name = f"{model_cfg['name']}_{view_name}"
            model = _MODELS[model_cfg["cls"]](**model_cfg["params"])
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_s = time.perf_counter() - t0

            preds = model.predict(X_test)
            metrics = binary_metrics(y_test, preds)
            metrics["fit_seconds"] = round(fit_s, 2)
            results[name] = metrics
            print(f"[{name:>22}] macro-F1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f}")

    # Persist the best-per-view pipelines for the unified benchmark/serving.
    out_dir = cfg["output"]["model_dir"]
    os.makedirs(out_dir, exist_ok=True)

    import joblib

    best = (-1.0, None, None)
    for view_name, vectorizer in views.items():
        X_train = vectorizer.fit_transform(train_df["text"])
        y_train = train_df["label"].values
        for model_cfg in cfg["baselines"]["models"]:
            name = f"{model_cfg['name']}_{view_name}"
            if results[name]["macro_f1"] > best[0]:
                model = _MODELS[model_cfg["cls"]](**model_cfg["params"]).fit(X_train, y_train)
                best = (results[name]["macro_f1"], name, (vectorizer, model))
    f1, name, (vectorizer, model) = best
    pipe = Pipeline([("tfidf", vectorizer), ("clf", model)])
    joblib.dump(pipe, os.path.join(out_dir, "best_classical.joblib"))

    results["_best"] = {"name": name, "macro_f1": f1, "path": os.path.join(out_dir, "best_classical.joblib")}
    with open(cfg["output"]["metrics_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"Best classical: {name} (macro-F1={f1:.4f}) -> {out_dir}/best_classical.joblib")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    train_classical_baselines(args.config)


if __name__ == "__main__":
    main()
