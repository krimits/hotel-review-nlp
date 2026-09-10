from __future__ import annotations

import joblib
import pandas as pd
import yaml

from reviewnlp.baselines.classical import train_classical_baselines


def test_classical_uses_character_features_and_selects_only_on_dev(tmp_path):
    config = {
        "seed": 42,
        "data": {"processed_dir": "unused"},
        "baselines": {
            "tfidf_word": {"max_features": 100, "ngram_range": [1, 2], "min_df": 1, "sublinear_tf": True},
            "tfidf_char": {"max_features": 100, "ngram_range": [3, 5], "min_df": 1, "sublinear_tf": True},
            "models": [
                {"name": "nb", "cls": "MultinomialNB", "params": {"alpha": 0.3}},
                {"name": "lr", "cls": "SGDClassifier", "params": {"loss": "log_loss", "random_state": 42}},
            ],
        },
        "output": {"model_dir": str(tmp_path / "models"), "metrics_path": str(tmp_path / "metrics.json")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    data = {
        split: pd.DataFrame({
            "text": [f"{split} wonderful clean room", f"{split} terrible dirty room", f"{split} great friendly staff", f"{split} awful rude staff"],
            "label": ["positive", "negative", "positive", "negative"],
        })
        for split in ("train", "dev", "test")
    }
    first = train_classical_baselines(str(path), df_override=data)
    assert first["_best"]["selection_split"] == "dev"
    for name in ("nb_char", "lr_char"):
        pipeline = joblib.load(tmp_path / "models" / f"{name}.joblib")
        assert pipeline.named_steps["tfidf"].analyzer == "char"
        assert pipeline.predict(data["test"]["text"]).shape == (4,)
    dev_best = max((n for n in first if not n.startswith("_")), key=lambda n: first[n]["dev"]["macro_f1"])
    assert first["_best"]["name"] == dev_best
    data["test"]["label"] = data["test"]["label"].map({"positive": "negative", "negative": "positive"})
    second = train_classical_baselines(str(path), df_override=data)
    assert second["_best"]["name"] == first["_best"]["name"]
    assert second[dev_best]["macro_f1"] != first[dev_best]["macro_f1"]
