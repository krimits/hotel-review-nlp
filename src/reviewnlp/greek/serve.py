"""Load a saved GreekBERT checkpoint once for Greek sentiment inference."""

from __future__ import annotations

import os
import threading

from reviewnlp.greek.data import LABEL_NAMES, clean_text


class GreekModelManager:
    def __init__(self, model_path: str | None = None, max_length: int = 128):
        self.model_path = model_path or os.getenv("GREEK_MODEL_PATH", "")
        self.max_length = max_length
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not self.model_path:
                raise FileNotFoundError("Set GREEK_MODEL_PATH to a trained GreekBERT checkpoint")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            ).to(self.device).eval()
            if self.model.config.num_labels != len(LABEL_NAMES):
                raise ValueError("Greek checkpoint must have exactly two sentiment labels")
            if self.model.config.id2label != dict(enumerate(LABEL_NAMES)):
                raise ValueError("Greek checkpoint label mapping is not negative=0, positive=1")
            self._loaded = True

    def predict_batch(self, texts: list[str]) -> list[dict]:
        self.load()
        import torch

        with self._lock, torch.inference_mode():
            encoded = self.tokenizer(
                [clean_text(text) for text in texts], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            logits = self.model(**{key: value.to(self.device)
                                   for key, value in encoded.items()}).logits
            confidence, labels = torch.softmax(logits, dim=-1).max(dim=-1)
        return [{"label": LABEL_NAMES[int(label)], "confidence": float(probability)}
                for label, probability in zip(labels.cpu(), confidence.cpu(), strict=True)]


_manager = GreekModelManager()


def get_greek_model() -> GreekModelManager:
    return _manager
