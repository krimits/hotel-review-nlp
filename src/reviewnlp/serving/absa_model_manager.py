"""Model manager for ABSA inference - loads the model once, serves many requests."""

from __future__ import annotations

import os
import threading
from typing import Any

from reviewnlp.absa.pipeline import generate_aspect_records, load_absa_model

DEFAULT_BATCH_SIZE = 8


class AbsaModelManager:
    """Holds one loaded ABSA model and runs batched generation against it.

    The load happens on first use rather than at import, and is guarded so
    concurrent first requests load once between them. Every request after that
    reuses the same weights, which is the point: reloading Qwen2.5-0.5B per
    request would dominate latency entirely.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        self.adapter_dir = os.getenv("ABSA_ADAPTER_DIR")
        self.device = os.getenv("ABSA_DEVICE")
        self.batch_size = batch_size or int(os.getenv("ABSA_BATCH_SIZE", DEFAULT_BATCH_SIZE))
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._loaded = False
        self.tokenizer: Any = None
        self.model: Any = None
        self.variant: str = "base"

    def load(self) -> None:
        """Load the model if it is not loaded yet. Safe to call concurrently."""
        if self._loaded:  # fast path: no lock once warm
            return
        with self._lock:
            if self._loaded:
                return
            self.tokenizer, self.model, self.variant = load_absa_model(
                adapter_dir=self.adapter_dir,
                device=self.device,
            )
            self._loaded = True

    def analyze(self, text: str) -> dict:
        """Analyze one review.

        Returns:
            The record generate_aspect_records produces: the parse result plus
            text, raw_generation and generation_hit_token_budget.
        """
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str], batch_size: int | None = None) -> list[dict]:
        """Analyze several reviews, generating them in batches.

        This is one padded forward pass per batch, not a loop of single
        generations - which is what makes the batch endpoint worth calling.

        Returns:
            One record per input text, in the order given.
        """
        self.load()
        with self._inference_lock:
            return generate_aspect_records(
                self.tokenizer,
                self.model,
                texts,
                batch_size=batch_size or self.batch_size,
            )
