"""Singleton model manager for ABSA inference - loads model once at startup."""

from __future__ import annotations

import os
import threading
from typing import Any

from reviewnlp.absa.pipeline import load_absa_model
from reviewnlp.absa.extract import format_absa_messages, parse_absa_output


class AbsaModelManager:
    """Thread-safe singleton that loads the ABSA model once and serves inference requests.
    
    This prevents reloading the model on every request, which is critical for production
    latency and throughput.
    """

    def __init__(self) -> None:
        self.adapter_dir = os.getenv("ABSA_ADAPTER_DIR")
        self.device = os.getenv("ABSA_DEVICE")
        self._lock = threading.Lock()
        self._loaded = False
        self.tokenizer: Any = None
        self.model: Any = None
        self.variant: str = "base"

    def load(self) -> None:
        """Load the model if not already loaded. Thread-safe."""
        with self._lock:
            if self._loaded:
                return

            self.tokenizer, self.model, self.variant = load_absa_model(
                adapter_dir=self.adapter_dir,
                device=self.device,
            )
            self._loaded = True

    def analyze(self, text: str) -> dict:
        """Analyze a single review text and return parsed aspects.
        
        Args:
            text: The review text to analyze
            
        Returns:
            Dictionary with aspects list and metadata (json_valid, salvaged, etc.)
        """
        self.load()

        normalized_text = " ".join(str(text).split())[:4000]
        messages = format_absa_messages(normalized_text)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        encoded = {
            key: value.to(self.model.device)
            for key, value in encoded.items()
        }

        generated = self.model.generate(
            **encoded,
            max_new_tokens=320,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_length = encoded["input_ids"].shape[1]
        generated_tokens = generated[:, input_length:]
        raw_output = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        )

        return parse_absa_output(raw_output, review=normalized_text)

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze multiple review texts in a batch.
        
        Args:
            texts: List of review texts to analyze
            
        Returns:
            List of dictionaries with aspects and metadata for each review
        """
        return [self.analyze(text) for text in texts]
