"""Utility helpers: seeding and config loading."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed every RNG we touch, and keep CUDA deterministic when present."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    """Load a YAML config file into a plain dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg
