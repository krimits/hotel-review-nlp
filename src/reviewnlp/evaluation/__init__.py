"""Evaluation toolkit: metrics, significance tests, unified benchmark, plots."""

from .compare import matches
from .metrics import binary_metrics
from .significance import mcnemar_exact

__all__ = ["binary_metrics", "matches", "mcnemar_exact"]
