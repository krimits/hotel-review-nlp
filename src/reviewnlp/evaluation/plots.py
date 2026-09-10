"""Benchmark plots: confusion matrices and latency comparison."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe for CI and Colab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


def plot_confusion_matrices(results: dict, out_path: str) -> None:
    """One heatmap per model, normalized per true class."""
    names = [n for n, r in results.items() if "confusion_matrix" in r]
    if not names:
        return
    n = len(names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    labels = ["negative", "positive"]
    for ax, name in zip(axes, names, strict=False):
        cm = np.asarray(results[name]["confusion_matrix"], dtype=float)
        cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
        sn.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False,
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
    for ax in axes[len(names):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_latency(results: dict, out_path: str) -> None:
    """p50/p95 per-text latency bar chart across models."""
    names, p50, p95 = [], [], []
    for name, r in results.items():
        if "latency" not in r:
            continue
        names.append(name)
        p50.append(r["latency"]["p50_ms_per_text"])
        p95.append(r["latency"]["p95_ms_per_text"])
    if not names:
        return

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, p50, width=0.36, label="p50")
    ax.bar(x + 0.18, p95, width=0.36, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("ms per review (batch of 64, sample)")
    ax.set_title("Inference latency per model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")
