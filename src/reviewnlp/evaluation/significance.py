"""McNemar's exact test - are two models *significantly* different?

On the same test set, model A and model B agree on most examples. The only
informative rows are the discordant ones:

    n01 = A wrong, B right
    n10 = A right, B wrong

Under H0 (both models equally good), n01 ~ Binomial(n01 + n10, 0.5). We use
the exact binomial version (valid even when n01 + n10 < 25, unlike the
chi-square approximation) - this is the standard recommendation from
Dietterich (1998), "Approximate statistical tests for comparing supervised
classification learning algorithms".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(n01: int, n10: int) -> dict:
    """Exact McNemar test on discordant pair counts.

    Returns {'statistic', 'p_value', 'significant_at_0.05', 'n_discordant'}.
    """
    if min(n01, n10) < 0:
        raise ValueError("counts must be non-negative")
    n = n01 + n10
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_0.05": False, "n_discordant": 0}

    result = binomtest(min(n01, n10), n, p=0.5)
    return {
        "statistic": float(min(n01, n10)),
        "p_value": float(result.pvalue),
        "significant_at_0.05": bool(result.pvalue < 0.05),
        "n_discordant": n,
    }


def pairwise_mcnemar(y_true, preds: dict[str, np.ndarray]) -> dict:
    """All-pairs McNemar over {'model_name': predictions}."""
    y_true = np.asarray(y_true)
    names = list(preds)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n01 = int(np.sum((preds[a] != y_true) & (preds[b] == y_true)))
            n10 = int(np.sum((preds[a] == y_true) & (preds[b] != y_true)))
            res = mcnemar_exact(n01, n10)
            out[f"{a} vs {b}"] = res
    return out
