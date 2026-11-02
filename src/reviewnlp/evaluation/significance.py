"""McNemar's exact test and the cluster bootstrap.

McNemar: on the same test set, model A and model B agree on most examples. The only
informative rows are the discordant ones:

    n01 = A wrong, B right
    n10 = A right, B wrong

Under H0 (both models equally good), n01 ~ Binomial(n01 + n10, 0.5). We use
the exact binomial version (valid even when n01 + n10 < 25, unlike the
chi-square approximation) - this is the standard recommendation from
Dietterich (1998), "Approximate statistical tests for comparing supervised
classification learning algorithms".

Cluster bootstrap: `reviewnlp.evaluation.metrics.bootstrap_ci` resamples *rows*
(reviews) independently. Rows are not independent when reviews share a hotel - the
unseen-hotels split holds about ninety reviews per hotel, and reviews of one hotel share
its staff, building and clientele - so the row bootstrap understates the sampling
variability of a score on that split. `cluster_resample_statistics`,
`cluster_bootstrap_ci` and `cluster_bootstrap_paired_ci` resample whole clusters
(hotels) instead, so the interval describes drawing a different set of hotels, which is
the quantity a claim about unseen hotels is about.

Design-effect assumption: clusters are independent and exchangeable, and rows inside a
cluster may depend on each other arbitrarily. The interval's width *is* the design
effect; it is not estimated separately. Two consequences are worth stating with any
number taken from here:

- the interval rests on the number of clusters, not the number of rows: with k clusters
  each tail is pinned by about k/2 distinct clusters, so a small k gives an unstable
  interval even when the row count is in the thousands;
- resampling pools whole clusters, so the statistic is computed on the pooled rows and a
  large cluster contributes proportionally more rows than a small one. That is the
  row-level estimand under cluster sampling, not an average of per-cluster scores.

The interval is the percentile interval, not BCa: no bias correction and no acceleration,
which is the same choice `metrics.bootstrap_ci` makes, so a clustered and an iid interval
computed here are directly comparable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest

Statistic = Callable[[np.ndarray], float]


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


def _cluster_codes(clusters: Sequence[Any]) -> tuple[np.ndarray, int]:
    """Integer code per row, in first-appearance order, and the cluster count."""
    order: dict[Any, int] = {}
    codes = np.empty(len(clusters), dtype=np.intp)
    for position, cluster in enumerate(clusters):
        try:
            code = order.setdefault(cluster, len(order))
        except TypeError as error:
            raise TypeError(f"cluster labels must be hashable, got {cluster!r}") from error
        codes[position] = code
    return codes, len(order)


def _row_groups(codes: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """Row positions of each cluster, grouped in code order."""
    ordered = np.argsort(codes, kind="stable")
    edges = np.searchsorted(codes[ordered], np.arange(n_clusters + 1))
    return [ordered[edges[g] : edges[g + 1]] for g in range(n_clusters)]


def _resample_rows(groups: list[np.ndarray], n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """One whole-cluster resample: draw clusters with replacement, keep all their rows.

    With one row per cluster in first-appearance order, group g holds row g, so the pooled
    rows are exactly the drawn row indices - the degenerate case in the tests.
    """
    draws = rng.integers(0, n_clusters, size=n_clusters)
    if all(len(group) == 1 for group in groups):
        return draws.astype(np.intp, copy=False)
    return np.concatenate([groups[draw] for draw in draws])


def _validate(values: np.ndarray, clusters: Sequence[Any], n_resamples: int) -> tuple[list[Any], int]:
    if values.ndim == 0:
        raise ValueError("values must hold one entry (row) per observation")
    n = int(values.shape[0])
    clusters = list(clusters)
    if n == 0:
        raise ValueError("need at least one observation")
    if len(clusters) != n:
        raise ValueError(f"length mismatch: {n} values vs {len(clusters)} clusters")
    if n_resamples < 1:
        raise ValueError(f"need at least one resample, got {n_resamples}")
    return clusters, n


def cluster_resample_statistics(
    values,
    clusters: Sequence[Any],
    statistic: Statistic = np.mean,
    n_resamples: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """The statistic on each of `n_resamples` whole-cluster resamples of `values`.

    Parameters
    ----------
    values
        One row per observation. A 1-D array is the usual case for a mean; pass a 2-D
        array (for example stacked label and prediction codes) with a custom `statistic`
        when the quantity is not a plain mean, such as macro-F1.
    clusters
        One hashable cluster label per row (a hotel name or id).
    statistic
        Callable on the resampled rows, returning one number per resample.
    n_resamples, seed
        Fixed seed, so the result is deterministic.

    The design-effect assumption is stated in the module docstring: clusters are
    exchangeable, rows within a cluster may be arbitrarily dependent, and the pooled
    sample is what the statistic sees. One cluster gives a constant distribution.
    """
    values = np.asarray(values)
    clusters, _ = _validate(values, clusters, n_resamples)
    codes, n_clusters = _cluster_codes(clusters)
    groups = _row_groups(codes, n_clusters)
    rng = np.random.default_rng(seed)
    return np.array(
        [float(statistic(values[_resample_rows(groups, n_clusters, rng)])) for _ in range(n_resamples)],
        dtype=float,
    )


def cluster_bootstrap_ci(
    values,
    clusters: Sequence[Any],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    statistic: Statistic = np.mean,
) -> tuple[float, float]:
    """Percentile interval of `statistic` resampling whole clusters (hotels).

    Returns the (alpha/2, 1 - alpha/2) quantiles of `cluster_resample_statistics`, rounded
    to four decimals like the intervals in `metrics`. The width reflects cluster sampling:
    it widens with the within-cluster dependence and, with few clusters, is itself noisy.
    This does not replace a paired test on the same reviews; use
    `cluster_bootstrap_paired_ci` for a difference between two models on the same split.

    The design-effect assumption is stated in the module docstring.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    statistics = cluster_resample_statistics(values, clusters, statistic, n_resamples, seed)
    low, high = np.quantile(statistics, [alpha / 2, 1 - alpha / 2])
    return round(float(low), 4), round(float(high), 4)


def cluster_bootstrap_paired_ci(
    values_a,
    values_b,
    clusters: Sequence[Any],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    statistic: Statistic = np.mean,
) -> tuple[float, float, float]:
    """Paired cluster bootstrap: statistic(b) - statistic(a), same resampled clusters for both.

    The point estimate is `statistic(values_b) - statistic(values_a)` on all rows. Each
    resample draws whole clusters once and scores both models on the same pooled rows, so
    the interval carries the models' correlation on the same reviews *and* the clusters'
    dependence. Resampling the two models independently would overstate the uncertainty of
    the difference, because the same hotel's reviews are judged by both.

    Returns (point, low, high) rounded to four decimals, as `metrics.paired_bootstrap_diff`.
    The design-effect assumption is stated in the module docstring.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    a, b = np.asarray(values_a), np.asarray(values_b)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("values must hold one entry (row) per observation")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {b.shape[0]}")
    clusters, _ = _validate(a, clusters, n_resamples)
    point = float(statistic(b)) - float(statistic(a))
    codes, n_clusters = _cluster_codes(clusters)
    groups = _row_groups(codes, n_clusters)
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        rows = _resample_rows(groups, n_clusters, rng)
        differences[i] = float(statistic(b[rows])) - float(statistic(a[rows]))
    low, high = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    return round(point, 4), round(float(low), 4), round(float(high), 4)
