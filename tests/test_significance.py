"""Cluster-bootstrap correctness: hand-worked cases, determinism, whole clusters, degenerate case.

The row-level intervals in `reviewnlp.evaluation.metrics` are tested in test_metrics.py.
These tests pin the property that separates the clustered interval from them: whole
clusters are drawn, never split, and the degenerate case (one cluster per row) reduces to
the ordinary row bootstrap.
"""

from __future__ import annotations

import numpy as np
import pytest

from reviewnlp.evaluation.metrics import _macro_f1, bootstrap_ci
from reviewnlp.evaluation.significance import (
    cluster_bootstrap_ci,
    cluster_bootstrap_paired_ci,
    cluster_resample_statistics,
)


def test_hand_worked_two_clusters_of_two():
    # Values 1 and 0, one cluster each. Drawing two clusters with replacement gives (1,1),
    # (1,0), (0,1) or (0,0) with equal chance, so the resampled mean is 1, 0.5 or 0 with
    # probabilities 1/4, 1/2, 1/4. The 2.5th and 97.5th percentiles are therefore 0 and 1.
    values = np.array([1.0, 1.0, 0.0, 0.0])
    clusters = ["a", "a", "b", "b"]
    assert values.mean() == 0.5
    assert cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=0) == (0.0, 1.0)
    outcomes = np.unique(cluster_resample_statistics(values, clusters, n_resamples=2000, seed=0))
    assert list(outcomes) == [0.0, 0.5, 1.0]


def test_unequal_clusters_are_resampled_whole():
    # Cluster "big" holds three zeros, "small" one 10. Drawing two clusters whole can only
    # pool 6 zeros (0), 3 zeros and one 10 (2.5), or two 10s (10). Splitting a cluster
    # would produce other values, so the exact outcome set proves clusters stay whole.
    values = np.array([0.0, 0.0, 0.0, 10.0])
    clusters = ["big", "big", "big", "small"]
    outcomes = np.unique(cluster_resample_statistics(values, clusters, n_resamples=2000, seed=3))
    assert list(outcomes) == [0.0, 2.5, 10.0]
    assert cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=3) == (0.0, 10.0)


def test_same_seed_gives_the_same_distribution_and_interval():
    rng = np.random.default_rng(11)
    values = rng.random(300)
    clusters = rng.integers(0, 7, 300).tolist()
    first = cluster_resample_statistics(values, clusters, n_resamples=200, seed=5)
    second = cluster_resample_statistics(values, clusters, n_resamples=200, seed=5)
    assert np.array_equal(first, second)
    assert cluster_bootstrap_ci(values, clusters, n_resamples=200, seed=5) == cluster_bootstrap_ci(
        values, clusters, n_resamples=200, seed=5
    )
    other = cluster_resample_statistics(values, clusters, n_resamples=200, seed=6)
    assert not np.array_equal(first, other)


def test_degenerate_one_cluster_per_row_equals_the_iid_bootstrap():
    # Every review is its own cluster, so drawing rows and drawing clusters are the same
    # operation. With the same generator protocol the two distributions are identical
    # element by element, not merely close.
    rng = np.random.default_rng(7)
    values = rng.random(200)
    clusters = np.arange(200)
    low, high = cluster_bootstrap_ci(values, clusters, n_resamples=500, seed=7)
    iid_rng = np.random.default_rng(7)
    iid_means = np.array([values[iid_rng.integers(0, 200, 200)].mean() for _ in range(500)])
    iid_low, iid_high = np.quantile(iid_means, [0.025, 0.975])
    assert (low, high) == (round(float(iid_low), 4), round(float(iid_high), 4))


def test_paired_interval_resamples_one_set_of_clusters_for_both_models():
    # Model b is model a plus 5 on every row. Independent draws would make the resampled
    # difference of means vary; shared draws make every replicate difference exactly 5.
    values_a = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    values_b = values_a + 5.0
    clusters = ["h1", "h1", "h2", "h2", "h3", "h3"]
    assert cluster_bootstrap_paired_ci(values_a, values_b, clusters, n_resamples=500, seed=0) == (5.0, 5.0, 5.0)


def test_paired_point_estimate_and_interval_of_a_real_difference():
    rng = np.random.default_rng(3)
    n, n_hotels = 600, 30
    clusters = rng.integers(0, n_hotels, n).tolist()
    truth = rng.integers(0, 2, n)
    difficulty = rng.normal(0.0, 1.0, n_hotels)[np.array(clusters)]  # rows of a hotel share it
    right_a = rng.random(n) + 0.15 * difficulty < 0.90
    right_b = rng.random(n) + 0.15 * difficulty < 0.95
    values_a = (np.where(right_a, truth, 1 - truth) == truth).astype(float)
    values_b = (np.where(right_b, truth, 1 - truth) == truth).astype(float)
    point, low, high = cluster_bootstrap_paired_ci(values_a, values_b, clusters, n_resamples=500, seed=0)
    assert point == pytest.approx(float(values_b.mean() - values_a.mean()), abs=1e-4)
    assert low <= point <= high


def test_macro_f1_statistic_works_on_stacked_codes():
    # The split_views.py use: two columns of codes and the same macro-F1 the iid interval
    # uses. A perfect model scores 1.0 whatever the resampling.
    labels = np.array([0] * 50 + [1] * 50)
    values = np.column_stack([labels, labels])

    def statistic(rows: np.ndarray) -> float:
        return _macro_f1(rows[:, 0], rows[:, 1], (0, 1))

    assert cluster_bootstrap_ci(values, ["h"] * 100, n_resamples=50, seed=0, statistic=statistic) == (1.0, 1.0)
    assert statistic(values) == 1.0


def test_hotel_clustering_widens_the_interval_relative_to_rows():
    # 30 hotels of 20 reviews. A hotel is either fully classified correctly or fully wrong,
    # so rows inside a hotel are perfectly dependent and there are only 30 effective units.
    # The row bootstrap treats the 600 rows as independent and must be too narrow.
    rng = np.random.default_rng(0)
    hotels = np.repeat(np.arange(30), 20).tolist()
    good = rng.random(30) < 0.8
    labels = np.tile([0] * 10 + [1] * 10, 30)
    predicted = np.where(good[np.repeat(np.arange(30), 20)], labels, 1 - labels)

    def statistic(rows: np.ndarray) -> float:
        return _macro_f1(rows[:, 0], rows[:, 1], (0, 1))

    values = np.column_stack([labels, predicted])
    rows_low, rows_high = bootstrap_ci(labels, predicted, labels=(0, 1), n_resamples=1000, seed=0)
    clustered_low, clustered_high = cluster_bootstrap_ci(
        values, hotels, n_resamples=1000, seed=0, statistic=statistic
    )
    # Every good hotel scores 1.0, every bad one 0.0, so the pooled macro-F1 is the share of
    # good hotels and the row-level interval cannot see that only 30 draws drive it.
    assert statistic(values) == pytest.approx(float(good.mean()), abs=1e-9)
    assert clustered_high - clustered_low > rows_high - rows_low
    assert clustered_low < statistic(values) < clustered_high


def test_single_cluster_gives_a_point_interval():
    values = np.array([0.0, 1.0, 2.0])
    assert cluster_bootstrap_ci(values, ["only", "only", "only"], n_resamples=50, seed=0) == (1.0, 1.0)


def test_validation():
    values = np.arange(4.0)
    clusters = ["a", "a", "b", "b"]
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(values, clusters[:3])
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(np.array([]), [])
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(values, clusters, alpha=0.0)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(values, clusters, alpha=1.0)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(values, clusters, n_resamples=0)
    with pytest.raises(ValueError):
        cluster_bootstrap_paired_ci(values, values[:3], clusters)
    with pytest.raises(ValueError):
        cluster_bootstrap_paired_ci(1.0, 1.0, clusters)
    with pytest.raises(TypeError):
        cluster_resample_statistics(values, [["a"], ["a"], ["b"], ["b"]])
