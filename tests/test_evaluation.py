"""Tests for the shared evaluation primitives and the objective convention."""
import numpy as np

from evaluation import assign_l2, dead_clusters, final_mse, imbalance_factor, set_threads


def test_faiss_returns_squared_l2():
    # Faiss L2 returns *squared* distances: verify against a manual computation.
    rng = np.random.RandomState(0)
    x = rng.randn(200, 5).astype("float32")
    c = rng.randn(7, 5).astype("float32")
    dist, labels = assign_l2(x, c)
    manual_sq = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    manual_lab = manual_sq.argmin(1)
    assert np.array_equal(labels, manual_lab)
    assert np.allclose(dist, manual_sq[np.arange(len(x)), manual_lab], rtol=1e-5)


def test_final_mse_matches_manual():
    rng = np.random.RandomState(1)
    x = rng.randn(300, 8).astype("float32")
    c = rng.randn(10, 8).astype("float32")
    manual = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).min(1).mean()
    assert abs(final_mse(x, c) - manual) / manual < 1e-5


def test_dead_and_imbalance():
    labels = np.array([0, 0, 0, 1, 1, 2])
    assert dead_clusters(labels, k=5, threshold=1) == 2
    assert imbalance_factor(labels, k=3) > 1.0
    balanced = np.array([0, 0, 1, 1, 2, 2])
    assert imbalance_factor(balanced, k=3) == 1.0


def test_set_threads_returns_value():
    assert set_threads(2) == 2
