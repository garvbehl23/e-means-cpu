"""Correctness tests for the official CPU e-means (``NumpyEmeans``)."""
import numpy as np
import pytest

from datasets import make_blobs
from e_means import EMeansCPU
from evaluation import assign_l2, final_mse


def _separated_blobs(n=4000, d=8, k=5, sep=12.0, noise=0.5, seed=0):
    x, centers = make_blobs(n, d, k, seed=seed, sep=sep, noise=noise)
    return x.astype("float32"), centers.astype("float32")


def test_recovers_well_separated_clusters():
    x, centers = _separated_blobs()
    model = EMeansCPU(k=5, nepochs=10, seed=1).fit(x)
    assert np.isfinite(model.centroids_).all()
    # every true center is matched by a distinct learned centroid
    dist = ((centers[:, None, :] - model.centroids_[None, :, :]) ** 2).sum(-1)
    matched = dist.argmin(1)
    assert len(set(matched.tolist())) == len(centers)
    assert dist.min(1).max() < 1.0
    # near-optimal objective (within 5% of the generating centers)
    assert final_mse(x, model.centroids_) <= 1.05 * final_mse(x, centers)


def test_small_2d_dataset():
    x, centers = _separated_blobs(n=3000, d=2, k=4, sep=20.0, noise=0.5, seed=3)
    model = EMeansCPU(k=4, nepochs=15, seed=3).fit(x)
    assert np.isfinite(model.centroids_).all()
    assert len(np.unique(model.labels_)) == 4
    assert final_mse(x, model.centroids_) <= 1.1 * final_mse(x, centers)


def test_reproducible_with_fixed_seed():
    x, _ = _separated_blobs(n=2000, d=16, k=8, seed=5)
    a = EMeansCPU(k=8, nepochs=4, seed=42).fit(x)
    b = EMeansCPU(k=8, nepochs=4, seed=42).fit(x)
    assert np.array_equal(a.centroids_, b.centroids_)
    c = EMeansCPU(k=8, nepochs=4, seed=43).fit(x)
    assert not np.array_equal(a.centroids_, c.centroids_)


def test_explicit_init_is_respected():
    x, _ = _separated_blobs(n=2000, d=8, k=6, seed=7)
    init = x[np.random.RandomState(0).permutation(len(x))[:6]].copy()
    model = EMeansCPU(k=6, nepochs=2, seed=0).fit(x, init_centroids=init)
    # after only 2 epochs from a given init, centroids must move but stay finite
    assert np.isfinite(model.centroids_).all()
    assert not np.array_equal(model.centroids_, init)


def test_objective_matches_shared_evaluator():
    x, _ = _separated_blobs(n=3000, d=12, k=10, seed=11)
    model = EMeansCPU(k=10, nepochs=5, seed=0).fit(x)
    # e-means.assign returns squared L2 distances; must agree with the shared routine
    d_em, _ = model.model_.assign(x)
    shared = final_mse(x, model.centroids_)
    assert np.allclose(d_em.mean(), shared, rtol=1e-4)


def test_dead_cluster_revival_produces_no_nan():
    # k far larger than the number of real clusters -> many dead clusters
    x, _ = make_blobs(3000, 8, 3, seed=0, sep=10.0, noise=0.5)
    model = EMeansCPU(k=40, nepochs=5, seed=0).fit(x.astype("float32"))
    assert np.isfinite(model.centroids_).all()
    assert model.n_splits_ >= 0
    _, labels = assign_l2(x.astype("float32"), model.centroids_)
    assert len(np.unique(labels)) <= 40


def test_duplicate_points_no_nan():
    rng = np.random.RandomState(0)
    x = np.repeat(rng.randn(5, 6).astype("float32"), 200, axis=0)  # heavy ties
    model = EMeansCPU(k=10, nepochs=3, seed=0).fit(x)
    assert np.isfinite(model.centroids_).all()


def test_ablation_variants_run():
    x, _ = _separated_blobs(n=3000, d=16, k=8, seed=2)
    for variant in ("full", "no_cosine", "no_revival", "no_ema"):
        model = EMeansCPU(k=8, nepochs=3, seed=0, variant=variant).fit(x)
        assert np.isfinite(model.centroids_).all()
        assert final_mse(x, model.centroids_) > 0
