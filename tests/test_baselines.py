"""Tests that the baseline implementations agree with each other and sklearn."""
import numpy as np

from datasets import make_blobs
from baselines import FaissKMeans, FaissLloydLoop, SklearnKMeans
from evaluation import final_mse


def _data(n=4000, d=10, k=6, seed=0):
    x, _ = make_blobs(n, d, k, seed=seed, sep=8.0, noise=0.7)
    return x.astype("float32")


def test_lloyd_implementations_agree():
    x = _data()
    k = 6
    init = x[np.random.RandomState(0).permutation(len(x))[:k]].copy()
    a = FaissKMeans(k, n_iter=25, seed=0).fit(x, init_centroids=init)
    b = SklearnKMeans(k, n_iter=25, seed=0, init="random").fit(x, init_centroids=init)
    c = FaissLloydLoop(k, n_iter=25, seed=0).fit(x, init_centroids=init)
    mse_a = final_mse(x, a.centroids_)
    mse_b = final_mse(x, b.centroids_)
    mse_c = final_mse(x, c.centroids_)
    # All three Lloyd variants converge to the same objective up to the small
    # differences caused by their distinct empty-cluster handling policies.
    assert abs(mse_a - mse_c) / mse_a < 1e-2
    assert abs(mse_b - mse_c) / mse_b < 1e-2


def test_faiss_loop_objective_is_monotone_non_increasing():
    x = _data(n=3000, d=8, k=8, seed=1)
    model = FaissLloydLoop(8, n_iter=10, seed=0).fit(x)
    objs = [o for (_, _, o) in model.obj_history_]
    diffs = np.diff(objs)
    assert (diffs <= 1e-6).all()


def test_faiss_loop_history_shape():
    x = _data(n=2000, d=8, k=5, seed=2)
    model = FaissLloydLoop(5, n_iter=7, seed=0).fit(x)
    assert len(model.obj_history_) == 8  # initial + n_iter
    times = [t for (_, t, _) in model.obj_history_]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))
