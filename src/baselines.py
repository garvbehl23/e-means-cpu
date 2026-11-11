"""Baseline clustering methods with a uniform ``fit`` interface.

All baselines accept an optional explicit ``init_centroids`` so that every
method can be started from identical initial centroids, which is the fairest
way to compare convergence. Every baseline reports:

* ``centroids_``   : (k, d) float32
* ``labels_``      : (n,) int64 last-pass labels (where the library provides them)
* ``fit_time_``    : wall-clock seconds for the optimization only
* ``n_iter_``      : iterations actually performed
* ``obj_history_`` : list of (iteration, cumulative_seconds, objective)
"""
from __future__ import annotations

import time

import numpy as np
import faiss
from sklearn.cluster import KMeans as _SKLearnKMeans
from sklearn.cluster import MiniBatchKMeans as _SKLearnMiniBatch

from evaluation import assign_l2


class SklearnKMeans:
    """Lloyd's algorithm via scikit-learn (Cython + OpenMP)."""

    name = "Lloyd k-means (sklearn)"

    def __init__(self, k, n_iter=25, init="random", seed=0, tol=0.0, n_init=1):
        self.k = int(k)
        self.n_iter = int(n_iter)
        self.init = init
        self.seed = int(seed)
        self.tol = tol
        self.n_init = n_init

    def fit(self, x, init_centroids=None):
        x = np.ascontiguousarray(x, dtype="float32")
        init = init_centroids if init_centroids is not None else self.init
        model = _SKLearnKMeans(
            n_clusters=self.k,
            init=init,
            n_init=self.n_init,
            max_iter=self.n_iter,
            tol=self.tol,
            algorithm="lloyd",
            random_state=self.seed,
        )
        t0 = time.perf_counter()
        model.fit(x)
        self.fit_time_ = time.perf_counter() - t0
        self.model_ = model
        self.centroids_ = np.ascontiguousarray(model.cluster_centers_, dtype="float32")
        self.labels_ = model.labels_.astype("int64")
        self.n_iter_ = int(model.n_iter_)
        self.inertia_ = float(model.inertia_)
        self.obj_history_ = None
        return self


class FaissKMeans:
    """Lloyd's algorithm via Faiss (the paper's CPU baseline)."""

    name = "Lloyd k-means (Faiss)"

    def __init__(self, k, n_iter=25, seed=0):
        self.k = int(k)
        self.n_iter = int(n_iter)
        self.seed = int(seed)

    def fit(self, x, init_centroids=None):
        x = np.ascontiguousarray(x, dtype="float32")
        model = faiss.Kmeans(
            x.shape[1], self.k, niter=self.n_iter, nredo=1, seed=self.seed, verbose=False
        )
        t0 = time.perf_counter()
        if init_centroids is not None:
            model.train(x, init_centroids=np.ascontiguousarray(init_centroids, dtype="float32"))
        else:
            model.train(x)
        self.fit_time_ = time.perf_counter() - t0
        self.model_ = model
        self.centroids_ = np.ascontiguousarray(model.centroids, dtype="float32")
        _, self.labels_ = assign_l2(x, self.centroids_)
        obj = list(np.asarray(model.obj, dtype="float64")) if model.obj is not None else []
        self.obj_history_ = obj
        self.n_iter_ = len(obj) if obj else self.n_iter
        return self


class SklearnMiniBatch:
    """Sculley-style mini-batch k-means via scikit-learn.

    The paper disables scikit-learn's early stopping so every run completes its
    full budget (``max_no_improvement=None``, ``tol=0``). We follow that.
    """

    name = "MiniBatchKMeans (sklearn)"

    def __init__(self, k, n_iter=25, batch_size=1024, init="random", seed=0):
        self.k = int(k)
        self.n_iter = int(n_iter)
        self.batch_size = int(batch_size)
        self.init = init
        self.seed = int(seed)

    def fit(self, x, init_centroids=None):
        x = np.ascontiguousarray(x, dtype="float32")
        init = init_centroids if init_centroids is not None else self.init
        model = _SKLearnMiniBatch(
            n_clusters=self.k,
            init=init,
            n_init=1,
            max_iter=self.n_iter,
            batch_size=self.batch_size,
            tol=0.0,
            max_no_improvement=None,
            reassignment_ratio=0.0,
            random_state=self.seed,
        )
        t0 = time.perf_counter()
        model.fit(x)
        self.fit_time_ = time.perf_counter() - t0
        self.model_ = model
        self.centroids_ = np.ascontiguousarray(model.cluster_centers_, dtype="float32")
        self.labels_ = model.labels_.astype("int64")
        self.n_iter_ = int(model.n_iter_)
        self.obj_history_ = None
        return self


class FaissLloydLoop:
    """Lloyd's algorithm written against the Faiss ``knn`` assignment primitive.

    This is not a strawman: assignment uses the same optimized C++ primitive
    (``faiss.knn``) as the official e-means CPU path. It exists to produce a
    clean per-iteration (iteration, seconds, objective) trace, which the
    library wrappers do not expose uniformly. Its final objective is
    cross-checked against ``FaissKMeans`` and ``SklearnKMeans`` in the tests.
    """

    name = "Lloyd k-means (faiss.knn loop)"

    def __init__(self, k, n_iter=25, seed=0):
        self.k = int(k)
        self.n_iter = int(n_iter)
        self.seed = int(seed)

    def fit(self, x, init_centroids=None):
        x = np.ascontiguousarray(x, dtype="float32")
        k, d = self.k, x.shape[1]
        if init_centroids is None:
            idx = np.random.RandomState(self.seed).permutation(len(x))[:k]
            centroids = x[idx].copy()
        else:
            centroids = np.ascontiguousarray(init_centroids, dtype="float32").copy()

        history = []
        t0 = time.perf_counter()
        dist, labels = assign_l2(x, centroids)
        history.append((0, time.perf_counter() - t0, float(dist.mean())))
        for it in range(1, self.n_iter + 1):
            counts = np.bincount(labels, minlength=k)
            sums = np.zeros((k, d), dtype="float64")
            np.add.at(sums, labels, x)
            nonempty = counts > 0
            centroids[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype("float32")
            dist, labels = assign_l2(x, centroids)
            history.append((it, time.perf_counter() - t0, float(dist.mean())))
        self.fit_time_ = time.perf_counter() - t0
        self.centroids_ = np.ascontiguousarray(centroids, dtype="float32")
        self.labels_ = labels
        self.n_iter_ = self.n_iter
        self.obj_history_ = history
        return self
