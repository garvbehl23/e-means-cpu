"""Common, algorithm-independent evaluation primitives.

Every method in this repository is scored with the *same* exact assignment
routine so that the reported objective is comparable across algorithms. We use
Faiss' ``knn`` primitive, which is the same C++ primitive the official
``NumpyEmeans`` CPU implementation uses for assignment.

Faiss' L2 metric returns *squared* Euclidean distances, so the k-means
objective (mean squared distance to the nearest centroid) is the mean of the
returned distances, with no further squaring.
"""
from __future__ import annotations

import os

import numpy as np

import faiss


def set_threads(n):
    """Pin Faiss (OpenMP) and BLAS/OpenMP-backed libraries to ``n`` threads."""
    n = int(n)
    faiss.omp_set_num_threads(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n)
    except Exception:
        pass
    return n


def assign_l2(x, centroids, chunk=65536):
    """Exact nearest-centroid assignment under squared L2.

    Returns ``(sq_distances, labels)`` as float32 / int64 1-D arrays.
    """
    x = np.ascontiguousarray(x, dtype="float32")
    centroids = np.ascontiguousarray(centroids, dtype="float32")
    dist = np.empty(len(x), dtype="float32")
    labels = np.empty(len(x), dtype="int64")
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))
        d, l = faiss.knn(x[start:end], centroids, 1)
        dist[start:end] = d.ravel()
        labels[start:end] = l.ravel()
    return dist, labels


def final_mse(x, centroids):
    """Mean squared distance from every row of ``x`` to its nearest centroid."""
    dist, _ = assign_l2(x, centroids)
    return float(dist.mean())


def dead_clusters(labels, k, threshold=10):
    """Number of clusters with fewer than ``threshold`` assigned rows."""
    counts = np.bincount(labels, minlength=k)
    return int((counts < threshold).sum())


def imbalance_factor(labels, k):
    """Faiss' imbalance factor: k * sum_j n_j^2 / n^2 (1.0 is perfectly balanced)."""
    n = len(labels)
    counts = np.bincount(labels, minlength=k).astype("float64")
    return float(k * (counts ** 2).sum() / (n * n))


class PeakMemory:
    """Context manager reporting peak resident set size delta in MiB."""

    def __enter__(self):
        self._start = _peak_rss_mib()
        return self

    def __exit__(self, *exc):
        self.peak_mib = max(0.0, _peak_rss_mib() - self._start)


def _peak_rss_mib():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def current_rss_mib():
    """Current resident set size in MiB (Linux /proc)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")
