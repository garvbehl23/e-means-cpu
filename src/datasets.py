"""Dataset generation and loading for the e-means CPU benchmark.

Two families of datasets are provided:

* Synthetic Gaussian mixtures (``make_blobs``) with controllable sample count,
  dimension and separation. These let us sweep (n, d, k) cheaply and with a
  known ground-truth structure.
* A real, publicly available embedding dataset: arXiv paper titles embedded
  with the ``instructor-xl`` sentence-transformer model, 768 dimensions, 50,000
  vectors. Provenance:
      Qdrant/arxiv-titles-instructorxl-embeddings (HuggingFace)
      https://huggingface.co/datasets/Qdrant/arxiv-titles-instructorxl-embeddings
  It is the same modality/dimension as the arXiv dataset used by the paper
  (Appendix A of arXiv:2611.09501). We cache it locally as a float32 ``.npy``.
"""
from __future__ import annotations

import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ARXIV_PARQUET = os.path.join(DATA_DIR, "arxiv_instructorxl_shard0.parquet")
ARXIV_NPY = os.path.join(DATA_DIR, "arxiv_instructorxl_768.npy")
ARXIV_URL = (
    "https://huggingface.co/api/datasets/"
    "Qdrant/arxiv-titles-instructorxl-embeddings/parquet/default/train/0.parquet"
)


def make_blobs(n, d, k, seed=0, sep=1.0, noise=1.0, dtype="float32"):
    """Sample ``n`` points from ``k`` isotropic Gaussian clusters in ``d`` dims.

    Cluster centers are drawn from N(0, sep^2 I) and points from
    N(center, noise^2 I). ``sep``/``noise`` control how overlapped the clusters
    are; at high dimension distances concentrate, so even moderate ``sep``
    yields non-trivial clustering problems similar to real embeddings.
    """
    rng = np.random.RandomState(seed)
    centers = (rng.randn(k, d) * sep).astype(dtype)
    idx = rng.randint(0, k, size=n)
    x = centers[idx] + (rng.randn(n, d) * noise).astype(dtype)
    return x.astype(dtype), centers


def load_arxiv_768(dtype="float32"):
    """Return the cached 768-d arXiv instructor-xl embeddings, shape (50000, 768)."""
    if os.path.exists(ARXIV_NPY):
        return np.load(ARXIV_NPY).astype(dtype, copy=False)
    if not os.path.exists(ARXIV_PARQUET):
        _download_arxiv_parquet()
    return _parquet_to_npy(ARXIV_PARQUET, ARXIV_NPY).astype(dtype, copy=False)


def _download_arxiv_parquet():
    import urllib.request

    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = ARXIV_PARQUET + ".part"
    urllib.request.urlretrieve(ARXIV_URL, tmp)
    os.replace(tmp, ARXIV_PARQUET)


def _parquet_to_npy(parquet_path, npy_path):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["vector"])
    column = table.column("vector").combine_chunks()
    flat = column.values.to_numpy(zero_copy_only=False).astype("float32")
    x = flat.reshape(len(column), -1)
    np.save(npy_path, x)
    return x


def subsample(x, n, seed=0):
    """Deterministic contiguous-free subsample of ``n`` rows from ``x``."""
    if n >= len(x):
        return x
    idx = np.random.RandomState(seed).choice(len(x), size=n, replace=False)
    idx.sort()
    return x[idx]


def split_train_test(x, n_test=20000, seed=0):
    """Split off a held-out test set, following the paper's holdout protocol."""
    n = len(x)
    if n_test >= n:
        raise ValueError("n_test must be smaller than the dataset")
    idx = np.random.RandomState(seed).permutation(n)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], x[test_idx]


def get_dataset(name, n, d, k, seed=0):
    """Return ``(X_train, X_test, centers_or_None)`` for a named dataset.

    ``name`` is one of ``"blobs"`` (synthetic, uses ``d`` and ``k``) or
    ``"arxiv"`` (real, 768-d; ``d`` selects a prefix of the embedding
    dimensions and ``k`` is ignored here).
    """
    if name == "blobs":
        x, centers = make_blobs(n, d, k, seed=seed, sep=1.0, noise=1.0)
        n_test = min(20000, max(1000, n // 5))
        train, test = split_train_test(x, n_test=n_test, seed=seed + 99)
        return train, test, centers
    if name == "arxiv":
        x = load_arxiv_768()
        if d is not None and d < x.shape[1]:
            x = x[:, :d]
        x = subsample(x, n, seed=seed)
        n_test = min(20000, max(1000, len(x) // 5))
        train, test = split_train_test(x, n_test=n_test, seed=seed + 99)
        return train, test, None
    raise ValueError(f"unknown dataset: {name!r}")
