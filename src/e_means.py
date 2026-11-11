"""Thin, benchmark-friendly wrapper around the *official* CPU e-means.

The official CPU implementation is ``faiss.contrib.e_means.NumpyEmeans``
(shipped in faiss >= 1.16.0, CPU path via ``faiss.knn``). We do not
re-implement the algorithm: this module only exposes a uniform ``fit``
interface and a few documented ablations.

Ablations
---------
``variant="full"``          official defaults (cosine schedule + revival).
``variant="no_cosine"``     the step is held constant at ``alpha0`` (the
                            schedule of Equation 5 is disabled by patching
                            ``_scheduled_alpha``).
``variant="no_revival"``    dead-cluster revival disabled
                            (``unused_threshold=0``).
``variant="no_ema"``        the EMA is replaced by a hard reset to each
                            batch's mean (constant step alpha=1). This is no
                            longer the e-means algorithm; it is reported as an
                            approximate "EMA off" reference.

See ``docs/paper_notes.md`` for the mapping between these switches and the
paper's equations.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means import base as _emeans_base


@contextmanager
def _patched_schedule(fn):
    original = _emeans_base._scheduled_alpha
    _emeans_base._scheduled_alpha = fn
    try:
        yield
    finally:
        _emeans_base._scheduled_alpha = original


class EMeansCPU:
    """Official ``NumpyEmeans`` with a scikit-learn-like interface."""

    name = "e-means (official, faiss CPU)"

    def __init__(
        self,
        k,
        nepochs=5,
        seed=0,
        batch_law="auto",
        batch_size=None,
        alpha0=None,
        unused_threshold=None,
        variant="full",
        verbose=False,
    ):
        self.k = int(k)
        self.nepochs = nepochs
        self.seed = int(seed)
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.unused_threshold = unused_threshold
        self.variant = variant
        self.verbose = verbose

    def _schedule_patch(self):
        if self.variant == "no_cosine":
            return lambda alpha0, step, total: float(alpha0)
        if self.variant == "no_ema":
            return lambda alpha0, step, total: 1.0
        return None

    def fit(self, x, init_centroids=None):
        x = np.ascontiguousarray(x, dtype="float32")
        d = x.shape[1]
        unused_threshold = self.unused_threshold
        if self.variant == "no_revival":
            unused_threshold = 0.0
        model = NumpyEmeans(
            d,
            self.k,
            nepochs=self.nepochs,
            seed=self.seed,
            batch_law=self.batch_law,
            batch_size=self.batch_size,
            alpha0=self.alpha0,
            unused_threshold=unused_threshold,
            verbose=self.verbose,
        )
        patch = self._schedule_patch()
        t0 = time.perf_counter()
        if patch is None:
            centroids, labels = model.train(x, init_centroids=init_centroids)
        else:
            with _patched_schedule(patch):
                centroids, labels = model.train(x, init_centroids=init_centroids)
        self.fit_time_ = time.perf_counter() - t0

        self.model_ = model
        self.centroids_ = np.ascontiguousarray(centroids, dtype="float32")
        self.labels_ = labels
        self.recipe_ = model.recipe
        self.n_splits_ = int(model.n_splits)
        self.iteration_stats_ = list(model.iteration_stats)
        self.n_iter_ = int(model.recipe["total_steps"])
        self.n_epochs_ = self.nepochs
        return self
