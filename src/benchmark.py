"""Benchmark harness: budget sweeps, fair timing and metric collection.

Design notes
------------
* A "budget" is expressed in **epochs** (full passes over the training set) for
  every method, so the x-axis is directly comparable:
    - e-means      : ``nepochs``
    - Lloyd        : number of iterations (= passes)
    - MiniBatch    : number of epochs
* All methods are started from **identical initial centroids** (``k`` rows
  sampled from the training set), which is the fairest convergence comparison.
* Timing uses ``time.perf_counter`` and excludes dataset generation. Data
  generation time and initialization time are recorded separately.
* The reported objective is always the exact mean squared distance to the
  nearest centroid, computed by a single shared routine (``evaluation.final_mse``).
* ``time_to_target`` uses the earliest sweep budget whose objective is at or
  below a target (by default, the converged Lloyd objective). If a method
  never reaches the target within the swept budgets it is marked
  ``reached_target=False``; no speedup is fabricated.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from evaluation import (
    PeakMemory,
    assign_l2,
    current_rss_mib,
    dead_clusters,
    final_mse,
    imbalance_factor,
)
from e_means import EMeansCPU
from baselines import FaissKMeans, FaissLloydLoop, SklearnKMeans, SklearnMiniBatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Methods that must use their *own* initialization. k-means++ is defined by its
# seeding, so forcing a shared random init onto it would defeat the baseline.
OWN_INIT_METHODS = {"lloyd-sklearn-pp"}


def make_init(x, k, seed):
    """Sample ``k`` distinct rows of ``x`` as initial centroids (fixed RNG)."""
    idx = np.random.RandomState(seed).permutation(len(x))[:k]
    return np.ascontiguousarray(x[idx], dtype="float32")


def build_method(name, k, budget, seed, mini_batch_size=1024):
    if name == "e-means":
        return EMeansCPU(k, nepochs=budget, seed=seed)
    if name == "e-means-no-cosine":
        return EMeansCPU(k, nepochs=budget, seed=seed, variant="no_cosine")
    if name == "e-means-no-revival":
        return EMeansCPU(k, nepochs=budget, seed=seed, variant="no_revival")
    if name == "e-means-no-ema":
        return EMeansCPU(k, nepochs=budget, seed=seed, variant="no_ema")
    if name == "lloyd-faiss":
        return FaissKMeans(k, n_iter=budget, seed=seed)
    if name == "lloyd-sklearn":
        return SklearnKMeans(k, n_iter=budget, seed=seed, init="random")
    if name == "lloyd-sklearn-pp":
        return SklearnKMeans(k, n_iter=budget, seed=seed, init="k-means++")
    if name == "lloyd-faiss-loop":
        return FaissLloydLoop(k, n_iter=budget, seed=seed)
    if name == "minibatch":
        return SklearnMiniBatch(k, n_iter=budget, batch_size=mini_batch_size, seed=seed)
    raise ValueError(f"unknown method: {name!r}")


def run_one(method_name, x_train, x_test, k, budget, seed, init, mini_batch_size=1024):
    """Run one method at one budget and return a metric record."""
    model = build_method(method_name, k, budget, seed, mini_batch_size)
    init_arg = None if method_name in OWN_INIT_METHODS else init
    with PeakMemory() as mem:
        rss_before = current_rss_mib()
        model.fit(x_train, init_centroids=init_arg)
        rss_after = current_rss_mib()
    centroids = model.centroids_
    train_mse = final_mse(x_train, centroids)
    test_mse = final_mse(x_test, centroids) if x_test is not None else float("nan")
    _, labels = assign_l2(x_train, centroids)
    return {
        "method": method_name,
        "k": int(k),
        "budget": budget,
        "seed": int(seed),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "fit_time_s": float(model.fit_time_),
        "n_iter": int(getattr(model, "n_iter_", 0)),
        "peak_mem_mib": float(mem.peak_mib),
        "rss_delta_mib": float(rss_after - rss_before),
        "dead_clusters": dead_clusters(labels, k),
        "imbalance": imbalance_factor(labels, k),
        "n_splits": int(getattr(model, "n_splits_", 0)),
        "batch_size": int(model.recipe_["batch_size"]) if hasattr(model, "recipe_") else np.nan,
        "alpha0": float(model.recipe_["alpha0"]) if hasattr(model, "recipe_") else np.nan,
    }


def converged_lloyd_mse(x_train, x_test, k, seed, n_iter=50):
    """Reference objective: Lloyd (Faiss) run to convergence on the train set."""
    init = make_init(x_train, k, seed)
    model = FaissKMeans(k, n_iter=n_iter, seed=seed).fit(x_train, init_centroids=init)
    return final_mse(x_train, model.centroids_)


def budget_sweep(
    method_names,
    x_train,
    x_test,
    k,
    budgets,
    seed,
    dataset_meta,
    target_mse=None,
    mini_batch_size=1024,
    init=None,
):
    """Run every method across a list of budgets and return a tidy DataFrame.

    ``target_mse`` defaults to the converged Lloyd objective. ``time_to_target``
    and ``epoch_to_target`` are derived per method from the sweep itself.
    """
    if init is None:
        init = make_init(x_train, k, seed)
    if target_mse is None:
        target_mse = converged_lloyd_mse(x_train, x_test, k, seed)

    rows = []
    for name in method_names:
        for budget in budgets:
            rec = run_one(name, x_train, x_test, k, budget, seed, init, mini_batch_size)
            rec.update(dataset_meta)
            rec["target_mse"] = target_mse
            rows.append(rec)
    df = pd.DataFrame(rows)

    df["reached_target"] = df["train_mse"] <= df["target_mse"]
    df = _add_time_to_target(df)
    return df


def _add_time_to_target(df):
    """For each (method) group, earliest budget that reaches the target."""
    df = df.copy()
    df["time_to_target_s"] = np.nan
    df["epoch_to_target"] = np.nan
    for _, grp in df.groupby("method"):
        hit = grp[grp["reached_target"]].sort_values("budget")
        if len(hit):
            first = hit.iloc[0]
            df.loc[hit.index, ["time_to_target_s", "epoch_to_target"]] = [
                first["fit_time_s"],
                first["budget"],
            ]
    return df


def save_results(df, name):
    os.makedirs(os.path.join(RESULTS_DIR, "raw"), exist_ok=True)
    path = os.path.join(RESULTS_DIR, "raw", f"{name}.csv")
    df.to_csv(path, index=False)
    key = "method" if "method" in df.columns else ("arm" if "arm" in df.columns else None)
    meta = {
        "name": name,
        "rows": int(len(df)),
        "methods": sorted(df[key].unique().tolist()) if (len(df) and key) else [],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(RESULTS_DIR, "raw", f"{name}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return path
