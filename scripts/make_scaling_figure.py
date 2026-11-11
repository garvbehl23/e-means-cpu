#!/usr/bin/env python
"""Consolidate all benchmark cells into one table and a scaling figure.

Computes, per cell, the speedup of e-means over Faiss-Lloyd to reach Lloyd's
own converged objective, and how many epochs each needed. Writes
``results/processed/benchmark_table.csv`` and two figures.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import plotting as P  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "results", "raw")
PROC = os.path.join(REPO_ROOT, "results", "processed")

CELLS = [
    "blobs_n10000_d128_k64",
    "blobs_n100000_d128_k64",
    "blobs_n100000_d384_k128",
    "blobs_n100000_d128_k1024",
    "arxiv_n50000_d768_k128",
]


def cell_metrics(df, method="e-means", ref="lloyd-faiss"):
    med = df.groupby(["method", "budget"]).agg(mse=("train_mse", "median"),
                                               t=("fit_time_s", "median")).reset_index()
    conv = med[med["budget"] == med["budget"].max()].set_index("method")["mse"]
    ref_mse = float(conv[ref])
    ref_t = float(med[(med["method"] == ref) & (med["budget"] == med["budget"].max())]["t"].iloc[0])
    g = med[med["method"] == method]
    hit = g[g["mse"] <= ref_mse].sort_values("budget")
    em_ep = int(hit.iloc[0]["budget"]) if len(hit) else np.nan
    em_t = float(hit.iloc[0]["t"]) if len(hit) else np.nan
    return dict(
        em_converged=round(float(conv[method]), 4),
        lloyd_converged=round(ref_mse, 4),
        em_best=round(float(g["mse"].min()), 4),
        epochs_to_lloyd=em_ep,
        time_to_lloyd_s=round(em_t, 4) if em_t == em_t else np.nan,
        lloyd_time_s=round(ref_t, 4),
        speedup=round(ref_t / em_t, 3) if em_t == em_t else np.nan,
        em_beats_lloyd_quality=bool(conv[method] < ref_mse),
    )


def main():
    rows = []
    for cell in CELLS:
        path = os.path.join(RAW, f"{cell}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        m = cell_metrics(df)
        m.update(cell=cell, dataset=df["dataset"].iloc[0], n=int(df["n"].iloc[0]),
                 d=int(df["d"].iloc[0]), k=int(df["k"].iloc[0]))
        rows.append(m)
    table = pd.DataFrame(rows)
    os.makedirs(PROC, exist_ok=True)
    table.to_csv(os.path.join(PROC, "benchmark_table.csv"), index=False)
    print(table.to_string(index=False))

    syn = table[table["dataset"] == "blobs"]
    if len(syn) >= 2:
        P.plot_scaling_curve(
            syn["k"].tolist(), syn["speedup"].tolist(),
            [f"n={n} d={d}" for n, d in zip(syn["n"], syn["d"])],
            "number of clusters k", "speedup to reach Lloyd's converged loss",
            "e-means CPU speedup vs k (synthetic)", os.path.join(P.FIG_DIR, "scaling_speedup_vs_k.png"),
            logx=True)
    real = table[table["dataset"] == "arxiv"]
    if len(real):
        P.plot_scaling_curve(
            table["n"].tolist(), table["speedup"].tolist(),
            [f"{c}" for c in table["cell"]],
            "training samples n", "speedup to reach Lloyd's converged loss",
            "e-means CPU speedup across cells", os.path.join(P.FIG_DIR, "scaling_speedup_overview.png"),
            logx=True)
    return table


if __name__ == "__main__":
    main()
