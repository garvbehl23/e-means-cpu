#!/usr/bin/env python
"""Consolidate benchmark cells into one table and the two scaling figures.

Reads the per-cell raw CSVs, recomputes the per-cell speedup of e-means over
Faiss-Lloyd to reach Lloyd's own converged objective, writes
``results/processed/benchmark_table.csv`` and renders:

  * ``figures/scaling_speedup_vs_k.png``      (speedup vs cluster count)
  * ``figures/scaling_speedup_overview.png``  (speedup per configuration)

The script re-reads the CSV it just wrote and independently recomputes every
value, so a mismatch between the figures and the committed CSV is a hard error.
No benchmark data is modified here: this only reads ``results/raw``.
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


def build_table():
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
    return pd.DataFrame(rows)


def validate(table, csv_path):
    """Recompute every value from the raw CSVs and compare to the written CSV."""
    written = pd.read_csv(csv_path)
    fresh = build_table()
    assert len(written) == len(fresh) == len(CELLS), "row count mismatch"
    assert list(written["cell"]) == list(fresh["cell"]), "cell order mismatch"
    for col in ["em_converged", "lloyd_converged", "em_best", "speedup",
                "time_to_lloyd_s", "lloyd_time_s", "epochs_to_lloyd"]:
        a = written[col].to_numpy(dtype="float64")
        b = fresh[col].to_numpy(dtype="float64")
        if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
            raise AssertionError(f"column {col!r} does not match the raw CSVs: {a} vs {b}")
    # plotted arrays must equal the CSV columns exactly
    assert table["speedup"].tolist() == written["speedup"].tolist()
    assert table["k"].tolist() == written["k"].tolist()
    print("validation: benchmark_table.csv matches the raw CSVs (values unchanged)")


def main():
    table = build_table()
    os.makedirs(PROC, exist_ok=True)
    csv_path = os.path.join(PROC, "benchmark_table.csv")
    table.to_csv(csv_path, index=False)
    print(table.to_string(index=False))
    validate(table, csv_path)

    P.plot_speedup_vs_k(table)
    P.plot_speedup_overview(table)
    return table


if __name__ == "__main__":
    main()
