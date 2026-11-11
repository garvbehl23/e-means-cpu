#!/usr/bin/env python
"""Generate the standard figure set and speedup tables for result tags.

Example:
    python scripts/make_figures.py blobs_n10000_d128_k64 arxiv_n50000_d768_k64

Two targets are reported, both computed from medians over seeds:
  * target_lloyd : Lloyd's (Faiss) converged objective (the paper's convention).
  * target_best  : the best converged objective reached by any method (a
                   stricter, quality-matched bar). Methods that never reach it
                   within the swept budgets are reported as "not reached".
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


def _budget_medians(df, ycol="train_mse"):
    return (
        df.groupby(["method", "budget"])
        .agg(mse=(ycol, "median"), t=("fit_time_s", "median"))
        .reset_index()
    )


def _time_to_target(med, target):
    out = {}
    for method, g in med.groupby("method"):
        hit = g[g["mse"] <= target].sort_values("budget")
        if len(hit):
            row = hit.iloc[0]
            out[method] = (int(row["budget"]), float(row["t"]), True)
        else:
            out[method] = (np.nan, np.nan, False)
    return out


def summarize(df, ref_method="lloyd-faiss", budget=None):
    if budget is None:
        budget = int(df["budget"].max())
    if ref_method not in set(df["method"]):
        ref_method = "e-means"
    med = _budget_medians(df)
    converged = med[med["budget"] == budget].set_index("method")["mse"]
    best_per_method = med.groupby("method")["mse"].min()
    target_lloyd = float(converged.get(ref_method, df["target_mse"].iloc[0]))
    target_best = float(best_per_method.min())

    ref_t = med[(med["method"] == ref_method) & (med["budget"] == budget)]["t"]
    ref_time = float(ref_t.iloc[0]) if len(ref_t) else float("nan")

    t_lloyd = _time_to_target(med, target_lloyd)
    t_best = _time_to_target(med, target_best)

    rows = []
    for method in sorted(df["method"].unique()):
        ep_l, tt_l, ok_l = t_lloyd[method]
        ep_b, tt_b, ok_b = t_best[method]
        rows.append(
            {
                "method": method,
                "converged_mse": round(float(converged.get(method, np.nan)), 4),
                "best_mse": round(float(best_per_method.get(method, np.nan)), 4),
                "epochs_to_lloyd": ep_l,
                "time_to_lloyd_s": round(tt_l, 5) if ok_l else np.nan,
                "speedup_lloyd": round(ref_time / tt_l, 3) if ok_l else np.nan,
                "epochs_to_best": ep_b,
                "time_to_best_s": round(tt_b, 5) if ok_b else np.nan,
                "reached_best": ok_b,
            }
        )
    return pd.DataFrame(rows), target_lloyd, target_best, ref_time


def main(tags):
    for tag in tags:
        csv = os.path.join(RAW, f"{tag}.csv")
        if not os.path.exists(csv):
            print(f"missing {csv}")
            continue
        df = pd.read_csv(csv)
        target = float(df["target_mse"].iloc[0])
        budget = int(df["budget"].max())
        # titles are derived from the actual evaluated configuration (n, d, k)
        P.plot_mse_vs_epochs(df, path=os.path.join(P.FIG_DIR, f"{tag}_mse_vs_epochs.png"),
                             ycol="test_mse", target_mse=target)
        P.plot_mse_vs_time(df, path=os.path.join(P.FIG_DIR, f"{tag}_mse_vs_time.png"),
                           ycol="test_mse", target_mse=target)
        P.plot_runtime_bars(df, budget, path=os.path.join(P.FIG_DIR, f"{tag}_runtime_bars.png"))
        P.plot_time_to_target(df, path=os.path.join(P.FIG_DIR, f"{tag}_time_to_target.png"))

        table, t_lloyd, t_best, ref_time = summarize(df, budget=budget)
        print(f"\n=== {tag} ===")
        print(f"target_lloyd (Lloyd@{budget}) = {t_lloyd:.5g}  |  "
              f"target_best = {t_best:.5g}  |  ref_time(Lloyd@{budget}) = {ref_time:.5g}s")
        print(table.to_string(index=False))
        out = os.path.join(REPO_ROOT, "results", "processed", f"{tag}_summary.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        table.to_csv(out, index=False)
        print(f"summary -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["blobs_n10000_d128_k64"])
