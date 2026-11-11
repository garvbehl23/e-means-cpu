#!/usr/bin/env python
"""Fair ablation of the cosine step schedule (paper Appendix G.3).

The paper's point: alpha0 is fitted as the *peak* of a decaying schedule, so
comparing it against a constant step at that same value is unfair. The constant
arm must be retuned on validation data. This script does exactly that:

  * cosine arm  : the official schedule, alpha annealed alpha0 -> 0.1*alpha0.
  * constant arm: a constant step alpha, swept over a grid and selected on a
                  held-out validation split, then reported on test.

Only the constant arm is retuned; the cosine arm keeps its shipped alpha0.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from benchmark import make_init, save_results  # noqa: E402
from datasets import get_dataset, split_train_test  # noqa: E402
from e_means import EMeansCPU  # noqa: E402
from evaluation import final_mse, set_threads  # noqa: E402

ALPHA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def run(args):
    set_threads(args.threads)
    x_train_full, x_test, _ = get_dataset(args.dataset, args.n, args.d, args.k, seed=0)
    x_train, x_val = split_train_test(x_train_full, n_test=max(2000, len(x_train_full) // 5), seed=123)
    init = make_init(x_train, args.k, seed=args.seeds[0])
    rows = []
    for seed in args.seeds:
        # cosine arm (official schedule)
        cos = EMeansCPU(args.k, nepochs=args.epochs, seed=seed).fit(x_train, init_centroids=init)
        cos_val = final_mse(x_val, cos.centroids_)
        cos_test = final_mse(x_test, cos.centroids_)
        rows.append(dict(seed=seed, arm="cosine", alpha=np.nan, val_mse=cos_val, test_mse=cos_test))
        # constant arm: sweep alpha, select on validation
        best = None
        for a in ALPHA_GRID:
            m = EMeansCPU(args.k, nepochs=args.epochs, seed=seed, alpha0=a,
                          variant="no_cosine").fit(x_train, init_centroids=init)
            v = final_mse(x_val, m.centroids_)
            if best is None or v < best[0]:
                best = (v, a, final_mse(x_test, m.centroids_))
        rows.append(dict(seed=seed, arm="constant_retuned", alpha=best[1],
                         val_mse=best[0], test_mse=best[2]))
        print(f"  seed {seed}: cosine val={cos_val:.5f} test={cos_test:.5f} | "
              f"best constant alpha={best[1]} val={best[0]:.5f} test={best[2]:.5f}")
    df = pd.DataFrame(rows)
    df["dataset"] = args.dataset
    df["n"] = len(x_train)
    df["d"] = x_train.shape[1]
    df["k"] = args.k
    df["epochs"] = args.epochs
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="blobs", choices=["blobs", "arxiv"])
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    tag = f"cosine_ablation_{args.dataset}_n{args.n}_d{args.d}_k{args.k}_T{args.epochs}"
    print(f"[{tag}]")
    df = run(args)
    med = df.groupby("arm")[["val_mse", "test_mse"]].median().round(5)
    print("\nmedian by arm:\n", med.to_string())
    save_results(df, tag)


if __name__ == "__main__":
    main()
