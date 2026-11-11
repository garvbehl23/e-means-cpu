#!/usr/bin/env python
"""Run a budget sweep for one (dataset, n, d, k) cell across seeds and methods.

Example:
    python scripts/run_benchmark.py --dataset blobs --n 10000 --d 128 --k 64 \
        --seeds 0 1 2 --budgets 1 2 3 5 10 25
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from benchmark import budget_sweep, save_results  # noqa: E402
from datasets import get_dataset  # noqa: E402
from evaluation import set_threads  # noqa: E402

DEFAULT_METHODS = ["e-means", "lloyd-faiss", "lloyd-sklearn", "lloyd-sklearn-pp", "minibatch"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="blobs", choices=["blobs", "arxiv"])
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5, 10, 25])
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_threads(args.threads)
    tag = args.tag or f"{args.dataset}_n{args.n}_d{args.d}_k{args.k}"

    t_data = time.perf_counter()
    x_train, x_test, _ = get_dataset(args.dataset, args.n, args.d, args.k, seed=0)
    data_time = time.perf_counter() - t_data
    meta = {
        "dataset": args.dataset,
        "n": int(len(x_train)),
        "d": int(x_train.shape[1]),
        "k": int(args.k),
        "threads": args.threads,
        "data_time_s": data_time,
    }
    print(f"[{tag}] train={x_train.shape} test={x_test.shape} data_time={data_time:.3f}s")

    frames = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        df = budget_sweep(
            args.methods,
            x_train,
            x_test,
            args.k,
            args.budgets,
            seed,
            meta,
            mini_batch_size=args.minibatch_size,
        )
        df["cell"] = tag
        frames.append(df)
        print(f"  seed {seed}: {len(df)} runs in {time.perf_counter() - t0:.1f}s")

    all_df = pd.concat(frames, ignore_index=True)
    path = save_results(all_df, tag)
    print(f"saved -> {path}")
    summary = (
        all_df.groupby(["method", "budget"])[["train_mse", "test_mse", "fit_time_s"]]
        .median()
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
