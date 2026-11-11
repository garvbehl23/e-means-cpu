#!/usr/bin/env python
"""Run every cell of a JSON benchmark config.

Example:
    python scripts/run_grid.py configs/benchmark_grid.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from benchmark import budget_sweep, save_results  # noqa: E402
from datasets import get_dataset  # noqa: E402
from evaluation import set_threads  # noqa: E402


def main(path):
    with open(path) as fh:
        cfg = json.load(fh)
    set_threads(cfg.get("threads", 4))
    for cell in cfg["cells"]:
        tag = cell.get("tag") or f"{cell['dataset']}_n{cell['n']}_d{cell['d']}_k{cell['k']}"
        t0 = time.perf_counter()
        x_train, x_test, _ = get_dataset(cell["dataset"], cell["n"], cell["d"], cell["k"], seed=0)
        meta = {
            "dataset": cell["dataset"],
            "n": int(len(x_train)),
            "d": int(x_train.shape[1]),
            "k": int(cell["k"]),
            "threads": cfg.get("threads", 4),
            "data_time_s": time.perf_counter() - t0,
        }
        print(f"[{tag}] train={x_train.shape} test={x_test.shape}")
        frames = []
        for seed in cfg["seeds"]:
            df = budget_sweep(cfg["methods"], x_train, x_test, cell["k"], cfg["budgets"], seed, meta,
                              mini_batch_size=cfg.get("minibatch_size", 1024))
            df["cell"] = tag
            frames.append(df)
        all_df = pd.concat(frames, ignore_index=True)
        print(f"  saved -> {save_results(all_df, tag)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark_grid.json")
