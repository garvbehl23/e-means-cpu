# e-means-cpu — a CPU reproduction of *e-means: outperforming k-means (in one-tenth the epochs)*

A CPU-only reproduction and fair benchmark of **e-means** (arXiv:2611.09501,
FAIR at Meta) against **Lloyd's k-means** and **MiniBatchKMeans**, using the
**official** `faiss.contrib.e_means.NumpyEmeans` implementation.

No GPU is used or required. This is a research reproduction, not a
re-implementation: the official class ships in `faiss-cpu >= 1.16.0` and is used
directly.

---

## TL;DR — what was verified

| claim | verdict |
|---|---|
| e-means reaches a **lower objective** than Lloyd | ✅ reproduced on all 5 cells |
| reaches Lloyd's converged loss in **~10× fewer epochs** (1–2 epochs) | ✅ reproduced |
| **10–45× wall-clock speedup** | ❌ that is a **GPU** result; on this 4-core CPU: **0.31×–4.27×** (scales with `k`), **1.09×** on real 768-d data |
| **dead-cluster revival** is the dominant component | ✅ reproduced (largest ablation effect) |
| **cosine step schedule** helps | ⚠️ not confirmed — tie/harmful after fair retuning |
| **no hyperparameter tuning** needed | ✅ confirmed (shipped scaling laws, defaults) |

## Hardware used (measured)

Intel **i7-1185G7, 4 physical / 8 logical cores**, 30 GiB RAM, Ubuntu 24.04,
Python 3.13, `faiss-cpu 1.16.0`. **No CUDA GPU is present.** Benchmarks pin
OpenMP/BLAS to 4 threads for stable timings.

## Install

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# or:  uv pip install --python .venv/bin/python numpy scipy scikit-learn faiss-cpu matplotlib pandas pyarrow pytest
```

The real dataset (arXiv titles embedded with `instructor-xl`, 768-d, 50k
vectors) downloads automatically on first use from
`https://huggingface.co/datasets/Qdrant/arxiv-titles-instructorxl-embeddings`.

## Reproduce everything

```bash
export OMP_NUM_THREADS=4

# 0) correctness tests
.venv/bin/python -m pytest -q

# 1) the headline grid (5 cells × 3 seeds × 6 budgets)
.venv/bin/python scripts/run_grid.py configs/benchmark_grid.json

# 2) component ablations
.venv/bin/python scripts/run_grid.py configs/ablations.json

# 3) fair cosine-schedule ablation (retuned constant step)
.venv/bin/python scripts/run_cosine_ablation.py --dataset arxiv  --n 50000 --d 768 --k 128  --epochs 5
.venv/bin/python scripts/run_cosine_ablation.py --dataset blobs  --n 100000 --d 128 --k 1024 --epochs 5

# 4) figures + speedup tables
.venv/bin/python scripts/make_figures.py blobs_n10000_d128_k64 blobs_n100000_d128_k64 \
    blobs_n100000_d384_k128 blobs_n100000_d128_k1024 arxiv_n50000_d768_k128
.venv/bin/python scripts/make_scaling_figure.py
```

Single-cell run (as in the brief — 10 000 × 128 × 64):

```bash
OMP_NUM_THREADS=4 .venv/bin/python scripts/run_benchmark.py \
    --dataset blobs --n 10000 --d 128 --k 64 --seeds 0 1 2 --budgets 1 2 3 5 10 25
```

## Benchmark table

Times are medians over 3 seeds on 4 threads. **Speedup** = Lloyd time ÷ e-means
time to reach **Lloyd's own converged objective**.

| dataset | n | d | k | e-means conv. | Lloyd conv. | epochs to Lloyd | e-means s | Lloyd s | speedup |
|---|---|---|---|---|---|---|---|---|---|
| blobs | 8 000 | 128 | 64 | **134.50** | 147.50 | 1 | 0.033 | 0.017 | 0.52× |
| blobs | 80 000 | 128 | 64 | **142.18** | 145.14 | 1 | 0.220 | 0.067 | 0.31× |
| blobs | 80 000 | 384 | 128 | **403.11** | 454.88 | 1 | 0.757 | 0.379 | 0.50× |
| blobs | 80 000 | 128 | 1024 | **134.26** | 145.12 | 2 | 0.814 | 3.474 | **4.27×** |
| **arxiv** | 40 000 | **768** | 128 | **0.3432** | 0.3444 | 2 | 1.473 | 1.602 | **1.09×** |

e-means reaches a **lower** objective than Lloyd in every row and reaches
Lloyd's converged loss in **1–2 epochs** (vs 10–25). Its CPU wall-clock
advantage grows with `k` (per-batch overhead amortises) and is far below the
paper's GPU figures.

## Figures (`figures/`)

* `scaling_speedup_vs_k.png`, `scaling_speedup_overview.png` — speedup vs k / across cells
* `<cell>_mse_vs_epochs.png`, `<cell>_mse_vs_time.png` — objective vs epochs / wall-clock
* `<cell>_runtime_bars.png`, `<cell>_time_to_target.png`
* `*_ablation_*` — component and cosine-schedule ablations

All figures are 300 DPI and are generated **only** from the benchmark CSVs in
`results/raw/`; no illustrative data is plotted.

## Repository layout

```
e-means-cpu/
├── configs/          benchmark_grid.json, ablations.json
├── src/              e_means.py (official wrapper), baselines.py, datasets.py,
│                     evaluation.py, benchmark.py, plotting.py
├── scripts/          run_benchmark.py, run_grid.py, run_cosine_ablation.py,
│                     make_figures.py, make_scaling_figure.py
├── tests/            22 correctness tests
├── results/raw|processed/   CSVs + speedup tables
├── figures/          300-DPI PNGs
└── docs/             paper_notes.md, research_report.md, linkedin_post.md
```

## Implementation notes & fairness

* **Objective:** exact mean squared distance to the nearest centroid, computed by
  a single shared routine (`evaluation.final_mse`, via `faiss.knn`). Faiss L2
  returns *squared* distances — tested.
* **Initialization:** every method starts from the **same** sampled centroids,
  except the k-means++ baseline, which uses its own seeding (its cost is counted).
* **Timing:** `time.perf_counter`, optimization only; data-gen and init recorded
  separately.
* **Not a strawman:** the Lloyd baseline uses Faiss/sklearn C++ (the paper's CPU
  baseline). A `faiss.knn` Lloyd loop is included for fine-grained curves and is
  cross-checked against Faiss/sklearn to agree within 1%.

## Deviations from the official implementation (documented)

The official code itself differs from the paper in places: initial mass `2**-16`
vs `1e-10`; revival disabled for `d ≤ 2`; shipped scaling-law coefficients differ
from Table 2; `alpha0` is clamped and frequently saturates at `1.0`. See
`docs/paper_notes.md §2`.

## Citation & attribution

Official code: `facebookresearch/faiss`, `contrib/e_means/` (MIT, Meta). Please
cite:

```bibtex
@misc{bessa2026emeans,
  title  = {{e-means}: outperforming {k-means} (in one-tenth the epochs)},
  author = {Bessa, Swann and Mazar{\'e}, Pierre-Emmanuel and Szilvasy, Gergely and
            Touvron, Hugo and Picard, David and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  year   = {2026}, note = {Preprint}, url = {https://arxiv.org/abs/2611.09501}
}
```

## Limitations

CPU-only, 4-core laptop; real data limited to 40k×768 (the paper reaches 10B);
3 seeds. Full discussion in `docs/research_report.md`.

## License

This reproduction is released under the **MIT License** (see `LICENSE`). The
official e-means implementation it depends on is also MIT (Meta Platforms,
Inc.); attribution and dataset provenance are in `NOTICE`.
