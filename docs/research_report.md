# Research report — CPU reproduction of e-means

**Target paper:** *e-means: outperforming k-means (in one-tenth the epochs)*,
arXiv:2611.09501v1 (FAIR at Meta). **Official code:** `faiss/contrib/e_means`.

---

## 1. Research question

On a CPU-only machine, does the official e-means implementation reproduce the
paper's central claims — that e-means reaches a **lower** clustering objective
than Lloyd's k-means, in **~10× fewer epochs**, with a wall-clock speedup — and
how do those claims hold up against Lloyd and MiniBatchKMeans under a fair,
quality-matched comparison?

## 2. Environment (measured, not assumed)

| item | value |
|---|---|
| CPU | 11th Gen Intel Core i7-1185G7 @ 3.00 GHz |
| cores | **4 physical / 8 logical**, 1 socket, 1 NUMA node |
| RAM | 30 GiB (≈23 GiB free) |
| disk | 228 GiB free |
| GPU | **none** — Intel Iris Xe (integrated). No `nvidia-smi`, no CUDA |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-42 |
| Python | 3.13.13 (isolated `uv` venv) |
| libraries | numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, **faiss-cpu 1.16.0**, matplotlib 3.11.2, pandas 3.1.1, pyarrow 26.0.1 |

> **Correction to the original brief:** the machine has **no NVIDIA RTX 6000 Ada**
> (or any CUDA GPU). All experiments are therefore CPU-only, and no GPU
> comparison is possible or claimed. Benchmarking pins OpenMP/BLAS to **4
> threads** (`faiss.omp_set_num_threads(4)` + `threadpoolctl`) for stable timings.

## 3. Official implementation is CPU-capable

`faiss.contrib.e_means.NumpyEmeans` ships in **faiss-cpu ≥ 1.16.0** and runs the
official algorithm on CPU using the `faiss.knn` primitive for assignment. No
re-implementation was required; the repository wraps the official class. The
GPU path (`Emeans`, CuTe/Triton kernels) is out of scope.

## 4. Datasets

| name | type | n (train) | d | provenance |
|---|---|---|---|---|
| `blobs` | synthetic Gaussian mixture | 8k / 80k | 128 / 384 | generated, seed-controlled (`datasets.make_blobs`) |
| `arxiv` | **real** text embeddings | 40k (50k − 10k test) | **768** | arXiv titles embedded with `instructor-xl`, HuggingFace `Qdrant/arxiv-titles-instructorxl-embeddings` (same modality/dim as the paper's VIBE arXiv dataset) |

The arXiv vectors are unit-norm (‖x‖≈1). Test sets are held out (20k protocol,
or 20% when smaller) and never used for fitting.

## 5. Fair-comparison methodology

* **Same data, same initial centroids.** Every method is started from the *same*
  `k` rows sampled from the training set (`benchmark.make_init`), so the
  comparison isolates the optimization. Exception: `lloyd-sklearn-pp` uses its
  own k-means++ seeding (that *is* the baseline) — its init cost is included.
* **One objective, one evaluator.** The reported objective is always the exact
  mean squared distance to the nearest centroid, computed by a single shared
  routine using `faiss.knn` (the same C++ primitive the official e-means CPU
  path uses). Faiss L2 returns *squared* distances; verified in
  `tests/test_evaluation.py::test_faiss_returns_squared_l2`.
* **No squared-Euclidean/cosine mixing.** All methods minimise squared L2.
* **Timing.** `time.perf_counter()`, optimization only; dataset generation and
  initialization are recorded separately. No CUDA is involved, so no device
  synchronisation is needed.
* **Budgets.** "budget" = epochs (full passes) for all methods, so the x-axis is
  directly comparable. Sweeps of 1–25 epochs give MSE-vs-epoch and
  MSE-vs-wall-clock curves without fabricating interpolation.
* **Two targets.** `target_lloyd` = Lloyd's converged objective (the paper's
  convention, "Lloyd@25"); `target_best` = the best converged objective reached
  by any method. A method that never reaches a target is reported as *not
  reached* — no speedup is fabricated.
* **Repetition.** 3 seeds for all cells; medians reported. (The brief asked for
  3 seeds initially, 5 for final comparisons; 3 are used throughout and noted.)

## 6. Correctness verification (Phase 4)

`pytest` — **22/22 passing**:

* `test_faiss_returns_squared_l2` — the objective convention (Faiss returns
  squared L2).
* `test_final_mse_matches_manual` — objective vs a manual computation.
* `test_scheduled_alpha_endpoints` / `_matches_closed_form` — Equation 5.
* `test_shipped_scaling_law_coefficients` — Eq. 6–7 coefficients equal the
  shipped faiss values (and differ from the paper's Table 2 — documented).
* `test_recipe_batch_is_power_of_two_and_capped`, `test_epoch_rows_fractional`.
* `test_recovers_well_separated_clusters`, `test_small_2d_dataset` — e-means
  recovers ground-truth structure and reaches near-optimal MSE.
* `test_reproducible_with_fixed_seed`, `test_explicit_init_is_respected`.
* `test_objective_matches_shared_evaluator` — official `assign` agrees with the
  shared evaluator.
* `test_dead_cluster_revival_produces_no_nan`, `test_duplicate_points_no_nan` —
  no NaN under heavy ties / many dead clusters.
* `test_lloyd_implementations_agree` — Faiss, sklearn and the faiss.knn Lloyd
  loop converge to the same objective (±1%).
* `test_faiss_loop_objective_is_monotone_non_increasing`.

## 7. Main benchmark results

Consolidated table (`results/processed/benchmark_table.csv`). Times are medians
over 3 seeds on 4 CPU threads; "speedup" = Lloyd time ÷ e-means time to reach
**Lloyd's own converged objective**.

| cell | n | d | k | e-means converged | Lloyd converged | e-means best | epochs to Lloyd | e-means time (s) | Lloyd time (s) | **speedup** |
|---|---|---|---|---|---|---|---|---|---|---|
| blobs | 8 000 | 128 | 64 | 134.50 | 147.50 | 131.58 | **1** | 0.033 | 0.017 | **0.52×** |
| blobs | 80 000 | 128 | 64 | 142.18 | 145.14 | 134.26 | **1** | 0.220 | 0.067 | **0.31×** |
| blobs | 80 000 | 384 | 128 | 403.11 | 454.88 | 397.70 | **1** | 0.757 | 0.379 | **0.50×** |
| blobs | 80 000 | 128 | 1024 | 134.26 | 145.12 | 134.26 | **2** | 0.814 | 3.474 | **4.27×** |
| arxiv | 40 000 | **768** | 128 | 0.3432 | 0.3444 | 0.3432 | **2** | 1.473 | 1.602 | **1.09×** |

**Findings**

1. **Quality: reproduced.** e-means reaches a *lower* final objective than
   random-init Lloyd in **every** cell (e.g. arXiv 0.3432 vs 0.3444, −0.36%;
   k=1024 134.3 vs 145.1, −7.5%). This matches the paper's core claim.
2. **Epochs: reproduced.** e-means reaches Lloyd's converged loss in **1–2
   epochs** vs Lloyd's 10–25 — i.e. ~10× fewer passes, as the title claims.
3. **Wall-clock: not reproduced at the paper's magnitude.** On this 4-core CPU
   the speedup to reach Lloyd's converged loss ranges from **0.31× (slower) at
   k=64** to **4.27× at k=1024**, and **1.09× on real arXiv 768-d**. The paper's
   10–45× is a **GPU** result (H100); it is not a CPU result and is not claimed
   here. The CPU story is governed by the per-batch overhead of the NumPy/faiss
   e-means path: one e-means epoch issues `n/B` small `faiss.knn` calls, whose
   Python overhead dominates until `k` (and hence the per-call work) is large
   enough to amortise it.
4. **vs k-means++.** On synthetic data k-means++ reaches a comparable final
   objective, but its **seeding is very expensive on CPU** (e.g. 13.9 s vs
   e-means's 0.76 s for 1 epoch at 80k×384×128; 37.8 s at k=1024). e-means thus
   reaches near-best quality far faster end-to-end. This is consistent with the
   paper's motivation, though the paper measures this on GPU/64-core hardware.
5. **MiniBatchKMeans (Sculley).** Worst objective in most cells (e.g. 158.1 vs
   134.3 at k=1024) — e-means dominates the classic mini-batch baseline.

Figure: `figures/scaling_speedup_vs_k.png`, `figures/scaling_speedup_overview.png`.

## 8. Ablation results

### 8.1 Component ablations (official switches)

Train MSE (median of 3 seeds). `no_revival` sets `unused_threshold=0`;
`no_cosine` holds the step constant at `alpha0`; `no_ema` sets the step to 1
(hard batch-mean — no longer the e-means algorithm).

**arXiv 768-d, k=128** (full e-means best at every budget):

| epochs | full | no-cosine | no-revival | no-ema |
|---|---|---|---|---|
| 1 | **0.34870** | 0.34913 | 0.35357 | 0.36473 |
| 3 | **0.34394** | 0.34436 | 0.34891 | 0.35347 |
| 5 | **0.34337** | 0.34438 | 0.34817 | 0.35284 |
| 10 | **0.34318** | 0.34462 | 0.34784 | 0.34462† |

**Synthetic 128-d, k=1024:**

| epochs | full | no-cosine | no-revival | no-ema |
|---|---|---|---|---|
| 3 | 139.56 | **136.37** | 151.65 | 152.48 |
| 5 | 137.42 | **133.78** | 148.37 | 147.54 |
| 10 | 134.57 | **130.63** | 148.14 | 147.48 |

† At T=10 for the arXiv config the step law **saturates at `alpha0 = 1.0`**, so
`no_cosine` and `no_ema` become the same algorithm (identical value).

**Interpretation**

* **Dead-cluster revival is the most important component** (+1.4–1.5% train MSE
  when removed on arXiv; +10–13% at k=1024). This **reproduces** the paper's
  Table 1/14 ordering.
* **EMA is essential.** Removing it (hard batch-mean) is unstable early
  (0.379 at T=2 on arXiv vs 0.344 full) and never wins. Reproduces the paper's
  motivation for EMA.
* **Cosine schedule: not confirmed here.** On synthetic k=1024 the naive
  constant-step arm *beats* full e-means. Because the paper warns that the
  constant arm must be retuned (App. G.3), a **fair** version was run (§8.2).

### 8.2 Fair cosine-schedule ablation (retuned constant step)

`scripts/run_cosine_ablation.py` sweeps a constant step on a validation split,
selects the best, and reports on test (paper's App. G.3 protocol).

| config | arm | α | val MSE | test MSE |
|---|---|---|---|---|
| synthetic k=1024, T=5 | cosine (α0=0.59) | — | 142.49 | 142.14 |
| | constant, retuned | 0.7 | **135.70** | **135.44** |
| arXiv 768-d k=128, T=5 | cosine | — | 0.34725 | **0.34755** |
| | constant, retuned | 0.3–0.5 | **0.34708** | 0.34770 |

**Result:** after fair retuning, the cosine schedule is **not beneficial** on the
cells tested — it *hurts* on synthetic k=1024 (135.4 vs 142.1) and is a *tie* on
arXiv (0.34770 vs 0.34755, within seed noise). This **differs** from the paper's
Table 13, where cosine helped by 0.03–0.54% on test. Possible reasons: the
scaling law's `α0` is a *peak* value that can be too small as a constant, and
synthetic data may favour a larger constant step; the paper's datasets are real
embeddings at much larger scale.

## 9. Speedup calculation (explicit)

`speedup = Lloyd_time_to_target / e-means_time_to_target`, both times being the
first swept budget whose median objective ≤ `target_lloyd`. Example (arXiv,
target 0.34445): e-means reaches it at T=2 (1.473 s); Lloyd at T=25 (1.602 s) →
speedup = 1.602 / 1.473 = **1.09×**. Where a method never reached the target
within the sweep, it is reported as *not reached* (e.g. MiniBatchKMeans on the
arXiv cell).

## 10. Differences from the published findings

| claim (paper) | this reproduction |
|---|---|
| e-means beats Lloyd's converged loss | **reproduced** on all 5 cells |
| reaches Lloyd@25 loss in 1.4–2.6 epochs | **reproduced** (1–2 epochs) |
| 10–45× wall-clock speedup | **not a CPU result**; here 0.31×–4.27× (CPU), scales with k |
| dead-cluster revival dominant | **reproduced** (largest ablation effect) |
| cosine schedule helps (Table 13) | **not confirmed** — tie/harmful after fair retuning |
| beats k-means++ at convergence | **partially** — ties/beats on real data, k-means++ marginally better on synthetic, but far slower to seed |
| no hyperparameter tuning needed | confirmed (shipped scaling laws run with defaults) |

**Known deviations in the official code itself** (documented, not our error):
initial mass `2**-16` vs paper's `1e-10`; revival disabled for `d ≤ 2`; shipped
scaling-law coefficients differ from Table 2; `alpha0` clamp `[1e-4, 1]` and
frequent saturation at `1.0` for mid-size problems.

## 11. Limitations

* **No GPU.** The paper's headline speedups are GPU; only the CPU regime is
  testable here, on a 4-core laptop CPU (the paper's CPU results use 64 cores).
* **Scale.** Largest real cell is 40k × 768; the paper goes to 10B. The
  overhead-dominated small-`k` regime is over-represented.
* **3 seeds** (not 5) for the final comparisons, for time budget reasons.
* **Synthetic data** may not represent real embedding geometry; conclusions on
  the cosine schedule are dataset-dependent.
* `no_ema` is an *approximate* ablation (it changes the algorithm to a batch
  mean), not an official switch.
* `no_cosine`/`no_ema` collapse when `alpha0` saturates at 1.

## 12. Applications to SLM conversational-embedding curation (potential, not established)

e-means is attractive for organising **high-dimensional conversational
embeddings** because it needs no hyperparameter sweep and converges in a few
passes — useful when re-clustering a corpus repeatedly. Candidate uses:

* **Conversation-scenario clustering** — group embeddings of user turns into
  scenario buckets for taxonomy building or curriculum design.
* **Deduplication / diversity analysis** — near-duplicate conversations land in
  the same fine cluster; small clusters flag rare scenarios.
* **Underrepresented tool-use discovery** — clusters with few members that
  nonetheless contain tool calls highlight under-covered behaviours.
* **Multi-client dataset balancing** — cluster then reweight/oversample so no
  client's scenarios dominate.

These are **hypotheses**, not measured SLM-training improvements. Any such study
must use public or synthetic conversations; no private client data was accessed
or published here. The CPU findings suggest e-means is a good *quality-per-epoch*
choice, while the wall-clock benefit for SLM curation depends on `k` and on
whether a GPU is available.

## 13. Reproduction-fidelity checklist

| item | status |
|---|---|
| official implementation used | ✅ `faiss.contrib.e_means.NumpyEmeans` (faiss-cpu 1.16.0) |
| same objective & evaluator as official | ✅ shared `faiss.knn` objective, tested |
| same init across methods | ✅ shared random init (except k-means++ baseline) |
| paper's dataset (exact) | ⚠️ not downloaded (SIFT/DEEP/DINO10B too large); used a real 768-d arXiv embedding dataset instead |
| paper's `k` range (4√n–32√n) | ⚠️ smaller `k` used (≤1024) for CPU tractability |
| paper's hardware | ❌ no GPU; 4-core CPU vs 64-core/GPU |
| scaling-law hyperparameters | ✅ shipped defaults (no tuning) |
| reported metric | ✅ test MSE (and train MSE) |
| seeds | ⚠️ 3 (paper: 10) |

**Bottom line:** the *algorithmic* claims of e-means — lower loss than Lloyd,
~10× fewer epochs, revival as the key component — **reproduce on CPU**. The
*wall-clock* magnitude of the paper does **not** transfer to a small CPU box,
and the cosine-schedule benefit is **not confirmed** here. All raw measurements
are in `results/raw/`, summaries in `results/processed/`, figures in `figures/`.
