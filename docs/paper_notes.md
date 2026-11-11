# e-means — paper breakdown and implementation map

**Paper:** *e-means: outperforming k-means (in one-tenth the epochs)*.
Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou (FAIR at Meta; École
Nationale des Ponts et Chaussées). arXiv:2611.09501v1, 06 Nov 2026.
**Official code:** `facebookresearch/faiss`, `contrib/e_means/` (MIT).
**Official site:** https://e-means.github.io

> Provenance of this note: the arXiv HTML and the official source in
> `faiss-cpu 1.16.0` (`faiss/contrib/e_means/`) were read directly. Every
> equation and number below is taken from those sources; anything that could
> not be verified is flagged explicitly.

---

## 1. Problem

Given `X = {x_1..x_n} ⊂ R^d` and `k`, k-means minimises the mean squared
distance to the nearest centroid:

```
MSE(X, C) = (1/n) Σ_i || x_i - c_{a(i)} ||^2 ,   a(i) = argmin_j || x_i - c_j ||^2   (Eq. 1)
```

Lloyd alternates full-batch assignment and centroid-mean updates. Mini-batch
k-means (MacQueen 1967; Sculley 2010) updates from mini-batches. e-means is a
mini-batch method that targets the *canonical, fixed-dataset* k-means problem,
retargeting the EMA centroid rule used in VQ-VAEs.

## 2. Algorithm (paper Algorithm 1)

```
c_1..c_k ← RandomChoice(X, k)                  # random init from the data
N_j ← 1e-10 ; S_j ← N_j * c_j ;  t ← 0
for T epochs:
    for each batch B of B points from a fresh shuffle of X:
        t ← t+1 ; γ ← γ(t)                      # cosine schedule, Eq. 5
        a(x) ← argmin_j || x - S_j/N_j ||^2      # assign (Eq. 4 readout)
        n_j ← |{x : a(x)=j}| ; s_j ← Σ_{a(x)=j} x
        N_j ← γ N_j + (1-γ) n_j                 # EMA update, Eq. 2
        S_j ← γ S_j + (1-γ) s_j
        if t mod 5 == 0: ReviveDeadClusters(τ)  # Eq. 8
c_j ← mean of last-epoch points assigned to j   # last-epoch-mean finalization
```

### 2.1 EMA updates (Eq. 2–4)

Each cluster keeps a running **count** `N_j` and **sum** `S_j`; the centroid is
their ratio `c_j = S_j / N_j` (Eq. 4). A batch contributes `n_j` (count) and
`s_j` (sum) for cluster `j` (Eq. 3). The update is

```
N_j ← γ N_j + (1-γ) n_j          S_j ← γ S_j + (1-γ) s_j          (Eq. 2)
```

with `γ ∈ [0,1)` the decay, and `α = 1-γ` the step. Two limits are important:
`α = 1` makes `c_j` the batch mean (no memory); `α → 0` freezes the centroid.
The shipped code stores `(N_j, c_j)` instead of `(N_j, S_j)` and rewrites the
update as `c_new = c + (α n_j / N_new)(s_j/n_j − c)`; algebraically identical.
*Implementation:* `faiss/contrib/e_means/e_means.py::NumpyEmeans._apply_statistics`.

### 2.2 Initialization

Paper: `N_j = 1e-10`, `S_j = N_j c_j^{(0)}` so Eq. 4 returns the prescribed
initial centroids while the initialization carries negligible mass.
**Shipped code uses `N_j = 2**-16`** (a documented deviation;
`base.py::_INIT_USAGE`). Initial centroids are `k` rows sampled uniformly
without replacement from `X` (random init, not k-means++).

### 2.3 Cosine step schedule (Eq. 5)

The step `α` is annealed on a cosine from a peak `α0` to `ε α0`:

```
α(t) = α0 ( ε + (1-ε) · ½ (1 + cos(π t / M)) ) ,   ε = 0.1 ,  t ∈ {0..M-1}   (Eq. 5)
```

`M` is the total number of updates. *Implementation:* `base.py::_scheduled_alpha`
(`_ALPHA_MIN_FACTOR = 0.1`).

### 2.4 Dead-cluster revival (Eq. 8, Appendix H)

Every 5 updates, clusters with `N_j < τ · mean(N)`, `τ = 0.1`, are revived by
splitting a **donor** drawn without replacement with probability ∝ `N_j`. The
donor and the revived slot receive two perturbed copies of the donor centroid:

```
c_child      = c_donor ⊙ [1+δ, 1−δ, 1+δ, …]
c_donor      = c_donor ⊙ [1−δ, 1+δ, 1−δ, …]        δ ≪ 1                 (Eq. 8)
```

with `δ` the machine epsilon of the lowest supported precision (BF16).
Each child inherits `N_donor/2`. Cost is `O(d)` — no data re-scan.
*Implementation:* `e_means.py::NumpyEmeans._choose_splits` / `_apply_splits`
(`_SPLIT_EPSILON = 1e-4`), triggered from `base.py::_EmeansBase._finish_step`.
**Shipped-code deviations:** initial mass `2**-16`, and revival is *disabled for
`d ≤ 2`* (`_NO_REVIVAL_MAX_D = 2`).

### 2.5 Finalization (Section 3.2)

Two options, both cheap:
* **last-epoch-mean** (default): set `c_j` to the mean of the points the *last
  epoch* assigned to it. Guaranteed not to increase train MSE; reuses existing
  assignments. *Implementation:* `e_means.py::NumpyEmeans.finalize`.
* **slow-EMA**: a second, slower EMA of `(N_j, S_j)` tracked in parallel and
  used at the end for better *out-of-sample* MSE (half-life `h`). Not enabled by
  default in the CPU quick-start path.

Because of last-epoch-mean, the returned centroids depend on the **last epoch's
partition**, not on the full EMA trajectory — an important detail for ablation
interpretation (see `docs/research_report.md`).

### 2.6 Scaling laws (Section 3.3, Eq. 6–7)

Two log-linear laws set `B` and `α0` from the problem shape `(k, n, T)`:

```
α0 = min( a_c · k^a_k · B^a_B · n^a_n · T^a_T , 1 )        (Eq. 6)
B  =       b_c · k^b_k · n^b_n · T^b_T                     (Eq. 7, knee at +0.2%)
```

**Paper Table 2 coefficients** (fitted on DEEP):

| law | c | k | B | n | T |
|-----|---|---|---|---|---|
| step α0 | 0.824 | −0.46 | 0.90 | −0.45 | −0.11 |
| batch B (knee 0.2%) | 0.035 | 0.13 | — | 0.88 | 0.81 |

**Shipped-code coefficients** (`base.py::_BATCH_LAWS`, `_ALPHA_LAW`) — the code
comment states these *differ* from Table 2:

```
throughput batch : (0.035134, 0.134749, 0.876473, 0.813344)
quality   batch  : (0.00439175, 0.134749, 0.876473, 0.813344)   # = throughput/8
step α0          : (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
```

`B` is rounded to the nearest power of two and capped (`≈ n/2`); `α0` is clamped
to `[1e-4, 1]`. `batch_law="auto"` selects `"quality"` for `d ≤ 16` else
`"throughput"`. At `n > 2M` the paper divides the batch by 8.
*Implementation:* `base.py::_resolve_recipe`, `_step_law`, `_round_power_of_two`,
`_resolve_batch_law`, `_resolve_unused_threshold`.

## 3. How e-means differs from Lloyd and MiniBatchKMeans

| aspect | Lloyd | MiniBatch (Sculley) | e-means |
|---|---|---|---|
| update | exact mean of all assigned pts | running mean over batches | EMA of (count, sum) with decay γ |
| step | implicit (α=1, full batch) | 1/`n_j` running average | annealed `α=1-γ` (cosine, Eq. 5) |
| hyperparams | none | batch size, (reassignment) | none — set by scaling laws (Eq. 6–7) |
| dead clusters | not addressed | optional `reassignment_ratio` | split-donor revival every 5 steps (Eq. 8) |
| finalize | last iteration | last state | last-epoch mean / slow-EMA |
| per-epoch FLOPs | assignment-dominated | assignment-dominated | assignment-dominated (same order) |

## 4. Experimental protocol in the paper

* **Datasets (App. A):** BigANN (SIFT 128-d, DEEP 96-d, SPACEV 100-d,
  SimSearchNet++ 256-d), VIBE (AG-News 1024-d, ImageNet-CLIP 512-d, arXiv 768-d),
  and DINO10B (1024-d). `k` from `4√n` to `32√n`.
* **Baselines (App. C):** Faiss CPU/GPU (Lloyd), Flash-KMeans, cuML,
  scikit-learn MiniBatchKMeans (Sculley; early stopping off, batch + reassignment
  swept), Staleness-Reduction, SuperKMeans. k-means++ init from scikit-learn.
* **Metric:** test MSE; relative to **Lloyd@25** (Faiss default).
* **Hardware:** H100 80 GB (GPU); Xeon Platinum 8462Y+ 64 physical cores (CPU);
  CPU assignment via Faiss.
* **No hyperparameter tuning:** B and γ come from the scaling laws.

## 5. Reported results (paper claims — not yet independently verified here)

* Beats Lloyd@25 in **1.4–2.6 epochs**; **10–45×** GPU wall-clock speedups.
* DINO10B (10B × 1024-d, 16M clusters) trained in ~2.5 epochs.
* Ablation (Table 1, DEEP* n=1e6 k=16384 T=3): dead-cluster revival largest
  effect (+2.97% train if removed); cosine +0.26%; finalization +0.14%;
  k-means++ init −0.18%.
* Dead-cluster counts (Table 14, DEEP 1M): on 0.3%, resample 14.3%, off 26.3%.
* Cosine vs retuned constant step (Table 13): cosine better on train
  everywhere, on test most of the time (0.03–0.54%).

## 6. What this repository verifies (see `docs/research_report.md`)

* **Verified on CPU:** e-means reaches Lloyd's converged loss in 1–2 epochs and
  a lower final objective on every cell tested.
* **Verified:** revival is the most important component; EMA is essential
  (hard batch-mean is unstable early).
* **Divergence:** the *wall-clock* speedup on this 4-core CPU is far smaller than
  the paper's GPU numbers (0.31× at k=64 → 4.27× at k=1024; 1.09× on arXiv 768-d).
* **Divergence:** with a *retuned* constant step, the cosine schedule was not
  beneficial on the cells tested (tie on arXiv, worse on synthetic k=1024).
