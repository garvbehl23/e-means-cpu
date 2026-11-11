# LinkedIn post draft (for review — not published)

> Based exclusively on the experiments in this repository. Replace `[repo link]`
> and `[figure]` before posting. Numbers are medians over 3 seeds on the hardware
> stated.

---

Meta FAIR just posted a clustering paper with a bold title — **"e-means:
outperforming k-means (in one-tenth the epochs)"** (Bessa, Mazaré, Szilvasy,
Touvron, Picard, Douze, Jégou; arXiv:2611.09501). The idea is to retarget the
exponential-moving-average centroid rule from VQ-VAEs at the *classic* k-means
problem, and to ship scaling laws so you never tune a hyperparameter.

I reproduced it — **on CPU only** (I don't have a GPU on this machine), using the
official implementation, which actually ships inside `faiss-cpu >= 1.16.0`
(`faiss.contrib.e_means.NumpyEmeans`). Here's what I found, the honest version.

**The algorithm, in one paragraph.** Instead of recomputing each centroid as the
exact mean of its points (Lloyd), e-means keeps an exponential moving average of
each cluster's count and sum: `N ← γN + (1−γ)n`, `S ← γS + (1−γ)s`, centroid
`= S/N`. It anneals the step `α = 1−γ` on a cosine, revives dead clusters by
splitting a busy donor, and finalises with the last epoch's exact mean. Batch
size and peak step come from two log-linear scaling laws — no tuning.

**What reproduced (5 cells, 3 seeds, i7-1185G7, 4 threads):**
• e-means reached a **lower final objective than Lloyd in every cell** (e.g. on
real 768-d embeddings: 0.3432 vs 0.3444; at k=1024: 134.3 vs 145.1).
• It hit Lloyd's converged loss in **1–2 epochs** vs Lloyd's 10–25 — the "one
tenth the epochs" claim holds.
• The ablation ordering matched the paper: **dead-cluster revival mattered
most**, EMA was essential.

**What did *not* transfer:**
• The **10–45× speedup is a GPU result**. On my 4-core CPU it was **0.31× at
  k=64 (slower!), rising to 4.27× at k=1024, and 1.09× on real 768-d data.** The
  NumPy/faiss per-batch overhead dominates until k is large enough to amortise
  it.
• The **cosine schedule** did not help. After doing the paper's own fair
  protocol (retune a constant step on validation), the constant step *beat*
  cosine on synthetic k=1024 and *tied* it on real data.

**What I built:** a reproducible CPU repo — shared-objective evaluator, identical
initialisation across methods, budget sweeps for MSE-vs-epochs and
MSE-vs-wall-clock, 22 correctness tests, and 300-DPI figures. [repo link]

Takeaway: e-means is a genuinely better *optimizer* for k-means — better loss,
far fewer epochs, no tuning. But "10× faster" is hardware-dependent, and the
cosine schedule deserves a second look on your data.

Paper: https://arxiv.org/abs/2611.09501
Code: https://github.com/facebookresearch/faiss/tree/main/contrib/e_means

#MachineLearning #Clustering #KMeans #Research #Reproducibility
