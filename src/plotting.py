"""Publication-quality figures (Matplotlib, 300 DPI).

All curves show the median over seeds; shaded bands show the inter-quartile
range. Nothing is plotted from synthetic/illustrative numbers: every series
comes from a benchmark CSV produced by ``scripts/run_benchmark.py``.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")

STYLE = {
    "e-means": dict(color="#d62728", marker="o", lw=2.4, zorder=5),
    "lloyd-faiss": dict(color="#1f77b4", marker="s", lw=1.8),
    "lloyd-sklearn": dict(color="#17becf", marker="^", lw=1.6, ls="--"),
    "lloyd-sklearn-pp": dict(color="#2ca02c", marker="D", lw=1.8),
    "lloyd-faiss-loop": dict(color="#7f7f7f", marker="v", lw=1.4, ls=":"),
    "minibatch": dict(color="#ff7f0e", marker="P", lw=1.8),
    "e-means-no-cosine": dict(color="#9467bd", marker="x", lw=1.6, ls="--"),
    "e-means-no-revival": dict(color="#8c564b", marker="+", lw=1.6, ls="--"),
    "e-means-no-ema": dict(color="#e377c2", marker="*", lw=1.6, ls=":"),
}

LABELS = {
    "e-means": "e-means (official)",
    "lloyd-faiss": "Lloyd (Faiss)",
    "lloyd-sklearn": "Lloyd (sklearn, random init)",
    "lloyd-sklearn-pp": "Lloyd (sklearn, k-means++)",
    "lloyd-faiss-loop": "Lloyd (faiss.knn loop)",
    "minibatch": "MiniBatchKMeans (sklearn)",
    "e-means-no-cosine": "e-means, no cosine schedule",
    "e-means-no-revival": "e-means, no revival",
    "e-means-no-ema": "e-means, no EMA (batch mean)",
}


def _style(name):
    return STYLE.get(name, dict(color="black", marker=".", lw=1.2))


def _label(name):
    return LABELS.get(name, name)


def _median_iqr(df, method, xcol, ycol):
    sub = df[df["method"] == method]
    g = sub.groupby(xcol)[ycol]
    med = g.median()
    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)
    return med.index.values, med.values, q1.values, q3.values


def plot_mse_vs_epochs(df, path=None, ycol="test_mse", target_mse=None, title=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method in sorted(df["method"].unique()):
        x, med, q1, q3 = _median_iqr(df, method, "budget", ycol)
        order = np.argsort(x)
        st = _style(method)
        ax.plot(x[order], med[order], label=_label(method), **st)
        ax.fill_between(x[order], q1[order], q3[order], color=st["color"], alpha=0.15)
    if target_mse is not None:
        ax.axhline(target_mse, color="k", ls="--", lw=1.2, alpha=0.7,
                   label=f"Lloyd@convergence ({target_mse:.4g})")
    ax.set_xlabel("epochs / Lloyd iterations")
    ax.set_ylabel("mean squared error (test)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_title(title or "Clustering objective vs epochs", fontsize=11)
    fig.tight_layout()
    return _save(fig, path or os.path.join(FIG_DIR, "mse_vs_epochs.png"))


def plot_mse_vs_time(df, path=None, ycol="test_mse", target_mse=None, title=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        g = sub.groupby("budget")
        t = g["fit_time_s"].median()
        m = g[ycol].median()
        order = np.argsort(t.values)
        st = _style(method)
        ax.plot(t.values[order], m.values[order], label=_label(method), **st)
    if target_mse is not None:
        ax.axhline(target_mse, color="k", ls="--", lw=1.2, alpha=0.7,
                   label=f"Lloyd@convergence ({target_mse:.4g})")
    ax.set_xlabel("wall-clock time (s, optimization only)")
    ax.set_ylabel("mean squared error (test)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_title(title or "Clustering objective vs wall-clock time", fontsize=11)
    fig.tight_layout()
    return _save(fig, path or os.path.join(FIG_DIR, "mse_vs_time.png"))


def plot_runtime_bars(df, budget, path=None, title=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sub = df[df["budget"] == budget]
    methods = sorted(sub["method"].unique())
    med = [sub[sub["method"] == m]["fit_time_s"].median() for m in methods]
    err = [
        sub[sub["method"] == m]["fit_time_s"].quantile(0.75)
        - sub[sub["method"] == m]["fit_time_s"].quantile(0.25)
        for m in methods
    ]
    colors = [_style(m)["color"] for m in methods]
    ax.bar(range(len(methods)), med, yerr=err, color=colors, capsize=4, alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_label(m) for m in methods], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title(title or f"Runtime at {budget} epochs/iterations", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, path or os.path.join(FIG_DIR, "runtime_bars.png"))


def plot_time_to_target(df, path=None, title=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    methods = sorted(df["method"].unique())
    ttt = [df[df["method"] == m]["time_to_target_s"].median() for m in methods]
    reached = [df[df["method"] == m]["reached_target"].any() for m in methods]
    colors = [_style(m)["color"] for m in methods]
    ax.bar(range(len(methods)), ttt, color=colors, alpha=0.85)
    finite = [t for t in ttt if not np.isnan(t)]
    ymax = max(finite) if finite else 1.0
    ax.set_ylim(1e-3, ymax * 3.0)
    for i, (t, r) in enumerate(zip(ttt, reached)):
        if not r or np.isnan(t):
            ax.text(i, 1.5e-3, "not reached", ha="center", va="bottom", fontsize=7, color="#777")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_label(m) for m in methods], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("time to reach Lloyd's converged objective (s)")
    ax.set_yscale("log")
    ax.set_title(title or "Time to target objective", fontsize=11)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    return _save(fig, path or os.path.join(FIG_DIR, "time_to_target.png"))


def plot_scaling_curve(xvals, yvals, labels, xlabel, ylabel, title, path,
                       logx=False, hline=1.0, hline_label="parity (1x)"):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(xvals, yvals, "-o", color="#d62728", lw=2.2, ms=7, zorder=5)
    for x, y, lab in zip(xvals, yvals, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1.2, alpha=0.7, label=hline_label)
        ax.legend(fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")
    return path
