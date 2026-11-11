"""Publication-quality figures (Matplotlib, 300 DPI, colorblind-friendly).

Design rules
------------
* One shared style: white background, subtle grid, consistent typography.
* Colorblind-friendly palette (Okabe-Ito) with shape as a redundant channel.
* Every series is read from the benchmark CSVs; nothing is invented, smoothed
  or interpolated. Figures are the only thing this module changes.
* Point labels use deterministic annotation offsets plus leader lines (the
  scaling figures have <= 5 points, so manual offsets are more reliable and
  fully reproducible than a force-directed layout).
* ``_save`` writes 300 DPI PNG and can also emit PDF/SVG vector exports.
"""
from __future__ import annotations

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")

# --- Okabe-Ito colorblind-friendly palette -------------------------------
VERMILLION = "#D55E00"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
GREY = "#999999"
DARKGREY = "#4D4D4D"
BROWN = "#8C6D31"
INDIGO = "#5A4FCF"

FASTER = "#1B7A3E"
SLOWER = "#B23A2E"
PARITY_C = "#444444"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlepad": 9,
    "axes.labelsize": 10.5,
    "axes.labelpad": 6,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#E2E2E2",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "lines.solid_capstyle": "round",
})

# --- method styling ------------------------------------------------------
STYLE = {
    "e-means": dict(color=VERMILLION, marker="o", lw=2.4, ls="-"),
    "lloyd-faiss": dict(color=BLUE, marker="s", lw=1.9, ls="-"),
    "lloyd-sklearn": dict(color=SKY, marker="^", lw=1.7, ls="--"),
    "lloyd-sklearn-pp": dict(color=GREEN, marker="D", lw=1.9, ls="-"),
    "lloyd-faiss-loop": dict(color=GREY, marker="v", lw=1.5, ls=":"),
    "minibatch": dict(color=ORANGE, marker="P", lw=1.9, ls="-"),
    "e-means-no-cosine": dict(color=PURPLE, marker="X", lw=1.7, ls="--"),
    "e-means-no-revival": dict(color=BROWN, marker="+", lw=1.7, ls="--"),
    "e-means-no-ema": dict(color=INDIGO, marker="*", lw=1.7, ls=":"),
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

SHORT = {
    "e-means": "e-means",
    "lloyd-faiss": "Lloyd (Faiss)",
    "lloyd-sklearn": "Lloyd (random)",
    "lloyd-sklearn-pp": "Lloyd (k-means++)",
    "lloyd-faiss-loop": "Lloyd (loop)",
    "minibatch": "MiniBatch",
    "e-means-no-cosine": "e-means (no cosine)",
    "e-means-no-revival": "e-means (no revival)",
    "e-means-no-ema": "e-means (no EMA)",
}

# dimension -> (marker, color) for the scaling figures
DIM_MARKER = {128: "o", 384: "s", 768: "^"}
DIM_COLOR = {128: BLUE, 384: ORANGE, 768: GREEN}

_TAG_RE = re.compile(r"^(?P<ds>[a-z]+)_n(?P<n>\d+)_d(?P<d>\d+)_k(?P<k>\d+)(?P<suffix>.*)$")


def _style(name):
    return STYLE.get(name, dict(color=DARKGREY, marker=".", lw=1.2, ls="-"))


def _label(name):
    return LABELS.get(name, name)


def _short(name):
    return SHORT.get(name, name)


def _fmt_n(n):
    n = int(n)
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _dataset_title(ds):
    return "arXiv embeddings" if ds == "arxiv" else "synthetic blobs"


def _cell_label(row, with_k=True):
    """Short, human-readable configuration label, e.g. ``80k samples | d=128 | k=1024``."""
    ds = "arXiv " if row["dataset"] == "arxiv" else ""
    label = f"{ds}{_fmt_n(row['n'])} samples | d={int(row['d'])}"
    if with_k:
        label += f" | k={int(row['k'])}"
    return label


def _derive_title(df, kind):
    """Build a readable title from the actual evaluated configuration in ``df``."""
    tag = str(df["cell"].iloc[0]) if "cell" in df.columns else ""
    m = _TAG_RE.match(tag)
    ds = df["dataset"].iloc[0] if "dataset" in df.columns else (m["ds"] if m else "blobs")
    n = int(df["n"].iloc[0]) if "n" in df.columns else int(m["n"])
    d = int(df["d"].iloc[0]) if "d" in df.columns else int(m["d"])
    k = int(df["k"].iloc[0]) if "k" in df.columns else int(m["k"])
    return f"{_dataset_title(ds)} · {_fmt_n(n)} samples · d={d} · k={k} — {kind}"


def _median_iqr(df, method, xcol, ycol):
    sub = df[df["method"] == method]
    g = sub.groupby(xcol)[ycol]
    return g.median().index.values, g.median().values, g.quantile(0.25).values, g.quantile(0.75).values


# --- objective-vs-budget / vs-time --------------------------------------
def plot_mse_vs_epochs(df, path=None, ycol="test_mse", target_mse=None, title=None):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for method in sorted(df["method"].unique()):
        x, med, q1, q3 = _median_iqr(df, method, "budget", ycol)
        order = np.argsort(x)
        st = _style(method)
        ax.plot(x[order], med[order], label=_label(method), zorder=4, **st)
        ax.fill_between(x[order], q1[order], q3[order], color=st["color"], alpha=0.14, zorder=2)
    if target_mse is not None:
        ax.axhline(target_mse, color=PARITY_C, ls="--", lw=1.2, zorder=1,
                   label=f"Lloyd@convergence ({target_mse:.4g})")
    ax.set_xlabel("epochs / Lloyd iterations")
    ax.set_ylabel("mean squared error (test)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_title(title or _derive_title(df, "test MSE vs epochs"))
    return _save(fig, path or os.path.join(FIG_DIR, "mse_vs_epochs.png"))


def plot_mse_vs_time(df, path=None, ycol="test_mse", target_mse=None, title=None):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        g = sub.groupby("budget")
        t = g["fit_time_s"].median()
        m = g[ycol].median()
        order = np.argsort(t.values)
        st = _style(method)
        ax.plot(t.values[order], m.values[order], label=_label(method), zorder=4, **st)
    if target_mse is not None:
        ax.axhline(target_mse, color=PARITY_C, ls="--", lw=1.2, zorder=1,
                   label=f"Lloyd@convergence ({target_mse:.4g})")
    ax.set_xlabel("wall-clock time (s, optimization only)")
    ax.set_ylabel("mean squared error (test)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_title(title or _derive_title(df, "test MSE vs wall-clock time"))
    return _save(fig, path or os.path.join(FIG_DIR, "mse_vs_time.png"))


# --- horizontal bar charts ----------------------------------------------
def plot_runtime_bars(df, budget, path=None, title=None):
    sub = df[df["budget"] == budget]
    methods = sorted(sub["method"].unique(),
                     key=lambda m: sub[sub["method"] == m]["fit_time_s"].median())
    med = [sub[sub["method"] == m]["fit_time_s"].median() for m in methods]
    err = [sub[sub["method"] == m]["fit_time_s"].quantile(0.75)
           - sub[sub["method"] == m]["fit_time_s"].quantile(0.25) for m in methods]
    colors = [_style(m)["color"] for m in methods]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(methods))
    ax.barh(y, med, xerr=err, color=colors, height=0.62, alpha=0.92,
            capsize=3, error_kw=dict(ecolor="#555555", lw=1.1), zorder=3)
    pad = max(med) * 0.02
    for i, (v, e) in enumerate(zip(med, err)):
        ax.text(v + (e if e == e else 0.0) + pad, i, f"{v:.2f} s",
                va="center", ha="left", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock time (s, optimization only)")
    ax.set_xlim(0, (max(med) + max(err)) * 1.16)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_title(title or _derive_title(df, f"runtime at {budget} epochs"))
    return _save(fig, path or os.path.join(FIG_DIR, "runtime_bars.png"))


def plot_time_to_target(df, path=None, title=None):
    target = float(df["target_mse"].iloc[0]) if "target_mse" in df.columns else float("nan")
    methods = sorted(df["method"].unique())
    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        reached = bool(sub["reached_target"].any())
        t = float(sub["time_to_target_s"].median()) if reached else float("nan")
        rows.append((m, t, reached))
    rows.sort(key=lambda r: (not r[2], r[1] if r[1] == r[1] else np.inf))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    finite = [t for _, t, r in rows if r and t == t]
    xlo = min(finite) * 0.55 if finite else 0.1
    xhi = max(finite) * 1.95 if finite else 1.0
    for i, (m, t, reached) in enumerate(rows):
        if reached:
            ax.barh(i, t, color=_style(m)["color"], height=0.62, alpha=0.92, zorder=3)
            ax.text(t * 1.07, i, f"{t:.2f} s", va="center", ha="left", fontsize=8.5)
        else:
            ax.text(np.sqrt(xlo * xhi), i, "not reached within the swept epochs",
                    va="center", ha="center", fontsize=8.5, color="#777777", style="italic")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([_short(m) for m, _, _ in rows])
    ax.invert_yaxis()  # fastest on top
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("time to reach the converged-Lloyd objective (s, log scale)")
    ax.grid(True, axis="x", which="major")
    ax.grid(True, axis="x", which="minor", alpha=0.35)
    ax.grid(False, axis="y")
    base = title or _derive_title(df, "time to target objective")
    if target == target:
        base += f"\ntarget = converged Lloyd reference ({target:.5g})"
    ax.set_title(base, fontsize=11.5)
    return _save(fig, path or os.path.join(FIG_DIR, "time_to_target.png"))


# --- scaling figures -----------------------------------------------------
def plot_speedup_vs_k(table, path=None, title=None, exports=("png", "pdf")):
    """Figure A: CPU speedup vs cluster count (log x, log y).

    Two configurations share ``k = 64``; they are distinguished by marker/colour
    (dimension) and separated with leader-line annotations rather than by
    moving the points, so every plotted value is exact.
    """
    syn = table[table["dataset"] == "blobs"].copy().sort_values(["n", "d", "k"])
    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    ax.axhline(1.0, color=PARITY_C, ls="--", lw=1.2, zorder=1)

    # Points are plotted individually: the configurations differ in (n, d, k),
    # so a line through them would imply a trend between unrelated experiments.
    for d, g in syn.groupby("d"):
        d = int(d)
        ax.scatter(g["k"], g["speedup"], marker=DIM_MARKER.get(d, "o"), s=150,
                   color=DIM_COLOR.get(d, DARKGREY), edgecolor="white", linewidth=1.0,
                   zorder=5, label=f"d = {d}")

    # deterministic annotation offsets (points, ha, va) keyed by (n, d, k)
    offsets = {
        (8000, 128, 64): (16, 3, "left", "center"),
        (80000, 128, 64): (16, -3, "left", "center"),
        (80000, 384, 128): (16, 3, "left", "center"),
        (80000, 128, 1024): (-16, -6, "right", "top"),
    }
    for _, r in syn.iterrows():
        key = (int(r["n"]), int(r["d"]), int(r["k"]))
        dx, dy, ha, va = offsets.get(key, (12, 12, "left", "bottom"))
        text = f"{_fmt_n(r['n'])} samples\nd = {int(r['d'])}"
        ax.annotate(
            text, xy=(r["k"], r["speedup"]), xytext=(dx, dy),
            textcoords="offset points", ha=ha, va=va, fontsize=8.2,
            color="#222222", zorder=6,
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8,
                            shrinkA=1, shrinkB=5, connectionstyle="arc3,rad=0"),
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024])
    ax.set_xticklabels(["64", "128", "256", "512", "1024"])
    ax.set_xlim(54, 1350)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "4"])
    ax.set_ylim(0.24, 6.0)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("number of clusters k (log scale)")
    ax.set_ylabel("CPU speedup vs Lloyd\n(time to reach Lloyd's converged loss)")

    # parity guidance, kept inside the axes
    ax.text(0.985, 0.62, "above 1x: e-means faster", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=FASTER)
    ax.text(0.985, 0.38, "below 1x: e-means slower", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=SLOWER)
    ax.text(0.985, 0.455, "1x parity", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=PARITY_C)
    ax.legend(title="embedding dimension", loc="upper left", framealpha=0.95)
    ax.set_title(title or "e-means CPU speedup across cluster counts (synthetic)")
    return _save(fig, path or os.path.join(FIG_DIR, "scaling_speedup_vs_k.png"), exports=exports)


def plot_speedup_overview(table, path=None, title=None, exports=("png", "pdf")):
    """Figure B: categorical horizontal dot plot of speedup per configuration.

    A categorical layout is used because three configurations share ``n = 80000``;
    a speedup-vs-n line would imply a trend between unrelated configurations.
    """
    t = table.copy()
    t["label"] = t.apply(_cell_label, axis=1)
    t = t.sort_values("speedup").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = np.arange(len(t))
    xlo = 0.26
    for i, r in t.iterrows():
        d = int(r["d"])
        color = DIM_COLOR.get(d, DARKGREY)
        ax.hlines(i, xlo, r["speedup"], color="#DDDDDD", lw=1.3, zorder=1)
        ax.scatter(r["speedup"], i, marker=DIM_MARKER.get(d, "o"), s=160,
                   color=color, edgecolor="white", linewidth=1.0, zorder=5)
        faster = r["speedup"] >= 1.0
        ax.text(r["speedup"] * 1.09, i, f"{r['speedup']:.2f}x", va="center", ha="left",
                fontsize=9.5, color=FASTER if faster else SLOWER,
                fontweight="bold" if faster else "normal")

    ax.axvline(1.0, color=PARITY_C, ls="--", lw=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(t["label"])
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "4"])
    ax.set_xlim(xlo, 9.5)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("CPU speedup vs Lloyd to reach Lloyd's converged loss (log scale)")
    ax.grid(True, axis="x", which="major")
    ax.grid(True, axis="x", which="minor", alpha=0.35)
    ax.grid(False, axis="y")
    ax.text(1.06, -0.45, "1x parity", color=PARITY_C, fontsize=8.5, va="center")

    handles = [plt.Line2D([], [], marker=DIM_MARKER[d], color="none",
                          markerfacecolor=DIM_COLOR[d], markeredgecolor="white",
                          markersize=9, label=f"d = {d}") for d in sorted(DIM_MARKER)]
    ax.legend(handles=handles, title="embedding dimension", loc="lower right", framealpha=0.95)
    ax.set_title(title or "e-means CPU speedup by benchmark configuration")
    return _save(fig, path or os.path.join(FIG_DIR, "scaling_speedup_overview.png"), exports=exports)


def _save(fig, path, exports=("png",)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    written = []
    for fmt in exports:
        out = f"{base}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
        print(f"figure -> {out}")
    plt.close(fig)
    return written[0]
