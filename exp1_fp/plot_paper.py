"""Column-width paper figures for Experiment 1 (FIG 9 and FIG 10).

Why this module exists
----------------------
``exp1_fp/plot.py`` authors its figures on a 12.6 in canvas -- three panels
across at 15-19 pt type -- which is right for reading on screen and wrong for
the manuscript.  Composed and included in a 246 pt column that artwork prints
at scale 0.266, so its tick labels land at 3.4-3.9 pt and its 2 pt strokes at
0.53 pt.  Widening the float to ``figure*`` fixes the type but spends the full
text width on two figures.

This module authors the same content at the printed width instead, keeping the
original HORIZONTAL panel grid: the three permutation families run across as
columns, the two quantities (or the two error rates) as rows.  Three panels
across a 246 pt column is only workable because the axes are shared -- y within
each row, x down each column -- so tick labels appear once per row and once per
column instead of in all six panels.  ``build`` takes the target width, so the
same code emits the column-width and full-width variants.

Data preparation is NOT reimplemented -- ``_aggregate``, ``_get_perm_kinds``
and ``_plot_baselines`` are imported from ``plot.py``, so both figure families
derive their numbers from exactly one code path.  Only the drawing differs.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from exp1_fp.plot import (
    BASELINES_ORDER,
    BASELINE_STYLES,
    PERM_TITLES,
    STIM_PLOT_P_VALUES,
    _aggregate,
    _get_perm_kinds,
    _plot_baselines,
    _validate_native_data,
)

# --- print geometry --------------------------------------------------------
# revtex4-2 single column is 246 pt = 3.417 in.  Authoring here means the sizes
# below are the sizes the reader gets.
COL_WIDTH_IN = 3.417          # revtex4-2 single column, 246pt
FULL_WIDTH_IN = 7.083         # revtex4-2 full text width, 510pt
ROW_ASPECT = 0.74             # panel height / panel width
HEADROOM_IN = 0.50            # shared key strip above the panels

# --- type ladder -----------------------------------------------------------
# The same three sizes the TikZ figures use: 8 pt titles, 7 pt labels and
# ticks, 6.5 pt keys and annotations.
FS_TITLE, FS_LABEL, FS_TICK, FS_KEY = 8.0, 7.0, 7.0, 6.5

# --- stroke ladder ---------------------------------------------------------
# Identical rungs to paper/figures/src/gamma-fig-style.tex, fixed against the
# manuscript's 0.667 pt body-text stem.
W_HAIR, W_DATA, W_EMPH = 0.55, 0.70, 0.90
MARKER_PT = 2.4               # small enough for a 105 pt panel, large enough
                              # that the five baselines stay distinguishable


def _rc():
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_KEY,
        "axes.linewidth": W_HAIR,
        "xtick.major.width": W_HAIR,
        "ytick.major.width": W_HAIR,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "grid.linewidth": W_HAIR,
        "grid.alpha": 0.30,
        "lines.linewidth": W_DATA,
        "lines.markersize": MARKER_PT,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.012,
        "axes.formatter.use_mathtext": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# --- shot-noise gate for the Stim fidelity series ---------------------------
# stim_fidelity is a Monte-Carlo estimate from stim_shots samples, so its
# relative standard error is sqrt((1-F)/(n F)): with 10^6 shots a fidelity of
# 1e-6 rests on a single surviving shot and one of 3e-6 on three.  Those points
# sit at the bottom of the p = 1e-4 row and bend the apparent slope, so they are
# withheld rather than drawn.
#
# Threshold: at least MIN_SUCCESSES = 100 surviving shots, i.e. relative
# standard error at or below 10%.  Each structured point is one Stim run and
# each random point is the mean over 20 permutations.  With the process-fidelity
# estimator (sampling model v3) no surviving random point has a replicate with
# zero surviving shots; the worst spread across the 20 replicates at a surviving
# point is 60% of its mean, at the very tail of the p = 1e-4 row.
#
# Checked against the full table: the points failing this test form a contiguous
# TAIL in N for every curve (fidelity is monotone decreasing in N), so each
# series is truncated rather than punctured -- no gaps, and nothing is dropped
# from the interior.  It withholds 27 of the 306 plotted points, all in the
# p = 1e-4 row, the earliest at N = 225.
MIN_SUCCESSES = 100.0


def _shot_noise_cutoff(df_sub: pd.DataFrame, baseline: str) -> float:
    """Smallest N at which this baseline's fidelity estimate is unreliable.

    Returns +inf when every point clears the threshold.
    """
    b = df_sub[df_sub["baseline"] == baseline]
    if b.empty or "stim_shots" not in b.columns:
        return float("inf")
    per_n = b.groupby("N").apply(
        lambda d: float(d["stim_fidelity"].mean()) * float(d["stim_shots"].sum()),
        include_groups=False,
    )
    bad = per_n.index[per_n < MIN_SUCCESSES]
    return float(bad.min()) if len(bad) else float("inf")


def _truncate(data: dict, df_sub: pd.DataFrame) -> dict:
    """Drop each series' shot-noise-limited tail, in place of drawing it."""
    out = {}
    for baseline, d in data.items():
        cut = _shot_noise_cutoff(df_sub, baseline)
        if cut == float("inf"):
            out[baseline] = d
            continue
        keep = (np.asarray(d["L"]) ** 2) < cut
        out[baseline] = {
            "L": np.asarray(d["L"])[keep],
            "mean": np.asarray(d["mean"])[keep],
            "std": None if d["std"] is None else np.asarray(d["std"])[keep],
            **{k: np.asarray(d[k])[keep] for k in ("ci_lo", "ci_hi") if k in d},
        }
    return out


def _series(ax, data, logy=False):
    """Draw one panel's baselines.  Returns handles/labels for a shared key."""
    handles, labels = [], []
    for baseline in _plot_baselines(data):
        if baseline not in data:
            continue
        d = data[baseline]
        style = BASELINE_STYLES[baseline]
        x = d["L"] ** 2
        kw = dict(color=style["color"], marker=style["marker"],
                  linewidth=W_DATA, markersize=MARKER_PT,
                  markeredgewidth=W_HAIR, label=style["label"])
        if d["std"] is not None:
            h = ax.errorbar(x, d["mean"], yerr=d["std"], capsize=1.2,
                            elinewidth=W_HAIR, **kw)
        else:
            (h,) = ax.plot(x, d["mean"], **kw)
        handles.append(h)
        labels.append(style["label"])
    if logy:
        ax.set_yscale("log")
        # A log axis spanning under a decade gets minor labels such as
        # "6 x 10^-1"; in a shared-y row they are redundant clutter.
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        # Spacetime volume reaches 3e5; six-digit tick labels cost more panel
        # width than the curves do.  Factor the decade into an axis multiplier.
        ax.yaxis.set_major_formatter(
            mticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(style="sci", scilimits=(-2, 3), axis="y")
        ax.yaxis.get_offset_text().set_fontsize(FS_KEY)
    # Gridlines stay: the reader has to be able to recover values off these axes.
    ax.grid(True, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return handles, labels


def _shared_key(fig, handles, labels, ncol):
    """One key for the whole figure, above the panels.

    The screen version repeats an identical key inside every panel; at column
    width that costs a quarter of each panel's data area.
    """
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    fig.legend(h2, l2, loc="upper center", ncol=ncol, frameon=False,
               fontsize=FS_KEY, handlelength=1.6, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 1.0))


def _grid(width_in, n_rows, n_cols):
    """A horizontal panel grid at the given printed width."""
    panel_w = width_in / n_cols
    h = n_rows * panel_w * ROW_ASPECT + HEADROOM_IN
    return plt.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey="row",
                        figsize=(width_in, h))


def figure_depth_spacetime(df: pd.DataFrame, out: str,
                           width_in: float = COL_WIDTH_IN,
                           suffix: str = "") -> str:
    """FIG 9: CNOT depth and spacetime volume, one column per permutation."""
    _rc()
    kinds = _get_perm_kinds(df)
    df_sub = df[df["p_2q"] == df["p_2q"].min()]
    quantities = [("cnot_depth", "CNOT depth"),
                  ("spacetime_volume", "spacetime vol.")]
    fig, axes = _grid(width_in, len(quantities), len(kinds))
    H, L_ = [], []
    for row, (ycol, ylabel) in enumerate(quantities):
        for col, kind in enumerate(kinds):
            ax = axes[row, col]
            h, l = _series(ax, _aggregate(df_sub[df_sub["perm_kind"] == kind],
                                          ycol, kind))
            H += h
            L_ += l
            if row == 0:
                # pad clears the axis multiplier, which matplotlib draws above
                # the axes box and which otherwise displaces the title
                ax.set_title(PERM_TITLES[kind], pad=2.5)
            if row == len(quantities) - 1:
                ax.set_xlabel(r"$N = L^2$", labelpad=1.5)
        axes[row, 0].set_ylabel(ylabel, labelpad=2.0)
        # The caption and the prose address the rows as (a) and (b), but the
        # figure never carried the letters -- the reader had to infer which
        # row was which.  A bold letter at the outer top-left corner of each
        # row, in the axis-label size, matches how the FFFT/SYK panels are
        # lettered in their titles.
        axes[row, 0].text(-0.36, 1.02, f"({'ab'[row]})", transform=axes[row, 0].transAxes,
                          ha="left", va="bottom", fontsize=FS_LABEL, fontweight="bold")
    _shared_key(fig, H, L_, ncol=3)
    fig.subplots_adjust(left=0.128, right=0.988, top=0.845, bottom=0.135,
                        hspace=0.22, wspace=0.10)
    path = os.path.join(out, f"FP-exp1_depth_spacetime{suffix}.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def _aggregate_ci(df_sub: pd.DataFrame, perm_kind: str) -> dict:
    """Mean process fidelity per (baseline, L) with a 95% finite-shot interval.

    Each row carries a Wilson interval from its own 10^6 shots.  For the random
    family the plotted point is the mean over the 20 instances, so the shot-noise
    half-width is propagated through that mean: hw = sqrt(sum hw_i^2) / n.  The
    instance-to-instance spread is NOT drawn here; it is a property of the
    permutation family, not of the estimate.
    """
    out = {}
    for baseline in BASELINES_ORDER:
        b = df_sub[df_sub["baseline"] == baseline]
        if b.empty:
            continue
        hw = (b["stim_fidelity_ci_hi"] - b["stim_fidelity_ci_lo"]) / 2.0
        b = b.assign(_hw2=hw ** 2)
        g = b.groupby("L")
        n = g.size()
        mean = g["stim_fidelity"].mean()
        hw_mean = np.sqrt(g["_hw2"].sum()) / n
        out[baseline] = {
            "L": mean.index.values,
            "mean": mean.values,
            "std": None,
            "ci_lo": np.clip(mean.values - hw_mean.values, 0.0, 1.0),
            "ci_hi": np.clip(mean.values + hw_mean.values, 0.0, 1.0),
        }
    return out


def _ci_bars(ax, fig, data, marker_pt: float):
    """Draw 95% intervals only where they extend beyond the marker.

    The test is done in display space after the axis scale is set, so the same
    rule applies on the linear and the logarithmic rows.  A bar that would be
    hidden under its marker carries no information and is omitted.
    """
    fig.canvas.draw()
    r_px = 0.5 * marker_pt * fig.dpi / 72.0
    for baseline in _plot_baselines(data):
        d = data.get(baseline)
        if d is None or "ci_lo" not in d:
            continue
        x = np.asarray(d["L"], dtype=float) ** 2
        y, lo, hi = (np.asarray(d[k], dtype=float) for k in ("mean", "ci_lo", "ci_hi"))
        ok = (y > 0) & (lo > 0) & (hi > 0)
        if not ok.any():
            continue
        to_px = ax.transData.transform
        py = to_px(np.c_[x[ok], y[ok]])[:, 1]
        plo = to_px(np.c_[x[ok], lo[ok]])[:, 1]
        phi = to_px(np.c_[x[ok], hi[ok]])[:, 1]
        show = np.maximum(py - plo, phi - py) > r_px
        if not show.any():
            continue
        xs, ys = x[ok][show], y[ok][show]
        yerr = np.vstack([ys - lo[ok][show], hi[ok][show] - ys])
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=BASELINE_STYLES[baseline]["color"],
                    elinewidth=W_HAIR, capsize=1.2, capthick=W_HAIR, zorder=2.5)


def figure_stim(df: pd.DataFrame, out: str,
                width_in: float = COL_WIDTH_IN,
                suffix: str = "") -> str:
    """FIG 11: Stim process fidelity at two error rates, one column per permutation,
    with 95% finite-shot intervals where they exceed the marker."""
    import math

    _rc()
    kinds = _get_perm_kinds(df)
    p_values = list(STIM_PLOT_P_VALUES)
    fig, axes = _grid(width_in, len(p_values), len(kinds))
    H, L_ = [], []
    for row, p_2q in enumerate(p_values):
        for col, kind in enumerate(kinds):
            ax = axes[row, col]
            sub = df[(df["p_2q"] == p_2q) & (df["perm_kind"] == kind)]
            agg = _truncate(_aggregate_ci(sub, kind), sub)
            h, l = _series(ax, agg, logy=True)
            _ci_bars(ax, fig, agg, MARKER_PT)
            H += h
            L_ += l
            if row == 0:
                ax.set_title(PERM_TITLES[kind], pad=2.5)
            if row == len(p_values) - 1:
                ax.set_xlabel(r"$N = L^2$", labelpad=1.5)
        axes[row, 0].set_ylabel(
            rf"fidelity, $p = 10^{{{int(round(math.log10(p_2q)))}}}$", labelpad=2.0)
    _shared_key(fig, H, L_, ncol=3)
    fig.subplots_adjust(left=0.148, right=0.988, top=0.845, bottom=0.135,
                        hspace=0.22, wspace=0.10)
    path = os.path.join(out, f"FP-exp1_stim_fidelity{suffix}.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="exp1_fp/results/data.csv")
    ap.add_argument("--paper-fig-dir", required=True)
    ap.add_argument("--width", choices=("column", "full"), default="column")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    w = COL_WIDTH_IN if args.width == "column" else FULL_WIDTH_IN
    df = pd.read_csv(args.data)
    _validate_native_data(df)
    os.makedirs(args.paper_fig_dir, exist_ok=True)
    for f in (figure_depth_spacetime, figure_stim):
        print(f(df, args.paper_fig_dir, width_in=w, suffix=args.suffix))


if __name__ == "__main__":
    main()
