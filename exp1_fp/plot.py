"""Experiment 1: plotting functions.

Generates publication-quality plots for FP benchmarking results.
Each plot type has subplots for each permutation type (reverse, transpose, random).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style & constants
# ---------------------------------------------------------------------------

BASELINE_STYLES = {
    "1d":        {"color": "#555555", "marker": "s", "label": "FSWAP baseline"},
    "ancilla":   {"color": "#1f77b4", "marker": "^", "label": "FP w/ ancillas"},
    "primitive":  {"color": "#ff7f0e", "marker": "D", "label": "Primitive \u0393"},
    "pipelined":  {"color": "#d62728", "marker": "o", "label": "FP w/o ancillas"},
}

PERM_TITLES = {
    "reverse": "Reversal",
    "transpose": "2D Reflection",
    "random": "Random",
}

# Full list (used for data aggregation, keeps primitive in the CSV).
BASELINES_ORDER = ["1d", "ancilla", "primitive", "pipelined"]

# Baselines shown in Exp 1 plots (primitive hidden).
PLOT_BASELINES = ["1d", "ancilla", "pipelined"]


def _setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "font.weight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save_fig(fig, name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("svg",):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"))
    plt.close(fig)


def _get_perm_kinds(df: pd.DataFrame) -> list:
    """Return ordered list of perm kinds present in data."""
    kinds = []
    for k in ["reverse", "transpose", "random"]:
        if k in df["perm_kind"].values:
            kinds.append(k)
    return kinds


def _aggregate(df_sub: pd.DataFrame, y_col: str, perm_kind: str):
    """Aggregate data: for random perms compute mean/std; for structured just values."""
    results = {}
    for baseline in BASELINES_ORDER:
        bdf = df_sub[df_sub["baseline"] == baseline]
        if bdf.empty:
            continue
        if perm_kind == "random":
            grouped = bdf.groupby("L")[y_col]
            results[baseline] = {
                "L": grouped.mean().index.values,
                "mean": grouped.mean().values,
                "std": grouped.std().values,
            }
        else:
            results[baseline] = {
                "L": bdf["L"].values,
                "mean": bdf[y_col].values,
                "std": None,
            }
    return results


def _shared_y_row(axes, row: int, n_cols: int, ylabel: str):
    """Share y-axis across a row: tailor range to the middle subplot,
    show label/ticks only on the leftmost column."""
    mid_ax = axes[row, n_cols // 2]
    y_lo, y_hi = mid_ax.get_ylim()
    for col in range(n_cols):
        axes[row, col].set_ylim(y_lo, y_hi)
        if col == 0:
            axes[row, col].set_ylabel(ylabel)
        else:
            axes[row, col].set_ylabel("")
            axes[row, col].tick_params(labelleft=False, which="both")


# ---------------------------------------------------------------------------
# Depth plot
# ---------------------------------------------------------------------------

def plot_depth_vs_L(df: pd.DataFrame, output_dir: str = "exp1_fp/figures"):
    """CNOT depth vs N = L^2 for each permutation type."""
    _setup_style()
    perm_kinds = _get_perm_kinds(df)

    # Depth is noise-independent; pick one representative p_2q.
    p_2q_val = df["p_2q"].min()
    df_sub = df[df["p_2q"] == p_2q_val]

    n_cols = len(perm_kinds)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4), squeeze=False)

    for col, kind in enumerate(perm_kinds):
        ax = axes[0, col]
        data = _aggregate(df_sub[df_sub["perm_kind"] == kind], "cnot_depth", kind)

        for baseline in PLOT_BASELINES:
            if baseline not in data:
                continue
            d = data[baseline]
            style = BASELINE_STYLES[baseline]
            N_vals = d["L"] ** 2
            if d["std"] is not None:
                ax.errorbar(N_vals, d["mean"], yerr=d["std"],
                            color=style["color"], marker=style["marker"],
                            label=style["label"], capsize=3, linewidth=2, markersize=5)
            else:
                ax.plot(N_vals, d["mean"],
                        color=style["color"], marker=style["marker"],
                        label=style["label"], linewidth=2, markersize=5)

        ax.set_xlabel(r"$N = L^2$")
        ax.set_title(PERM_TITLES[kind])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _shared_y_row(axes, 0, n_cols, "CNOT depth")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    _save_fig(fig, "depth_vs_L", output_dir)


# ---------------------------------------------------------------------------
# Spacetime plot
# ---------------------------------------------------------------------------

def plot_spacetime_vs_N(df: pd.DataFrame, output_dir: str = "exp1_fp/figures"):
    """Spacetime volume vs N = L^2 for each permutation type."""
    _setup_style()
    perm_kinds = _get_perm_kinds(df)
    p_2q_val = df["p_2q"].min()
    df_sub = df[df["p_2q"] == p_2q_val]

    n_cols = len(perm_kinds)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4), squeeze=False)

    for col, kind in enumerate(perm_kinds):
        ax = axes[0, col]
        data = _aggregate(df_sub[df_sub["perm_kind"] == kind], "spacetime_volume", kind)

        for baseline in PLOT_BASELINES:
            if baseline not in data:
                continue
            d = data[baseline]
            style = BASELINE_STYLES[baseline]
            N_vals = d["L"] ** 2
            if d["std"] is not None:
                ax.errorbar(N_vals, d["mean"], yerr=d["std"],
                            color=style["color"], marker=style["marker"],
                            label=style["label"], capsize=3, linewidth=2, markersize=5)
            else:
                ax.plot(N_vals, d["mean"],
                        color=style["color"], marker=style["marker"],
                        label=style["label"], linewidth=2, markersize=5)

        ax.set_xlabel(r"$N = L^2$")
        ax.set_title(PERM_TITLES[kind])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Shared y-axis with scientific notation.
    _shared_y_row(axes, 0, n_cols, "Spacetime volume")
    for col in range(n_cols):
        axes[0, col].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].yaxis.get_offset_text().set_fontsize(9)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    _save_fig(fig, "spacetime_vs_N", output_dir)


# ---------------------------------------------------------------------------
# Stim fidelity plot
# ---------------------------------------------------------------------------

def plot_stim_fidelity(df: pd.DataFrame, output_dir: str = "exp1_fp/figures"):
    """Stim noisy-Clifford fidelity vs N = L^2, one row per noise rate.

    Log-scale y-axis.  Zero-valued fidelities (simulation floor) are dropped
    so that lines end cleanly instead of plunging.  Each row's x-range is
    tailored to the rightmost non-zero datapoint across all baselines.
    """
    _setup_style()
    perm_kinds = _get_perm_kinds(df)
    p_values = sorted(df["p_2q"].unique())
    n_cols = len(perm_kinds)

    fig, axes = plt.subplots(len(p_values), n_cols,
                             figsize=(4.91 * n_cols, 2.31 * len(p_values)),
                             squeeze=False)

    for row, p_2q in enumerate(p_values):
        df_p = df[df["p_2q"] == p_2q]
        max_N_in_row = 0

        for col, kind in enumerate(perm_kinds):
            ax = axes[row, col]
            data = _aggregate(df_p[df_p["perm_kind"] == kind], "stim_fidelity", kind)

            for baseline in PLOT_BASELINES:
                if baseline not in data:
                    continue
                d = data[baseline]
                style = BASELINE_STYLES[baseline]
                N_vals = d["L"] ** 2
                y_vals = d["mean"]
                # Drop zero-valued points (avoid vertical plunge on log scale).
                mask = y_vals > 0
                N_vals, y_vals = N_vals[mask], y_vals[mask]
                if len(N_vals) == 0:
                    continue
                max_N_in_row = max(max_N_in_row, N_vals.max())
                ax.plot(N_vals, y_vals,
                        color=style["color"], marker=style["marker"],
                        label=style["label"], linewidth=2, markersize=5)

            ax.set_yscale("log")
            ax.set_xlabel("")
            ax.set_title(f"{PERM_TITLES[kind]}, p = {p_2q:.0e}")
            ax.grid(True, alpha=0.3)

        # Shared y per row, tailored to middle subplot.
        _shared_y_row(axes, row, n_cols, "Fidelity (Stim)")

        # First row: use plain decimals (0.2, 0.6) instead of scientific
        if row == 0:
            from matplotlib.ticker import FixedLocator, ScalarFormatter
            for col in range(n_cols):
                ax = axes[row, col]
                ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.4, 0.6, 1.0]))
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.get_major_formatter().set_scientific(False)

        # Tailor x-range to effective (non-zero) data for this row.
        if max_N_in_row > 0:
            for col in range(n_cols):
                axes[row, col].set_xlim(left=None, right=max_N_in_row * 1.05)

        axes[row, 0].legend(fontsize=9)

        # "N = L²" at the bottom-right corner, below and right of ticks
        ax_right = axes[row, n_cols - 1]
        ax_right.annotate(
            r"$N\!=\!L^2$", xy=(1, 0), xycoords="axes fraction",
            xytext=(20, -6), textcoords="offset points",
            ha="right", va="top", fontweight="bold",
            fontsize=ax_right.xaxis.get_ticklabels()[0].get_fontsize(),
            annotation_clip=False,
        )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)

    # Align y-axis labels across all rows to the same horizontal position
    label_x = -0.12
    for row in range(len(p_values)):
        axes[row, 0].yaxis.set_label_coords(label_x, 0.5)

    _save_fig(fig, "stim_fidelity", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame, output_dir: str = "exp1_fp/figures"):
    """Generate all Experiment 1 plots."""
    print(f"Generating plots in {output_dir}/")
    plot_depth_vs_L(df, output_dir)
    plot_spacetime_vs_N(df, output_dir)
    plot_stim_fidelity(df, output_dir)
    print("All plots saved.")
