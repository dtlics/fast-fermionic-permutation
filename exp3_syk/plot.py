"""Experiment 3: plotting functions for sparse SYK Trotter step benchmarking.

Generates 4 plots:
1. Spacetime volume vs N
2. Estimated fidelity vs N (log scale, per-panel y-axis)
3. Depth breakdown trend (methods + shared interaction line, vs N=L^2)
4. Number of colors vs N (multiple k values)
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_STYLES = {
    "naive_pauli": {"color": "#2ca02c", "marker": "v", "label": "Naive Pauli"},
    "1d":          {"color": "#555555", "marker": "s", "label": "FSWAP baseline"},
    "ancilla":     {"color": "#1f77b4", "marker": "^", "label": "FP w/ ancillas"},
    "primitive":   {"color": "#ff7f0e", "marker": "D", "label": r"Primitive $\Gamma$"},
    "pipelined":   {"color": "#d62728", "marker": "o", "label": "FP w/o ancillas"},
}

BASELINES_ORDER = ["naive_pauli", "1d", "ancilla", "primitive", "pipelined"]

# Baselines shown in plots (primitive hidden).
PLOT_BASELINES = ["naive_pauli", "1d", "ancilla", "pipelined"]


def _setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save_fig(fig, name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("svg",):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"))
    plt.close(fig)


def _aggregate(df_sub: pd.DataFrame, y_col: str):
    """Aggregate over instances: compute mean and std grouped by (baseline, N)."""
    results = {}
    for baseline in BASELINES_ORDER:
        bdf = df_sub[df_sub["baseline"] == baseline]
        if bdf.empty:
            continue
        grouped = bdf.groupby("N")[y_col]
        results[baseline] = {
            "N": grouped.mean().index.values,
            "mean": grouped.mean().values,
            "std": grouped.std().values,
        }
    return results


def _aggregate_by_L(df_sub: pd.DataFrame, y_col: str):
    """Aggregate over instances grouped by (baseline, L)."""
    results = {}
    for baseline in BASELINES_ORDER:
        bdf = df_sub[df_sub["baseline"] == baseline]
        if bdf.empty:
            continue
        grouped = bdf.groupby("L")[y_col]
        results[baseline] = {
            "L": grouped.mean().index.values,
            "mean": grouped.mean().values,
            "std": grouped.std().values,
        }
    return results


def _primary_k(df: pd.DataFrame) -> float:
    """Return the primary (most common) k value."""
    return df["k"].mode().iloc[0]


def plot_spacetime_vs_N(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Plot spacetime volume vs N (log-log), focusing on larger system sizes."""
    _setup_style()
    k_val = _primary_k(df)
    df_sub = df[df["k"] == k_val]
    df_sub = df_sub[df_sub["p_2q"] == df_sub["p_2q"].min()]

    # Focus on N >= 36 (L >= 6) for clearer scaling
    df_sub = df_sub[df_sub["N"] >= 36]

    data = _aggregate(df_sub, "spacetime_volume")

    fig, ax = plt.subplots(figsize=(7, 5))
    for baseline in PLOT_BASELINES:
        if baseline not in data:
            continue
        d = data[baseline]
        style = BASELINE_STYLES[baseline]
        ax.errorbar(d["N"], d["mean"], yerr=d["std"],
                     color=style["color"], marker=style["marker"],
                     label=style["label"], capsize=3, linewidth=1.5)

    # Crossover vertical line where FP w/o ancillas becomes best
    N_cross = _find_pipelined_crossover_N(data, lower_is_better=True)
    if N_cross is not None:
        _draw_crossover(ax, N_cross, ax.get_ylim()[0])

    # Asymptotic guide lines
    Ns = np.array(sorted(df_sub["N"].unique()), dtype=float)
    ax.loglog(Ns, 50.0 * Ns**2, "--", color="gray", alpha=0.4,
              label=r"$\propto N^2$")
    ax.loglog(Ns, 700.0 * Ns**1.5, ":", color="gray", alpha=0.4,
              label=r"$\propto N\sqrt{N}$")

    ax.set_xlabel(r"$N = L^2$")
    ax.set_ylabel(r"Spacetime volume (qubits $\times$ CNOT depth)")
    ax.set_title(f"Spacetime Volume Per Trotter Step (k={k_val})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "spacetime_vs_N", output_dir)


def _find_pipelined_crossover_N(data: dict, lower_is_better: bool = False) -> float | None:
    """Find the first N where 'pipelined' becomes the best method.

    lower_is_better=False: pipelined must have the highest value (fidelity).
    lower_is_better=True:  pipelined must have the lowest value (depth/spacetime).
    """
    if "pipelined" not in data:
        return None
    others = [bl for bl in PLOT_BASELINES if bl != "pipelined" and bl in data]
    if not others:
        return None

    pip = data["pipelined"]
    for N, val in zip(pip["N"], pip["mean"]):
        best = True
        for bl in others:
            d = data[bl]
            idx = np.where(d["N"] == N)[0]
            if len(idx) == 0:
                continue
            other_val = d["mean"][idx[0]]
            if lower_is_better:
                if other_val <= val:
                    best = False
                    break
            else:
                if other_val >= val:
                    best = False
                    break
        if best:
            return float(N)
    return None


def _draw_crossover(ax, N_cross: float, y_pos: float):
    """Draw a vertical crossover guide line with N label."""
    ax.axvline(N_cross, color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"N={int(N_cross)}", xy=(N_cross, y_pos),
                fontsize=7, color="gray", ha="center", va="bottom",
                xytext=(0, 2), textcoords="offset points")


def plot_fidelity_vs_N(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Plot fidelity vs N (log-log), panels for p=1e-5 and p=1e-4.

    y-axis is tailored using the p_2q=1e-5 panel as reference.
    Curves that drop below y_floor stop early.
    Vertical crossover line where FP w/o ancillas becomes the best method.
    """
    _setup_style()
    k_val = _primary_k(df)
    df_k = df[df["k"] == k_val]
    # Only show p=1e-5 and p=1e-4 (drop 1e-3)
    p_values = sorted(p for p in df_k["p_2q"].unique() if p <= 1e-4)

    fp_baselines = ["1d", "ancilla", "pipelined"]

    fig, axes = plt.subplots(1, len(p_values),
                              figsize=(5 * len(p_values), 4.5),
                              squeeze=False, sharey=True)

    # y-floor from the p_2q=1e-5 panel (reference case)
    ref_sub = df_k[df_k["p_2q"] == 1e-5]
    ref_data = _aggregate(ref_sub, "mult_fidelity")
    fp_min = 1.0
    for bl in fp_baselines:
        if bl in ref_data:
            vals = ref_data[bl]["mean"]
            positive = vals[vals > 0]
            if len(positive) > 0:
                fp_min = min(fp_min, positive.min())
    y_floor = max(fp_min * 0.1, 1e-300)

    for col, p_2q in enumerate(p_values):
        ax = axes[0, col]
        df_sub = df_k[df_k["p_2q"] == p_2q]
        data = _aggregate(df_sub, "mult_fidelity")

        for baseline in PLOT_BASELINES:
            if baseline not in data:
                continue
            d = data[baseline]
            style = BASELINE_STYLES[baseline]
            mask = d["mean"] >= y_floor
            if not mask.any():
                continue
            ax.plot(d["N"][mask], d["mean"][mask],
                    color=style["color"], marker=style["marker"],
                    label=style["label"], linewidth=1.5, markersize=5)

        # Crossover vertical line where FP w/o ancillas becomes best
        N_cross = _find_pipelined_crossover_N(data)
        if N_cross is not None:
            _draw_crossover(ax, N_cross, y_floor)

        ax.set_xlabel(r"$N = L^2$")
        if col == 0:
            ax.set_ylabel("Fidelity estimate")
        ax.set_title(f"k={k_val}, p = {p_2q:.0e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, which="major", alpha=0.3)

    axes[0, 0].set_ylim(bottom=y_floor, top=2.0)
    fig.suptitle(r"Estimated Fidelity $(1-p)^G$ — Per Trotter Step", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    _save_fig(fig, "fidelity_vs_N", output_dir)


def plot_depth_breakdown_trend(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Methods' total depth vs N=L^2, with shared interaction-only line.

    Solid lines = total CNOT depth per method.
    Dashed gray line = interaction depth (shared across all FP methods).
    For naive Pauli, the total IS the interaction depth (no FP).
    """
    _setup_style()
    k_val = _primary_k(df)
    df_k = df[df["k"] == k_val]
    p_min = df_k["p_2q"].min()
    df_sub = df_k[df_k["p_2q"] == p_min]

    # Focus on N >= 36 (L >= 6) for clearer scaling
    df_sub = df_sub[df_sub["N"] >= 36]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plot total depth for each method
    data_total = _aggregate(df_sub, "cnot_depth")
    for baseline in PLOT_BASELINES:
        if baseline not in data_total:
            continue
        d = data_total[baseline]
        style = BASELINE_STYLES[baseline]
        ax.errorbar(d["N"], d["mean"], yerr=d["std"],
                     color=style["color"], marker=style["marker"],
                     label=style["label"], capsize=3, linewidth=1.8)

    # Plot shared interaction depth (use any FP baseline, they're identical)
    fp_baseline = "pipelined"
    bdf = df_sub[df_sub["baseline"] == fp_baseline]
    if not bdf.empty:
        grouped = bdf.groupby("L")["interaction_cnot_depth"]
        L_vals = grouped.mean().index.values
        N_vals = L_vals ** 2
        int_mean = grouped.mean().values
        int_std = grouped.std().values
        ax.errorbar(N_vals, int_mean, yerr=int_std,
                     color="#888888", marker="x", linestyle="--",
                     label="Local Rotations", capsize=3, linewidth=1.5,
                     alpha=0.8)

    # Crossover vertical line where FP w/o ancillas becomes best
    N_cross = _find_pipelined_crossover_N(data_total, lower_is_better=True)
    if N_cross is not None:
        _draw_crossover(ax, N_cross, ax.get_ylim()[0])

    # Asymptotic guide lines
    Ns = np.array(sorted(df_sub["N"].unique()), dtype=float)
    ax.loglog(Ns, 60.0 * Ns, "--", color="gray", alpha=0.4,
              label=r"$\propto N$")
    ax.loglog(Ns, 800.0 * np.sqrt(Ns), ":", color="gray", alpha=0.4,
              label=r"$\propto \sqrt{N}$")

    ax.set_xlabel(r"$N = L^2$")
    ax.set_ylabel("CNOT Depth Per Trotter Step")
    ax.set_title(f"Depth Breakdown Trend (k={k_val})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "depth_breakdown_trend", output_dir)


def plot_colors_vs_N(
    df: pd.DataFrame,
    output_dir: str = "exp3_syk/figures",
    colors_df: Optional[pd.DataFrame] = None,
):
    """Number of colors used vs N for multiple k values.

    If colors_df is provided, it is used instead of df for this plot.
    colors_df should have columns: L, N, k, instance_idx, n_colors.
    """
    _setup_style()
    src = colors_df if colors_df is not None else df

    fig, ax = plt.subplots(figsize=(6, 4))

    for k_val in sorted(src["k"].unique()):
        df_k = src[src["k"] == k_val]
        df_unique = df_k.drop_duplicates(subset=["L", "k", "instance_idx"])
        grouped = df_unique.groupby("N")["n_colors"]
        N_vals = grouped.mean().index.values
        means = grouped.mean().values
        stds = grouped.std().values

        ax.errorbar(N_vals, means, yerr=stds,
                     marker="o", capsize=3, linewidth=1.5,
                     label=f"k = {k_val}")

    ax.set_xlabel("N (qubits)")
    ax.set_ylabel("Number of colors")
    ax.set_title("Parent-Level Graph Coloring")
    ax.legend()
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "colors_vs_N", output_dir)


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str = "exp3_syk/figures",
    colors_df: Optional[pd.DataFrame] = None,
):
    """Generate all Experiment 3 plots."""
    print(f"Generating plots in {output_dir}/")
    plot_spacetime_vs_N(df, output_dir)
    plot_fidelity_vs_N(df, output_dir)
    plot_depth_breakdown_trend(df, output_dir)
    plot_colors_vs_N(df, output_dir, colors_df=colors_df)
    print("All plots saved.")
