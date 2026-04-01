"""Experiment 3: plotting functions for sparse SYK Trotter step benchmarking.

Generates 4 plots:
1. Spacetime volume vs N
2. Estimated fidelity vs N (log scale, per-panel y-axis)
3. Depth breakdown trend (all 5 methods + shared interaction line, vs L)
4. Number of colors vs N (multiple k values)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_STYLES = {
    "naive_pauli": {"color": "#2ca02c", "marker": "v", "label": "Naive Pauli (no FP)"},
    "1d":          {"color": "#555555", "marker": "s", "label": "1D (OET sort)"},
    "ancilla":     {"color": "#1f77b4", "marker": "^", "label": r"Ancilla $\Gamma$"},
    "primitive":   {"color": "#ff7f0e", "marker": "D", "label": r"Primitive $\Gamma$"},
    "pipelined":   {"color": "#d62728", "marker": "o", "label": r"Pipelined $\Gamma$"},
}

BASELINES_ORDER = ["naive_pauli", "1d", "ancilla", "primitive", "pipelined"]


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
    for ext in ("pdf", "png"):
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
    """Plot spacetime volume vs N for all 5 baselines."""
    _setup_style()
    k_val = _primary_k(df)
    df_sub = df[df["k"] == k_val]
    df_sub = df_sub[df_sub["p_2q"] == df_sub["p_2q"].min()]

    data = _aggregate(df_sub, "spacetime_volume")

    fig, ax = plt.subplots(figsize=(7, 5))
    for baseline in BASELINES_ORDER:
        if baseline not in data:
            continue
        d = data[baseline]
        style = BASELINE_STYLES[baseline]
        ax.errorbar(d["N"], d["mean"], yerr=d["std"],
                     color=style["color"], marker=style["marker"],
                     label=style["label"], capsize=3, linewidth=1.5)

    ax.set_xlabel("N (qubits)")
    ax.set_ylabel(r"Spacetime volume (qubits $\times$ CNOT depth)")
    ax.set_title(f"Spacetime Volume vs System Size (k={k_val})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "spacetime_vs_N", output_dir)


def plot_fidelity_vs_N(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Plot fidelity vs N on log scale, one subplot per noise rate.

    y-axis is tuned per panel to show FP-method separation clearly.
    Naive Pauli naturally clips when it falls below the floor.
    No error bars (they distort heavily on log scale).
    """
    _setup_style()
    k_val = _primary_k(df)
    df_k = df[df["k"] == k_val]
    p_values = sorted(df_k["p_2q"].unique())

    fp_baselines = ["1d", "ancilla", "primitive", "pipelined"]

    fig, axes = plt.subplots(1, len(p_values),
                              figsize=(5 * len(p_values), 4.5), squeeze=False)

    for col, p_2q in enumerate(p_values):
        ax = axes[0, col]
        df_sub = df_k[df_k["p_2q"] == p_2q]
        data = _aggregate(df_sub, "mult_fidelity")

        # Determine y-floor from the FP baselines' minimum mean value
        fp_min = 1.0
        for bl in fp_baselines:
            if bl in data:
                vals = data[bl]["mean"]
                positive = vals[vals > 0]
                if len(positive) > 0:
                    fp_min = min(fp_min, positive.min())
        # Set floor 2 decades below the FP minimum, but not below 1e-300
        y_floor = max(fp_min * 1e-2, 1e-300)

        for baseline in BASELINES_ORDER:
            if baseline not in data:
                continue
            d = data[baseline]
            style = BASELINE_STYLES[baseline]
            # Only plot points above the floor
            mask = d["mean"] >= y_floor
            if not mask.any():
                continue
            ax.plot(d["N"][mask], d["mean"][mask],
                    color=style["color"], marker=style["marker"],
                    label=style["label"], linewidth=1.5, markersize=5)

        ax.set_xlabel("N (qubits)")
        ax.set_ylabel("Fidelity estimate")
        ax.set_title(f"k={k_val}, p = {p_2q:.0e}")
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_floor, top=2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Estimated Fidelity $(1-p)^G$ — One Trotter Step", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fidelity_vs_N", output_dir)


def plot_depth_breakdown_trend(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """All 5 methods' total depth vs L, with shared interaction-only line.

    Solid lines = total CNOT depth per method.
    Dashed gray line = interaction depth (shared across all FP methods).
    For naive Pauli, the total IS the interaction depth (no FP).
    """
    _setup_style()
    k_val = _primary_k(df)
    df_k = df[df["k"] == k_val]
    p_min = df_k["p_2q"].min()
    df_sub = df_k[df_k["p_2q"] == p_min]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plot total depth for each method
    data_total = _aggregate_by_L(df_sub, "cnot_depth")
    for baseline in BASELINES_ORDER:
        if baseline not in data_total:
            continue
        d = data_total[baseline]
        style = BASELINE_STYLES[baseline]
        ax.errorbar(d["L"], d["mean"], yerr=d["std"],
                     color=style["color"], marker=style["marker"],
                     label=style["label"], capsize=3, linewidth=1.8)

    # Plot shared interaction depth (use any FP baseline, they're identical)
    fp_baseline = "pipelined"
    bdf = df_sub[df_sub["baseline"] == fp_baseline]
    if not bdf.empty:
        grouped = bdf.groupby("L")["interaction_cnot_depth"]
        L_vals = grouped.mean().index.values
        int_mean = grouped.mean().values
        int_std = grouped.std().values
        ax.errorbar(L_vals, int_mean, yerr=int_std,
                     color="#888888", marker="x", linestyle="--",
                     label="Local rotations only", capsize=3, linewidth=1.5,
                     alpha=0.8)

    ax.set_xlabel("L")
    ax.set_ylabel("CNOT depth (one Trotter step)")
    ax.set_title(f"Depth Breakdown Trend (k={k_val})")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "depth_breakdown_trend", output_dir)


def plot_colors_vs_N(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Number of colors used vs N for multiple k values."""
    _setup_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    for k_val in sorted(df["k"].unique()):
        df_k = df[df["k"] == k_val]
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
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "colors_vs_N", output_dir)


def generate_all_plots(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Generate all Experiment 3 plots."""
    print(f"Generating plots in {output_dir}/")
    plot_spacetime_vs_N(df, output_dir)
    plot_fidelity_vs_N(df, output_dir)
    plot_depth_breakdown_trend(df, output_dir)
    plot_colors_vs_N(df, output_dir)
    print("All plots saved.")
