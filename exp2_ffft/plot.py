"""Experiment 2: figure generation for FFFT benchmarking."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -- Style & method configuration -------------------------------------------

METHOD_STYLES = {
    "1d_baseline":   {"color": "C0", "marker": "o", "label": "CT-FFFT"},
    "gamma_2d_core": {"color": "C1", "marker": "^", "label": "2D (no reorder)"},
    "gamma_2d_proper": {"color": "C3", "marker": "s", "label": "FP-FFFT"},
}

# Methods shown in plots (gamma_2d_core kept in data/code but hidden).
_PLOT_METHODS = {"1d_baseline", "gamma_2d_proper"}


def _setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _save_fig(fig, fig_dir: str, name: str):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("svg",):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)


def _find_crossover_N(sub: pd.DataFrame, metric: str) -> float | None:
    """Return the log-interpolated N where FP-FFFT first beats CT-FFFT."""
    bl = sub[sub["method"] == "1d_baseline"].sort_values("N")
    pr = sub[sub["method"] == "gamma_2d_proper"].sort_values("N")
    mg = bl[["N", metric]].merge(pr[["N", metric]], on="N",
                                  suffixes=("_bl", "_pr"))
    diff = mg[f"{metric}_bl"].values - mg[f"{metric}_pr"].values
    signs = diff[:-1] * diff[1:]
    idx = np.where(signs < 0)[0]
    if not len(idx):
        return None
    i = idx[0]
    N1, N2 = mg["N"].iloc[i], mg["N"].iloc[i + 1]
    return float(np.exp(np.interp(0, [diff[i], diff[i + 1]],
                                  [np.log(N1), np.log(N2)])))


# -- Plot 1: circuit depth vs N ---------------------------------------------

def plot_depth_vs_N(df: pd.DataFrame, fig_dir: str):
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))

    p_val = df["p_2q"].iloc[0]
    sub = df[df["p_2q"] == p_val]

    for method, style in METHOD_STYLES.items():
        if method not in _PLOT_METHODS:
            continue
        data = sub[sub["method"] == method].sort_values("N")
        if data.empty:
            continue
        ax.loglog(data["N"], data["cnot_depth"],
                  marker=style["marker"], color=style["color"],
                  label=style["label"], linewidth=1.5, markersize=6)

    Ns = np.array(sorted(sub["N"].unique()), dtype=float)
    ax.loglog(Ns, 5.0 * Ns, "--", color="gray", alpha=0.4,
              label=r"$\propto N$")
    ax.loglog(Ns, 50.0 * np.sqrt(Ns), ":", color="gray", alpha=0.4,
              label=r"$\propto \sqrt{N}$")

    N_cross = _find_crossover_N(sub, "cnot_depth")
    if N_cross is not None:
        ax.axvline(N_cross, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel(r"$N = L^2$")
    ax.set_ylabel("CNOT depth")
    ax.set_title("Circuit Depth")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="major", alpha=0.2)
    _save_fig(fig, fig_dir, "depth_vs_N")


# -- Plot 2: spacetime volume vs N ------------------------------------------

def plot_spacetime_vs_N(df: pd.DataFrame, fig_dir: str):
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))

    p_val = df["p_2q"].iloc[0]
    sub = df[df["p_2q"] == p_val]

    for method, style in METHOD_STYLES.items():
        if method not in _PLOT_METHODS:
            continue
        data = sub[sub["method"] == method].sort_values("N")
        data = data[data["N"] >= 16]
        if data.empty:
            continue
        ax.loglog(data["N"], data["spacetime_volume"],
                  marker=style["marker"], color=style["color"],
                  label=style["label"], linewidth=1.5, markersize=6)

    Ns = np.array(sorted(n for n in sub["N"].unique() if n >= 16), dtype=float)
    ax.loglog(Ns, 6.0 * Ns**2, "--", color="gray", alpha=0.4,
              label=r"$\propto N^2$")
    ax.loglog(Ns, 60.0 * Ns**1.5, ":", color="gray", alpha=0.4,
              label=r"$\propto N\sqrt{N}$")

    N_cross = _find_crossover_N(sub, "spacetime_volume")
    if N_cross is not None:
        ax.axvline(N_cross, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel(r"$N = L^2$")
    ax.set_ylabel(r"Spacetime volume  ($N_{\mathrm{qubits}} \times D$)")
    ax.set_title("Spacetime Volume")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="major", alpha=0.2)
    _save_fig(fig, fig_dir, "spacetime_vs_N")


# -- Plot 3: fidelity (3 panels, one per p_2q) ------------------------------

def plot_fidelity_vs_N(df: pd.DataFrame, fig_dir: str):
    _setup_style()
    p_values = sorted(df["p_2q"].unique())

    # y-floor from the p_2q=1e-4 panel (reference case)
    ref_sub = df[df["p_2q"] == 1e-4]
    ref_fid = ref_sub.loc[
        ref_sub["method"].isin(_PLOT_METHODS - {"1d_baseline"})
        & (ref_sub["mult_fidelity"] > 0),
        "mult_fidelity",
    ]
    y_floor = ref_fid.min() * 0.1 if not ref_fid.empty else 1e-6

    fig, axes = plt.subplots(1, len(p_values),
                              figsize=(5 * len(p_values), 4.2), sharey=True)
    if len(p_values) == 1:
        axes = [axes]

    for ax, p_2q in zip(axes, p_values):
        sub = df[df["p_2q"] == p_2q]
        for method, style in METHOD_STYLES.items():
            if method not in _PLOT_METHODS:
                continue
            data = sub[sub["method"] == method].sort_values("N")
            data = data[data["mult_fidelity"] >= y_floor]
            if data.empty:
                continue
            ax.semilogy(data["N"], data["mult_fidelity"],
                        marker=style["marker"], color=style["color"],
                        label=style["label"], linewidth=1.5, markersize=5)
        ax.set_xlabel(r"$N = L^2$")
        ax.set_title(f"$p_{{2q}} = {p_2q:.0e}$")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylim(bottom=y_floor, top=2.0)
    axes[0].set_ylabel("Estimated fidelity")
    axes[-1].legend(loc="lower left", framealpha=0.9)
    fig.suptitle("Fidelity Estimate", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fidelity_vs_N")


# -- Plot 4: depth breakdown (FP-FFFT stages) -------------------------------

def plot_depth_breakdown(df: pd.DataFrame, fig_dir: str):
    _setup_style()

    from common.fp_2d import GammaMethod, build_fp_2d
    from common.grid import make_system_qubits
    from common.gamma_pipeline import build_gamma_pipelined
    from exp2_ffft.collect import count_cnot_resources
    from exp2_ffft.twiddle import build_twiddle_circuit
    from exp2_ffft.col_ffft_bare import build_bare_column_fffts, build_row_fffts
    from exp2_ffft.ffft_proper import (
        _build_odd_row_reversal,
        _col_major_raster_to_row_major_snake_perm,
    )

    p_val = df["p_2q"].iloc[0]
    sub = df[(df["p_2q"] == p_val) & (df["method"] == "gamma_2d_proper")]
    L_values = sorted(v for v in sub["L"].unique() if v <= 20)
    N_values = [L * L for L in L_values]

    stage_names = [
        r"Odd-row rev",
        r"$\Gamma$ ($\times 2$, DFT)",
        "Col FFT",
        "Twiddle",
        "Row FFT",
        r"FP reorder (incl. $\Gamma \times 2$)",
    ]
    stage_data = {s: [] for s in stage_names}

    for L in L_values:
        sq = make_system_qubits(L)
        _d = lambda circ: count_cnot_resources(circ, L, 0)["cnot_depth"]

        rev = _build_odd_row_reversal(L, sq)
        gamma, _ = build_gamma_pipelined(L, sq=sq)
        col = build_bare_column_fffts(L, sq)
        tw = build_twiddle_circuit(L, sq)
        row = build_row_fffts(L, sq)
        fp_perm = _col_major_raster_to_row_major_snake_perm(L)
        fp = build_fp_2d(L, fp_perm, GammaMethod.PIPELINED)

        stage_data[r"Odd-row rev"].append(_d(rev))
        stage_data[r"$\Gamma$ ($\times 2$, DFT)"].append(2 * _d(gamma))
        stage_data["Col FFT"].append(_d(col))
        stage_data["Twiddle"].append(_d(tw))
        stage_data["Row FFT"].append(_d(row))
        stage_data[r"FP reorder (incl. $\Gamma \times 2$)"].append(_d(fp.circuit))

    fig, ax = plt.subplots(figsize=(max(6, len(L_values) * 0.7 + 1), 4.5))
    x = np.arange(len(L_values))
    bottom = np.zeros(len(L_values))
    colors = ["C7", "C3", "C1", "C2", "C4", "C9"]

    for (label, values), color in zip(stage_data.items(), colors):
        vals = np.array(values, dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += vals

    # Asymptotic guide: straight line slightly above all bar tops
    bar_totals = bottom.copy()
    a, b = np.polyfit(x, bar_totals, 1)
    min_residual = min(a * xi + b - bt for xi, bt in zip(x, bar_totals))
    ax.plot(x, a * x + (b - min_residual + 15), "--", color="gray", alpha=0.5,
            label=r"$\propto \sqrt{N}$")

    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in N_values], rotation=45, ha="right")
    ax.set_xlabel(r"$N = L^2$")
    ax.set_ylabel("CNOT depth")
    ax.set_title(r"FP-FFFT — Depth Breakdown")
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    _save_fig(fig, fig_dir, "depth_breakdown")


# -- Entry point -------------------------------------------------------------

def generate_all_plots(
    csv_path: str = "exp2_ffft/results/data.csv",
    fig_dir: str = "exp2_ffft/figures",
):
    """Generate all Exp 2 plots from saved CSV data."""
    df = pd.read_csv(csv_path)
    plot_depth_vs_N(df, fig_dir)
    plot_spacetime_vs_N(df, fig_dir)
    plot_fidelity_vs_N(df, fig_dir)
    plot_depth_breakdown(df, fig_dir)
    print(f"All plots saved to {fig_dir}/")
