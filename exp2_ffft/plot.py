"""Experiment 2: figure generation for FFFT benchmarking."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_STYLES = {
    "1d_baseline":      {"color": "C0", "marker": "o", "label": "1D chain (OpenFermion)"},
    "gamma_2d_core":    {"color": "C1", "marker": "^", "label": r"2D core (no reorder)"},
    "gamma_2d_proper":  {"color": "C3", "marker": "s", "label": r"2D proper (ours)"},
}


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
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)


def _one_row(df: pd.DataFrame, p_2q: float) -> pd.DataFrame:
    """Return one row per (method, L) at a given p_2q."""
    return df[df["p_2q"] == p_2q]


# -----------------------------------------------------------------------
# Plot 1: depth vs N
# -----------------------------------------------------------------------

def plot_depth_vs_N(df: pd.DataFrame, fig_dir: str):
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))

    p_val = df["p_2q"].iloc[0]
    sub = _one_row(df, p_val)

    for method, style in METHOD_STYLES.items():
        data = sub[sub["method"] == method].sort_values("N")
        if data.empty:
            continue
        ax.loglog(data["N"], data["cnot_depth"],
                  marker=style["marker"], color=style["color"],
                  label=style["label"], linewidth=1.5, markersize=6)

    Ns = np.array(sorted(sub["N"].unique()), dtype=float)
    ax.loglog(Ns, 3.0 * Ns, "--", color="gray", alpha=0.4, label=r"$\propto N$")
    ax.loglog(Ns, 50.0 * np.sqrt(Ns), ":", color="gray", alpha=0.4,
              label=r"$\propto \sqrt{N}$")

    ax.set_xlabel(r"$N = L^2$ (number of fermionic modes)")
    ax.set_ylabel("CNOT depth")
    ax.set_title("FFFT Circuit Depth")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    _save_fig(fig, fig_dir, "depth_vs_N")


# -----------------------------------------------------------------------
# Plot 2: spacetime volume vs N
# -----------------------------------------------------------------------

def plot_spacetime_vs_N(df: pd.DataFrame, fig_dir: str):
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))

    p_val = df["p_2q"].iloc[0]
    sub = _one_row(df, p_val)

    for method, style in METHOD_STYLES.items():
        data = sub[sub["method"] == method].sort_values("N")
        if data.empty:
            continue
        ax.loglog(data["N"], data["spacetime_volume"],
                  marker=style["marker"], color=style["color"],
                  label=style["label"], linewidth=1.5, markersize=6)

    ax.set_xlabel(r"$N = L^2$ (number of fermionic modes)")
    ax.set_ylabel(r"Spacetime volume  ($N_{\mathrm{qubits}} \times D$)")
    ax.set_title("FFFT Spacetime Volume")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    _save_fig(fig, fig_dir, "spacetime_vs_N")


# -----------------------------------------------------------------------
# Plot 3: fidelity — 3 panels, one per p_2q
# -----------------------------------------------------------------------

def plot_fidelity_vs_N(df: pd.DataFrame, fig_dir: str):
    _setup_style()
    p_values = sorted(df["p_2q"].unique())

    fig, axes = plt.subplots(1, len(p_values), figsize=(5 * len(p_values), 4.2),
                             sharey=True)
    if len(p_values) == 1:
        axes = [axes]

    for ax, p_2q in zip(axes, p_values):
        sub = _one_row(df, p_2q)
        for method, style in METHOD_STYLES.items():
            data = sub[sub["method"] == method].sort_values("N")
            if data.empty:
                continue
            ax.semilogy(data["N"], data["mult_fidelity"],
                        marker=style["marker"], color=style["color"],
                        label=style["label"], linewidth=1.5, markersize=5)

        ax.set_xlabel(r"$N = L^2$ (number of modes)")
        ax.set_title(f"$p_{{2q}} = {p_2q:.0e}$")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Estimated fidelity")
    axes[-1].legend(loc="lower left", framealpha=0.9)
    fig.suptitle("Multiplicative Fidelity", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fidelity_vs_N")


# -----------------------------------------------------------------------
# Plot 4: depth breakdown (proper circuit)
# -----------------------------------------------------------------------

def plot_depth_breakdown(df: pd.DataFrame, fig_dir: str):
    _setup_style()

    from common.fp_2d import GammaMethod, build_fp_2d
    from common.grid import make_system_qubits
    from common.gamma_pipeline import build_gamma_pipelined
    from exp2_ffft.collect import count_cnot_resources
    from exp2_ffft.twiddle import build_twiddle_circuit
    from exp2_ffft.col_ffft_bare import build_bare_column_fffts, build_row_fffts
    from exp2_ffft.ffft_proper import _build_odd_row_reversal, _col_major_raster_to_row_major_snake_perm

    p_val = df["p_2q"].iloc[0]
    sub = df[(df["p_2q"] == p_val) & (df["method"] == "gamma_2d_proper")]
    L_values = sorted(v for v in sub["L"].unique() if v <= 20)

    stage_names = [r"Odd-row rev", r"$\Gamma$ (x4)", "Col FFT", "Twiddle",
                   "Row FFT", "FP transpose"]
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
        stage_data[r"$\Gamma$ (x4)"].append(4 * _d(gamma))
        stage_data["Col FFT"].append(_d(col))
        stage_data["Twiddle"].append(_d(tw))
        stage_data["Row FFT"].append(_d(row))
        # FP transpose has its own 2 Gammas inside; subtract to avoid
        # double-counting with the x4 above. Actually the FP circuit includes
        # 2 Gammas already, so report net FP depth (includes its own Gammas).
        # Re-do: report the 2 DFT Gammas and FP total separately.
        stage_data[r"$\Gamma$ (x4)"][-1] = 2 * _d(gamma)  # only DFT Gammas
        stage_data["FP transpose"].append(_d(fp.circuit))

    fig, ax = plt.subplots(figsize=(max(6, len(L_values) * 0.7 + 1), 4.5))
    x = np.arange(len(L_values))
    bottom = np.zeros(len(L_values))
    colors = ["C7", "C3", "C1", "C2", "C4", "C9"]

    for (label, values), color in zip(stage_data.items(), colors):
        vals = np.array(values, dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}" for L in L_values])
    ax.set_xlabel(r"$L$ (grid side length)")
    ax.set_ylabel("CNOT depth")
    ax.set_title("Proper 2D FFFT — Depth Breakdown")
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    _save_fig(fig, fig_dir, "depth_breakdown")


# -----------------------------------------------------------------------
# Plot 5: verification error
# -----------------------------------------------------------------------

def plot_verification(df: pd.DataFrame, fig_dir: str):
    _setup_style()

    p_val = df["p_2q"].iloc[0]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for method, style in METHOD_STYLES.items():
        sub = df[(df["p_2q"] == p_val) & (df["method"] == method)]
        sub = sub.dropna(subset=["verification_error"])
        sub = sub[sub["verification_error"] > 0]
        if sub.empty:
            continue
        ax.semilogy(sub["L"], sub["verification_error"],
                    marker=style["marker"], color=style["color"],
                    markersize=8, label=style["label"], linewidth=1.5)

    ax.axhline(1e-5, color="gray", linestyle="--", alpha=0.5,
               label="Threshold ($10^{-5}$)")
    ax.set_xlabel(r"$L$ (grid side length)")
    ax.set_ylabel(r"$\|M_{\mathrm{circuit}} - F_N\|_F$")
    ax.set_title("Statevector Verification Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, fig_dir, "verification_error")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def generate_all_plots(
    csv_path: str = "exp2_ffft/results/data.csv",
    fig_dir: str = "exp2_ffft/figures",
):
    """Generate all plots from saved CSV data."""
    df = pd.read_csv(csv_path)
    plot_depth_vs_N(df, fig_dir)
    plot_spacetime_vs_N(df, fig_dir)
    plot_fidelity_vs_N(df, fig_dir)
    plot_depth_breakdown(df, fig_dir)
    plot_verification(df, fig_dir)
    print(f"All plots saved to {fig_dir}/")
