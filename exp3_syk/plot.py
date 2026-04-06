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


def _find_fidelity_p(df_k: pd.DataFrame, target_frac: float = 0.80,
                      p_idle_factor: float = 0.1) -> float:
    """Binary-search for the largest p_2q where *pipelined* has ≥ target_frac
    of its N-values with mean fidelity > 0.5.

    Recomputes fidelity from gate counts so we are not limited to the p_2q
    values stored in the CSV.
    """
    # Grab gate counts (same for every stored p_2q).
    ref_p = df_k["p_2q"].min()
    sub = df_k[(df_k["p_2q"] == ref_p) & (df_k["baseline"] == "pipelined")]
    g2q = sub.groupby("N")["total_2q_gates"].mean()
    gidle = sub.groupby("N")["total_idle_slots"].mean()
    n_points = len(g2q)

    def frac_above(p: float) -> float:
        above = 0
        for N in g2q.index:
            F = (1 - p) ** g2q[N] * (1 - p * p_idle_factor) ** gidle[N]
            if F > 0.5:
                above += 1
        return above / n_points

    lo, hi = 1e-10, 1e-4
    for _ in range(80):
        mid = np.sqrt(lo * hi)          # geometric midpoint
        if frac_above(mid) >= target_frac:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def _recompute_fidelity(df_k: pd.DataFrame, p_2q: float,
                         p_idle_factor: float = 0.1) -> pd.DataFrame:
    """Return a copy of df_k (at any single stored p_2q) with mult_fidelity
    recomputed for the given *p_2q*."""
    ref_p = df_k["p_2q"].min()
    out = df_k[df_k["p_2q"] == ref_p].copy()
    p_idle = p_2q * p_idle_factor
    out["mult_fidelity"] = (
        (1 - p_2q) ** out["total_2q_gates"]
        * (1 - p_idle) ** out["total_idle_slots"]
    )
    out["p_2q"] = p_2q
    return out


def plot_fidelity_vs_N(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Plot fidelity vs N (log-log) at automatically chosen p_2q.

    p_2q is the largest value for which ≥ 80 % of system sizes keep
    the *pipelined* fidelity above 0.5, so the plot stays in a
    visually meaningful range.
    """
    _setup_style()
    k_val = _primary_k(df)
    df_k = df[df["k"] == k_val]

    # Fixed p_2q chosen so the plot stays in a meaningful fidelity range.
    p_2q = 1e-6

    # Recompute fidelity at this p for all baselines
    df_sub = _recompute_fidelity(df_k, p_2q)
    data = _aggregate(df_sub, "mult_fidelity")

    y_floor = 1e-3  # nothing meaningful below this at the chosen p

    fig, ax = plt.subplots(figsize=(6, 4.5))

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
                label=style["label"], linewidth=2.5, markersize=6)

    # Crossover vertical line where FP w/o ancillas becomes best
    N_cross = _find_pipelined_crossover_N(data)
    if N_cross is not None:
        _draw_crossover(ax, N_cross, y_floor)

    ax.set_xlabel(r"$\mathbf{N = L^2}$", fontsize=14)
    ax.set_ylabel(r"$\mathbf{Fidelity\;estimate}$", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_floor, top=2.0)
    ax.tick_params(axis="both", labelsize=14)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.legend(fontsize=9, prop={"weight": "bold"})
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "fidelity_vs_N", output_dir)


def plot_depth_breakdown(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Grouped bar chart: total CNOT depth per method.

    Three methods compared side-by-side for each N = L^2:
      0  FP w/o ancillas   (ours)
      1  FP w/ ancillas
      2  FSWAP baseline

    Light pastel fills matching each method's identity color;
    markers on bar tops; single unified legend.

    Font sizes (matching exp2):
      title 16, axis labels 14, tick labels 14, legend 9.  All bold.
    """
    _setup_style()

    k_val = _primary_k(df)
    df_k = df[df["k"] == k_val]
    p_min = df_k["p_2q"].min()
    df_sub = df_k[df_k["p_2q"] == p_min]

    # -- Aggregate mean depths per method per N ---------------------------------
    methods = ["pipelined", "ancilla", "1d"]   # ours, with ancilla, baseline
    N_values = sorted(df_sub["N"].unique())

    depth_data = {}   # total cnot_depth
    rot_data   = {}   # interaction_cnot_depth (local rotations)
    for bl in methods:
        bdf = df_sub[df_sub["baseline"] == bl]
        tot_vals, rot_vals = [], []
        for N in N_values:
            ndf = bdf[bdf["N"] == N]
            tot_vals.append(ndf["cnot_depth"].mean() if not ndf.empty else 0)
            rot_vals.append(ndf["interaction_cnot_depth"].mean() if not ndf.empty else 0)
        depth_data[bl] = np.array(tot_vals)
        rot_data[bl]   = np.array(rot_vals)

    # -- Light pastel palette (based on BASELINE_STYLES identity colors) -------
    #   pipelined  #d62728 (red)   -> light rose
    #   ancilla    #1f77b4 (blue)  -> light sky
    #   1d         #555555 (gray)  -> light silver
    method_cfg = [
        ("FP w/o ancillas", "o", "#d62728", "#f0b0b0"),   # red marker, rose fill
        ("FP w/ ancillas",  "^", "#1f77b4", "#a8cee8"),   # blue marker, sky fill
        ("FSWAP baseline",  "s", "#555555", "#c8c8c8"),   # gray marker, silver fill
    ]

    # -- Bar layout ------------------------------------------------------------
    bar_w   = 0.15
    gap     = 0.025
    step    = bar_w + gap
    offsets = [-step, 0.0, step]
    x       = np.arange(len(N_values))

    fig, ax = plt.subplots(figsize=(max(9, len(N_values) * 1.8 + 1), 5.5))

    # -- Local rotation overlay color ------------------------------------------
    _c_rot = "#ffe040"   # bright yellow — visible on any pastel

    bar_tops = []
    legend_handles = []
    for i, bl in enumerate(methods):
        label, marker, mc, fc = method_cfg[i]
        xpos = x + offsets[i]

        # Full-height bar (total depth)
        ax.bar(xpos, depth_data[bl], bar_w, color=fc,
               edgecolor=mc, linewidth=0.6, label="_nolegend_")

        # Local rotation overlay at the bottom
        ax.bar(xpos, rot_data[bl], bar_w, color=_c_rot,
               edgecolor="none", label="_nolegend_")

        tops = depth_data[bl]
        h = ax.scatter(xpos, tops, marker=marker, color=mc, s=20, zorder=5,
                       label=label)
        legend_handles.append(h)
        bar_tops.append(tops)

    # -- Axes & labels ---------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in N_values], rotation=45, ha="right",
                       fontsize=16, fontweight="bold")
    ax.set_xlabel(r"$\mathbf{N = L^2}$", fontsize=16)
    ax.set_ylabel(r"$\mathbf{CNOT\;depth}$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")

    # -- Single legend: methods + local rotation indicator ----------------------
    import matplotlib.patches as mpatches
    rot_patch = mpatches.Patch(facecolor=_c_rot, edgecolor="none",
                               label="Local Rotation")
    all_handles = legend_handles + [rot_patch]
    ax.legend(handles=all_handles, fontsize=11, ncol=1, framealpha=0.9,
              scatterpoints=1, handletextpad=0.3,
              loc="upper left", bbox_to_anchor=(0.01, 0.98),
              prop={"weight": "bold"})

    _save_fig(fig, "depth_breakdown", output_dir)


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
    plot_depth_breakdown(df, output_dir)
    plot_colors_vs_N(df, output_dir, colors_df=colors_df)
    print("All plots saved.")
