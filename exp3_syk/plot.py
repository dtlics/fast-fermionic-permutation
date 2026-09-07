"""Experiment 3: plotting functions for sparse SYK Trotter step benchmarking.

Generates 4 plots:
1. Spacetime volume vs N
2. Independent-location no-fault probability vs N (log scale)
3. Depth breakdown trend (methods + shared interaction line, vs N=L^2)
4. Number of colors vs N (multiple k values)
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_STYLES = {
    "naive_pauli": {"color": "#2ca02c", "marker": "v", "label": "Naive Pauli"},
    "1d":          {"color": "#555555", "marker": "s", "label": "FSWAP baseline"},
    "ancilla":     {"color": "#1f77b4", "marker": "^", "label": "FP w/ ancillas"},
    "primitive":   {"color": "#ff7f0e", "marker": "D", "label": r"Primitive $\Gamma$"},
    "pipelined":   {"color": "#d62728", "marker": "o", "label": "FP w/o ancillas"},
    "folded":      {"color": "#d62728", "marker": "o", "label": r"FP w/o ancillas (folded $\Gamma$)"},
}

BASELINES_ORDER = [
    "naive_pauli", "1d", "ancilla", "primitive", "pipelined", "folded"
]

# Primitive is hidden. Folded is the current ancilla-free method; pipelined
# remains a fallback so historical CSVs still render without relabeling data.
PLOT_BASELINES_PREFIX = ["naive_pauli", "1d", "ancilla"]
PUBLICATION_L_VALUES = tuple(range(4, 21))
PUBLICATION_P_VALUES = (1e-3, 1e-4, 1e-5)
PUBLICATION_N_INSTANCES = 10


def _preferred_ancilla_free(available) -> str:
    names = set(available)
    if "folded" in names:
        return "folded"
    return "pipelined"


def _plot_baselines(data: dict) -> list[str]:
    return PLOT_BASELINES_PREFIX + [_preferred_ancilla_free(data)]


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
    for baseline in _plot_baselines(data):
        if baseline not in data:
            continue
        d = data[baseline]
        style = BASELINE_STYLES[baseline]
        ax.errorbar(d["N"], d["mean"], yerr=d["std"],
                     color=style["color"], marker=style["marker"],
                     label=style["label"], capsize=3, linewidth=1.5)

    # Crossover vertical line where FP w/o ancillas becomes best
    N_cross = _find_ancilla_free_crossover_N(data, lower_is_better=True)
    if N_cross is not None:
        _draw_crossover(ax, N_cross, ax.get_ylim()[0])

    # Asymptotic guide lines
    Ns = np.array(sorted(df_sub["N"].unique()), dtype=float)
    ax.loglog(Ns, 50.0 * Ns**2, "--", color="gray", alpha=0.4,
              label=r"$\propto N^2$")
    ax.loglog(Ns, 700.0 * Ns**1.5, ":", color="gray", alpha=0.4,
              label=r"$\propto N\sqrt{N}$")

    ax.set_xlabel(r"$N = L^2$")
    ax.set_ylabel(r"Spacetime volume (qubits $\times$ CNOT-equivalent depth)")
    ax.set_title(f"Spacetime Volume Per Trotter Step (k={k_val})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels(
        [str(int(value)) for value in Ns], rotation=35, ha="right"
    )
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend()
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "spacetime_vs_N", output_dir)


def _find_ancilla_free_crossover_N(
    data: dict, lower_is_better: bool = False
) -> float | None:
    """Find the first N where the current ancilla-free method is best.

    Folded data are preferred when present; otherwise historical pipelined
    data are used.
    """
    target = _preferred_ancilla_free(data)
    if target not in data:
        return None
    others = [
        baseline
        for baseline in _plot_baselines(data)
        if baseline != target and baseline in data
    ]
    if not others:
        return None

    ancilla_free = data[target]
    for N, val in zip(ancilla_free["N"], ancilla_free["mean"]):
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
    # Anchor the label in axes-relative y coordinates so changing to a log
    # scale cannot clip it against the lower boundary.
    ax.annotate(
        f"N={int(N_cross)}",
        xy=(N_cross, 0.56),
        xycoords=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="bold",
        color="gray",
        ha="left",
        va="center",
        xytext=(3, 0),
        textcoords="offset points",
        rotation=90,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65,
              "pad": 0.5},
    )


def _find_fidelity_p(df_k: pd.DataFrame, target_frac: float = 0.80,
                      p_idle_factor: float = 1.0) -> float:
    """Find the largest p_2q meeting the ancilla-free no-fault threshold.

    Folded data are preferred; pipelined data are the historical fallback.
    """
    # Grab gate counts (same for every stored p_2q).
    ref_p = df_k["p_2q"].min()
    target = _preferred_ancilla_free(df_k["baseline"].unique())
    sub = df_k[
        (df_k["p_2q"] == ref_p) & (df_k["baseline"] == target)
    ].copy()
    gates, idle = _canonical_accounting_series(sub)
    sub["_native_gates"] = gates
    sub["_layered_idle"] = idle
    g2q = sub.groupby("N")["_native_gates"].mean()
    gidle = sub.groupby("N")["_layered_idle"].mean()
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


def _canonical_accounting_series(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return publication counts, never legacy raw-gate approximations."""
    required = {"total_cnots", "total_idle_slots_layered"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"SYK data lack CNOT-equivalent accounting columns {missing}; "
            "rerun collection"
        )
    if frame[list(required)].isna().any().any():
        raise ValueError("SYK data contain missing CNOT-equivalent counts")
    return frame["total_cnots"], frame["total_idle_slots_layered"]


def validate_publication_data(
    df: pd.DataFrame,
    *,
    require_publication_coverage: bool = True,
) -> None:
    """Validate the complete strict-FSWAP SYK table before plotting."""
    from exp3_syk.collect import (
        ACCOUNTING_MODEL,
        BASELINE_CONFIGS,
        RESOURCE_DEPENDENCY_VERSIONS,
        no_fault_values_match,
        validate_gamma_schedule_provenance,
        validate_hall_provenance,
        validate_reproducibility_provenance,
        validate_resource_dependency_provenance,
    )

    required = {
        "L", "N", "k", "instance_idx", "baseline", "p_2q", "p_idle",
        "accounting_model", "hall_decomposition_model", "n_ancillas",
        "gamma_schedule_model",
        *RESOURCE_DEPENDENCY_VERSIONS,
        "total_qubits", "cnot_depth", "fp_cnot_depth",
        "interaction_cnot_depth", "total_cnots", "n_fswap",
        "n_2q_cnot_cz", "total_idle_slots_layered", "spacetime_volume",
        "n_1q_hadamard", "n_1q_other", "n_1q_s_sdag", "n_1q_rz",
        "mult_fidelity",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"SYK CSV lacks publication-accounting columns {missing}; "
            "rerun collection"
        )
    if df[list(required)].isna().any().any():
        raise ValueError("SYK CSV has missing publication-accounting values")
    if set(df["accounting_model"].astype(str)) != {ACCOUNTING_MODEL}:
        raise ValueError("SYK CSV mixes legacy and current accounting models")
    validate_hall_provenance(df)
    validate_gamma_schedule_provenance(df)
    validate_reproducibility_provenance(df)
    validate_resource_dependency_provenance(df)

    if not (df["N"] == df["L"] ** 2).all():
        raise ValueError("SYK CSV violates N = L^2")
    if not (df["total_qubits"] == df["N"] + df["n_ancillas"]).all():
        raise ValueError("SYK CSV violates Q = N + n_ancillas")
    expected_depth = df["fp_cnot_depth"] + df["interaction_cnot_depth"]
    if not (df["cnot_depth"] == expected_depth).all():
        raise ValueError(
            "SYK CSV violates D = D_FP + D_interaction"
        )
    if df.duplicated(
        ["L", "k", "instance_idx", "baseline", "p_2q"]
    ).any():
        raise ValueError("SYK CSV has duplicate benchmark keys")
    if not np.allclose(df["p_idle"], df["p_2q"], rtol=1e-12, atol=0):
        raise ValueError("SYK CSV does not use p_idle = p_2q")

    expected_gates = 2 * df["n_fswap"] + df["n_2q_cnot_cz"]
    if not (df["total_cnots"] == expected_gates).all():
        raise ValueError("SYK CSV violates G = 2 n_FSWAP + n_CNOT/CZ")
    expected_idle = df["total_qubits"] * df["cnot_depth"] - 2 * df["total_cnots"]
    if not (df["total_idle_slots_layered"] == expected_idle).all():
        raise ValueError("SYK CSV violates I = QD - 2G")
    expected_volume = df["total_qubits"] * df["cnot_depth"]
    if not (df["spacetime_volume"] == expected_volume).all():
        raise ValueError("SYK CSV violates spacetime volume QD")

    expected_no_fault = (
        (1.0 - df["p_2q"]) ** df["total_cnots"]
        * (1.0 - df["p_idle"]) ** df["total_idle_slots_layered"]
    )
    if not no_fault_values_match(df["mult_fidelity"], expected_no_fault):
        raise ValueError("SYK CSV no-fault estimates do not match its counts")

    resource_columns = [
        "N", "n_ancillas", "total_qubits", "cnot_depth",
        "fp_cnot_depth", "interaction_cnot_depth", "total_cnots",
        "n_fswap", "n_2q_cnot_cz", "total_idle_slots_layered",
        "spacetime_volume", "n_1q_hadamard", "n_1q_other", "n_1q_s_sdag",
        "n_1q_rz",
    ]
    repeated = df.groupby(
        ["L", "k", "instance_idx", "baseline"], sort=False
    )[resource_columns].nunique(dropna=False)
    if (repeated != 1).any().any():
        raise ValueError("SYK CSV has noise-dependent resource counts")

    if require_publication_coverage:
        baselines = {name for name, _ in BASELINE_CONFIGS}
        expected = {
            (L, 1.0, instance_idx, baseline, round(float(p), 15))
            for L in PUBLICATION_L_VALUES
            for instance_idx in range(PUBLICATION_N_INSTANCES)
            for baseline in baselines
            for p in PUBLICATION_P_VALUES
        }
        observed = {
            (int(row.L), float(row.k), int(row.instance_idx),
             str(row.baseline), round(float(row.p_2q), 15))
            for row in df[
                ["L", "k", "instance_idx", "baseline", "p_2q"]
            ].itertuples(index=False)
        }
        if observed != expected:
            raise ValueError(
                "SYK CSV does not have exact publication coverage "
                f"({len(expected - observed)} missing, "
                f"{len(observed - expected)} extra keys)"
            )


def _recompute_fidelity(df_k: pd.DataFrame, p_2q: float,
                         p_idle_factor: float = 1.0) -> pd.DataFrame:
    """Return a copy with the legacy ``mult_fidelity`` column recomputed as
    a no-fault estimate from CNOT-equivalent gate and idle exponents."""
    ref_p = df_k["p_2q"].min()
    out = df_k[df_k["p_2q"] == ref_p].copy()
    p_idle = p_2q * p_idle_factor
    gates, idle = _canonical_accounting_series(out)
    out["mult_fidelity"] = (1 - p_2q) ** gates * (1 - p_idle) ** idle
    out["p_2q"] = p_2q
    return out


def plot_fidelity_vs_N(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Plot the independent-location no-fault estimate versus N.

    p_2q is the largest value for which ≥ 80 % of system sizes keep the
    current ancilla-free method's no-fault estimate above 0.5, so the plot stays in a
    visually meaningful range.
    """
    _setup_style()
    k_val = _primary_k(df)
    df_k = df[df["k"] == k_val]

    # Fixed p_2q chosen so the plot stays in a meaningful probability range.
    p_2q = 1e-6

    # Recompute the no-fault estimate at this p for all baselines.
    df_sub = _recompute_fidelity(df_k, p_2q)
    data = _aggregate(df_sub, "mult_fidelity")

    y_floor = 1e-3  # nothing meaningful below this at the chosen p

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for baseline in _plot_baselines(data):
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
    N_cross = _find_ancilla_free_crossover_N(data)
    if N_cross is not None:
        _draw_crossover(ax, N_cross, y_floor)

    ax.set_xlabel(r"$N = L^2$")
    ax.set_ylabel("No-fault probability")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_floor, top=1.1)
    n_values = sorted(int(value) for value in df_sub["N"].unique())
    ax.set_xticks(n_values)
    ax.set_xticklabels(
        [str(value) for value in n_values], rotation=35, ha="right"
    )
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(markerscale=1.2, fontsize=9, loc="lower left")
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "fidelity_vs_N", output_dir)


def plot_depth_breakdown(df: pd.DataFrame, output_dir: str = "exp3_syk/figures"):
    """Grouped bar chart: total CNOT-equivalent depth per method.

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
    ancilla_free = _preferred_ancilla_free(df_sub["baseline"].unique())
    methods = ["1d", "ancilla", ancilla_free]
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
    #   1d         #555555 (gray)  -> light silver
    #   ancilla    #1f77b4 (blue)  -> light sky
    #   folded/pipelined  #d62728 (red) -> light rose
    method_cfg = [
        ("FSWAP baseline",  "s", "#555555", "#c8c8c8"),   # gray marker, silver fill
        ("FP w/ ancillas",  "^", "#1f77b4", "#a8cee8"),   # blue marker, sky fill
        (BASELINE_STYLES[ancilla_free]["label"], "o", "#d62728", "#f0b0b0"),
    ]

    # -- Bar layout ------------------------------------------------------------
    bar_w   = 0.15
    gap     = 0.025
    step    = bar_w + gap
    offsets = [-step, 0.0, step]
    x       = np.arange(len(N_values))

    # Keep the auxiliary plot at the same two-column journal width as the
    # paper-facing FT figure.  Scaling width with every sampled N made the
    # publication sweep more than 21 inches wide.
    fig, ax = plt.subplots(figsize=(12.4, 5.5))

    # -- Local rotation overlay color ------------------------------------------
    _c_rot = "#ffe040"   # bright yellow — visible on any pastel

    bar_tops = []    # list of (xpos_array, tops_array)
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
        bar_tops.append((xpos.copy(), tops.copy()))

    # -- Reduction curves from FSWAP to FP w/o ancillas ------------------------
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MPath

    fswap_xpos, fswap_tops = bar_tops[0]    # FSWAP baseline
    fpwo_xpos,  fpwo_tops  = bar_tops[2]    # FP w/o ancillas

    for i in range(len(N_values)):
        x0, y0 = fswap_xpos[i], fswap_tops[i]
        x1, y1 = fpwo_xpos[i],  fpwo_tops[i]
        reduction = (y0 - y1) / y0 * 100

        if reduction <= 0:
            continue

        local_max = max(bar_tops[j][1][i] for j in range(3))
        dx = x1 - x0
        xm = (x0 + x1) / 2
        pad = 0.04 * local_max

        # Cubic Bézier arch: both controls at same height h.
        # B_y(t) = (1-t)³y0 + 3t(1-t)h + t³y1
        P1x = x0 + dx * 0.3
        P2x = x1 - dx * 0.3

        t_samples = np.linspace(0, 1, 1000)
        Bx = (1-t_samples)**3*x0 + 3*t_samples*(1-t_samples)**2*P1x \
           + 3*t_samples**2*(1-t_samples)*P2x + t_samples**3*x1

        required_h = max(y0, y1) + pad
        # Ensure curve clears middle bar (index 1: FP w/ ancillas)
        xj = bar_tops[1][0][i]
        hj = bar_tops[1][1][i]
        t_bar = t_samples[np.argmin(np.abs(Bx - xj))]
        coeff = 3 * t_bar * (1 - t_bar)
        baseline = (1 - t_bar)**3 * y0 + t_bar**3 * y1
        needed = (hj + pad - baseline) / coeff
        required_h = max(required_h, needed)

        P0 = np.array([x0, y0])
        P1 = np.array([P1x, required_h])
        P2 = np.array([P2x, required_h])
        P3 = np.array([x1, y1])

        path = MPath(
            [P0, P1, P2, P3],
            [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
        )
        arrow = FancyArrowPatch(
            path=path, arrowstyle="-|>", color="#555555",
            lw=1.5, mutation_scale=12, zorder=6,
        )
        ax.add_patch(arrow)

        mid_y = 0.125 * y0 + 0.75 * required_h + 0.125 * y1
        text_offset = 0.015 * local_max
        # Shift text right for larger N to avoid crowding
        N = N_values[i]
        if N >= 576:
            x_shift = 0.24
        elif N >= 400:
            x_shift = 0.12
        else:
            x_shift = 0.0
        ax.text(xm + x_shift, mid_y + text_offset, f"{reduction:.0f}%",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#555555")

    # -- Axes & labels ---------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in N_values], rotation=45, ha="right",
                       fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$\mathbf{N = L^2}$", fontsize=14)
    ax.set_ylabel("CNOT-equivalent depth", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.yaxis.get_offset_text().set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")

    # -- Single legend: methods + local rotation indicator ----------------------
    import matplotlib.patches as mpatches
    rot_patch = mpatches.Patch(facecolor=_c_rot, edgecolor="none",
                               label="Local Rotation")
    all_handles = legend_handles + [rot_patch]
    ax.legend(handles=all_handles, ncol=4, framealpha=0.9,
              scatterpoints=1, handletextpad=0.3, markerscale=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              prop={"weight": "bold", "size": 9})

    fig.tight_layout()

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
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "colors_vs_N", output_dir)


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str = "exp3_syk/figures",
    colors_df: Optional[pd.DataFrame] = None,
    *,
    require_publication_coverage: bool = True,
):
    """Generate all Experiment 3 plots."""
    validate_publication_data(
        df, require_publication_coverage=require_publication_coverage,
    )
    if colors_df is not None:
        from exp3_syk.collect import COLORS_K_VALUES, validate_colors_data

        if require_publication_coverage:
            colors_L = PUBLICATION_L_VALUES
            colors_k = COLORS_K_VALUES
            colors_instances = PUBLICATION_N_INSTANCES
        else:
            colors_L = tuple(sorted(int(L) for L in colors_df["L"].unique()))
            colors_k = tuple(sorted(float(k) for k in colors_df["k"].unique()))
            colors_instances = int(colors_df["instance_idx"].max()) + 1
        validate_colors_data(
            colors_df, colors_L, colors_k, colors_instances,
        )
    print(f"Generating plots in {output_dir}/")
    plot_spacetime_vs_N(df, output_dir)
    plot_fidelity_vs_N(df, output_dir)
    plot_depth_breakdown(df, output_dir)
    plot_colors_vs_N(df, output_dir, colors_df=colors_df)
    print("All plots saved.")
