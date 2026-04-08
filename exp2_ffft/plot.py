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
    "1d_baseline":   {"color": "C2", "marker": "o", "label": "CT-FFFT"},
    "gamma_2d_core": {"color": "C1", "marker": "^", "label": "2D (no reorder)"},
    "gamma_2d_proper": {"color": "C3", "marker": "s",
                        "label": "Gamma-FP-FFFT w/o ancillas"},
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

    from common.fp_2d import GammaMethod
    from exp2_ffft.collect import count_cnot_resources
    from exp2_ffft.ffft_proper import build_ffft_proper
    from common.metrics import multiplicative_fidelity

    p_values = sorted(df["p_2q"].unique())
    p_idle_factor = df["p_idle"].iloc[0] / df["p_2q"].iloc[0]

    # y-floor from the p_2q=1e-4 panel (reference case)
    ref_sub = df[df["p_2q"] == 1e-4]
    ref_fid = ref_sub.loc[
        ref_sub["method"].isin(_PLOT_METHODS - {"1d_baseline"})
        & (ref_sub["mult_fidelity"] > 0),
        "mult_fidelity",
    ]
    y_floor = ref_fid.min() * 0.1 if not ref_fid.empty else 1e-6

    # -- Pre-compute ancilla variant fidelity for each (L, p_2q) -------------
    L_all = sorted(df["L"].unique())
    anc_fidelity = {}  # (L, p_2q) -> fidelity
    for L in L_all:
        r_anc = build_ffft_proper(L, GammaMethod.ANCILLA)
        n_anc = len(r_anc.anc_qubits)
        res = count_cnot_resources(r_anc.circuit, L, n_anc)
        for p_2q in p_values:
            p_idle = p_2q * p_idle_factor
            fid = multiplicative_fidelity(
                res["total_2q_gates"], res["total_idle_slots"],
                p_2q=p_2q, p_idle=p_idle)
            anc_fidelity[(L, p_2q)] = fid

    fig, axes = plt.subplots(len(p_values), 1,
                              figsize=(7, 4.2 * len(p_values)))
    if len(p_values) == 1:
        axes = [axes]

    for idx, (ax, p_2q) in enumerate(zip(axes, p_values)):
        sub = df[df["p_2q"] == p_2q]

        # Plot in explicit order: CT-FFFT, w/ ancillas, w/o ancillas
        # 1) CT-FFFT
        ct_style = METHOD_STYLES["1d_baseline"]
        ct_data = sub[sub["method"] == "1d_baseline"].sort_values("N")
        ct_data = ct_data[ct_data["mult_fidelity"] >= y_floor]
        if not ct_data.empty:
            ax.semilogy(ct_data["N"], ct_data["mult_fidelity"],
                        marker=ct_style["marker"], color=ct_style["color"],
                        label=ct_style["label"], linewidth=3.5, markersize=7)

        # 2) Gamma-FP-FFFT w/ ancillas
        anc_N = [L * L for L in L_all]
        anc_f = [anc_fidelity[(L, p_2q)] for L in L_all]
        anc_N_f = [(n, f) for n, f in zip(anc_N, anc_f) if f >= y_floor]
        if anc_N_f:
            ns, fs = zip(*anc_N_f)
            ax.semilogy(ns, fs, marker="P", color="C0",
                        label="Gamma-FP-FFFT w/ ancillas",
                        linewidth=3.5, markersize=7)

        # 3) Gamma-FP-FFFT w/o ancillas
        gp_style = METHOD_STYLES["gamma_2d_proper"]
        gp_data = sub[sub["method"] == "gamma_2d_proper"].sort_values("N")
        gp_data = gp_data[gp_data["mult_fidelity"] >= y_floor]
        if not gp_data.empty:
            ax.semilogy(gp_data["N"], gp_data["mult_fidelity"],
                        marker=gp_style["marker"], color=gp_style["color"],
                        label=gp_style["label"], linewidth=3.5, markersize=7)

        ax.set_ylim(bottom=y_floor, top=2.0)
        ax.set_ylabel(r"$\mathbf{Estimated\;fidelity}$", fontsize=20)
        ax.set_title(f"$\\mathbf{{p_{{2q}} = {p_2q:.0e}}}$", fontsize=21)
        ax.tick_params(axis="both", labelsize=17)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(r"$\mathbf{N = L^2}$", fontsize=20)

    axes[0].legend(loc="lower left", framealpha=0.9,
                   prop={"weight": "bold", "size": 15})
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fidelity_vs_N")


# -- Plot 4: depth breakdown (FP-FFFT stages) -------------------------------

def plot_depth_breakdown(df: pd.DataFrame, fig_dir: str):
    """Grouped stacked-bar chart: CNOT depth breakdown by circuit stage.

    Five methods compared side-by-side for each N = L^2:
      0  Gamma-FP-FFFT w/o ancillas  (Gamma sandwich, pipelined Gamma)
      1  Gamma-FP-FFFT w/ ancillas   (Gamma sandwich, ancilla Gamma)
      2  FP-FFFT w/o ancillas        (FP sandwich, pipelined 2D FP)
      3  FP-FFFT FSWAP baseline      (FP sandwich, 1D snake FP)
      4  CT-FFFT                     (OpenFermion 1D, split into FP + other)

    Font sizes (matplotlib defaults -> current):
      title       12 -> 16    axis labels  12 -> 14
      tick labels 10 -> 14    legend text   10 -> 9  (compact, fits in plot)
      time arrow   -  -> 12   All text bold.
    """
    _setup_style()

    # -- Imports -------------------------------------------------------------
    import cirq as _cirq
    import matplotlib.patches as mpatches
    from openfermion.circuits.gates import FSwapPowGate as _FSwapPow

    from common.fp_2d import GammaMethod, build_fp_2d
    from common.fp_1d import build_fp_1d
    from common.grid import make_system_qubits, make_ancilla_qubits
    from common.gamma_pipeline import build_gamma_pipelined
    from common.gamma_ancilla import build_gamma_with_ancillas
    from exp2_ffft.collect import count_cnot_resources, _gate_cnot_cost
    from exp2_ffft.twiddle import build_twiddle_circuit
    from exp2_ffft.col_ffft_bare import build_bare_column_fffts, build_row_fffts
    from exp2_ffft.ffft_baseline_1d import build_ffft_1d
    from exp2_ffft.ffft_proper import (
        _build_odd_row_reversal,
        _col_major_raster_to_row_major_snake_perm,
        _transpose_perm,
        _rev_transpose_perm,
    )

    # -- Helpers -------------------------------------------------------------

    def _is_perm_gate(gate):
        """True for FSWAP or SwapPermutationGate wrapping FSWAP."""
        if isinstance(gate, _FSwapPow):
            return True
        return hasattr(gate, 'swap_gate') and isinstance(gate.swap_gate, _FSwapPow)

    def _split_perm_comp(circuit, L):
        """Split total CNOT depth into permutation and the rest.

        Walks the original moment structure.  For each moment the
        permutation cost is the max CNOT cost among FSWAP gates (0 if
        none).  ``other = total - perm`` so the two sum exactly.
        """
        d_perm = 0
        for moment in circuit:
            max_perm = 0
            for op in moment:
                if len(op.qubits) >= 2 and _is_perm_gate(op.gate):
                    max_perm = max(max_perm, _gate_cnot_cost(op.gate))
            d_perm += max_perm
        d_total = count_cnot_resources(circuit, L, 0)["cnot_depth"]
        return d_perm, d_total - d_perm

    # -- Data selection ------------------------------------------------------

    p_val = df["p_2q"].iloc[0]
    sub = df[(df["p_2q"] == p_val) & (df["method"] == "gamma_2d_proper")]
    L_values = sorted(v for v in sub["L"].unique() if v <= 20)
    N_values = [L * L for L in L_values]

    # -- Color palette (Amber & Navy) ----------------------------------------
    # Muted pastels for secondary stages; saturated amber / navy for emphasis.
    _c = {
        "odd_rev":  "#c8c8c8",   # light gray
        "gamma":    "#d89850",   # amber  (emphasis)
        "col_fft":  "#e8d898",   # pale gold
        "twiddle":  "#b8d0c0",   # pale sage
        "row_fft":  "#b0b8d0",   # pale periwinkle
        "fp":       "#506888",   # navy   (emphasis)
        "other_ct": "#88a0c0",   # steel blue
    }

    # -- Stage definitions per method ----------------------------------------
    # Each entry: (legend_label, color, show_in_legend)
    # Duplicate stages within a method use in_legend=False.

    _G = r"$\mathbf{\Gamma}$"

    # Method 0: Gamma sandwich (pipelined, no ancilla)
    gamma_info = [
        ("Odd-row rev", _c["odd_rev"], True),
        (_G,            _c["gamma"],   True),
        ("Col FFT",     _c["col_fft"], True),
        (_G,            _c["gamma"],   False),
        ("Twiddle",     _c["twiddle"], True),
        ("Row FFT",     _c["row_fft"], True),
        ("FP",          _c["fp"],      True),
    ]
    # Method 1: Gamma sandwich (ancilla Gamma) — same stage layout
    gamma_anc_info = [
        ("Odd-row rev", _c["odd_rev"], False),
        (_G,            _c["gamma"],   False),
        ("Col FFT",     _c["col_fft"], False),
        (_G,            _c["gamma"],   False),
        ("Twiddle",     _c["twiddle"], False),
        ("Row FFT",     _c["row_fft"], False),
        ("FP",          _c["fp"],      False),
    ]
    # Method 2: FP sandwich (pipelined 2D FP, no ancilla)
    fp2d_info = [
        ("FP",      _c["fp"],      False),
        ("Col FFT", _c["col_fft"], False),
        ("FP",      _c["fp"],      False),
        ("Twiddle", _c["twiddle"], False),
        ("Row FFT", _c["row_fft"], False),
        ("FP",      _c["fp"],      False),
    ]
    # Method 3: FP sandwich (1D snake FSWAP baseline)
    fp1d_info = [
        ("FP",      _c["fp"],      False),
        ("Col FFT", _c["col_fft"], False),
        ("FP",      _c["fp"],      False),
        ("Twiddle", _c["twiddle"], False),
        ("Row FFT", _c["row_fft"], False),
        ("FP",      _c["fp"],      False),
    ]
    # Method 4: CT-FFFT (OpenFermion 1D baseline, split into FP + other)
    baseline_info = [
        ("FP",          _c["fp"],       False),
        ("Other (CT)",  _c["other_ct"], True),
    ]

    # -- Collect per-stage depths for each L ---------------------------------

    gamma_depths     = [[] for _ in gamma_info]
    gamma_anc_depths = [[] for _ in gamma_anc_info]
    fp2d_depths      = [[] for _ in fp2d_info]
    fp1d_depths      = [[] for _ in fp1d_info]
    baseline_depths  = [[] for _ in baseline_info]

    for L in L_values:
        sq = make_system_qubits(L)
        _d = lambda circ: count_cnot_resources(circ, L, 0)["cnot_depth"]

        # Shared sub-circuits
        rev   = _build_odd_row_reversal(L, sq)
        col   = build_bare_column_fffts(L, sq)
        tw    = build_twiddle_circuit(L, sq)
        row   = build_row_fffts(L, sq)
        tw_d  = _d(tw)
        row_d = _d(row)

        reorder_perm = _col_major_raster_to_row_major_snake_perm(L)
        trans_perm   = _transpose_perm(L)
        rt_perm      = _rev_transpose_perm(L)

        # M0: Gamma sandwich (pipelined)
        gamma, _   = build_gamma_pipelined(L, sq=sq)
        fp_reorder = build_fp_2d(L, reorder_perm, GammaMethod.PIPELINED)
        gamma_d    = _d(gamma)
        gamma_depths[0].append(_d(rev))
        gamma_depths[1].append(gamma_d)
        gamma_depths[2].append(_d(col))
        gamma_depths[3].append(gamma_d)
        gamma_depths[4].append(tw_d)
        gamma_depths[5].append(row_d)
        gamma_depths[6].append(_d(fp_reorder.circuit))

        # M1: Gamma sandwich (ancilla)
        aq = make_ancilla_qubits(L)
        gamma_anc_circ, _, _ = build_gamma_with_ancillas(L, sq=sq, aq=aq)
        gamma_anc_d    = _d(gamma_anc_circ)
        fp_reorder_anc = build_fp_2d(L, reorder_perm, GammaMethod.ANCILLA)
        gamma_anc_depths[0].append(_d(rev))
        gamma_anc_depths[1].append(gamma_anc_d)
        gamma_anc_depths[2].append(_d(col))
        gamma_anc_depths[3].append(gamma_anc_d)
        gamma_anc_depths[4].append(tw_d)
        gamma_anc_depths[5].append(row_d)
        gamma_anc_depths[6].append(_d(fp_reorder_anc.circuit))

        # M2: FP sandwich (pipelined 2D)
        fp_rt_2d = build_fp_2d(L, rt_perm,      GammaMethod.PIPELINED)
        fp_t_2d  = build_fp_2d(L, trans_perm,    GammaMethod.PIPELINED)
        fp_r_2d  = build_fp_2d(L, reorder_perm,  GammaMethod.PIPELINED)
        fp2d_depths[0].append(_d(fp_rt_2d.circuit))
        fp2d_depths[1].append(row_d)
        fp2d_depths[2].append(_d(fp_t_2d.circuit))
        fp2d_depths[3].append(tw_d)
        fp2d_depths[4].append(row_d)
        fp2d_depths[5].append(_d(fp_r_2d.circuit))

        # M3: FP sandwich (1D snake baseline)
        fp_rt_1d = build_fp_1d(L, rt_perm)
        fp_t_1d  = build_fp_1d(L, trans_perm)
        fp_r_1d  = build_fp_1d(L, reorder_perm)
        fp1d_depths[0].append(_d(fp_rt_1d.circuit))
        fp1d_depths[1].append(row_d)
        fp1d_depths[2].append(_d(fp_t_1d.circuit))
        fp1d_depths[3].append(tw_d)
        fp1d_depths[4].append(row_d)
        fp1d_depths[5].append(_d(fp_r_1d.circuit))

        # M4: CT-FFFT (OpenFermion 1D, perm/comp split)
        r_1d = build_ffft_1d(L)
        d_perm, d_comp = _split_perm_comp(r_1d.circuit, L)
        baseline_depths[0].append(d_perm)
        baseline_depths[1].append(d_comp)

    # -- Draw grouped bars ---------------------------------------------------

    bar_w   = 0.15
    gap     = 0.025                          # gap between bars in group
    step    = bar_w + gap
    offsets = [-2*step, -step, 0.0, step, 2*step]
    x       = np.arange(len(L_values))

    fig, ax = plt.subplots(figsize=(max(7, len(L_values) * 1.44 + 0.8), 7))

    all_methods = [
        (fp1d_info,      fp1d_depths),       # M0 – FP-FFFT FSWAP baseline
        (baseline_info,  baseline_depths),   # M1 – CT-FFFT
        (fp2d_info,      fp2d_depths),       # M2 – FP-FFFT w/o ancillas
        (gamma_anc_info, gamma_anc_depths),  # M3 – Gamma w/ ancillas
        (gamma_info,     gamma_depths),      # M4 – Gamma w/o ancillas
    ]

    # Method markers placed on bar tops
    method_markers = [
        (r"FP-FFFT FSWAP baseline: $\mathbf{O(N)}$",            "^", "#333333"),
        (r"CT-FFFT: $\mathbf{O(N)}$",                           "o", "#333333"),
        (r"FP-FFFT w/o ancillas: $\mathbf{O(N^{1/2})}$",       "D", "#333333"),
        (r"Gamma-FP-FFFT w/ ancillas: $\mathbf{O(N^{1/2})}$",  "P", "#333333"),
        (r"Gamma-FP-FFFT w/o ancillas: $\mathbf{O(N^{1/2})}$", "s", "#333333"),
    ]

    bar_tops = []
    for method_idx, (info, depths) in enumerate(all_methods):
        bottom = np.zeros(len(L_values))
        xpos   = x + offsets[method_idx]
        is_ct  = (method_idx == 1)
        for (label, color, in_legend), values in zip(info, depths):
            vals  = np.array(values, dtype=float)
            extra = dict(hatch="//", edgecolor="#aaaaaa", linewidth=0.3
                         ) if is_ct else {}
            ax.bar(xpos, vals, bar_w, bottom=bottom,
                   label=label if in_legend else "_nolegend_",
                   color=color, **extra)
            bottom += vals
        bar_tops.append((xpos, bottom.copy()))

    marker_handles = []
    for (xpos, tops), (mlabel, marker, mc) in zip(bar_tops, method_markers):
        h = ax.scatter(xpos, tops, marker=marker, color=mc, s=20, zorder=5)
        marker_handles.append((h, mlabel))

    # -- Reduction arrows from CT-FFFT to Gamma w/o ancillas -----------------
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MPath

    ct_xpos, ct_tops       = bar_tops[1]   # CT-FFFT
    gamma_xpos, gamma_tops = bar_tops[4]   # Gamma w/o ancillas

    for i in range(len(L_values)):
        x0, y0 = ct_xpos[i], ct_tops[i]
        x1, y1 = gamma_xpos[i], gamma_tops[i]
        reduction = (y0 - y1) / y0 * 100

        # Only draw when there's a positive reduction
        if reduction <= 0:
            continue

        local_max = max(bar_tops[j][1][i] for j in range(5))
        dx = x1 - x0
        xm = (x0 + x1) / 2
        pad = 0.04 * local_max

        # Cubic Bézier arch with both controls at the same height h.
        # B_y(t) = (1-t)³y0 + 3t(1-t)h + t³y1  (when P1_y = P2_y = h)
        # Solve for h so curve clears each middle bar at its x-position.
        P1x = x0 + dx * 0.3
        P2x = x1 - dx * 0.3
        P0 = np.array([x0, y0])
        P3 = np.array([x1, y1])

        t_samples = np.linspace(0, 1, 1000)
        Bx = (1-t_samples)**3*x0 + 3*t_samples*(1-t_samples)**2*P1x \
           + 3*t_samples**2*(1-t_samples)*P2x + t_samples**3*x1

        required_h = max(y0, y1) + pad          # minimum: visible arc
        for j in [2, 3]:
            xj = bar_tops[j][0][i]
            hj = bar_tops[j][1][i]
            t_bar = t_samples[np.argmin(np.abs(Bx - xj))]
            coeff = 3 * t_bar * (1 - t_bar)
            baseline = (1 - t_bar)**3 * y0 + t_bar**3 * y1
            needed = (hj + pad - baseline) / coeff
            required_h = max(required_h, needed)

        P1 = np.array([P1x, required_h])
        P2 = np.array([P2x, required_h])

        path = MPath(
            [P0, P1, P2, P3],
            [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
        )
        arrow = FancyArrowPatch(
            path=path, arrowstyle="-|>", color="#555555",
            lw=1.5, mutation_scale=12, zorder=6,
        )
        ax.add_patch(arrow)

        # Text at the curve midpoint (t=0.5): B_y = 0.125*y0 + 0.75*h + 0.125*y1
        mid_y = 0.125 * y0 + 0.75 * required_h + 0.125 * y1
        is_last = (i == len(L_values) - 1)
        text_offset = 0.04 * local_max if is_last else 0.015 * local_max
        ax.text(xm, mid_y + text_offset, f"{reduction:.0f}%",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#555555")

    # -- Axes & labels -------------------------------------------------------
    # Font sizes: title 16, axis labels 14, tick labels 14, legends 9.
    # All text bold.

    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in N_values], rotation=45, ha="right",
                       fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$\mathbf{N = L^2}$", fontsize=14)
    ax.set_ylabel(r"$\mathbf{CNOT\;depth}$", fontsize=14)
    ax.set_title(r"$\mathbf{FFFT\;—\;Depth\;Breakdown}$", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")

    # -- Legend 1 (top): stage colors + hatch swatch -------------------------

    bar_handles, bar_labels = ax.get_legend_handles_labels()
    hatch_patch = mpatches.Patch(facecolor="white", edgecolor="#aaaaaa",
                                 hatch="//", label="Not Time Ordered")
    bar_handles.append(hatch_patch)
    bar_labels.append("Not Time Ordered")

    leg1 = ax.legend(bar_handles, bar_labels,
                     loc="upper left", fontsize=9, ncol=2, framealpha=0.9,
                     bbox_to_anchor=(0.01, 0.98),
                     prop={"weight": "bold"})
    ax.add_artist(leg1)

    # -- Legend 2 (below leg1): method markers, single column ----------------

    ax.legend([h for h, _ in marker_handles],
              [l for _, l in marker_handles],
              fontsize=9, ncol=1, framealpha=0.9,
              scatterpoints=1, handletextpad=0.3,
              loc="upper left", bbox_to_anchor=(0.01, 0.73),
              prop={"weight": "bold"})

    # -- Time arrow below legend 2 ------------------------------------------

    ax.annotate("", xy=(0.045, 0.38), xytext=(0.045, 0.24),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#CBF3BB",
                                lw=3.5, mutation_scale=14))
    ax.text(0.045, 0.22, "Time", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, color="#90c080",
            fontweight="bold")

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
