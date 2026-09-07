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
    "gamma_2d_ancilla": {"color": "C0", "marker": "P",
                          "label": "Gamma-FP-FFFT w/ ancillas"},
    "gamma_2d_proper": {"color": "C3", "marker": "s",
                        "label": "Gamma-FP-FFFT w/o ancillas"},
}

# Methods shown in plots (gamma_2d_core kept in data/code but hidden).
_PLOT_METHODS = {
    "1d_baseline", "gamma_2d_ancilla", "gamma_2d_proper",
}


def _validate_folded_provenance(
    df: pd.DataFrame,
    *,
    require_publication_coverage: bool = False,
) -> None:
    """Reject legacy, incomplete, or internally inconsistent benchmark data."""
    import cirq
    import networkx as nx
    import openfermion

    from exp2_ffft.collect import (
        ACCOUNTING_MODEL,
        PUBLICATION_L_VALUES,
        PUBLICATION_P_VALUES,
        _hall_model_for_method,
    )
    from common.gamma_folded import (
        GAMMA_SCHEDULE_NOT_APPLICABLE,
        folded_gamma_schedule_model,
    )
    from exp2_ffft.ft_accounting import ROTATION_INVENTORY_MODEL

    required = {
        "L",
        "gamma_method",
        "gamma_schedule_model",
        "cirq_version",
        "openfermion_version",
        "numpy_version",
        "pandas_version",
        "matplotlib_version",
        "networkx_version",
        "accounting_model",
        "rotation_inventory_model",
        "hall_decomposition_model",
        "n_ancillas",
        "total_qubits",
        "cnot_depth",
        "total_cnot_equiv",
        "total_idle_slots_layered",
        "n_1q_nonz",
        "n_exact_t",
        "n_synth_rz",
        "p_2q",
        "p_idle",
        "p_1q",
        "mult_fidelity",
        "spacetime_volume",
        "verification_error",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "benchmark CSV lacks folded-Gamma provenance columns "
            f"{missing}; rerun `python -m exp2_ffft.run_experiment`"
        )

    expected_gamma = {
        "gamma_2d_core": "folded",
        "gamma_2d_proper": "folded",
        "gamma_2d_ancilla": "ancilla",
    }
    for method, expected in expected_gamma.items():
        rows = df[df["method"] == method]
        if rows.empty:
            continue
        observed = set(rows["gamma_method"].dropna().astype(str))
        if observed != {expected}:
            raise ValueError(
                f"paper plots require gamma_method={expected!r} for "
                f"{method!r}; found {sorted(observed)!r}. Rerun the experiment."
            )
    folded_rows = df["method"].isin({"gamma_2d_core", "gamma_2d_proper"})
    if not folded_rows.any():
        raise ValueError(
            "paper plots require folded 2D benchmark rows; rerun the experiment"
        )

    expected_schedule = df.apply(
        lambda row: (
            folded_gamma_schedule_model(int(row.L))
            if str(row.gamma_method) == "folded"
            else GAMMA_SCHEDULE_NOT_APPLICABLE
        ),
        axis=1,
    )
    observed_schedule = df["gamma_schedule_model"].astype(str)
    if not (observed_schedule == expected_schedule).all():
        raise ValueError(
            "benchmark CSV has stale folded-Gamma schedule provenance; "
            "rerun the experiment"
        )

    version_columns = [
        "cirq_version", "openfermion_version", "numpy_version",
        "pandas_version", "matplotlib_version", "networkx_version",
    ]
    if df[version_columns].isna().any().any():
        raise ValueError(
            "benchmark CSV has missing dependency-version provenance; "
            "rerun the experiment"
        )
    expected_versions = {
        "cirq_version": cirq.__version__,
        "openfermion_version": openfermion.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "networkx_version": nx.__version__,
    }
    mismatched_versions = {
        column: sorted(set(df[column].astype(str)))
        for column, expected in expected_versions.items()
        if set(df[column].astype(str)) != {expected}
    }
    if mismatched_versions:
        raise ValueError(
            "benchmark CSV dependency versions differ from the plotting "
            f"runtime: {mismatched_versions}; reproduce the pinned environment"
        )

    if (pd.to_numeric(df["N"], errors="coerce") !=
            pd.to_numeric(df["L"], errors="coerce") ** 2).any():
        raise ValueError("benchmark CSV violates N = L^2")
    n_ancillas = pd.to_numeric(df["n_ancillas"], errors="coerce")
    expected_ancillas = np.where(
        df["method"] == "gamma_2d_ancilla", df["L"], 0,
    )
    if not np.array_equal(n_ancillas.to_numpy(), expected_ancillas):
        raise ValueError(
            "benchmark CSV ancilla counts do not match the compiled methods"
        )
    if not (df["total_qubits"] == df["N"] + n_ancillas).all():
        raise ValueError("benchmark CSV violates Q = N + n_ancillas")

    key_columns = ["L", "method", "p_2q"]
    if df.duplicated(key_columns).any():
        raise ValueError(f"benchmark CSV has duplicate keys {key_columns}")

    accounting_models = set(df["accounting_model"].dropna().astype(str))
    if accounting_models != {ACCOUNTING_MODEL}:
        raise ValueError(
            "paper plots require CNOT-equivalent layered FFFT noise accounting; "
            f"found {sorted(accounting_models)!r}. Rerun the experiment."
        )

    rotation_models = set(
        df["rotation_inventory_model"].dropna().astype(str)
    )
    if rotation_models != {ROTATION_INVENTORY_MODEL}:
        raise ValueError(
            "paper plots require the current fail-closed FFFT rotation "
            f"inventory; found {sorted(rotation_models)!r}. Rerun the "
            "experiment."
        )

    for column in ("n_exact_t", "n_synth_rz"):
        values = pd.to_numeric(df[column], errors="coerce")
        if (
            values.isna().any()
            or (values < 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(
                f"benchmark CSV has invalid nonnegative integer {column}"
            )

    expected_hall = df["method"].astype(str).map(_hall_model_for_method)
    observed_hall = df["hall_decomposition_model"].astype(str)
    if not (observed_hall == expected_hall).all():
        raise ValueError(
            "paper plots require canonical Hall provenance for the proper 2D "
            "circuit and not-applicable provenance for other methods; rerun "
            "the experiment"
        )

    expected_idle = (
        df["total_qubits"] * df["cnot_depth"]
        - 2 * df["total_cnot_equiv"]
    )
    if not (df["total_idle_slots_layered"] == expected_idle).all():
        raise ValueError(
            "benchmark CSV violates native idle invariant Q*D - 2*G; "
            "rerun the experiment"
        )

    if not np.allclose(df["p_1q"], df["p_2q"], rtol=1e-12, atol=0):
        raise ValueError(
            "benchmark CSV does not use p_1q = p_2q; rerun the experiment"
        )
    if not np.allclose(df["p_idle"], df["p_2q"], rtol=1e-12, atol=0):
        raise ValueError(
            "benchmark CSV does not use p_idle = p_2q; rerun the experiment"
        )

    expected_volume = df["total_qubits"] * df["cnot_depth"]
    if not (df["spacetime_volume"] == expected_volume).all():
        raise ValueError("benchmark CSV violates spacetime volume QD")

    verification = pd.to_numeric(df["verification_error"], errors="coerce")
    verified_rows = (
        (pd.to_numeric(df["N"], errors="coerce") <= 16)
        & df["method"].isin({"1d_baseline", "gamma_2d_proper"})
    )
    if (
        verification[verified_rows].isna().any()
        or (verification[verified_rows] < 0).any()
        or (verification[verified_rows] > 1e-10).any()
    ):
        raise ValueError(
            "benchmark CSV has missing or excessive small-system FFFT "
            "verification error; rerun the experiment"
        )
    if verification[~verified_rows].notna().any():
        raise ValueError(
            "benchmark CSV reports FFFT verification for an unsupported "
            "method or system size"
        )

    expected_fidelity = (
        (1.0 - df["p_2q"]) ** df["total_cnot_equiv"]
        * (1.0 - df["p_idle"]) ** df["total_idle_slots_layered"]
    )
    if not np.isclose(
        df["mult_fidelity"],
        expected_fidelity,
        rtol=5e-12,
        atol=0,
        equal_nan=False,
    ).all():
        raise ValueError(
            "benchmark CSV no-fault estimate does not match its "
            "CNOT-equivalent layered counts; "
            "rerun the experiment"
        )

    resource_columns = [
        "total_qubits", "cnot_depth", "total_cnot_equiv",
        "total_idle_slots_layered", "n_1q_nonz", "n_exact_t",
        "n_synth_rz", "spacetime_volume",
        "verification_error",
    ]
    repeated = df.groupby(["L", "method"], sort=False)[
        resource_columns
    ].nunique(dropna=False)
    if (repeated != 1).any().any():
        raise ValueError(
            "benchmark CSV has physical-error-dependent resource or "
            "rotation counts"
        )

    if require_publication_coverage:
        expected_methods = {
            "1d_baseline", "gamma_2d_core", "gamma_2d_ancilla",
            "gamma_2d_proper",
        }
        expected = {
            (int(L), method, round(float(p), 15))
            for L in PUBLICATION_L_VALUES
            for method in expected_methods
            for p in PUBLICATION_P_VALUES
        }
        observed = {
            (int(row.L), str(row.method), round(float(row.p_2q), 15))
            for row in df[["L", "method", "p_2q"]].itertuples(index=False)
        }
        if observed != expected:
            missing_keys = len(expected - observed)
            extra_keys = len(observed - expected)
            raise ValueError(
                "benchmark CSV does not have exact publication coverage "
                f"({missing_keys} missing, {extra_keys} extra keys)"
            )


def _method_depths_from_csv(df: pd.DataFrame, method: str) -> dict[int, int]:
    """Return one authoritative CNOT-equivalent depth per ``L`` for ``method``.

    Resource counts are repeated once per physical error rate in ``data.csv``.
    A paper plot must therefore reject inconsistent repetitions rather than
    silently choosing one of them.
    """
    rows = df.loc[df["method"] == method, ["L", "cnot_depth"]]
    if rows.empty:
        raise ValueError(f"benchmark CSV has no rows for method {method!r}")

    inconsistent = rows.groupby("L")["cnot_depth"].nunique(dropna=False)
    bad_L = inconsistent[inconsistent != 1].index.tolist()
    if bad_L:
        raise ValueError(
            f"benchmark CSV has inconsistent {method!r} depths at L={bad_L}"
        )

    depths: dict[int, int] = {}
    for row in rows.drop_duplicates("L").itertuples(index=False):
        value = float(row.cnot_depth)
        if not np.isfinite(value) or value < 0 or value != round(value):
            raise ValueError(
                f"benchmark CSV has invalid {method!r} depth {value!r} at "
                f"L={row.L}"
            )
        depths[int(row.L)] = int(value)
    return depths


def _residual_depth_from_csv(
    total: int,
    preceding_stages: list[int],
    *,
    method: str,
    L: int,
) -> int:
    """Close a plotted stage decomposition exactly against its CSV total."""
    residual = int(total) - sum(int(value) for value in preceding_stages)
    if residual < 0:
        raise ValueError(
            f"CSV depth {total} for {method!r} at L={L} is smaller than "
            f"the {sum(preceding_stages)} layers in its preceding stages"
        )
    return residual


def _assert_breakdown_totals(
    depths: list[list[int]],
    L_values: list[int],
    expected: dict[int, int],
    *,
    method: str,
) -> None:
    """Guard the paper-facing stacked bars against a second source of truth."""
    observed = {
        int(L): sum(int(stage[j]) for stage in depths)
        for j, L in enumerate(L_values)
    }
    mismatches = {
        L: (observed[L], expected[L])
        for L in L_values
        if observed[L] != expected[L]
    }
    if mismatches:
        raise AssertionError(
            f"{method!r} depth-breakdown totals disagree with data.csv: "
            f"{mismatches}"
        )


def _ancilla_fidelity_from_resources(
    resources: dict,
    p_2q: float,
    p_idle_factor: float,
) -> float:
    """Apply the same CNOT-equivalent model as the saved benchmark rows."""
    from exp2_ffft.collect import cnot_equivalent_no_fault

    return cnot_equivalent_no_fault(
        resources,
        p_2q=p_2q,
        p_idle=p_2q * p_idle_factor,
        p_1q=p_2q,
    )


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
    for ext in ("svg", "pdf"):
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
    ax.set_ylabel("CNOT-equivalent depth")
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


# -- Plot 3: independent-location no-fault probability ---------------------

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

    # A horizontal strip remains legible at a two-column journal width.  The
    # legend and axis labels are shared so the three panels spend their space
    # on data rather than repeated furniture.
    fig, axes = plt.subplots(
        1,
        len(p_values),
        figsize=(11.5, 3.6),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

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
                        label=ct_style["label"], linewidth=2.2, markersize=5.5)

        # 2) Gamma-FP-FFFT w/ ancillas
        anc_style = METHOD_STYLES["gamma_2d_ancilla"]
        anc_data = sub[sub["method"] == "gamma_2d_ancilla"].sort_values("N")
        anc_data = anc_data[anc_data["mult_fidelity"] >= y_floor]
        if not anc_data.empty:
            ax.semilogy(
                anc_data["N"], anc_data["mult_fidelity"],
                marker=anc_style["marker"], color=anc_style["color"],
                label=anc_style["label"], linewidth=2.2, markersize=5.5,
            )

        # 3) Gamma-FP-FFFT w/o ancillas
        gp_style = METHOD_STYLES["gamma_2d_proper"]
        gp_data = sub[sub["method"] == "gamma_2d_proper"].sort_values("N")
        gp_data = gp_data[gp_data["mult_fidelity"] >= y_floor]
        if not gp_data.empty:
            ax.semilogy(gp_data["N"], gp_data["mult_fidelity"],
                        marker=gp_style["marker"], color=gp_style["color"],
                        label=gp_style["label"], linewidth=2.2, markersize=5.5)

        ax.set_ylim(bottom=y_floor, top=2.0)
        exponent = int(round(np.log10(p_2q)))
        ax.set_title(rf"$p_{{2q}}=10^{{{exponent}}}$", fontsize=13)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        framealpha=0.9,
        fontsize=10,
        columnspacing=1.6,
        handlelength=2.4,
    )
    fig.supxlabel(r"$N=L^2$", fontsize=13, y=0.025)
    fig.supylabel("Independent-location no-fault probability", fontsize=13, x=0.008)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.20,
        top=0.76,
        wspace=0.10,
    )
    _save_fig(fig, fig_dir, "fidelity_vs_N")


# -- Plot 4: depth breakdown (FP-FFFT stages) -------------------------------

def plot_depth_breakdown(df: pd.DataFrame, fig_dir: str):
    """Grouped stacked-bar chart: CNOT-equivalent depth by circuit stage.

    Five methods compared side-by-side for each N = L^2:
      0  Gamma-FP-FFFT w/o ancillas  (Gamma sandwich, folded Gamma)
      1  Gamma-FP-FFFT w/ ancillas   (Gamma sandwich, ancilla Gamma)
      2  FP-FFFT w/o ancillas        (FP sandwich, folded 2D FP)
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
    from common.gamma_folded import build_gamma_folded
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
        """Split total CNOT-equivalent depth into permutation and the rest.

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

    # The provenance-validated CSV is the source of truth for the two paper
    # methods.  The canonical Hall router is deterministic, and the
    # cross-process regression separately requires its explicit stage sum to
    # equal this total.  Deriving the last plotted segment by closure also
    # ensures a plot-only run cannot drift from the collected full circuit.
    gamma_csv_depth = _method_depths_from_csv(df, "gamma_2d_proper")
    gamma_anc_csv_depth = _method_depths_from_csv(df, "gamma_2d_ancilla")
    baseline_csv_depth = _method_depths_from_csv(df, "1d_baseline")
    L_values = sorted(L for L in gamma_csv_depth if L <= 20)
    missing_ancilla = sorted(set(L_values) - set(gamma_anc_csv_depth))
    if missing_ancilla:
        raise ValueError(
            "benchmark CSV lacks gamma_2d_ancilla depth rows at "
            f"L={missing_ancilla}"
        )
    missing_baseline = sorted(set(L_values) - set(baseline_csv_depth))
    if missing_baseline:
        raise ValueError(
            "benchmark CSV lacks 1d_baseline depth rows at "
            f"L={missing_baseline}"
        )
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

    # Method 0: Gamma sandwich (folded, no ancilla)
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
    # Method 2: FP sandwich (folded 2D FP, no ancilla)
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
        _d = lambda circ, n_anc=0: count_cnot_resources(
            circ, L, n_anc
        )["cnot_depth"]

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

        # M0: Gamma sandwich (folded)
        gamma, _ = build_gamma_folded(L, sq=sq)
        gamma_d = _d(gamma)
        gamma_fixed = [
            _d(rev), gamma_d, _d(col), gamma_d, tw_d, row_d,
        ]
        for stage, value in zip(gamma_depths[:-1], gamma_fixed):
            stage.append(value)
        gamma_depths[-1].append(_residual_depth_from_csv(
            gamma_csv_depth[L],
            gamma_fixed,
            method="gamma_2d_proper",
            L=L,
        ))

        # M1: Gamma sandwich (ancilla)
        aq = make_ancilla_qubits(L)
        gamma_anc_circ, _, _ = build_gamma_with_ancillas(L, sq=sq, aq=aq)
        gamma_anc_d    = _d(gamma_anc_circ, len(aq))
        gamma_anc_fixed = [
            _d(rev), gamma_anc_d, _d(col), gamma_anc_d, tw_d, row_d,
        ]
        for stage, value in zip(gamma_anc_depths[:-1], gamma_anc_fixed):
            stage.append(value)
        gamma_anc_depths[-1].append(_residual_depth_from_csv(
            gamma_anc_csv_depth[L],
            gamma_anc_fixed,
            method="gamma_2d_ancilla",
            L=L,
        ))

        # M2: FP sandwich (folded 2D)
        fp_rt_2d = build_fp_2d(L, rt_perm,      GammaMethod.FOLDED)
        fp_t_2d  = build_fp_2d(L, trans_perm,    GammaMethod.FOLDED)
        fp_r_2d  = build_fp_2d(L, reorder_perm,  GammaMethod.FOLDED)
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
        d_perm, _ = _split_perm_comp(r_1d.circuit, L)
        baseline_depths[0].append(d_perm)
        baseline_depths[1].append(_residual_depth_from_csv(
            baseline_csv_depth[L],
            [d_perm],
            method="1d_baseline",
            L=L,
        ))

    _assert_breakdown_totals(
        gamma_depths,
        L_values,
        gamma_csv_depth,
        method="gamma_2d_proper",
    )
    _assert_breakdown_totals(
        baseline_depths,
        L_values,
        baseline_csv_depth,
        method="1d_baseline",
    )
    _assert_breakdown_totals(
        gamma_anc_depths,
        L_values,
        gamma_anc_csv_depth,
        method="gamma_2d_ancilla",
    )

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
    ax.set_ylabel("CNOT-equivalent depth", fontsize=14, fontweight="bold")
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

    # -- Zoom inset: magnify the "ours" bar (Gamma w/o ancillas) at N=144 -----
    # The rightmost bar of the N=144 cluster, blown up so its time-ordered
    # stages are readable.  Stage names sit to the right of each segment; a
    # green "Time" arrow on the left marks the bottom-to-top execution order.
    _draw_n144_zoom_inset(ax, bar_tops[4], gamma_info, gamma_depths,
                          N_values, bar_w)

    _save_fig(fig, fig_dir, "depth_breakdown")


def _draw_n144_zoom_inset(ax, ours_bar_tops, stage_info, stage_depths,
                          N_values, bar_w):
    """Inset that magnifies the Gamma-FP-FFFT w/o ancillas bar at N=144.

    ``ours_bar_tops`` is ``bar_tops[4]`` = (xpositions, cumulative tops) for
    that method.  The inset reuses the exact per-stage depths so it is a
    faithful zoom of the real bar (zero-depth stages such as Twiddle at this
    N are omitted).
    """
    if 144 not in N_values:
        return
    zi = N_values.index(144)

    z_xpos, z_tops = ours_bar_tops
    src_xc  = z_xpos[zi]
    src_xr  = src_xc + bar_w / 2.0        # right edge of the source bar
    src_top = z_tops[zi]

    seg_vals  = [d[zi] for d in stage_depths]
    seg_names = [lbl for (lbl, _, _) in stage_info]
    seg_cols  = [c   for (_, c, _) in stage_info]

    # -- Inset axes in the empty band to the right of the legends -----------
    inset = ax.inset_axes([0.340, 0.50, 0.245, 0.46])
    inset.set_xlim(0, 1)
    inset.set_ylim(0, src_top * 1.05)
    for sp in inset.spines.values():
        sp.set_visible(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.patch.set_alpha(0.0)

    # -- Stacked bar (large) -------------------------------------------------
    bar_cx, bar_hw = 0.34, 0.085          # center & half-width (inset x)
    cum, mids = 0.0, []
    for val, col in zip(seg_vals, seg_cols):
        if val > 0:
            inset.bar(bar_cx, val, bar_hw * 2, bottom=cum, color=col,
                      edgecolor="white", linewidth=0.7, zorder=3)
        mids.append(cum + val / 2.0)
        cum += val

    # -- Stage labels on the right, spread to avoid overlap -----------------
    keep   = [k for k in range(len(seg_vals)) if seg_vals[k] > 0]
    mid_ys = [mids[k] for k in keep]
    min_gap = src_top * 0.092
    lab_ys = list(mid_ys)
    for _ in range(80):                   # iterative de-overlap
        for a in range(1, len(lab_ys)):
            d = lab_ys[a] - lab_ys[a - 1]
            if d < min_gap:
                shift = (min_gap - d) / 2.0
                lab_ys[a - 1] -= shift
                lab_ys[a]     += shift
        lab_ys[0] = max(lab_ys[0], min_gap * 0.55)

    label_x = bar_cx + bar_hw + 0.06
    for k, ly in zip(keep, lab_ys):
        inset.plot([bar_cx + bar_hw, label_x - 0.015], [mids[k], ly],
                   color="#9a9a9a", lw=0.7, zorder=2, clip_on=False)
        inset.text(label_x, ly, seg_names[k], ha="left", va="center",
                   fontsize=9, fontweight="bold", color="#2b2b2b",
                   clip_on=False)

    # -- Green "Time" arrow on the left (bottom -> top execution order) ------
    inset.annotate("", xy=(0.085, 0.80), xytext=(0.085, 0.10),
                   xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color="#8ec96e",
                                   lw=3.0, mutation_scale=14))
    inset.text(0.02, 0.45, "Time", rotation=90, transform=inset.transAxes,
               ha="center", va="center", fontsize=11,
               color="#5a9a3a", fontweight="bold")

    # -- Highlight the source bar + light zoom-connector lines ---------------
    # Connectors are drawn *behind* the bars (low zorder) so they tuck behind
    # intervening clusters and only show across empty gaps; the highlight box
    # (on top) marks exactly which bar is magnified.
    from matplotlib.patches import ConnectionPatch, Rectangle

    src_xl = src_xc - bar_w / 2.0
    pad_x  = bar_w * 0.22
    pad_y  = src_top * 0.035
    ax.add_patch(Rectangle(
        (src_xl - pad_x, -pad_y), bar_w + 2 * pad_x, src_top + 2 * pad_y,
        fill=False, edgecolor="#8f8f8f", lw=1.1, alpha=0.95, zorder=6))

    rxr = src_xl + bar_w + pad_x          # rectangle right edge (x, data)
    for yA, xyB in [(src_top + pad_y, (1.0, 0.0)), (-pad_y, (0.0, 0.0))]:
        ax.add_patch(ConnectionPatch(
            xyA=(rxr, yA), coordsA=ax.transData,
            xyB=xyB, coordsB=inset.transAxes,
            color="#a6a6a6", lw=1.0, alpha=0.9, zorder=0.5))


# -- Entry point -------------------------------------------------------------

def generate_all_plots(
    csv_path: str = "exp2_ffft/results/data.csv",
    fig_dir: str = "exp2_ffft/figures",
    *,
    require_publication_coverage: bool = True,
):
    """Generate all Exp 2 plots from folded-Gamma benchmark data."""
    df = pd.read_csv(csv_path)
    _validate_folded_provenance(
        df, require_publication_coverage=require_publication_coverage,
    )
    plot_depth_vs_N(df, fig_dir)
    plot_spacetime_vs_N(df, fig_dir)
    plot_fidelity_vs_N(df, fig_dir)
    plot_depth_breakdown(df, fig_dir)
    print(f"All plots saved to {fig_dir}/")
