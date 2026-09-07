"""Experiment 1: plotting functions.

Generates publication-quality plots for FP benchmarking results.
Each plot type has subplots for each permutation type (reverse, transpose, random).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp1_fp.collect import (
    ACCOUNTING_MODEL,
    REPRODUCIBILITY_COLUMNS,
    SAMPLING_MODEL,
    _gamma_schedule_model_for_baseline,
    _hall_model_for_baseline,
    _stim_seed_for_row,
    has_valid_reproducibility_provenance,
)

# ---------------------------------------------------------------------------
# Style & constants
# ---------------------------------------------------------------------------

BASELINE_STYLES = {
    "1d":        {"color": "#555555", "marker": "s", "label": "FSWAP baseline"},
    "ancilla":   {"color": "#1f77b4", "marker": "^", "label": "FP w/ ancillas"},
    "primitive":  {"color": "#ff7f0e", "marker": "D", "label": "Primitive \u0393"},
    "pipelined":  {"color": "#d62728", "marker": "o", "label": "FP w/o ancillas"},
    # The paper caption identifies this curve as the folded construction;
    # keeping the in-panel label short prevents the shared legend from
    # intruding into the neighboring subplot.
    "folded":     {"color": "#d62728", "marker": "o",
                   "label": "FP w/o ancillas"},
}

PERM_TITLES = {
    "reverse": "Reversal",
    "transpose": "2D Reflection",
    "random": "Random",
}

# Full list (used for data aggregation, keeps primitive in the CSV).
BASELINES_ORDER = ["1d", "ancilla", "primitive", "pipelined", "folded"]

# Primitive is hidden.  New data use folded; pipelined remains an explicit
# fallback so historical CSVs can be audited without being relabeled.
PLOT_BASELINES_PREFIX = ["1d", "ancilla"]
STIM_PLOT_P_VALUES = (1e-5, 1e-4)


def _validate_native_data(df: pd.DataFrame) -> None:
    """Reject stale or internally inconsistent data before plotting."""
    required = {
        "L", "N", "perm_kind", "perm_idx", "baseline", "accounting_model",
        "hall_decomposition_model", "gamma_schedule_model", "sampling_model",
        "stim_seed",
        *REPRODUCIBILITY_COLUMNS,
        "total_qubits", "cnot_depth", "total_cnots", "n_fswap",
        "n_2q_cnot_cz", "total_idle_slots_layered", "spacetime_volume",
        "p_2q", "p_idle", "mult_fidelity", "stim_fidelity", "stim_shots",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Experiment 1 data lack CNOT-equivalent-accounting columns: "
            + ", ".join(sorted(missing))
        )
    if df.empty:
        raise ValueError("Experiment 1 data are empty")
    if set(df["accounting_model"].astype(str)) != {ACCOUNTING_MODEL}:
        raise ValueError(
            f"Experiment 1 plots require accounting_model={ACCOUNTING_MODEL}"
        )
    hall_expected = df["baseline"].astype(str).map(_hall_model_for_baseline)
    if not (df["hall_decomposition_model"].astype(str) == hall_expected).all():
        raise ValueError("Experiment 1 data have stale Hall-decomposition provenance")
    gamma_schedule_expected = df.apply(
        lambda row: _gamma_schedule_model_for_baseline(
            str(row.baseline), int(row.L)
        ),
        axis=1,
    )
    if not (
        df["gamma_schedule_model"].astype(str) == gamma_schedule_expected
    ).all():
        raise ValueError("Experiment 1 data have stale Gamma-schedule provenance")
    if set(df["sampling_model"].astype(str)) != {SAMPLING_MODEL}:
        raise ValueError(
            f"Experiment 1 plots require sampling_model={SAMPLING_MODEL}"
        )
    seed_expected = df.apply(
        lambda row: _stim_seed_for_row(
            int(row.L), str(row.perm_kind), int(row.perm_idx),
            str(row.baseline), float(row.p_2q), float(row.p_idle),
        ),
        axis=1,
    )
    if not np.array_equal(
        df["stim_seed"].astype("int64"), seed_expected.astype("int64")
    ):
        raise ValueError("Experiment 1 data contain stale row-specific Stim seeds")
    if not has_valid_reproducibility_provenance(df):
        raise ValueError(
            "Experiment 1 data have stale permutation or dependency provenance"
        )
    if set(df["baseline"].astype(str)) != set(BASELINES_ORDER):
        raise ValueError(
            "Experiment 1 data must contain every baseline, including folded"
        )
    keys = ["L", "perm_kind", "perm_idx", "baseline", "p_2q"]
    if df.duplicated(keys).any():
        raise ValueError("Experiment 1 data contain duplicate experiment rows")
    if not np.allclose(df["p_idle"], df["p_2q"], rtol=1e-12, atol=0):
        raise ValueError(
            "Experiment 1 publication data do not use p_idle = p_2q"
        )

    expected_cnots = 2 * df["n_fswap"] + df["n_2q_cnot_cz"]
    expected_idle = df["total_qubits"] * df["cnot_depth"] - 2 * df["total_cnots"]
    expected_volume = df["total_qubits"] * df["cnot_depth"]
    expected_fidelity = (
        (1 - df["p_2q"]) ** df["total_cnots"]
        * (1 - df["p_idle"]) ** df["total_idle_slots_layered"]
    )
    if not (df["N"] == df["L"] ** 2).all():
        raise ValueError("Experiment 1 data violate N=L^2")
    if not (df["total_cnots"] == expected_cnots).all():
        raise ValueError(
            "Experiment 1 data violate FSWAP=2 CNOT-equivalent accounting"
        )
    if not (df["total_idle_slots_layered"] == expected_idle).all():
        raise ValueError("Experiment 1 data violate the layered idle invariant")
    if not (df["spacetime_volume"] == expected_volume).all():
        raise ValueError("Experiment 1 data contain inconsistent spacetime volume")
    if not np.allclose(df["mult_fidelity"], expected_fidelity, rtol=1e-12, atol=0):
        raise ValueError("Experiment 1 data contain an inconsistent native no-fault estimate")
    if (df["stim_shots"] <= 0).any() or df["stim_fidelity"].isna().any():
        raise ValueError("Experiment 1 plots require completed Stim samples")
    if not df["stim_fidelity"].between(0, 1).all():
        raise ValueError("Experiment 1 Stim return probabilities must lie in [0,1]")


def _preferred_ancilla_free(available) -> str:
    names = set(available)
    if "folded" in names:
        return "folded"
    return "pipelined"


def _plot_baselines(data: dict) -> list[str]:
    return PLOT_BASELINES_PREFIX + [_preferred_ancilla_free(data)]


def _setup_style():
    plt.rcParams.update({
        "font.size": 15,
        "font.weight": "bold",
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "axes.titlesize": 19,
        "axes.titleweight": "bold",
        "legend.fontsize": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save_fig(fig, name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "svg"):
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
    """Share the full visible y-range across a row and label it once."""
    limits = [axes[row, col].get_ylim() for col in range(n_cols)]
    y_lo = min(limit[0] for limit in limits)
    y_hi = max(limit[1] for limit in limits)
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
    """CNOT-equivalent entangling depth vs N = L^2 for each permutation."""
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

        for baseline in _plot_baselines(data):
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
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)

    _shared_y_row(axes, 0, n_cols, "CNOT depth")
    for col in range(n_cols):
        axes[0, col].ticklabel_format(
            axis="y", style="sci", scilimits=(2, 2)
        )
    axes[0, 0].yaxis.get_offset_text().set_fontsize(13)
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

        for baseline in _plot_baselines(data):
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
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)

    # Shared y-axis with scientific notation.
    _shared_y_row(axes, 0, n_cols, "Spacetime volume")
    for col in range(n_cols):
        axes[0, col].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].yaxis.get_offset_text().set_fontsize(13)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    _save_fig(fig, "spacetime_vs_N", output_dir)


# ---------------------------------------------------------------------------
# Stim all-zero return-probability plot
# ---------------------------------------------------------------------------

def plot_stim_fidelity(df: pd.DataFrame, output_dir: str = "exp1_fp/figures"):
    """Stim all-zero return probability vs N = L^2, one row per noise rate.

    Log-scale y-axis.  Zero-valued probabilities (simulation floor) are dropped
    so that lines end cleanly instead of plunging.  Each row's x-range is
    tailored to the rightmost non-zero datapoint across all baselines.
    """
    _setup_style()
    perm_kinds = _get_perm_kinds(df)
    available_p = set(float(p) for p in df["p_2q"].unique())
    missing_p = [p for p in STIM_PLOT_P_VALUES if p not in available_p]
    if missing_p:
        raise ValueError(
            "Experiment 1 data lack requested Stim display rates: "
            + ", ".join(f"{p:.0e}" for p in missing_p)
        )
    p_values = list(STIM_PLOT_P_VALUES)
    n_cols = len(perm_kinds)

    fig, axes = plt.subplots(len(p_values), n_cols,
                             figsize=(4.2 * n_cols, 3.55 * len(p_values)),
                             squeeze=False)

    for row, p_2q in enumerate(p_values):
        df_p = df[df["p_2q"] == p_2q]
        max_N_in_row = 0

        for col, kind in enumerate(perm_kinds):
            ax = axes[row, col]
            data = _aggregate(df_p[df_p["perm_kind"] == kind], "stim_fidelity", kind)

            for baseline in _plot_baselines(data):
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
            exponent = int(round(np.log10(float(p_2q))))
            ax.set_title(rf"{PERM_TITLES[kind]}, $p=10^{{{exponent}}}$")
            ax.grid(True, alpha=0.3)

        # Shared y per row, tailored to the middle subplot.  Retain the
        # established per-row label placement from the MICRO figure.
        _shared_y_row(axes, row, n_cols, "Process fidelity")

        # The high-fidelity row is easier to read as ordinary decimals.  A
        # fixed decimal formatter is required on both major and minor ticks;
        # ScalarFormatter on a log axis otherwise mixes labels such as 0.6
        # and 9 x 10^-1 in the same row.
        if row == 0:
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

            y_lo, y_hi = axes[row, 0].get_ylim()
            decimal_ticks = [
                value for value in (0.2, 0.4, 0.6, 0.8, 1.0)
                if y_lo <= value <= y_hi
            ]
            for col in range(n_cols):
                ax = axes[row, col]
                ax.yaxis.set_major_locator(FixedLocator(decimal_ticks))
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _position: f"{value:.1f}")
                )
                ax.yaxis.set_minor_formatter(NullFormatter())

        # Tailor x-range to effective (non-zero) data for this row.
        if max_N_in_row > 0:
            for col in range(n_cols):
                ax = axes[row, col]
                ax.set_xlim(left=None, right=max_N_in_row * 1.05)
                # Reserve the lower-right corner for the compact axis label.
                # Matplotlib otherwise places a terminal tick (for example,
                # 200 or 400) directly underneath ``N=L^2``.
                step = 100 if max_N_in_row >= 350 else 50
                ticks = np.arange(
                    0,
                    0.82 * ax.get_xlim()[1] + 0.5 * step,
                    step,
                )
                ax.set_xticks(ticks)

        axes[row, 0].legend(fontsize=13)

        # "N = L²" at the bottom-right corner, vertically centered on x-tick labels
        ax_right = axes[row, n_cols - 1]
        ax_right.annotate(
            r"$N\!=\!L^2$", xy=(1, 0), xycoords="axes fraction",
            xytext=(6, -14), textcoords="offset points",
            ha="right", va="center", fontweight="bold",
            fontsize=ax_right.xaxis.get_ticklabels()[0].get_fontsize(),
            annotation_clip=False,
        )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)

    # Align the three row labels exactly, as in the previous publication plot.
    for row in range(len(p_values)):
        axes[row, 0].yaxis.set_label_coords(-0.21, 0.5)

    _save_fig(fig, "stim_fidelity", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame, output_dir: str = "exp1_fp/figures"):
    """Generate all Experiment 1 plots."""
    _validate_native_data(df)
    print(f"Generating plots in {output_dir}/")
    plot_depth_vs_L(df, output_dir)
    plot_spacetime_vs_N(df, output_dir)
    plot_stim_fidelity(df, output_dir)
    print("All plots saved.")
