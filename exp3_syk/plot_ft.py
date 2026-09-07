"""Fault-tolerant sparse-SYK resource and success-estimate figure.

The model follows the FLASQ block convention: one block is one logical patch
for one logical CNOT-equivalent timestep, so active and idle patches are both
charged.  FSWAP depth is already two CNOT-equivalent layers in the collected
data.  Logical spacetime volume and factory volume are additive.

Usage:
    python -m exp3_syk.plot_ft
    python -m exp3_syk.plot_ft --alpha 1 --distance 11
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp3_syk.plot import validate_publication_data


C_T = 70
P_PHYS = 1e-3
P_THRESHOLD = 1e-2
PREFAC = 0.03
DISPLAY_FLOOR = 1e-2
PANEL_WIDTH_RATIOS = (3, 1)

# --- print geometry -----------------------------------------------------
# These figures are included in the manuscript at revtex4-2 \textwidth
# (510pt = 7.083in).  They used to be authored at 12.4in and left for LaTeX
# to shrink by 0.576, which dropped the 7.2pt legend text to 4.1pt on the
# printed page.  Authoring at the final width instead means the sizes below
# are the sizes the reader gets.  The ladder is set so that mathtext
# sub/superscripts, which matplotlib renders at ~0.69 of the base with no
# separate size knob, still clear 6pt: primary text >=8.7pt, scripts >=6pt.
PAPER_WIDTH_IN = 7.083

# --- stroke ladder ---------------------------------------------------------
# The same three rungs the TikZ figures use (paper/figures/src/gamma-fig-style.tex),
# fixed against the manuscript's body-text stem, measured at 0.667pt on the
# compiled page.  These figures are authored at the printed width, so a value
# here is the width the reader gets.
#
# Before this, the file carried seven distinct widths from 0.45 to 1.8pt and the
# figure printed at 1.20x the text stem at the median -- visibly heavier than the
# prose around it.  Marker sizes are deliberately NOT reduced: shrinking them is
# what made the three magic-inventory series in the inset overlap.
W_HAIR = 0.50   # 0.75x the stem: grids, bar edges, threshold rules, error bars
W_DATA = 0.70   # 1.05x: the series themselves
W_EMPH = 0.90   # 1.35x: reserved for a mark a caption points at
ANNOTATION_COLOR = "#555555"   # the drop bracket and its label

PAPER_HEIGHT_IN = 3.2
FS_BASE = 8.0      # tick labels and default text
FS_LABEL = 8.0     # axis labels
FS_TITLE = 8.5     # panel titles
FS_SMALL = 7.5     # legends, annotations, inset labels
FS_FLOOR = 7.5     # inset tick labels: the smallest type in the figure

VOLUME_DISPLAY_SCALE = 1e7
DEFAULT_ALPHA = 1.0
INSET_FACE_COLOR = "#f7f2e8"
INSET_EDGE_COLOR = "#b8aa92"
# The legend occupies roughly the upper 26% of the axes at upper left, so an
# inset topping out at 0.74 had its title AND its offset label drawn beneath
# the legend patch, which is semi-transparent -- both were washed out and the
# title unreadable.  exp2_ffft already carries this fix; this file did not.
# Top is now 0.30 + 0.34 = 0.64, matching exp2_ffft's clearance.
INSET_BOUNDS = (0.06, 0.30, 0.42, 0.34)

METHODS_VOLUME = ["1d", "ancilla", "folded"]
METHODS_FIDELITY = ["naive_pauli", "1d", "ancilla", "folded"]
METHOD_STYLE = {
    "naive_pauli": ("Naive Pauli", "v", "#2ca02c"),
    "1d": ("FSWAP baseline", "s", "#555555"),
    "ancilla": ("FP w/ ancillas", "^", "#1f77b4"),
    "folded": (r"FP w/o ancillas (folded $\Gamma$)", "o", "#d62728"),
}


def logical_cycle_error_rate(
    distance: int,
    p_phys: float = P_PHYS,
    p_threshold: float = P_THRESHOLD,
) -> float:
    """Return the logical error probability for one surface-code cycle."""
    return PREFAC * (p_phys / p_threshold) ** ((distance + 1) / 2)


def logical_block_error_rate(
    distance: int,
    p_phys: float = P_PHYS,
    p_threshold: float = P_THRESHOLD,
) -> float:
    """Return the error probability for a distance-``d`` logical block.

    A logical CNOT-equivalent block occupies ``distance`` code cycles.  The
    ``log1p``/``expm1`` form evaluates ``1-(1-p_cyc)**distance`` accurately
    when the per-cycle probability is small.
    """
    if int(distance) != distance or distance <= 0:
        raise ValueError("distance must be a positive integer")
    p_cycle = logical_cycle_error_rate(distance, p_phys, p_threshold)
    if not 0.0 <= p_cycle < 1.0:
        raise ValueError("logical cycle error probability must lie in [0, 1)")
    return float(-np.expm1(int(distance) * np.log1p(-p_cycle)))


def truncate_below_display_floor(
    values,
    floor: float = DISPLAY_FLOOR,
) -> np.ndarray:
    """Replace each sub-floor finite value by NaN for plotting.

    Masking pointwise, instead of slicing at the first small value, preserves
    any later above-floor segment in a nonmonotonic series.
    """
    if floor < 0:
        raise ValueError("display floor must be nonnegative")
    displayed = np.asarray(values, dtype=float).copy()
    displayed[np.isfinite(displayed) & (displayed < floor)] = np.nan
    return displayed


def load_ft(
    csv_path: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    distance: int = 11,
    require_publication_coverage: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Aggregate logical block volume and success estimates over instances."""
    df = pd.read_csv(csv_path)
    required = {"n_1q_rz"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "SYK CSV lacks CNOT-equivalent-accounting columns "
            f"{missing}; rerun collection"
        )
    validate_publication_data(
        df, require_publication_coverage=require_publication_coverage,
    )
    if "folded" not in set(df["baseline"]):
        raise ValueError("SYK CSV has no folded-Gamma rows")

    p_values = sorted(df["p_2q"].unique())
    reference_p = p_values[0]
    df = df[(df.k == 1.0) & (df.p_2q == reference_p)].copy()
    df = df[df.baseline.isin(METHODS_FIDELITY)]

    p_b = logical_block_error_rate(distance)
    df["S"] = df.total_qubits * df.cnot_depth
    df["n_T"] = C_T * df.n_1q_rz
    df["V_T"] = alpha * df.n_T
    df["S_plus_V_T"] = df.S + df.V_T
    # Conservative independent-block no-fault proxy.  S covers the
    # algorithmic schedule, n_T the state-consumption step, and V_T the
    # preparation service.  A separate cultivated-state factor would charge
    # the factory faults represented by V_T a second time.
    log_f = (df.S + df.n_T + df.V_T) * np.log1p(-p_b)
    df["F"] = np.exp(log_f)

    grouped = (
        df.groupby(["N", "baseline"])[
            ["S", "V_T", "S_plus_V_T", "n_T", "F"]
        ]
        .agg(["mean", "std"])
    )
    grouped.columns = ["_".join(col) for col in grouped.columns]
    return grouped.reset_index(), p_b


def _series(data: pd.DataFrame, baseline: str, column: str, n_values):
    sub = data[data.baseline == baseline].set_index("N")
    return np.array([sub.loc[n, column] for n in n_values], dtype=float)


def _draw_spacetime_reduction_arrows(
    ax,
    n_values,
    positions,
    arrow_heights,
    comparison_values,
) -> list[float]:
    """Draw routing-volume comparisons above the routing-volume bars."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MPath

    old_x = positions["1d"]
    new_x = positions["folded"]
    old_y = arrow_heights["1d"]
    new_y = arrow_heights["folded"]
    middle_y = arrow_heights["ancilla"]
    old_comparison = comparison_values["1d"]
    new_comparison = comparison_values["folded"]

    winning_indices = [
        i for i in range(len(n_values))
        if new_comparison[i] < old_comparison[i]
    ]
    if not winning_indices:
        return []
    requested_sizes = [
        n_values[winning_indices[0]],
        100,
        225,
        max(n_values),
    ]
    annotated_indices = []
    for size in requested_sizes:
        if size in n_values:
            i = n_values.index(size)
            if i in winning_indices and i not in annotated_indices:
                annotated_indices.append(i)

    tops = []
    global_max = max(
        float(np.nanmax(values)) for values in arrow_heights.values()
    )

    for i in annotated_indices:
        x0, x1 = old_x[i], new_x[i]
        y0, y1 = old_y[i], new_y[i]
        reduction = 100.0 * (
            old_comparison[i] - new_comparison[i]
        ) / old_comparison[i]

        # Two-segment quadratic path with a rounded plateau at the apex,
        # matching exp2.  A single quadratic control point sends the control
        # point far off the axis when the endpoints differ by most of the range,
        # producing a needle instead of an arch.  The apex clears the group's
        # tallest bar, which at N=225 is the ancilla bar rather than the
        # comparator, so the arc passes over the middle bar either way.
        start_y, end_y = float(y0), float(y1)
        middle_x = 0.5 * (x0 + x1)
        local_max = max(start_y, float(middle_y[i]), end_y)
        # The clearance needs a floor in AXIS units, not just a fraction of the
        # local bars: at N=25 the bars are ~1.5% of the axis, so a clearance of
        # 0.015*global_max made the arch about 2pt tall and effectively invisible.
        # 0.045*global_max keeps every arch a legible dome at its own scale.
        clearance = max(0.050 * local_max, 0.045 * global_max)
        apex_y = local_max + clearance
        left_ctrl_x = 0.5 * (x0 + middle_x)
        right_ctrl_x = 0.5 * (middle_x + x1)

        ax.add_patch(FancyArrowPatch(
            path=MPath(
                [
                    np.array([x0, start_y]),
                    np.array([left_ctrl_x, apex_y]),
                    np.array([middle_x, apex_y]),
                    np.array([right_ctrl_x, apex_y]),
                    np.array([x1, end_y]),
                ],
                [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.CURVE3, MPath.CURVE3],
            ),
            arrowstyle="-",
            color=ANNOTATION_COLOR,
            lw=W_DATA,
            zorder=2.8,
        ))
        ax.annotate(
            f"{reduction:.0f}%",
            xy=(middle_x, apex_y),
            xytext=(0, 2.2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FS_SMALL,
            fontweight="bold",
            color=ANNOTATION_COLOR,
            zorder=7,
        )
        tops.append(apex_y + 0.035 * global_max)

    return tops


def _add_magic_inset(ax, data: pd.DataFrame, n_values) -> None:
    """Show the raw magic inventory for every routing method."""
    l_values = np.sqrt(n_values).astype(int)
    inset = ax.inset_axes(INSET_BOUNDS)
    inset.set_facecolor(INSET_FACE_COLOR)
    inset.patch.set_alpha(0.98)

    inset_styles = {
        "1d": {
            "linestyle": "--", "linewidth": 1.45, "markersize": 3.8,
            "markerfacecolor": "#555555", "markeredgewidth": 0.8, "zorder": 3,
        },
        "ancilla": {
            "linestyle": "-.", "linewidth": 1.20, "markersize": 4.5,
            "markerfacecolor": "none", "markeredgewidth": 0.9, "zorder": 4,
        },
        "folded": {
            "linestyle": "-", "linewidth": 0.85, "markersize": 3.0,
            "markerfacecolor": "none", "markeredgewidth": 0.9,
            "alpha": 0.82, "zorder": 5,
        },
    }
    for method in METHODS_VOLUME:
        _, marker, color = METHOD_STYLE[method]
        inset.plot(
            l_values,
            _series(data, method, "n_T_mean", n_values),
            marker=marker,
            color=color,
            markeredgecolor=color,
            **inset_styles[method],
        )

    inset.set_xlim(3.6, 20.4)
    inset.set_xticks([4, 8, 12, 16, 20])
    inset.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True,
    )
    inset.yaxis.get_offset_text().set_fontsize(FS_FLOOR)
    inset.set_xlabel(r"$L$", fontsize=FS_SMALL, labelpad=0)
    inset.set_ylabel(r"$n_T$", fontsize=FS_SMALL, labelpad=0)
    inset.yaxis.set_label_coords(-0.075, 0.5)
    inset.set_title(r"Magic inventory $n_T$", fontsize=FS_SMALL, pad=2.0)
    inset.tick_params(axis="both", which="both", labelsize=FS_FLOOR, length=2.5, pad=1)
    inset.grid(color="#8b8174", alpha=0.16, linewidth=W_HAIR)
    for spine in inset.spines.values():
        spine.set_color(INSET_EDGE_COLOR)
        spine.set_linewidth(0.7)


def _panel_volume(ax, data: pd.DataFrame, alpha: float) -> None:
    n_values = sorted(data.N.unique())
    x = np.arange(len(n_values), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]
    arrow_heights = {}
    comparison_values = {}
    positions = {}
    for offset, method in zip(offsets, METHODS_VOLUME):
        label, marker, color = METHOD_STYLE[method]
        s_mean_raw = _series(data, method, "S_mean", n_values)
        s_std_raw = _series(data, method, "S_std", n_values)
        s_mean = s_mean_raw / VOLUME_DISPLAY_SCALE
        s_std = s_std_raw / VOLUME_DISPLAY_SCALE
        xpos = x + offset
        ax.bar(
            xpos, s_mean, width, color=color, alpha=0.26,
            edgecolor=color, linewidth=W_HAIR, zorder=2,
        )
        ax.errorbar(xpos, s_mean, yerr=s_std, fmt="none", ecolor=color,
                    elinewidth=W_HAIR, capsize=1.5, zorder=3)
        ax.scatter(xpos, s_mean, marker=marker, color=color, s=18,
                   label=label, zorder=4)
        arrow_heights[method] = (
            s_mean + np.nan_to_num(s_std, nan=0.0)
        )
        comparison_values[method] = s_mean_raw
        positions[method] = xpos

    arrow_tops = _draw_spacetime_reduction_arrows(
        ax, list(n_values), positions, arrow_heights, comparison_values,
    )

    ax.legend(fontsize=FS_SMALL, framealpha=0.95, loc="upper left")
    _add_magic_inset(ax, data, n_values)
    ax.set_xticks(x)
    ax.set_xticklabels(n_values, rotation=45, ha="right")
    ax.set_xlabel(r"$N=L^2$")
    ax.set_ylabel(r"Logical spacetime volume $S$ ($10^7$ blocks)")
    ax.set_title("(a) Logical spacetime volume")
    data_top = max(
        np.max(
            (
                _series(data, method, "S_mean", n_values)
            + np.nan_to_num(
                _series(data, method, "S_std", n_values), nan=0.0,
            )
            ) / VOLUME_DISPLAY_SCALE
        )
        for method in METHODS_VOLUME
    )
    if arrow_tops:
        data_top = max(data_top, max(arrow_tops))
    ax.set_ylim(0.0, 1.06 * data_top)
    ax.grid(axis="y", alpha=0.2)


def _panel_fidelity(ax, data: pd.DataFrame, distance: int, p_b: float) -> None:
    n_values = sorted(data.N.unique())
    compact_labels = {
        "naive_pauli": "Naive Pauli",
        "1d": "FSWAP",
        "ancilla": "FP w/ anc.",
        "folded": "Folded FP",
    }
    for method in METHODS_FIDELITY:
        _, marker, color = METHOD_STYLE[method]
        mean = _series(data, method, "F_mean", n_values)
        std = _series(data, method, "F_std", n_values)
        displayed = truncate_below_display_floor(mean)
        visible = np.isfinite(displayed)
        lower = np.where(visible, np.maximum(DISPLAY_FLOOR, mean - std), np.nan)
        upper = np.where(visible, np.minimum(1.0, mean + std), np.nan)
        ax.plot(
            n_values,
            displayed,
            marker=marker,
            color=color,
            linewidth=W_DATA,
            markersize=4,
            label=compact_labels[method],
        )
        ax.fill_between(n_values, lower, upper, color=color, alpha=0.10)

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=W_HAIR)
    ax.annotate(
        r"$F_{\mathrm{FT}}=0.5$",
        # Right-aligned: at the printed font size this label is wide enough
        # that the old left placement put it across the Naive Pauli curve,
        # which crosses 0.5 near N = 85.  The top right of the panel is clear.
        xy=(0.98, 0.5),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=FS_SMALL,
    )
    ax.set_xlabel(r"$N=L^2$")
    ax.set_ylabel("Estimated no-fault probability")
    ax.set_yscale("log")
    ax.set_ylim(DISPLAY_FLOOR, 1.05)
    ax.set_title(rf"(b) Estimated no-fault probability ($d={distance}$)")
    ax.grid(which="major", alpha=0.2)
    ax.legend(
        fontsize=FS_SMALL,
        loc="lower right",
        framealpha=0.95,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )


def make_ft_figure(
    data: pd.DataFrame,
    *,
    alpha: float,
    distance: int,
    p_b: float,
):
    """Build the 3:1 resource/success figure without saving it."""
    plt.rcParams.update({
        "font.size": FS_BASE,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(PAPER_WIDTH_IN, PAPER_HEIGHT_IN),
        gridspec_kw={"width_ratios": PANEL_WIDTH_RATIOS},
    )
    _panel_volume(axes[0], data, alpha)
    _panel_fidelity(axes[1], data, distance, p_b)
    fig.tight_layout(w_pad=1.4)
    return fig


def generate_ft_figure(
    csv_path: str = "exp3_syk/results/data.csv",
    output_dir: str = "exp3_syk/figures",
    *,
    alpha: float = DEFAULT_ALPHA,
    distance: int = 11,
    require_publication_coverage: bool = True,
) -> pd.DataFrame:
    data, p_b = load_ft(
        csv_path,
        alpha=alpha,
        distance=distance,
        require_publication_coverage=require_publication_coverage,
    )
    fig = make_ft_figure(
        data,
        alpha=alpha,
        distance=distance,
        p_b=p_b,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = "FP-exp3_combined_ft"
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="exp3_syk/results/data.csv")
    parser.add_argument("--output-dir", default="exp3_syk/figures")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--distance", type=int, default=11)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow a deliberately reduced non-publication benchmark table",
    )
    args = parser.parse_args()
    data = generate_ft_figure(
        args.csv,
        args.output_dir,
        alpha=args.alpha,
        distance=args.distance,
        require_publication_coverage=not args.allow_partial,
    )
    print(
        f"Saved FT figure for {data.N.nunique()} sizes to {args.output_dir}/"
    )


if __name__ == "__main__":
    main()
