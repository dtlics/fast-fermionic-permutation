"""Early-fault-tolerant resource figure for the fermionic FFT benchmark.

The model matches the sparse-SYK analysis in ``exp3_syk.plot_ft``.  One
logical block is one logical patch for one CNOT timestep, so the logical
spacetime volume is ``S = Q D``.  The two-CNOT skeleton of
each non-Clifford FFFT beam splitter is already included in ``D``.

The additional magic-state inventory is

    n_T = n_exact_t + c_T n_synth_rz,
    V_T = alpha n_T,
    F_FT = (1-p_B) ** (S+n_T+V_T),

where the logical block error is obtained from the per-cycle logical error as
``p_B = 1 - (1-p_cyc)**d`` for a distance-``d`` surface-code block.

Only the mode-preserving CT-FFFT, ancilla-Gamma, and folded-Gamma circuits are
plotted.  The relaxed-layout 2D arithmetic core remains in the validated CSV
and returned table but is not an equivalent end-to-end transform.

Usage:
    python -m exp2_ffft.plot_ft
    python -m exp2_ffft.plot_ft --alpha 1 --distance 11
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp2_ffft.plot import _validate_folded_provenance


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

VOLUME_DISPLAY_SCALE = 1e6
DEFAULT_ALPHA = 1.0
INSET_FACE_COLOR = "#f7f2e8"
INSET_EDGE_COLOR = "#b8aa92"
# The legend occupies roughly the upper 26% of the axes at upper left, so an
# inset topping out at 0.76 had its title drawn underneath the legend patch.
INSET_BOUNDS = (0.14, 0.30, 0.42, 0.34)

METHODS_COMPARABLE = (
    "1d_baseline", "gamma_2d_ancilla", "gamma_2d_proper",
)
METHODS_ALL = (
    "1d_baseline", "gamma_2d_core", "gamma_2d_ancilla", "gamma_2d_proper",
)
METHOD_STYLE = {
    "1d_baseline": ("CT-FFFT", "s", "#555555"),
    "gamma_2d_core": ("2D arithmetic core (relaxed layout)", "^", "#f39c12"),
    "gamma_2d_ancilla": (r"$\Gamma$-FP-FFFT w/ ancillas", "^", "#1f77b4"),
    "gamma_2d_proper": (
        r"$\Gamma$-FP-FFFT w/o ancillas (folded $\Gamma$)", "o", "#d62728",
    ),
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
    c_t: int = C_T,
    distance: int = 11,
    require_publication_coverage: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Load validated FFFT rows and compute early-FT resource estimates."""
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if int(c_t) != c_t or c_t < 0:
        raise ValueError("c_t must be a nonnegative integer")

    df = pd.read_csv(csv_path)
    _validate_folded_provenance(
        df, require_publication_coverage=require_publication_coverage,
    )
    missing_methods = sorted(set(METHODS_ALL) - set(df["method"].astype(str)))
    if missing_methods:
        raise ValueError(f"FFFT CSV lacks methods {missing_methods}")

    reference_p = float(pd.to_numeric(df["p_2q"], errors="raise").min())
    data = df[np.isclose(df["p_2q"], reference_p, rtol=0.0, atol=0.0)].copy()
    data = data[data["method"].isin(METHODS_ALL)].copy()
    if data.duplicated(["N", "method"]).any():
        raise ValueError("FFFT early-FT table has duplicate (N, method) rows")

    p_b = logical_block_error_rate(distance)
    data["S"] = data["total_qubits"] * data["cnot_depth"]
    if not (data["S"] == data["spacetime_volume"]).all():
        raise ValueError("FFFT early-FT table violates S = QD")
    data["n_T"] = data["n_exact_t"] + int(c_t) * data["n_synth_rz"]
    data["V_T"] = float(alpha) * data["n_T"]
    data["S_plus_V_T"] = data["S"] + data["V_T"]

    # Conservative independent-block no-fault proxy.  S covers the
    # algorithmic schedule, n_T the state-consumption step, and V_T the
    # preparation service.  We deliberately do not multiply by a separate
    # cultivated-state output infidelity: doing so would charge the factory
    # faults already represented by V_T a second time.
    log_success = (
        data["S"] + data["n_T"] + data["V_T"]
    ) * np.log1p(-p_b)
    data["P_FT"] = np.exp(log_success)
    return data.sort_values(["N", "method"]).reset_index(drop=True), p_b


def _series(data: pd.DataFrame, method: str, column: str, n_values) -> np.ndarray:
    indexed = data[data["method"] == method].set_index("N")
    return np.asarray([indexed.loc[n, column] for n in n_values], dtype=float)


def _add_reduction_arrow(
    ax,
    *,
    x0: float,
    x_blocker: float,
    x1: float,
    y0: float,
    y_blocker: float,
    y1: float,
    global_max: float,
    label: str,
) -> float:
    """Arch the volume reduction over the group, number on top of the arc.

    Geometry: a two-segment quadratic path whose apex is a rounded plateau --
    (x0, y0) -> (apex) -> (x1, y1) with control points at the quarter positions.
    A single quadratic control point does NOT work here: solving it to put the
    crown at a given height sends the control point far off the axis when the
    two endpoints differ by most of the range (at N=400 the drop is 74%), and
    the result is a needle rather than an arch.  The plateau keeps the arc
    smooth and readable at every problem size.

    The apex clears the group's TALLEST bar, not just the comparator, so the arc
    passes over the middle bar however the three are ordered -- at N=225 in the
    SYK panel the ancilla bar is the tallest of its group.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MPath

    start_y, end_y = float(y0), float(y1)
    middle_x = 0.5 * (x0 + x1)
    local_max = max(start_y, float(y_blocker), end_y)
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
            [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3],
        ),
        arrowstyle="-",
        color=ANNOTATION_COLOR,
        lw=W_DATA,
        zorder=3,
    ))
    ax.annotate(
        label,
        xy=(middle_x, apex_y),
        xytext=(0, 2.2),
        textcoords="offset points",
        color=ANNOTATION_COLOR,
        fontsize=FS_SMALL,
        ha="center",
        va="bottom",
        fontweight="bold",
        zorder=6,
    )
    return apex_y + 0.035 * global_max
def _add_magic_inset(ax, data: pd.DataFrame, n_values) -> None:
    """Show the three raw magic inventories inside the panel."""
    l_values = np.sqrt(n_values).astype(int)
    inset = ax.inset_axes(INSET_BOUNDS)
    inset.set_facecolor(INSET_FACE_COLOR)
    inset.patch.set_alpha(0.98)

    inset_styles = {
        "1d_baseline": {
            "linestyle": "--", "linewidth": 1.45, "markersize": 3.8,
            "markerfacecolor": "#555555", "markeredgewidth": 0.8, "zorder": 3,
        },
        "gamma_2d_ancilla": {
            "linestyle": "-.", "linewidth": 1.20, "markersize": 4.5,
            "markerfacecolor": "none", "markeredgewidth": 0.9, "zorder": 4,
        },
        "gamma_2d_proper": {
            "linestyle": "-", "linewidth": 0.85, "markersize": 3.0,
            "markerfacecolor": "none", "markeredgewidth": 0.9,
            "alpha": 0.82, "zorder": 5,
        },
    }
    for method in METHODS_COMPARABLE:
        _, marker, color = METHOD_STYLE[method]
        inset.plot(
            l_values,
            _series(data, method, "n_T", n_values),
            marker=marker,
            color=color,
            markeredgecolor=color,
            **inset_styles[method],
        )

    inset.set_yscale("log")
    inset.set_xlim(3.6, 20.4)
    inset.set_xticks([4, 8, 12, 16, 20])
    inset.set_xlabel(r"$L$", fontsize=FS_SMALL, labelpad=0)
    inset.set_ylabel(r"$n_T$", fontsize=FS_SMALL, labelpad=0)
    inset.yaxis.set_label_coords(-0.075, 0.5)
    inset.set_title(r"Magic inventory $n_T$", fontsize=FS_SMALL, pad=2.0)
    inset.tick_params(axis="both", which="both", labelsize=FS_FLOOR, length=2.5, pad=1)
    inset.grid(color="#8b8174", alpha=0.16, linewidth=W_HAIR)
    for spine in inset.spines.values():
        spine.set_color(INSET_EDGE_COLOR)
        spine.set_linewidth(0.7)


def _panel_volume(ax, data: pd.DataFrame, alpha: float, c_t: int) -> None:
    plotted = data[data["method"].isin(METHODS_COMPARABLE)]
    n_values = sorted(int(n) for n in plotted["N"].unique())
    x = np.arange(len(n_values), dtype=float)
    width = 0.22
    offsets = (-width, 0.0, width)
    spacetime = {}
    positions = {}

    for offset, method in zip(offsets, METHODS_COMPARABLE):
        label, marker, color = METHOD_STYLE[method]
        s = _series(plotted, method, "S", n_values) / VOLUME_DISPLAY_SCALE
        xpos = x + offset
        ax.bar(
            xpos, s, width, color=color, alpha=0.27,
            edgecolor=color, linewidth=W_HAIR, zorder=2,
        )
        ax.scatter(
            xpos, s, marker=marker, color=color, s=19,
            label=label, zorder=4,
        )
        spacetime[method] = s
        positions[method] = xpos

    first_win = next(
        (n_values[i] for i in range(len(n_values))
         if spacetime["gamma_2d_proper"][i]
         < spacetime["1d_baseline"][i]),
        None,
    )
    global_max = max(max(values) for values in spacetime.values())
    annotated_sizes = []
    for n in (first_win, 100, max(n_values)):
        if n is not None and n not in annotated_sizes:
            annotated_sizes.append(n)
    arrow_tops = []
    for n in annotated_sizes:
        if n not in n_values:
            continue
        i = n_values.index(n)
        old = spacetime["1d_baseline"][i]
        new = spacetime["gamma_2d_proper"][i]
        if new < old:
            reduction = 100.0 * (old - new) / old
            x0 = positions["1d_baseline"][i]
            x1 = positions["gamma_2d_proper"][i]
            x_middle = positions["gamma_2d_ancilla"][i]
            middle = spacetime["gamma_2d_ancilla"][i]
            arrow_tops.append(_add_reduction_arrow(
                ax,
                x0=x0,
                x_blocker=x_middle,
                x1=x1,
                y0=old,
                y_blocker=middle,
                y1=new,
                global_max=global_max,
                label=f"{reduction:.0f}%",
            ))

    ax.legend(
        fontsize=FS_SMALL, loc="upper left",
        framealpha=0.95,
    )
    _add_magic_inset(ax, plotted, n_values)

    ax.set_xticks(x)
    ax.set_xticklabels(n_values, rotation=45, ha="right")
    ax.set_xlabel(r"$N=L^2$")
    ax.set_ylabel(r"Logical spacetime volume $S$ ($10^6$ blocks)")
    ax.set_title("(a) Logical spacetime volume")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", alpha=0.2)


def _panel_success(ax, data: pd.DataFrame, distance: int, p_b: float) -> None:
    plotted = data[data["method"].isin(METHODS_COMPARABLE)]
    n_values = sorted(int(n) for n in plotted["N"].unique())
    compact_labels = {
        "1d_baseline": "CT-FFFT",
        "gamma_2d_ancilla": r"$\Gamma$ w/ anc.",
        "gamma_2d_proper": r"Folded $\Gamma$",
    }
    visible_values = []
    for method in METHODS_COMPARABLE:
        _, marker, color = METHOD_STYLE[method]
        success = _series(plotted, method, "P_FT", n_values)
        displayed = truncate_below_display_floor(success)
        visible_values.extend(displayed[np.isfinite(displayed)])
        ax.plot(
            n_values, displayed, marker=marker, color=color, linewidth=W_DATA,
            markersize=4, label=compact_labels[method],
        )

    y_min = max(DISPLAY_FLOOR, float(min(visible_values)) - 0.04)
    if y_min <= 0.5:
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=W_HAIR)
        ax.text(
            0.22, 0.515, r"$F_{\mathrm{FT}}=0.5$",
            transform=ax.get_yaxis_transform(),
            ha="center", va="bottom", color="#666666", fontsize=FS_SMALL,
        )
    ax.set_title(rf"(b) Estimated no-fault probability ($d={distance}$)")
    ax.set_xlabel(r"$N=L^2$")
    ax.set_ylabel("Estimated no-fault probability")
    ax.set_ylim(y_min, 1.01)
    ax.grid(alpha=0.2)
    ax.legend(
        fontsize=FS_SMALL,
        loc="lower left",
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
    c_t: int,
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
    _panel_volume(axes[0], data, alpha, int(c_t))
    _panel_success(axes[1], data, distance, p_b)
    fig.tight_layout(w_pad=1.4)
    return fig


def make_magic_diagnostic_preview(data: pd.DataFrame):
    """Build a non-paper preview of factorization and magic/routing balance."""
    plotted = data[data["method"].isin(("1d_baseline", "gamma_2d_proper"))]
    n_values = sorted(int(n) for n in plotted["N"].unique())
    preview_style = {
        "gamma_2d_proper": (
            r"$\Gamma$-FP-FFFT", "o", "#d62728", "-", 2,
        ),
        "1d_baseline": ("CT-FFFT", "s", "#555555", "--", 3),
    }
    l_values = np.sqrt(n_values).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for method, (label, marker, color, linestyle, zorder) in preview_style.items():
        n_t = _series(plotted, method, "n_T", n_values)
        v_t = _series(plotted, method, "V_T", n_values)
        s = _series(plotted, method, "S", n_values)
        axes[0].plot(
            l_values, n_t, label=label, marker=marker, color=color,
            linestyle=linestyle, linewidth=W_DATA, markersize=4,
            markerfacecolor=("white" if method == "gamma_2d_proper" else color),
            markeredgewidth=W_HAIR, alpha=(0.72 if method == "gamma_2d_proper" else 1.0),
            zorder=zorder,
        )
        axes[1].plot(
            l_values, v_t / s, label=label, marker=marker, color=color,
            linestyle=linestyle, linewidth=W_DATA, markersize=4,
            markerfacecolor=("white" if method == "gamma_2d_proper" else color),
            markeredgewidth=W_HAIR, alpha=(0.72 if method == "gamma_2d_proper" else 1.0),
            zorder=zorder,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Grid side length $L$")
    axes[0].set_ylabel(r"Magic inventory $n_T$")
    axes[0].set_title(r"(a) Magic inventory $n_T$")
    axes[0].text(
        0.98, 0.04, "Inventories coincide for every odd $L$",
        transform=axes[0].transAxes, ha="right", va="bottom", fontsize=FS_SMALL,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "pad": 2.0,
              "alpha": 0.90},
    )
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="#777777", linestyle=":", linewidth=W_HAIR)
    axes[1].set_xlabel(r"Grid side length $L$")
    axes[1].set_ylabel(r"Magic/routing ratio $V_T/S$")
    axes[1].set_title("(b) Relative magic workload")
    for ax in axes:
        for power_of_two in (4, 8, 16):
            ax.axvline(
                power_of_two, color="#9e9e9e", linestyle=":",
                linewidth=W_HAIR, alpha=0.45, zorder=0,
            )
        ax.set_xticks(l_values)
        ax.grid(which="major", alpha=0.22)
        ax.legend(fontsize=FS_SMALL, framealpha=0.92)
    fig.tight_layout(w_pad=1.8)
    return fig


def generate_ft_figure(
    csv_path: str = "exp2_ffft/results/data.csv",
    output_dir: str = "exp2_ffft/figures",
    *,
    alpha: float = DEFAULT_ALPHA,
    c_t: int = C_T,
    distance: int = 11,
    require_publication_coverage: bool = True,
) -> pd.DataFrame:
    """Generate the publication-quality two-panel FFFT early-FT figure."""
    data, p_b = load_ft(
        csv_path,
        alpha=alpha,
        c_t=c_t,
        distance=distance,
        require_publication_coverage=require_publication_coverage,
    )
    fig = make_ft_figure(
        data,
        alpha=alpha,
        c_t=int(c_t),
        distance=distance,
        p_b=p_b,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = "FP-exp2_combined_ft"
    for extension in ("pdf", "svg"):
        fig.savefig(
            os.path.join(output_dir, f"{stem}.{extension}"),
            bbox_inches="tight",
        )
    plt.close(fig)

    preview = make_magic_diagnostic_preview(data)
    preview_dir = os.path.join(output_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    preview.savefig(
        os.path.join(preview_dir, "FP-exp2_magic_diagnostics_preview.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(preview)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="exp2_ffft/results/data.csv")
    parser.add_argument("--output-dir", default="exp2_ffft/figures")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--c-t", type=int, default=C_T)
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
        c_t=args.c_t,
        distance=args.distance,
        require_publication_coverage=not args.allow_partial,
    )
    print(f"Saved FFFT FT figure for {data.N.nunique()} sizes to {args.output_dir}/")


if __name__ == "__main__":
    main()
