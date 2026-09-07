"""Regression tests for fault-tolerant SYK volume accounting."""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import pytest

from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from exp3_syk.collect import ACCOUNTING_MODEL, HALL_NOT_APPLICABLE
from exp3_syk import plot_ft


def _raw_ft_rows() -> pd.DataFrame:
    rows = []
    depths = {
        "naive_pauli": (30, 32),
        "1d": (20, 22),
        "ancilla": (16, 18),
        "folded": (12, 14),
    }
    for baseline, pair in depths.items():
        for instance_idx, depth in enumerate(pair):
            total_qubits = 18 if baseline == "ancilla" else 16
            total_cnots = 40 + instance_idx
            rows.append({
                "L": 4,
                "N": 16,
                "k": 1.0,
                "instance_idx": instance_idx,
                "baseline": baseline,
                "p_2q": 1e-5,
                "accounting_model": ACCOUNTING_MODEL,
                "hall_decomposition_model": (
                    HALL_DECOMPOSITION_MODEL
                    if baseline in {"ancilla", "folded"}
                    else HALL_NOT_APPLICABLE
                ),
                "total_qubits": total_qubits,
                "cnot_depth": depth,
                "total_cnots": total_cnots,
                "total_idle_slots_layered": (
                    total_qubits * depth - 2 * total_cnots
                ),
                "n_1q_rz": 3 + instance_idx,
            })
    return pd.DataFrame(rows)


def test_ft_aggregation_and_panel_use_routing_with_magic_inset(monkeypatch):
    raw = _raw_ft_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "validate_publication_data", lambda *args, **kwargs: None,
    )
    data, p_b = plot_ft.load_ft("unused.csv", alpha=1.0, distance=11)

    p_cycle = plot_ft.logical_cycle_error_rate(11)
    assert p_cycle == pytest.approx(3.0e-8)
    assert p_b == pytest.approx(1.0 - (1.0 - p_cycle) ** 11)

    folded_raw = raw[raw.baseline == "folded"]
    folded_totals = (
        folded_raw.total_qubits * folded_raw.cnot_depth
        + plot_ft.C_T * folded_raw.n_1q_rz
    )
    folded = data[data.baseline == "folded"].iloc[0]
    assert folded.S_plus_V_T_mean == np.mean(folded_totals)
    assert folded.S_plus_V_T_std == np.std(folded_totals, ddof=1)
    assert folded.S_plus_V_T_mean == folded.S_mean + folded.V_T_mean

    fig, ax = plt.subplots()
    try:
        plot_ft._panel_volume(ax, data, alpha=1.0)
        for method_index, method in enumerate(plot_ft.METHODS_VOLUME):
            row = data[data.baseline == method].iloc[0]
            routing = ax.patches[method_index]
            assert routing.get_height() == pytest.approx(
                row.S_mean / plot_ft.VOLUME_DISPLAY_SCALE
            )
            assert routing.get_y() == 0

        scatters = [
            item for item in ax.collections if isinstance(item, PathCollection)
        ]
        errorbars = [
            item for item in ax.collections if isinstance(item, LineCollection)
        ]
        assert len(scatters) == len(plot_ft.METHODS_VOLUME)
        assert len(errorbars) == len(plot_ft.METHODS_VOLUME)
        for scatter, errorbar, method in zip(
            scatters, errorbars, plot_ft.METHODS_VOLUME
        ):
            row = data[data.baseline == method].iloc[0]
            assert float(scatter.get_offsets()[0, 1]) == pytest.approx(
                row.S_mean / plot_ft.VOLUME_DISPLAY_SCALE
            )
            segment = errorbar.get_segments()[0]
            assert np.allclose(
                segment[:, 1],
                [
                    (row.S_mean - row.S_std)
                    / plot_ft.VOLUME_DISPLAY_SCALE,
                    (row.S_mean + row.S_std)
                    / plot_ft.VOLUME_DISPLAY_SCALE,
                ],
            )
        # Three routing bars are followed by the routing-volume comparison
        # curve(s).
        arcs = [
            patch for patch in ax.patches
            if isinstance(patch, FancyArrowPatch)
        ]
        assert len(arcs) == 1
        assert list(arcs[0].get_path().codes) == [
            MplPath.MOVETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.CURVE3, MplPath.CURVE3,
        ]
        vertices = arcs[0].get_path().vertices
        assert vertices[2, 0] == pytest.approx(
            0.5 * (vertices[0, 0] + vertices[4, 0])
        )
        assert vertices[1, 1] == pytest.approx(vertices[2, 1])
        assert vertices[3, 1] == pytest.approx(vertices[2, 1])
        assert vertices[2, 1] > max(vertices[0, 1], vertices[4, 1])
        savings_labels = [
            text.get_text() for text in ax.texts
            if text.get_text().endswith("%")
        ]
        assert savings_labels == ["38%"]
        assert all(text.get_bbox_patch() is None for text in ax.texts)
        assert len(ax.child_axes) == 1
        inset = ax.child_axes[0]
        assert inset.get_title() == r"Magic inventory $n_T$"
        assert len(inset.lines) == len(plot_ft.METHODS_VOLUME)
        for line, method in zip(inset.lines, plot_ft.METHODS_VOLUME):
            row = data[data.baseline == method].iloc[0]
            assert float(line.get_ydata()[0]) == pytest.approx(row.n_T_mean)
    finally:
        plt.close(fig)


def test_display_floor_masks_pointwise_without_discarding_recovery():
    values = np.array([0.2, 5e-4, 0.03, plot_ft.DISPLAY_FLOOR, np.nan])
    displayed = plot_ft.truncate_below_display_floor(values)

    assert displayed[0] == pytest.approx(0.2)
    assert np.isnan(displayed[1])
    assert displayed[2] == pytest.approx(0.03)
    assert displayed[3] == pytest.approx(plot_ft.DISPLAY_FLOOR)
    assert np.isnan(displayed[4])
    assert values[1] == pytest.approx(5e-4)


def test_alpha_changes_factory_volume_and_fidelity(monkeypatch):
    raw = _raw_ft_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "validate_publication_data", lambda *args, **kwargs: None,
    )

    alpha_one, _ = plot_ft.load_ft("unused.csv", alpha=1.0, distance=11)
    alpha_two, _ = plot_ft.load_ft("unused.csv", alpha=2.0, distance=11)
    one = alpha_one[alpha_one.baseline == "folded"].iloc[0]
    two = alpha_two[alpha_two.baseline == "folded"].iloc[0]
    assert two.V_T_mean == pytest.approx(2.0 * one.V_T_mean)
    assert two.F_mean < one.F_mean


def test_fidelity_panel_inserts_nan_gaps_below_display_floor():
    n_values = [16, 25, 36]
    rows = []
    for method in plot_ft.METHODS_FIDELITY:
        means = [0.2, 5e-4, 0.03] if method == "folded" else [0.4] * 3
        for n, mean in zip(n_values, means):
            rows.append({
                "N": n,
                "baseline": method,
                "F_mean": mean,
                "F_std": 0.01,
            })
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    try:
        plot_ft._panel_fidelity(
            ax,
            data,
            distance=11,
            p_b=plot_ft.logical_block_error_rate(11),
        )
        folded = next(line for line in ax.lines if line.get_label() == "Folded FP")
        y_data = np.asarray(folded.get_ydata(), dtype=float)
        assert y_data[0] == pytest.approx(0.2)
        assert np.isnan(y_data[1])
        assert y_data[2] == pytest.approx(0.03)
        assert ax.get_yscale() == "log"
        assert ax.get_legend()._loc == 4  # lower right
    finally:
        plt.close(fig)


def test_combined_figure_uses_three_to_one_panel_width(monkeypatch):
    raw = _raw_ft_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "validate_publication_data", lambda *args, **kwargs: None,
    )
    data, p_b = plot_ft.load_ft("unused.csv", alpha=1.0, distance=11)

    fig = plot_ft.make_ft_figure(
        data,
        alpha=1.0,
        distance=11,
        p_b=p_b,
    )
    try:
        grid = fig.axes[0].get_subplotspec().get_gridspec()
        assert tuple(grid.get_width_ratios()) == (3, 1)
        assert fig.axes[1].get_title() == r"(b) Estimated no-fault probability ($d=11$)"
    finally:
        plt.close(fig)
