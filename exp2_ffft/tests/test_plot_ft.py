"""Regression tests for the FFFT early-fault-tolerant figure."""

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import pytest

from exp2_ffft import plot_ft
from exp3_syk import plot_ft as syk_plot_ft


def _raw_rows() -> pd.DataFrame:
    rows = []
    configs = {
        "1d_baseline": (4, 20, 6, 2),
        "gamma_2d_core": (4, 12, 4, 1),
        "gamma_2d_ancilla": (6, 13, 5, 1),
        "gamma_2d_proper": (4, 14, 5, 1),
    }
    for p_2q in (1e-5, 1e-4):
        for method, (qubits, depth, exact_t, synth_rz) in configs.items():
            rows.append({
                "L": 2,
                "N": 4,
                "method": method,
                "p_2q": p_2q,
                "total_qubits": qubits,
                "cnot_depth": depth,
                "spacetime_volume": qubits * depth,
                "n_exact_t": exact_t,
                "n_synth_rz": synth_rz,
            })
    return pd.DataFrame(rows)


def test_early_ft_settings_match_exp3():
    """Keep the two publication experiments on one early-FT model."""
    for name in (
        "C_T",
        "P_PHYS",
        "P_THRESHOLD",
        "PREFAC",
        "DISPLAY_FLOOR",
        "PANEL_WIDTH_RATIOS",
        "DEFAULT_ALPHA",
        "INSET_FACE_COLOR",
        "INSET_EDGE_COLOR",
    ):
        assert getattr(plot_ft, name) == getattr(syk_plot_ft, name)
    assert plot_ft.VOLUME_DISPLAY_SCALE == 1e6
    assert syk_plot_ft.VOLUME_DISPLAY_SCALE == 1e7
    # insets lowered and shifted right so their title and offset label clear the
    # legend patch, which covers the upper-left ~26% of the axes
    assert plot_ft.INSET_BOUNDS == (0.14, 0.30, 0.42, 0.34)
    assert syk_plot_ft.INSET_BOUNDS == (0.06, 0.3, 0.42, 0.34)
    p_cycle = plot_ft.logical_cycle_error_rate(11)
    p_block = plot_ft.logical_block_error_rate(11)
    assert p_cycle == pytest.approx(3.0e-8)
    assert p_block == pytest.approx(1.0 - (1.0 - p_cycle) ** 11)
    assert p_cycle == syk_plot_ft.logical_cycle_error_rate(11)
    assert p_block == syk_plot_ft.logical_block_error_rate(11)


def test_load_ft_uses_exact_plus_synthesized_t_inventory(monkeypatch):
    raw = _raw_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "_validate_folded_provenance", lambda *args, **kwargs: None,
    )

    data, p_b = plot_ft.load_ft(
        "unused.csv", alpha=1.0, c_t=70, distance=11,
    )
    proper = data[data.method == "gamma_2d_proper"].iloc[0]
    assert p_b == pytest.approx(plot_ft.logical_block_error_rate(11))
    assert proper.S == 56
    assert proper.n_T == 5 + 70
    assert proper.V_T == 75
    assert proper.S_plus_V_T == 131
    expected = (1 - p_b) ** (56 + 75 + 75)
    assert proper.P_FT == pytest.approx(expected)


def test_volume_panel_uses_routing_bars_and_three_method_magic_inset(
    monkeypatch,
):
    raw = _raw_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "_validate_folded_provenance", lambda *args, **kwargs: None,
    )
    data, _ = plot_ft.load_ft("unused.csv")

    fig, ax = plt.subplots()
    try:
        plot_ft._panel_volume(ax, data, alpha=1.0, c_t=70)
        for method_index, method in enumerate(plot_ft.METHODS_COMPARABLE):
            row = data[data.method == method].iloc[0]
            routing = ax.patches[method_index]
            assert routing.get_height() == pytest.approx(
                row.S / plot_ft.VOLUME_DISPLAY_SCALE
            )
            assert routing.get_y() == 0

        markers = [
            item for item in ax.collections if isinstance(item, PathCollection)
        ]
        assert len(markers) == len(plot_ft.METHODS_COMPARABLE)
        for marker, method in zip(markers, plot_ft.METHODS_COMPARABLE):
            row = data[data.method == method].iloc[0]
            assert float(marker.get_offsets()[0, 1]) == pytest.approx(
                row.S / plot_ft.VOLUME_DISPLAY_SCALE
            )

        assert len(ax.child_axes) == 1
        inset = ax.child_axes[0]
        assert inset.get_title() == r"Magic inventory $n_T$"
        assert len(inset.lines) == len(plot_ft.METHODS_COMPARABLE)
        for line, method in zip(inset.lines, plot_ft.METHODS_COMPARABLE):
            row = data[data.method == method].iloc[0]
            assert float(line.get_ydata()[0]) == pytest.approx(row.n_T)
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))
        assert r"$10^6$" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_volume_savings_annotation_and_arrow_compare_routing_volume(monkeypatch):
    raw = _raw_rows()
    raw["N"] = 36
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "_validate_folded_provenance", lambda *args, **kwargs: None,
    )
    data, _ = plot_ft.load_ft("unused.csv")

    fig, ax = plt.subplots()
    try:
        plot_ft._panel_volume(ax, data, alpha=1.0, c_t=70)
        savings_labels = [text.get_text() for text in ax.texts]
        assert savings_labels == ["30%"]
        assert all(text.get_bbox_patch() is None for text in ax.texts)
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
        assert vertices[0, 1] == pytest.approx(80 / plot_ft.VOLUME_DISPLAY_SCALE)
        assert vertices[4, 1] == pytest.approx(56 / plot_ft.VOLUME_DISPLAY_SCALE)
        assert vertices[1, 1] == pytest.approx(vertices[2, 1])
        assert vertices[3, 1] == pytest.approx(vertices[2, 1])
        assert vertices[2, 1] > max(vertices[0, 1], vertices[4, 1])
    finally:
        plt.close(fig)


def test_relaxed_core_is_returned_but_not_plotted(monkeypatch):
    raw = _raw_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "_validate_folded_provenance", lambda *args, **kwargs: None,
    )
    data, _ = plot_ft.load_ft("unused.csv")
    assert set(data.method) == set(plot_ft.METHODS_ALL)

    fig, ax = plt.subplots()
    try:
        plot_ft._panel_success(
            ax,
            data,
            distance=11,
            p_b=plot_ft.logical_block_error_rate(11),
        )
        labels = {line.get_label() for line in ax.lines}
        assert plot_ft.METHOD_STYLE["gamma_2d_core"][0] not in labels
        assert "CT-FFFT" in labels
        assert r"Folded $\Gamma$" in labels
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
    raw = _raw_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "_validate_folded_provenance", lambda *args, **kwargs: None,
    )

    alpha_one, _ = plot_ft.load_ft("unused.csv", alpha=1.0)
    alpha_two, _ = plot_ft.load_ft("unused.csv", alpha=2.0)
    one = alpha_one[alpha_one.method == "gamma_2d_proper"].iloc[0]
    two = alpha_two[alpha_two.method == "gamma_2d_proper"].iloc[0]
    assert two.V_T == pytest.approx(2.0 * one.V_T)
    assert two.P_FT < one.P_FT


def test_success_panel_inserts_nan_gaps_below_display_floor():
    n_values = [16, 25, 36]
    rows = []
    for method in plot_ft.METHODS_COMPARABLE:
        success = (
            [0.2, 5e-4, 0.03]
            if method == "gamma_2d_proper"
            else [0.4] * 3
        )
        for n, probability in zip(n_values, success):
            rows.append({
                "N": n,
                "method": method,
                "P_FT": probability,
            })
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    try:
        plot_ft._panel_success(
            ax,
            data,
            distance=11,
            p_b=plot_ft.logical_block_error_rate(11),
        )
        folded = next(
            line for line in ax.lines
            if line.get_label() == r"Folded $\Gamma$"
        )
        y_data = np.asarray(folded.get_ydata(), dtype=float)
        assert y_data[0] == pytest.approx(0.2)
        assert np.isnan(y_data[1])
        assert y_data[2] == pytest.approx(0.03)
    finally:
        plt.close(fig)


def test_combined_figure_uses_three_to_one_panel_width(monkeypatch):
    raw = _raw_rows()
    monkeypatch.setattr(plot_ft.pd, "read_csv", lambda _: raw.copy())
    monkeypatch.setattr(
        plot_ft, "_validate_folded_provenance", lambda *args, **kwargs: None,
    )
    data, p_b = plot_ft.load_ft("unused.csv")

    fig = plot_ft.make_ft_figure(
        data,
        alpha=1.0,
        c_t=70,
        distance=11,
        p_b=p_b,
    )
    try:
        grid = fig.axes[0].get_subplotspec().get_gridspec()
        assert tuple(grid.get_width_ratios()) == (3, 1)
        assert fig.axes[1].get_title() == (
            r"(b) Estimated no-fault probability ($d=11$)"
        )
    finally:
        plt.close(fig)
