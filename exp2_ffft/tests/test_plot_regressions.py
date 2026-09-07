"""Regression tests for resource-aware FFFT plotting."""

import matplotlib.pyplot as plt
import pandas as pd

from exp2_ffft import collect, plot


def test_l2_depth_breakdown_counts_ancilla_qubits(monkeypatch):
    """The L=2 ancilla circuit must not be counted as a four-qubit circuit."""
    saved = []
    positive_ancilla_counts = []

    original_counter = collect.count_cnot_resources

    def recording_counter(circuit, L, n_ancillas=0):
        if n_ancillas:
            positive_ancilla_counts.append(n_ancillas)
        return original_counter(circuit, L, n_ancillas)

    def close_instead_of_save(fig, fig_dir, name):
        saved.append(name)
        plt.close(fig)

    monkeypatch.setattr(collect, "count_cnot_resources", recording_counter)
    monkeypatch.setattr(plot, "_save_fig", close_instead_of_save)
    df = pd.DataFrame({
        "L": [2, 2, 2],
        "N": [4, 4, 4],
        "method": [
            "gamma_2d_proper", "gamma_2d_ancilla", "1d_baseline",
        ],
        "cnot_depth": [32, 76, 12],
        "p_2q": [1e-3, 1e-3, 1e-3],
    })

    plot.plot_depth_breakdown(df, "unused")

    assert saved == ["depth_breakdown"]
    assert positive_ancilla_counts == [2]


def test_depth_breakdown_bar_tops_are_closed_to_csv(monkeypatch):
    """Rebuilt components must not replace either official CSV total."""
    captured = {}

    def capture_instead_of_save(fig, fig_dir, name):
        captured["fig"] = fig
        captured["name"] = name

    monkeypatch.setattr(plot, "_save_fig", capture_instead_of_save)
    df = pd.DataFrame({
        "L": [2, 2, 2],
        "N": [4, 4, 4],
        "method": [
            "gamma_2d_proper", "gamma_2d_ancilla", "1d_baseline",
        ],
        # Deliberately differ from the L=2 circuits rebuilt by the plotter.
        "cnot_depth": [35, 80, 15],
        "p_2q": [1e-3, 1e-3, 1e-3],
    })

    plot.plot_depth_breakdown(df, "unused")
    fig = captured["fig"]
    try:
        assert captured["name"] == "depth_breakdown"
        ax = fig.axes[0]
        # The first five PathCollections are the five method markers.  CT is
        # marker 1 and the official ancilla-free Gamma result is marker 4.
        assert float(ax.collections[1].get_offsets()[0, 1]) == 15
        assert float(ax.collections[4].get_offsets()[0, 1]) == 35
    finally:
        plt.close(fig)


def test_no_fault_plot_uses_horizontal_panels_and_shared_labels(monkeypatch):
    """The paper no-fault plot is one compact, consistently labelled strip."""
    captured = {}

    def capture_instead_of_save(fig, fig_dir, name):
        captured["fig"] = fig
        captured["name"] = name

    monkeypatch.setattr(plot, "_save_fig", capture_instead_of_save)

    rows = []
    for p_2q in (1e-5, 1e-4, 1e-3):
        for method in (
            "1d_baseline", "gamma_2d_ancilla", "gamma_2d_proper",
        ):
            rows.append({
                "L": 2,
                "N": 4,
                "method": method,
                "p_2q": p_2q,
                "p_idle": p_2q,
                "mult_fidelity": 0.9,
            })

    plot.plot_fidelity_vs_N(pd.DataFrame(rows), "unused")
    fig = captured["fig"]
    try:
        assert captured["name"] == "fidelity_vs_N"
        assert len(fig.axes) == 3
        assert [ax.get_title() for ax in fig.axes] == [
            r"$p_{2q}=10^{-5}$",
            r"$p_{2q}=10^{-4}$",
            r"$p_{2q}=10^{-3}$",
        ]
        assert len({round(ax.get_position().y0, 6) for ax in fig.axes}) == 1
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.texts] == [
            r"$N=L^2$",
            "Independent-location no-fault probability",
        ]
    finally:
        plt.close(fig)
