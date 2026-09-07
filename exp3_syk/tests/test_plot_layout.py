"""Focused visual-layout regressions for the auxiliary Exp3 plots."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from exp3_syk import plot


def _publication_span_depth_rows() -> pd.DataFrame:
    rows = []
    for L in plot.PUBLICATION_L_VALUES:
        N = L * L
        for baseline, scale in (("1d", 72), ("ancilla", 43), ("folded", 18)):
            rows.append(
                {
                    "L": L,
                    "N": N,
                    "k": 1.0,
                    "p_2q": 1e-5,
                    "baseline": baseline,
                    "cnot_depth": scale * L,
                    "interaction_cnot_depth": 8 * L,
                }
            )
    return pd.DataFrame(rows)


def test_depth_breakdown_stays_compact_and_keeps_legend_off_annotations(
    monkeypatch,
):
    captured = {}

    def capture(fig, name, output_dir):
        captured["fig"] = fig

    monkeypatch.setattr(plot, "_save_fig", capture)
    plot.plot_depth_breakdown(_publication_span_depth_rows(), "unused")

    fig = captured["fig"]
    try:
        width, height = fig.get_size_inches()
        assert width == pytest.approx(12.4)
        assert width / height < 2.5

        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        legend_box = ax.get_legend().get_window_extent(renderer)
        percentage_labels = [
            text for text in ax.texts if text.get_text().endswith("%")
        ]
        assert percentage_labels
        assert all(
            not legend_box.overlaps(text.get_window_extent(renderer))
            for text in percentage_labels
        )
    finally:
        plt.close(fig)
