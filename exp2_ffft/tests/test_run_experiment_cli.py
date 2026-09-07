"""Regressions for the canonical Exp2 publication CLI."""

import sys

import pandas as pd
import pytest

from exp2_ffft import collect, plot_ft, run_experiment as cli


@pytest.mark.parametrize("plot_only", [False, True])
def test_canonical_cli_emits_publication_ft_figure(
    monkeypatch, tmp_path, plot_only,
):
    events = []
    output_dir = tmp_path / "results"
    figure_dir = tmp_path / "figures"

    def fake_collect(**kwargs):
        events.append(("collect", kwargs))
        return pd.DataFrame([{
            "L": 2,
            "N": 4,
            "method": "1d_baseline",
            "gamma_method": "",
            "cnot_depth": 1,
            "total_2q_gates": 1,
            "spacetime_volume": 4,
            "verification_error": 0.0,
            "p_2q": 1e-5,
        }])

    def fake_ft(**kwargs):
        events.append(("ft", kwargs))

    monkeypatch.setattr(collect, "run_experiment", fake_collect)
    monkeypatch.setattr(plot_ft, "generate_ft_figure", fake_ft)
    argv = [
        "exp2_ffft.run_experiment",
        "--output-dir", str(output_dir),
        "--fig-dir", str(figure_dir),
    ]
    if plot_only:
        argv.append("--plot-only")
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    names = [name for name, _ in events]
    assert names == (["ft"] if plot_only else ["collect", "ft"])
    ft_args = next(args for name, args in events if name == "ft")
    expected_csv = f"{output_dir}/data.csv"
    assert ft_args == {
        "csv_path": expected_csv,
        "output_dir": str(figure_dir),
        "require_publication_coverage": True,
    }
