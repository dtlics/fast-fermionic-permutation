"""Experiment 3: Sparse SYK Trotter Step Benchmarking.

Usage:
    python -m exp3_syk.run_experiment                         # full sweep
    python -m exp3_syk.run_experiment --L 4 6 8               # specific sizes
    python -m exp3_syk.run_experiment --k 1.0 2.0             # sparsity values
    python -m exp3_syk.run_experiment --plot-only              # re-plot from CSV
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from exp3_syk.collect import (
    COLORS_K_VALUES,
    PUBLICATION_L_VALUES,
    collect_colors_data,
    run_experiment,
)
from exp3_syk.plot import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Sparse SYK Trotter Step Benchmarking"
    )
    parser.add_argument("--L", nargs="+", type=int,
                        default=list(PUBLICATION_L_VALUES),
                        help="grid side lengths (default: every L from 4 through 20)")
    parser.add_argument("--k", nargs="+", type=float, default=[1.0],
                        help="Sparsity parameter (expected terms ~ k * 2N)")
    parser.add_argument("--n-instances", type=int, default=10)
    parser.add_argument("--p-values", nargs="+", type=float,
                        default=[1e-3, 1e-4, 1e-5])
    parser.add_argument(
        "--p-idle-factor", type=float, default=1.0,
        help="non-entangling/CNOT logical-rate ratio (default: 1)",
    )
    parser.add_argument("--output-dir", default="exp3_syk/results")
    parser.add_argument("--fig-dir", default="exp3_syk/figures")
    parser.add_argument("--colors-k", nargs="+", type=float,
                        default=list(COLORS_K_VALUES),
                        help="k values for colors-vs-N plot (default: 0.5 1 2 3)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data collection, re-plot from existing CSV")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow plotting a deliberately reduced non-publication sweep",
    )
    args = parser.parse_args()
    csv_path = f"{args.output_dir}/data.csv"

    if args.plot_only:
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = run_experiment(
            L_values=args.L,
            k_values=args.k,
            n_instances=args.n_instances,
            p_values=args.p_values,
            p_idle_factor=args.p_idle_factor,
            output_dir=args.output_dir,
        )

    # Collect (or load) colors data for multiple k values
    colors_csv = f"{args.output_dir}/colors_data.csv"
    if args.plot_only and os.path.exists(colors_csv):
        colors_df = pd.read_csv(colors_csv)
    else:
        colors_df = collect_colors_data(
            L_values=args.L,
            k_values=args.colors_k,
            n_instances=args.n_instances,
            output_dir=args.output_dir,
        )

    generate_all_plots(
        df,
        args.fig_dir,
        colors_df=colors_df,
        require_publication_coverage=not args.allow_partial,
    )
    from exp3_syk.plot_ft import generate_ft_figure
    generate_ft_figure(csv_path=csv_path,
                       output_dir=args.fig_dir,
                       require_publication_coverage=not args.allow_partial)


if __name__ == "__main__":
    main()
