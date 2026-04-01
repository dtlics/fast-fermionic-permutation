"""Experiment 3: Sparse SYK Trotter Step Benchmarking.

Usage:
    python -m exp3_syk.run_experiment                         # full sweep
    python -m exp3_syk.run_experiment --L 4 6 8               # specific sizes
    python -m exp3_syk.run_experiment --k 1.0 2.0             # sparsity values
    python -m exp3_syk.run_experiment --plot-only              # re-plot from CSV
"""

from __future__ import annotations

import argparse

import pandas as pd

from exp3_syk.collect import run_experiment
from exp3_syk.plot import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Sparse SYK Trotter Step Benchmarking"
    )
    parser.add_argument("--L", nargs="+", type=int,
                        default=[4, 6, 8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--k", nargs="+", type=float, default=[1.0],
                        help="Sparsity parameter (expected terms ~ k * 2N)")
    parser.add_argument("--n-instances", type=int, default=10)
    parser.add_argument("--p-values", nargs="+", type=float,
                        default=[1e-3, 1e-4, 1e-5])
    parser.add_argument("--p-idle-factor", type=float, default=0.1)
    parser.add_argument("--output-dir", default="exp3_syk/results")
    parser.add_argument("--fig-dir", default="exp3_syk/figures")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data collection, re-plot from existing CSV")
    args = parser.parse_args()

    if args.plot_only:
        csv_path = f"{args.output_dir}/data.csv"
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

    generate_all_plots(df, args.fig_dir)


if __name__ == "__main__":
    main()
