"""Experiment 1: Fermionic Permutation Benchmarking.

CLI entrypoint for running the full experiment sweep and generating plots.

Usage:
    python -m exp1_fp.run_experiment                    # full experiment
    python -m exp1_fp.run_experiment --L 4 6 8          # subset of grid sizes
    python -m exp1_fp.run_experiment --plot-only         # re-plot from saved CSV
"""

import argparse
import os

import pandas as pd

from exp1_fp.collect import run_experiment
from exp1_fp.plot import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: FP Benchmarking")
    parser.add_argument("--L", nargs="+", type=int, default=[4, 6, 8, 10, 12, 14, 16],
                        help="Grid side lengths (default: 4 6 8 10 12 14 16)")
    parser.add_argument("--perms", nargs="+", default=["reverse", "transpose", "random"],
                        help="Permutation types")
    parser.add_argument("--n-random", type=int, default=20,
                        help="Number of random permutation instances (default: 20)")
    parser.add_argument("--p-values", nargs="+", type=float, default=[1e-3, 1e-4, 1e-5],
                        help="2-qubit gate error rates")
    parser.add_argument("--p-idle-factor", type=float, default=0.1,
                        help="p_idle = p_2q * factor (default: 0.1)")
    parser.add_argument("--shots", type=int, default=1000,
                        help="Stim shots per instance (default: 1000)")
    parser.add_argument("--output-dir", default="exp1_fp/results",
                        help="Output directory for data")
    parser.add_argument("--fig-dir", default="exp1_fp/figures",
                        help="Output directory for figures")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data collection, just re-plot from existing CSV")
    args = parser.parse_args()

    if args.plot_only:
        csv_path = os.path.join(args.output_dir, "data.csv")
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = run_experiment(
            L_values=args.L,
            perm_kinds=args.perms,
            n_random=args.n_random,
            p_values=args.p_values,
            p_idle_factor=args.p_idle_factor,
            shots=args.shots,
            output_dir=args.output_dir,
        )

    generate_all_plots(df, output_dir=args.fig_dir)


if __name__ == "__main__":
    main()
