"""Experiment 2: CLI entrypoint for FFFT benchmarking.

Usage:
    python -m exp2_ffft.run_experiment                    # full experiment
    python -m exp2_ffft.run_experiment --L 4 6 8          # subset of L values
    python -m exp2_ffft.run_experiment --plot-only         # re-plot from CSV
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: FFFT benchmarking")
    parser.add_argument("--L", type=int, nargs="+",
                        default=[2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                        help="Grid side lengths to sweep")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data collection; re-plot from CSV")
    parser.add_argument("--output-dir", default="exp2_ffft/results")
    parser.add_argument("--fig-dir", default="exp2_ffft/figures")
    args = parser.parse_args()

    if args.plot_only:
        from exp2_ffft.plot import generate_all_plots
        generate_all_plots(
            csv_path=f"{args.output_dir}/data.csv",
            fig_dir=args.fig_dir,
        )
        return

    from exp2_ffft.collect import run_experiment
    from exp2_ffft.plot import generate_all_plots

    df = run_experiment(
        L_values=args.L,
        output_dir=args.output_dir,
        verify=True,
    )

    generate_all_plots(
        csv_path=f"{args.output_dir}/data.csv",
        fig_dir=args.fig_dir,
    )

    # Print summary table
    p_val = df["p_2q"].iloc[0]
    summary = df[df["p_2q"] == p_val][["L", "N", "method", "cnot_depth",
                                        "total_2q_gates", "spacetime_volume",
                                        "verification_error"]]
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
