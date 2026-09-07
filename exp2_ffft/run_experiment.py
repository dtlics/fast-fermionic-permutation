"""Experiment 2: CLI entrypoint for FFFT benchmarking.

Usage:
    python -m exp2_ffft.run_experiment                    # full experiment
    python -m exp2_ffft.run_experiment --L 4 6 8          # subset of L values
    python -m exp2_ffft.run_experiment --plot-only         # re-plot the publication FT figure
"""

import argparse

from exp2_ffft.collect import PUBLICATION_L_VALUES


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: FFFT benchmarking")
    parser.add_argument("--L", type=int, nargs="+",
                        default=list(PUBLICATION_L_VALUES),
                        help="Grid side lengths (default: every L from 4 through 20)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data collection; re-plot from CSV")
    parser.add_argument("--output-dir", default="exp2_ffft/results")
    parser.add_argument("--fig-dir", default="exp2_ffft/figures")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow plotting a deliberately reduced non-publication sweep",
    )
    args = parser.parse_args()
    csv_path = f"{args.output_dir}/data.csv"

    if args.plot_only:
        from exp2_ffft.plot_ft import generate_ft_figure
        generate_ft_figure(
            csv_path=csv_path,
            output_dir=args.fig_dir,
            require_publication_coverage=not args.allow_partial,
        )
        return

    from exp2_ffft.collect import run_experiment

    df = run_experiment(
        L_values=args.L,
        output_dir=args.output_dir,
        verify=True,
    )

    from exp2_ffft.plot_ft import generate_ft_figure
    generate_ft_figure(
        csv_path=csv_path,
        output_dir=args.fig_dir,
        require_publication_coverage=not args.allow_partial,
    )

    # Print summary table
    p_val = df["p_2q"].iloc[0]
    summary = df[df["p_2q"] == p_val][[
        "L",
        "N",
        "method",
        "gamma_method",
        "cnot_depth",
        "total_2q_gates",
        "spacetime_volume",
        "verification_error",
    ]]
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
