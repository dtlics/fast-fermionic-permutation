"""Validate and atomically promote staged Experiment 1 data and figures."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from exp1_fp.validate_results import validate_results


FIGURE_NAMES = (
    "depth_vs_L.pdf",
    "depth_vs_L.svg",
    "spacetime_vs_N.pdf",
    "spacetime_vs_N.svg",
    "stim_fidelity.pdf",
    "stim_fidelity.svg",
)


def _prepare_copy(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    return temporary


def promote(
    staged_csv: Path,
    staged_figures: Path,
    official_csv: Path = Path("exp1_fp/results/data.csv"),
    official_figures: Path = Path("exp1_fp/figures"),
) -> dict:
    """Validate staging, replace each artifact atomically, and commit CSV last."""
    summary = validate_results(pd.read_csv(staged_csv))
    prepared: list[tuple[Path, Path]] = []
    try:
        for name in FIGURE_NAMES:
            destination = official_figures / name
            prepared.append(
                (_prepare_copy(staged_figures / name, destination), destination)
            )
        # The CSV is the publication-artifact commit marker and is replaced last.
        prepared_csv = _prepare_copy(staged_csv, official_csv)
        for temporary, destination in prepared:
            os.replace(temporary, destination)
        os.replace(prepared_csv, official_csv)
    finally:
        for temporary, _ in prepared:
            if temporary.exists():
                temporary.unlink()
        csv_temporary = official_csv.with_suffix(official_csv.suffix + ".tmp")
        if csv_temporary.exists():
            csv_temporary.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staged_csv", type=Path)
    parser.add_argument("staged_figures", type=Path)
    parser.add_argument(
        "--official-csv", type=Path,
        default=Path("exp1_fp/results/data.csv"),
    )
    parser.add_argument(
        "--official-figures", type=Path,
        default=Path("exp1_fp/figures"),
    )
    args = parser.parse_args()
    summary = promote(
        args.staged_csv,
        args.staged_figures,
        args.official_csv,
        args.official_figures,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
