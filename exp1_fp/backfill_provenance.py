"""Backfill reproducibility metadata without changing existing Exp1 values."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from common.metrics import multiplicative_fidelity
from exp1_fp.collect import _atomic_to_csv, add_reproducibility_provenance


def _canonicalize_derived_no_fault_values(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the exact scalar values produced by the collector formulas."""
    result = df.copy()
    native_columns = {
        "total_cnots", "total_idle_slots_layered", "p_2q", "p_idle",
        "mult_fidelity",
    }
    legacy_columns = {
        "total_2q_gates_legacy", "total_idle_slots_legacy", "p_2q", "p_idle",
        "mult_fidelity_legacy",
    }
    if native_columns.issubset(result.columns):
        result["mult_fidelity"] = [
            multiplicative_fidelity(int(gates), int(idles), float(p_2q), float(p_idle))
            for gates, idles, p_2q, p_idle in zip(
                result["total_cnots"], result["total_idle_slots_layered"],
                result["p_2q"], result["p_idle"],
            )
        ]
    if legacy_columns.issubset(result.columns):
        result["mult_fidelity_legacy"] = [
            multiplicative_fidelity(int(gates), int(idles), float(p_2q), float(p_idle))
            for gates, idles, p_2q, p_idle in zip(
                result["total_2q_gates_legacy"], result["total_idle_slots_legacy"],
                result["p_2q"], result["p_idle"],
            )
        ]
    return result


def backfill(source: Path, output: Path | None = None) -> pd.DataFrame:
    """Append provenance columns without altering Monte Carlo samples."""
    destination = output or source
    loaded = pd.read_csv(source, float_precision="round_trip")
    original = _canonicalize_derived_no_fault_values(loaded)
    enriched = add_reproducibility_provenance(original)
    assert_frame_equal(
        enriched[original.columns], original,
        check_exact=True, check_dtype=True,
    )
    sampled_columns = [
        column for column in ("stim_fidelity", "stim_shots", "stim_seed")
        if column in loaded.columns
    ]
    assert_frame_equal(
        enriched[sampled_columns], loaded[sampled_columns],
        check_exact=True, check_dtype=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_to_csv(enriched, str(destination))
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", nargs="?", type=Path,
        default=Path("exp1_fp/results/data.csv"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    enriched = backfill(args.csv, args.output)
    print(
        f"Atomically wrote {len(enriched)} rows and {len(enriched.columns)} "
        f"columns to {args.output or args.csv} without changing sampled values."
    )


if __name__ == "__main__":
    main()
