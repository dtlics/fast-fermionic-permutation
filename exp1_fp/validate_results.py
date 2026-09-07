"""Exhaustive validation for the publication Experiment 1 CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from exp1_fp.collect import (
    ACCOUNTING_MODEL,
    BASELINE_CONFIGS,
    DEPENDENCY_VERSIONS,
    DETERMINISTIC_PERMUTATION_SEED,
    PERMUTATION_MODEL,
    PUBLICATION_L_VALUES,
    SAMPLING_MODEL,
    _gamma_schedule_model_for_baseline,
    _hall_model_for_baseline,
    _stim_seed_for_row,
    _validated_resume_rows,
    has_valid_reproducibility_provenance,
)
from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from exp1_fp.plot import _validate_native_data


DEFAULT_L_VALUES = PUBLICATION_L_VALUES
DEFAULT_PERM_KINDS = ("reverse", "transpose", "random")
DEFAULT_P_VALUES = (1e-4, 1e-5)


def validate_results(
    df: pd.DataFrame,
    *,
    L_values: Sequence[int] = DEFAULT_L_VALUES,
    perm_kinds: Sequence[str] = DEFAULT_PERM_KINDS,
    n_random: int = 20,
    p_values: Sequence[float] = DEFAULT_P_VALUES,
    p_idle_factor: float = 1.0,
    shots: int = 1_000_000,
) -> dict:
    """Validate coverage, provenance, and every recorded resource identity."""
    _validate_native_data(df)
    baselines = tuple(name for name, _ in BASELINE_CONFIGS)
    n_permutations = sum(n_random if kind == "random" else 1 for kind in perm_kinds)
    rows_per_L = n_permutations * len(baselines) * len(p_values)
    expected_rows = len(L_values) * rows_per_L

    if len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(df)}")
    if set(df["L"].astype(int)) != set(L_values):
        raise ValueError("grid-size coverage differs from the requested sweep")
    if not (df.groupby("L").size() == rows_per_L).all():
        raise ValueError("one or more grid sizes lack their complete row block")

    complete_L, kept = _validated_resume_rows(
        df, perm_kinds, n_random, p_values, p_idle_factor, shots
    )
    if complete_L != set(L_values) or len(kept) != expected_rows:
        raise ValueError("resume-provenance validation rejected a grid-size block")

    keys = ["L", "perm_kind", "perm_idx", "baseline", "p_2q"]
    if df.duplicated(keys).any():
        raise ValueError("duplicate experiment keys found")
    if set(df["accounting_model"].astype(str)) != {ACCOUNTING_MODEL}:
        raise ValueError("mixed or stale accounting provenance found")
    hall_expected = df["baseline"].astype(str).map(_hall_model_for_baseline)
    if not (df["hall_decomposition_model"].astype(str) == hall_expected).all():
        raise ValueError("mixed or stale Hall-decomposition provenance found")
    gamma_schedule_expected = df.apply(
        lambda row: _gamma_schedule_model_for_baseline(
            str(row.baseline), int(row.L)
        ),
        axis=1,
    )
    if not (
        df["gamma_schedule_model"].astype(str) == gamma_schedule_expected
    ).all():
        raise ValueError("mixed or stale Gamma-schedule provenance found")
    if set(df["sampling_model"].astype(str)) != {SAMPLING_MODEL}:
        raise ValueError("mixed or stale Stim-sampling provenance found")
    seed_expected = df.apply(
        lambda row: _stim_seed_for_row(
            int(row.L), str(row.perm_kind), int(row.perm_idx),
            str(row.baseline), float(row.p_2q), float(row.p_idle),
        ),
        axis=1,
    )
    if not np.array_equal(
        df["stim_seed"].astype("int64"), seed_expected.astype("int64")
    ):
        raise ValueError("one or more Stim seeds do not match their row keys")
    if df["stim_seed"].duplicated().any():
        raise ValueError("distinct scientific rows unexpectedly share a Stim seed")
    if not has_valid_reproducibility_provenance(df):
        raise ValueError("permutation or dependency provenance is stale")
    if set(df["stim_shots"].astype(int)) != {shots}:
        raise ValueError("Stim shot provenance differs from the requested sweep")

    p_idle_expected = df["p_2q"] * p_idle_factor
    if not np.allclose(df["p_idle"], p_idle_expected, rtol=0, atol=1e-15):
        raise ValueError("idle-error rates do not match p_2q times the requested factor")
    if not (df["total_qubits"] == df["N"] + df["n_ancillas"]).all():
        raise ValueError("total-qubit counts are inconsistent")
    expected_ancillas = np.where(df["baseline"] == "ancilla", df["L"], 0)
    if not np.array_equal(df["n_ancillas"].to_numpy(), expected_ancillas):
        raise ValueError("ancilla counts are inconsistent with the method")
    if not (
        df["total_2q_gates_legacy"] == df["n_fswap"] + df["n_2q_cnot_cz"]
    ).all():
        raise ValueError("raw two-qubit gate counts are inconsistent")
    if not (
        df["total_cnots"] == 2 * df["n_fswap"] + df["n_2q_cnot_cz"]
    ).all():
        raise ValueError("FSWAP=2 CNOT-equivalent accounting is inconsistent")
    if not (
        df["total_idle_slots_layered"]
        == df["total_qubits"] * df["cnot_depth"] - 2 * df["total_cnots"]
    ).all():
        raise ValueError("layered idle-slot accounting is inconsistent")
    if not (
        df["spacetime_volume"] == df["total_qubits"] * df["cnot_depth"]
    ).all():
        raise ValueError("spacetime volume is inconsistent")

    no_fault = (
        (1 - df["p_2q"]) ** df["total_cnots"]
        * (1 - df["p_idle"]) ** df["total_idle_slots_layered"]
    )
    legacy_no_fault = (
        (1 - df["p_2q"]) ** df["total_2q_gates_legacy"]
        * (1 - df["p_idle"]) ** df["total_idle_slots_legacy"]
    )
    if not np.allclose(df["mult_fidelity"], no_fault, rtol=1e-12, atol=0):
        raise ValueError("native independent-location no-fault estimates are stale")
    if not np.allclose(
        df["mult_fidelity_legacy"], legacy_no_fault, rtol=1e-12, atol=0
    ):
        raise ValueError("legacy audit estimates are internally inconsistent")

    resource_columns = [
        "n_ancillas", "total_qubits", "cnot_depth", "total_cnots",
        "n_fswap", "n_2q_cnot_cz", "total_idle_slots_layered",
        "total_2q_gates_legacy", "total_idle_slots_legacy",
        "spacetime_volume",
    ]
    resource_variants = df.groupby(keys[:-1])[resource_columns].nunique(dropna=False)
    if (resource_variants > 1).any().any():
        raise ValueError("noise-independent resources vary across p_2q rows")

    for col in ("stim_fidelity_ci_lo", "stim_fidelity_ci_hi"):
        if col not in df.columns:
            raise ValueError(f"missing confidence-interval column {col}")
    if not ((df["stim_fidelity_ci_lo"] <= df["stim_fidelity"] + 1e-12)
            & (df["stim_fidelity"] <= df["stim_fidelity_ci_hi"] + 1e-12)).all():
        raise ValueError("a Stim confidence interval does not contain its estimate")
    measured_counts = df["stim_fidelity"].to_numpy() * shots
    if not np.allclose(measured_counts, np.rint(measured_counts), rtol=0, atol=1e-6):
        raise ValueError("Stim process fidelities are not quantized by the shot count")

    return {
        "rows": int(len(df)),
        "grid_sizes": [int(L) for L in sorted(complete_L)],
        "rows_per_grid_size": int(rows_per_L),
        "permutations_per_grid_size": int(n_permutations),
        "baselines": list(baselines),
        "p_2q_values": [float(p) for p in p_values],
        "stim_shots_per_row": int(shots),
        "accounting_model": ACCOUNTING_MODEL,
        "hall_decomposition_model_2d": HALL_DECOMPOSITION_MODEL,
        "gamma_schedule_model_folded_by_L": {
            str(int(L)): _gamma_schedule_model_for_baseline("folded", int(L))
            for L in sorted(set(L_values))
        },
        "sampling_model": SAMPLING_MODEL,
        "permutation_model": PERMUTATION_MODEL,
        "deterministic_permutation_seed_sentinel": DETERMINISTIC_PERMUTATION_SEED,
        "permutation_digest_model": "sha256_little_endian_uint64_destinations",
        "dependency_versions": DEPENDENCY_VERSIONS,
        "idle_invariant": "I = QD - 2G",
        "fswap_cnot_equivalent_cost": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("exp1_fp/results/data.csv"))
    args = parser.parse_args()
    summary = validate_results(pd.read_csv(args.csv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
