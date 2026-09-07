"""Strict publication-data guards for the sparse-SYK plots."""

import pandas as pd
import numpy as np
import pytest

from common.gamma_folded import folded_gamma_schedule_model
from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from exp3_syk.collect import ACCOUNTING_MODEL, RESOURCE_DEPENDENCY_VERSIONS
from exp3_syk.plot import validate_publication_data
from exp3_syk.syk_instance import (
    generate_sparse_syk,
    instance_reproducibility_provenance,
)


def _valid_row() -> dict:
    instance = generate_sparse_syk(4, 1.0, np.random.default_rng(seed=0))
    provenance = instance_reproducibility_provenance(instance, 0)
    p_2q = 1e-3
    p_idle = p_2q
    total_cnots = 7
    total_qubits = 16
    depth = 2
    idle = total_qubits * depth - 2 * total_cnots
    return {
        "L": 4,
        "N": 16,
        "k": 1.0,
        "instance_idx": 0,
        "baseline": "folded",
        "n_colors": instance.n_colors,
        "n_quartets": len(instance.quartets),
        **provenance,
        "p_2q": p_2q,
        "p_idle": p_idle,
        "accounting_model": ACCOUNTING_MODEL,
        "hall_decomposition_model": HALL_DECOMPOSITION_MODEL,
        "gamma_schedule_model": folded_gamma_schedule_model(4),
        **RESOURCE_DEPENDENCY_VERSIONS,
        "n_ancillas": 0,
        "total_qubits": total_qubits,
        "cnot_depth": depth,
        "fp_cnot_depth": 1,
        "interaction_cnot_depth": 1,
        "total_cnots": total_cnots,
        "n_fswap": 2,
        "n_2q_cnot_cz": 3,
        "total_idle_slots_layered": idle,
        "spacetime_volume": total_qubits * depth,
        "n_1q_hadamard": 1,
        "n_1q_other": 2,
        "n_1q_s_sdag": 3,
        "n_1q_rz": 4,
        "mult_fidelity": (
            (1 - p_2q) ** total_cnots
            * (1 - p_idle) ** idle
        ),
    }


def test_structurally_valid_strict_fswap_row_is_accepted():
    validate_publication_data(
        pd.DataFrame([_valid_row()]), require_publication_coverage=False,
    )


def test_subnormal_no_fault_round_trip_is_accepted():
    row = _valid_row()
    row["cnot_depth"] = 1_000_000
    row["fp_cnot_depth"] = 500_000
    row["interaction_cnot_depth"] = 500_000
    row["total_idle_slots_layered"] = (
        row["total_qubits"] * row["cnot_depth"]
        - 2 * row["total_cnots"]
    )
    row["spacetime_volume"] = row["total_qubits"] * row["cnot_depth"]
    row["mult_fidelity"] = np.nextafter(0.0, 1.0)
    validate_publication_data(
        pd.DataFrame([row]), require_publication_coverage=False,
    )


def test_raw_gate_fallback_is_rejected():
    row = _valid_row()
    row["total_2q_gates_legacy"] = row.pop("total_cnots")
    with pytest.raises(ValueError, match="publication-accounting columns"):
        validate_publication_data(
            pd.DataFrame([row]), require_publication_coverage=False,
        )


def test_broken_fswap_identity_is_rejected():
    row = _valid_row()
    row["n_fswap"] += 1
    with pytest.raises(ValueError, match="2 n_FSWAP"):
        validate_publication_data(
            pd.DataFrame([row]), require_publication_coverage=False,
        )


def test_broken_depth_breakdown_is_rejected():
    row = _valid_row()
    row["interaction_cnot_depth"] += 1
    with pytest.raises(ValueError, match=r"D_FP \+ D_interaction"):
        validate_publication_data(
            pd.DataFrame([row]), require_publication_coverage=False,
        )


def test_duplicate_keys_are_rejected():
    row = _valid_row()
    with pytest.raises(ValueError, match="duplicate"):
        validate_publication_data(
            pd.DataFrame([row, row]), require_publication_coverage=False,
        )


def test_noise_dependent_rz_count_is_rejected():
    first = _valid_row()
    second = _valid_row()
    second["p_2q"] = 1e-4
    second["p_idle"] = second["p_2q"]
    second["mult_fidelity"] = (
        (1 - second["p_2q"]) ** second["total_cnots"]
        * (1 - second["p_idle"]) ** second["total_idle_slots_layered"]
    )
    second["n_1q_rz"] += 1
    with pytest.raises(ValueError, match="noise-dependent resource counts"):
        validate_publication_data(
            pd.DataFrame([first, second]), require_publication_coverage=False,
        )


@pytest.mark.parametrize("column", ["cirq_version", "openfermion_version"])
def test_stale_circuit_dependency_is_rejected(column):
    row = _valid_row()
    row[column] = "stale-version"
    with pytest.raises(ValueError, match="dependency provenance mismatch"):
        validate_publication_data(
            pd.DataFrame([row]), require_publication_coverage=False,
        )


def test_stale_pauli_rotation_compiler_is_rejected():
    row = _valid_row()
    row["pauli_rotation_compiler_model"] = "lexicographic_nonlocal_legacy"
    with pytest.raises(ValueError, match="dependency provenance mismatch"):
        validate_publication_data(
            pd.DataFrame([row]), require_publication_coverage=False,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.pop("gamma_schedule_model"),
        lambda row: row.__setitem__(
            "gamma_schedule_model", "folded_midpoint_fallback_v0"
        ),
    ],
)
def test_stale_gamma_schedule_is_rejected(mutate):
    row = _valid_row()
    mutate(row)
    with pytest.raises(ValueError, match="publication-accounting|stale"):
        validate_publication_data(
            pd.DataFrame([row]), require_publication_coverage=False,
        )
