"""Tests for qp-to-Stim conversion and noisy Clifford fidelity simulation."""

import numpy as np
import pytest

from common.fp_1d import build_fp_1d, build_benchmark_permutation
from common.fp_2d import GammaMethod, build_fp_2d
from common.stim_convert import pennylane as qp_to_stim_circuit, simulate_clifford_fidelity


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _qubit_order(result):
    """Get qubit order: system qubits (raster) + ancilla qubits."""
    return list(result.sys_qubits) + list(result.anc_qubits)


# ---------------------------------------------------------------------------
# Noiseless identity test: forward + inverse = all zeros
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [3, 4, 5])
@pytest.mark.parametrize("perm_kind", ["reverse", "transpose", "random"])
def test_noiseless_fidelity_is_one_1d(L, perm_kind):
    """Noiseless forward + inverse must give fidelity = 1.0 (Baseline 1)."""
    perm = build_benchmark_permutation(L, perm_kind)
    result = build_fp_1d(L, perm)
    qo = _qubit_order(result)
    fid = simulate_clifford_fidelity(result.circuit, qo, p_2q=0.0, p_idle=0.0, shots=100)
    assert fid == 1.0, f"Expected fidelity 1.0, got {fid}"


@pytest.mark.parametrize("gamma", [GammaMethod.ANCILLA, GammaMethod.PRIMITIVE, GammaMethod.PIPELINED])
def test_noiseless_fidelity_is_one_2d(gamma):
    """Noiseless forward + inverse must give fidelity = 1.0 (Baselines 2-4)."""
    L = 4
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_2d(L, perm, gamma)
    qo = _qubit_order(result)
    fid = simulate_clifford_fidelity(result.circuit, qo, p_2q=0.0, p_idle=0.0, shots=100)
    assert fid == 1.0, f"Expected fidelity 1.0 for {gamma}, got {fid}"


# ---------------------------------------------------------------------------
# Noisy fidelity is < 1
# ---------------------------------------------------------------------------

def test_noisy_fidelity_less_than_one():
    """With noise, fidelity must be strictly less than 1."""
    L = 4
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_1d(L, perm)
    qo = _qubit_order(result)
    fid = simulate_clifford_fidelity(result.circuit, qo, p_2q=0.01, p_idle=0.001, shots=1000)
    assert fid < 1.0, f"Noisy fidelity should be < 1, got {fid}"
    assert fid > 0.0, f"Fidelity should be positive at modest noise"


# ---------------------------------------------------------------------------
# Fidelity decreases with higher noise
# ---------------------------------------------------------------------------

def test_fidelity_decreases_with_noise():
    """Higher noise rate should give lower fidelity."""
    L = 4
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_1d(L, perm)
    qo = _qubit_order(result)

    fid_low = simulate_clifford_fidelity(result.circuit, qo, p_2q=1e-3, p_idle=1e-4, shots=2000)
    fid_high = simulate_clifford_fidelity(result.circuit, qo, p_2q=1e-2, p_idle=1e-3, shots=2000)
    assert fid_low > fid_high, f"Lower noise should give higher fidelity: {fid_low} vs {fid_high}"


# ---------------------------------------------------------------------------
# Stim converter handles all gate types
# ---------------------------------------------------------------------------

def test_converter_all_baselines():
    """Converter should handle circuits from all 4 baselines without error."""
    L = 4
    perm = build_benchmark_permutation(L, "transpose")

    result_1d = build_fp_1d(L, perm)
    qp_to_stim_circuit(result_1d.circuit, _qubit_order(result_1d))

    for gamma in GammaMethod:
        result_2d = build_fp_2d(L, perm, gamma)
        qp_to_stim_circuit(result_2d.circuit, _qubit_order(result_2d))


# ---------------------------------------------------------------------------
# 1D has worse fidelity than pipelined at larger L
# ---------------------------------------------------------------------------

def test_pipelined_beats_1d_fidelity():
    """Pipelined (O(L) depth) should have better noisy fidelity than 1D (O(L^2)) at large L.

    The crossover is ~L=11 (2L^2 vs ~22L depth). At L=14, pipelined is clearly better.
    Use p=1e-3 to keep fidelities measurable.
    """
    L = 14
    perm = build_benchmark_permutation(L, "reverse")

    result_1d = build_fp_1d(L, perm)
    result_pipe = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    qo_1d = _qubit_order(result_1d)
    qo_pipe = _qubit_order(result_pipe)

    fid_1d = simulate_clifford_fidelity(result_1d.circuit, qo_1d, p_2q=1e-3, p_idle=1e-4, shots=2000)
    fid_pipe = simulate_clifford_fidelity(result_pipe.circuit, qo_pipe, p_2q=1e-3, p_idle=1e-4, shots=2000)

    assert fid_pipe > fid_1d, f"Pipelined should beat 1D at L=14: {fid_pipe} vs {fid_1d}"
