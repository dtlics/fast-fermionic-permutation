"""Tests for Cirq-to-Stim conversion and all-zero return sampling."""

import numpy as np
import pytest
import cirq

from common.fp_1d import build_fp_1d, build_benchmark_permutation
from common.fp_2d import GammaMethod, build_fp_2d
from common.stim_convert import (
    _packed_rows_are_zero,
    cirq_to_stim_circuit,
    simulate_all_zero_return_probability,
    simulate_clifford_fidelity,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _qubit_order(result):
    """Get qubit order: system qubits (raster) + ancilla qubits."""
    return list(result.sys_qubits) + list(result.anc_qubits)


def test_bit_packed_zero_rows_match_unpacked_for_partial_final_byte():
    """Packed all-zero decoding must ignore padding above the tenth bit."""
    unpacked = np.zeros((5, 10), dtype=np.uint8)
    unpacked[1, 0] = 1
    unpacked[2, 7] = 1
    unpacked[3, 8] = 1
    unpacked[4, 9] = 1
    expected = np.all(unpacked == 0, axis=1)

    packed = np.packbits(unpacked, axis=1, bitorder="little")
    # Exercise masking explicitly: these are padding, not measured bits.
    packed[:, -1] |= 0b11111100

    assert np.array_equal(_packed_rows_are_zero(packed, 10), expected)


# ---------------------------------------------------------------------------
# Noiseless identity test: forward + inverse = all zeros
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [3, 4, 5])
@pytest.mark.parametrize("perm_kind", ["reverse", "transpose", "random"])
def test_noiseless_return_probability_is_one_1d(L, perm_kind):
    """Noiseless forward + inverse must return all zeros (1-D baseline)."""
    perm = build_benchmark_permutation(L, perm_kind)
    result = build_fp_1d(L, perm)
    qo = _qubit_order(result)
    probability = simulate_all_zero_return_probability(
        result.circuit, qo, p_2q=0.0, p_idle=0.0, shots=100, seed=101 + L
    )
    assert probability == 1.0


@pytest.mark.parametrize(
    "gamma",
    [
        GammaMethod.ANCILLA,
        GammaMethod.PRIMITIVE,
        GammaMethod.PIPELINED,
        GammaMethod.FOLDED,
    ],
)
def test_noiseless_return_probability_is_one_2d(gamma):
    """Noiseless forward + inverse must return all zeros for every 2-D method."""
    L = 4
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_2d(L, perm, gamma)
    qo = _qubit_order(result)
    probability = simulate_all_zero_return_probability(
        result.circuit, qo, p_2q=0.0, p_idle=0.0, shots=100, seed=211
    )
    assert probability == 1.0, f"Expected probability 1.0 for {gamma}"


# ---------------------------------------------------------------------------
# Noisy return probability is < 1
# ---------------------------------------------------------------------------

def test_noisy_return_probability_less_than_one():
    """With noise, the all-zero return probability must be strictly below 1."""
    L = 4
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_1d(L, perm)
    qo = _qubit_order(result)
    probability = simulate_all_zero_return_probability(
        result.circuit,
        qo,
        p_2q=0.01,
        p_idle=0.001,
        shots=1000,
        seed=307,
    )
    assert 0.0 < probability < 1.0


def test_seeded_sampling_is_exactly_repeatable():
    """A fixed Stim seed must reproduce the identical Monte Carlo estimate."""
    L = 3
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_1d(L, perm)
    qo = _qubit_order(result)
    kwargs = dict(p_2q=0.02, p_idle=0.002, shots=4096, seed=8675309)

    first = simulate_all_zero_return_probability(result.circuit, qo, **kwargs)
    second = simulate_all_zero_return_probability(result.circuit, qo, **kwargs)

    assert first == second


def test_legacy_simulation_name_delegates_to_return_probability():
    """The historical API name must remain an exact compatibility wrapper."""
    result = build_fp_1d(3, build_benchmark_permutation(3, "transpose"))
    qo = _qubit_order(result)
    kwargs = dict(p_2q=0.01, p_idle=0.001, shots=1024, seed=401)

    assert simulate_clifford_fidelity(
        result.circuit, qo, **kwargs
    ) == simulate_all_zero_return_probability(result.circuit, qo, **kwargs)


# ---------------------------------------------------------------------------
# Return probability decreases with higher noise
# ---------------------------------------------------------------------------

def test_return_probability_decreases_with_noise():
    """Higher noise should reduce the all-zero return probability."""
    L = 4
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_1d(L, perm)
    qo = _qubit_order(result)

    probability_low = simulate_all_zero_return_probability(
        result.circuit,
        qo,
        p_2q=1e-3,
        p_idle=1e-4,
        shots=2000,
        seed=503,
    )
    probability_high = simulate_all_zero_return_probability(
        result.circuit,
        qo,
        p_2q=1e-2,
        p_idle=1e-3,
        shots=2000,
        seed=503,
    )
    assert probability_low > probability_high


# ---------------------------------------------------------------------------
# Stim converter handles all gate types
# ---------------------------------------------------------------------------

def test_converter_all_baselines():
    """Converter should handle circuits from all 4 baselines without error."""
    L = 4
    perm = build_benchmark_permutation(L, "transpose")

    result_1d = build_fp_1d(L, perm)
    cirq_to_stim_circuit(result_1d.circuit, _qubit_order(result_1d))

    for gamma in GammaMethod:
        result_2d = build_fp_2d(L, perm, gamma)
        cirq_to_stim_circuit(result_2d.circuit, _qubit_order(result_2d))


def test_converter_preserves_cirq_pauli_z_subclass():
    """The concrete ``cirq.Z`` singleton must not be dropped by conversion."""

    q = cirq.LineQubit(0)
    converted = cirq_to_stim_circuit(cirq.Circuit(cirq.Z(q)), [q])
    assert str(converted).splitlines() == ["Z 0", "TICK"]


# ---------------------------------------------------------------------------
# 1-D has lower return probability than pipelined at larger L
# ---------------------------------------------------------------------------

def test_pipelined_beats_1d_return_probability():
    """Pipelined O(L) depth should have higher return probability than 1-D.

    The crossover is ~L=11 (2L^2 vs ~22L depth). At L=14, pipelined is
    clearly better. Use a small rate so both probabilities remain resolved under
    the default FSWAP=2 layered noise schedule.
    """
    L = 14
    perm = build_benchmark_permutation(L, "reverse")

    result_1d = build_fp_1d(L, perm)
    result_pipe = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    qo_1d = _qubit_order(result_1d)
    qo_pipe = _qubit_order(result_pipe)

    probability_1d = simulate_all_zero_return_probability(
        result_1d.circuit,
        qo_1d,
        p_2q=1e-4,
        p_idle=1e-5,
        shots=4000,
        seed=601,
    )
    probability_pipe = simulate_all_zero_return_probability(
        result_pipe.circuit,
        qo_pipe,
        p_2q=1e-4,
        p_idle=1e-5,
        shots=4000,
        seed=607,
    )

    assert probability_pipe > probability_1d, (
        "Pipelined should have higher return probability than 1-D at L=14: "
        f"{probability_pipe} vs {probability_1d}"
    )


# --------------------------------------------------------------------------
# Process-fidelity estimator (Pauli-frame propagation with stim.FlipSimulator)
# --------------------------------------------------------------------------

from common.stim_convert import (  # noqa: E402
    simulate_process_fidelity,
    verify_ancilla_disentanglement,
    wilson_interval,
)


def _instance(L, method):
    perm = build_benchmark_permutation(L, "reverse")
    result = build_fp_1d(L, perm) if method == "1d" else build_fp_2d(L, perm, gamma_method=method)
    qo = list(result.sys_qubits) + list(result.anc_qubits)
    anc = list(range(len(result.sys_qubits), len(qo)))
    data = list(range(len(result.sys_qubits)))
    return result.circuit, qo, data, anc


@pytest.mark.parametrize("method", ["1d", GammaMethod.ANCILLA, GammaMethod.FOLDED])
def test_process_fidelity_is_one_without_noise(method):
    circuit, qo, data, anc = _instance(5, method)
    verify_ancilla_disentanglement(circuit, qo, anc)
    out = simulate_process_fidelity(circuit, qo, data, p_2q=0.0, p_idle=0.0, shots=2000, seed=3)
    assert out["fidelity"] == 1.0 and out["successes"] == 2000


@pytest.mark.parametrize("method", ["1d", GammaMethod.ANCILLA, GammaMethod.FOLDED])
def test_process_fidelity_detects_final_x_error(method):
    circuit, qo, data, _ = _instance(5, method)
    out = simulate_process_fidelity(
        circuit, qo, data, p_2q=0.0, p_idle=0.0, shots=500, seed=3,
        extra_tail=f"X_ERROR(1) {data[len(data) // 2]}",
    )
    assert out["fidelity"] == 0.0


@pytest.mark.parametrize("method", ["1d", GammaMethod.ANCILLA, GammaMethod.FOLDED])
def test_process_fidelity_detects_final_z_error(method):
    """A pure phase error is invisible to the all-zero return statistic but
    must register as a failure of the process fidelity."""
    circuit, qo, data, _ = _instance(5, method)
    out = simulate_process_fidelity(
        circuit, qo, data, p_2q=0.0, p_idle=0.0, shots=500, seed=3,
        extra_tail=f"Z_ERROR(1) {data[0]}",
    )
    assert out["fidelity"] == 0.0


def test_process_fidelity_ignores_errors_confined_to_ancillas():
    circuit, qo, data, anc = _instance(5, GammaMethod.ANCILLA)
    assert anc, "the ancilla method must carry ancillas"
    verify_ancilla_disentanglement(circuit, qo, anc)
    out = simulate_process_fidelity(
        circuit, qo, data, p_2q=0.0, p_idle=0.0, shots=300, seed=3,
        extra_tail=f"X_ERROR(1) {anc[0]}\nZ_ERROR(1) {anc[-1]}",
    )
    assert out["fidelity"] == 1.0


def test_ancilla_disentanglement_check_rejects_a_broken_circuit():
    circuit, qo, data, anc = _instance(5, GammaMethod.ANCILLA)
    broken = circuit + cirq.Circuit(cirq.CNOT(qo[data[0]], qo[anc[0]]))
    with pytest.raises(ValueError):
        verify_ancilla_disentanglement(broken, qo, anc)


def test_process_fidelity_is_below_one_with_noise_and_seeded():
    circuit, qo, data, _ = _instance(6, GammaMethod.FOLDED)
    a = simulate_process_fidelity(circuit, qo, data, p_2q=1e-3, p_idle=1e-4, shots=20000, seed=11)
    b = simulate_process_fidelity(circuit, qo, data, p_2q=1e-3, p_idle=1e-4, shots=20000, seed=11)
    assert 0.0 < a["fidelity"] < 1.0
    assert a == b
    assert a["ci_lo"] <= a["fidelity"] <= a["ci_hi"]


def test_process_fidelity_not_above_all_zero_return_probability():
    """The all-zero statistic ignores phase faults, so it can only over-estimate."""
    circuit, qo, data, _ = _instance(6, GammaMethod.FOLDED)
    fe = simulate_process_fidelity(circuit, qo, data, p_2q=1e-3, p_idle=1e-4, shots=50000, seed=5)["fidelity"]
    az = simulate_all_zero_return_probability(circuit, qo, p_2q=1e-3, p_idle=1e-4, shots=50000, seed=5)
    assert fe <= az + 0.01


def test_wilson_interval_basic_properties():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = wilson_interval(100, 100)
    assert 0.95 < lo < 1.0 and hi == 1.0
    lo, hi = wilson_interval(500, 1000)
    assert lo < 0.5 < hi and hi - lo < 0.07


def test_fswap_equals_h_cx_cx_h_and_simulator_form():
    """Item 4 of the revision: FSWAP executes as H_a, CX(a,b), CX(b,a), H_b."""
    from openfermion.circuits.gates import FSWAP
    a, b = cirq.LineQubit.range(2)
    u = cirq.unitary(cirq.Circuit(FSWAP(a, b)))
    seq = cirq.unitary(cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.H(b)))
    sim_form = cirq.unitary(cirq.Circuit(cirq.SWAP(a, b), cirq.CZ(a, b)))
    assert cirq.equal_up_to_global_phase(u, seq)
    assert cirq.equal_up_to_global_phase(u, sim_form)
