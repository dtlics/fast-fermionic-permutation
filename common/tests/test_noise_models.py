"""Deterministic tests for CNOT-equivalent gate and noise accounting."""

import cirq
import numpy as np
import pandas as pd
import pytest
from openfermion.circuits.gates import FSWAP

from common.fp_1d import build_benchmark_permutation
from common.fp_2d import GammaMethod, build_fp_2d
from common.metrics import (
    classify_single_qubit_gate,
    count_resources,
    count_single_qubit_gates,
    no_fault_probability,
)
from common.noise_models import clifford_idle_fidelity, model_rates
from common.stim_convert import cirq_to_stim_circuit
from exp1_fp.collect import BASELINE_CONFIGS as FP_BASELINES
from exp3_syk.collect import BASELINE_CONFIGS as SYK_BASELINES
from exp3_syk.plot import (
    _canonical_accounting_series,
    _preferred_ancilla_free,
)


def test_classify_single_qubit_gates():
    assert classify_single_qubit_gate(cirq.Z) == "pauli_z"
    assert classify_single_qubit_gate(cirq.S) == "s_sdag"
    assert classify_single_qubit_gate(cirq.inverse(cirq.S)) == "s_sdag"
    assert classify_single_qubit_gate(cirq.H) == "hadamard"
    assert classify_single_qubit_gate(cirq.rz(0.37)) == "rz"
    assert classify_single_qubit_gate(cirq.T) == "rz"
    assert classify_single_qubit_gate(cirq.X) == "other_1q"
    diagonal = cirq.MatrixGate(np.diag([1.0, np.exp(0.31j)]))
    assert classify_single_qubit_gate(diagonal) == "rz"


@pytest.mark.parametrize("gamma", list(GammaMethod))
def test_fp_single_qubit_gates_are_frame_trackable(gamma):
    perm = build_benchmark_permutation(4, "reverse")
    circuit = build_fp_2d(4, perm, gamma).circuit
    counts = count_single_qubit_gates(circuit)
    assert counts["hadamard"] == 0
    assert counts["s_sdag"] == 0
    assert counts["rz"] == 0
    assert counts["other_1q"] == 0
    assert counts["pauli_z"] == counts["total_1q"]


def test_native_counts_for_pure_cnot_and_fswap_moments():
    qubits = cirq.LineQubit.range(4)

    cnot = cirq.Circuit(cirq.Moment([cirq.CNOT(*qubits[:2])]))
    cnot_counts = count_resources(cnot, L=2)
    assert cnot_counts["cnot_depth"] == 1
    assert cnot_counts["total_cnots"] == 1
    assert cnot_counts["total_idle_slots_layered"] == 2

    fswap = cirq.Circuit(cirq.Moment([FSWAP(*qubits[:2])]))
    fswap_counts = count_resources(fswap, L=2)
    assert fswap_counts["cnot_depth"] == 2
    assert fswap_counts["total_cnots"] == 2
    assert fswap_counts["n_fswap"] == 1
    assert fswap_counts["total_idle_slots_layered"] == 4


def test_mixed_moment_and_global_idle_invariant():
    """Mixed moments charge completed CNOT pairs idle in layer two."""
    q = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(
        cirq.Moment([FSWAP(q[0], q[1]), cirq.CNOT(q[2], q[3])]),
        cirq.Moment([cirq.CZ(q[0], q[1]), cirq.CNOT(q[2], q[3])]),
        cirq.Moment(
            [FSWAP(q[0], q[1]), FSWAP(q[2], q[3]), FSWAP(q[4], q[5])]
        ),
    )
    counts = count_resources(circuit, L=2, n_ancillas=2)

    assert counts["cnot_depth"] == 5
    assert counts["total_cnots"] == 11
    assert counts["n_fswap"] == 4
    assert counts["n_2q_cnot_cz"] == 3
    assert counts["total_2q_gates"] == 7
    assert counts["total_idle_slots"] == 4
    assert counts["total_idle_slots_layered"] == 8
    assert counts["total_idle_slots_layered"] == (
        counts["total_qubits"] * counts["cnot_depth"]
        - 2 * counts["total_cnots"]
    )


@pytest.mark.parametrize("gamma", list(GammaMethod))
def test_global_idle_invariant_for_all_gamma_methods(gamma):
    L = 4
    perm = build_benchmark_permutation(L, "transpose")
    result = build_fp_2d(L, perm, gamma)
    counts = count_resources(result.circuit, L, len(result.anc_qubits))
    assert counts["total_idle_slots_layered"] == (
        counts["total_qubits"] * counts["cnot_depth"]
        - 2 * counts["total_cnots"]
    )


def test_named_model_uses_native_gate_and_layered_idle_exponents():
    p = 1e-3
    counts = {
        "total_cnots": 7,
        "total_idle_slots_layered": 11,
        "hadamard": 3,
        "s_sdag": 2,
        "pauli_z": 100,
    }
    expected_a = (1 - p) ** 7 * (1 - p / 10) ** (11 + 3 + 2)
    assert clifford_idle_fidelity(counts, "modelA", p) == pytest.approx(
        expected_a
    )
    assert clifford_idle_fidelity(
        counts, "modelA", p, p_idle=p / 20
    ) == pytest.approx(
        (1 - p) ** 7 * (1 - p / 20) ** 11 * (1 - p / 10) ** 5
    )
    # In the uniform early-FT model, the non-entangling-slot exponent already
    # covers H/S activity; it must not be charged a second time.
    expected_b = (1 - p) ** (7 + 11)
    assert clifford_idle_fidelity(counts, "modelB", p) == pytest.approx(
        expected_b
    )
    assert clifford_idle_fidelity(
        counts, "modelB_pauli", p
    ) == pytest.approx(expected_b * (1 - p) ** 100)

    assert model_rates("modelA", p)["p_idle"] == pytest.approx(p / 10)
    assert no_fault_probability(7, 11, p, p / 10) == pytest.approx(
        (1 - p) ** 7 * (1 - p / 10) ** 11
    )


def test_named_model_rejects_ambiguous_legacy_count_keys():
    with pytest.raises(KeyError):
        clifford_idle_fidelity(
            {"two_q": 7, "idle_slots": 11}, "modelA", 1e-3
        )


def test_collectors_include_folded_without_removing_pipelined():
    for configs in (FP_BASELINES, SYK_BASELINES):
        names = [name for name, _ in configs]
        assert "folded" in names
        assert "pipelined" in names


def test_plot_prefers_folded_and_rejects_legacy_accounting_csv():
    assert _preferred_ancilla_free(["pipelined", "folded"]) == "folded"
    assert _preferred_ancilla_free(["pipelined"]) == "pipelined"

    historical = pd.DataFrame(
        {
            "total_cnots": [3],
            "total_qubits": [6],
            "cnot_depth": [2],
            "total_idle_slots": [2],
        }
    )
    with pytest.raises(ValueError, match="total_idle_slots_layered"):
        _canonical_accounting_series(historical)

    current = historical.assign(total_idle_slots_layered=[6])
    gates, idle = _canonical_accounting_series(current)
    assert gates.iloc[0] == 3
    assert idle.iloc[0] == 6


def _noise_lines(circuit: cirq.Circuit, qubits, *, layered=True):
    stim_circuit = cirq_to_stim_circuit(
        circuit,
        qubits,
        p_2q=1e-3,
        p_idle=1e-4,
        layered=layered,
    )
    return [
        line
        for line in str(stim_circuit).splitlines()
        if line.startswith("DEPOLARIZE")
    ]


def test_stim_pure_fswap_has_two_gate_and_idle_layers():
    q = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.Moment([FSWAP(q[0], q[1])]))
    layered = _noise_lines(circuit, q)
    legacy = _noise_lines(circuit, q, layered=False)

    assert layered.count("DEPOLARIZE2(0.001) 0 1") == 2
    assert layered.count("DEPOLARIZE1(0.0001) 2 3") == 2
    assert legacy == [
        "DEPOLARIZE2(0.001) 0 1",
        "DEPOLARIZE1(0.0001) 2 3",
    ]


def test_stim_mixed_moment_has_second_layer_idle_on_cnot_pair():
    q = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(
        cirq.Moment([FSWAP(q[0], q[1]), cirq.CNOT(q[2], q[3])])
    )
    assert _noise_lines(circuit, q) == [
        "DEPOLARIZE2(0.001) 0 1 2 3",
        "DEPOLARIZE1(0.0001) 4 5",
        "DEPOLARIZE2(0.001) 0 1",
        "DEPOLARIZE1(0.0001) 2 3 4 5",
    ]
