"""Exact correctness and topology tests for the sparse-SYK phase compiler."""

from __future__ import annotations

import itertools

import numpy as np
import cirq
import pytest

from common.fp_2d import GammaMethod
from common.metrics import count_resources
from exp3_syk.syk_instance import generate_sparse_syk
from exp3_syk.trotter import (
    _assert_disjoint_packed_rotation_spans,
    _snake_qubits,
    assert_nearest_neighbor_circuit,
    build_pauli_exp_circuit,
    build_trotter_step_fp,
    build_trotter_step_naive,
)


@pytest.mark.parametrize(
    "pauli_word",
    [
        "".join(labels)
        for labels in itertools.product("IXYZ", repeat=4)
        if any(label != "I" for label in labels)
    ],
)
def test_dirty_bridge_sweep_is_exact_for_every_four_site_pauli(
    pauli_word: str,
):
    L = 2
    snake = _snake_qubits(L)
    support = {
        qubit: pauli_word[index]
        for index, qubit in enumerate(snake)
        if pauli_word[index] != "I"
    }
    angle = 0.173
    circuit = build_pauli_exp_circuit(support, angle, L)
    assert_nearest_neighbor_circuit(circuit)

    pauli = cirq.PauliString(
        {qubit: getattr(cirq, label) for qubit, label in support.items()}
    ).matrix(qubits=snake)
    expected = np.cos(angle) * np.eye(16) - 1j * np.sin(angle) * pauli
    actual = circuit.unitary(qubit_order=snake)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)

    support_positions = [
        index for index, label in enumerate(pauli_word) if label != "I"
    ]
    bridge_count = (
        support_positions[-1]
        - support_positions[0]
        + 1
        - len(support_positions)
    )
    expected_cnots = 2 * (len(support_positions) - 1 + 2 * bridge_count)
    observed_cnots = sum(
        isinstance(operation.gate, cirq.CNotPowGate)
        for operation in circuit.all_operations()
    )
    assert observed_cnots == expected_cnots


def test_disconnected_mixed_pauli_rotation_is_exact_and_nearest_neighbor():
    L = 3
    snake = _snake_qubits(L)
    support = {
        snake[0]: "X",
        snake[2]: "Y",
        snake[5]: "Z",
        snake[8]: "X",
    }
    angle = 0.173
    circuit = build_pauli_exp_circuit(support, angle, L)
    assert_nearest_neighbor_circuit(circuit)

    qubit_order = snake
    pauli = cirq.PauliString(
        {qubit: getattr(cirq, label) for qubit, label in support.items()}
    ).matrix(qubits=qubit_order)
    expected = (
        np.cos(angle) * np.eye(2 ** len(qubit_order))
        - 1j * np.sin(angle) * pauli
    )
    actual = circuit.unitary(qubit_order=qubit_order)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_packed_rotation_span_guard_accepts_disjoint_complete_spans():
    L = 2
    snake = _snake_qubits(L)
    first = {snake[0]: "X", snake[1]: "Z"}
    second = {snake[2]: "Y", snake[3]: "Z"}
    rotations = [
        (0, first, build_pauli_exp_circuit(first, 0.1, L)),
        (1, second, build_pauli_exp_circuit(second, 0.2, L)),
    ]
    _assert_disjoint_packed_rotation_spans(rotations, L, color_index=0)


def test_packed_rotation_span_guard_rejects_overlapping_dirty_bridges():
    L = 2
    snake = _snake_qubits(L)
    bridged = {snake[0]: "X", snake[2]: "Z"}
    disjoint_support = {snake[1]: "Y"}
    rotations = [
        (0, bridged, build_pauli_exp_circuit(bridged, 0.1, L)),
        (1, disjoint_support, build_pauli_exp_circuit(disjoint_support, 0.2, L)),
    ]
    with pytest.raises(ValueError, match="dirty-bridge spans overlap"):
        _assert_disjoint_packed_rotation_spans(rotations, L, color_index=0)


def test_compiler_rejects_support_outside_declared_grid():
    with pytest.raises(ValueError, match="outside"):
        build_pauli_exp_circuit({cirq.GridQubit(3, 0): "Z"}, 0.1, L=3)


def test_topology_guard_rejects_nonlocal_two_qubit_gate():
    circuit = cirq.Circuit(
        cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1))
    )
    with pytest.raises(ValueError, match="non-nearest-neighbor"):
        assert_nearest_neighbor_circuit(circuit)


def test_topology_guard_rejects_non_grid_two_qubit_gate():
    circuit = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))
    with pytest.raises(ValueError, match="not on GridQubits"):
        assert_nearest_neighbor_circuit(circuit)


def test_topology_guard_rejects_undecomposed_multiqubit_gate():
    qubits = cirq.LineQubit.range(3)
    operation = cirq.MatrixGate(
        np.eye(8), qid_shape=(2, 2, 2)
    ).on(*qubits)
    with pytest.raises(ValueError, match="undecomposed multi-qubit"):
        assert_nearest_neighbor_circuit(cirq.Circuit(operation))


@pytest.mark.parametrize(
    ("baseline", "gamma_method"),
    [
        ("1d", None),
        ("ancilla", GammaMethod.ANCILLA),
        ("folded", GammaMethod.FOLDED),
    ],
)
def test_complete_fp_trotter_circuit_is_nearest_neighbor(
    baseline: str, gamma_method: GammaMethod | None
):
    instance = generate_sparse_syk(4, 1.0, np.random.default_rng(seed=0))
    circuit, n_ancillas, fp_depth, interaction_depth = build_trotter_step_fp(
        instance, baseline, gamma_method
    )
    assert_nearest_neighbor_circuit(circuit)
    assert count_resources(circuit, 4, n_ancillas)["cnot_depth"] == (
        fp_depth + interaction_depth
    )


def test_complete_naive_trotter_circuit_is_nearest_neighbor():
    instance = generate_sparse_syk(4, 1.0, np.random.default_rng(seed=0))
    circuit, n_ancillas, fp_depth, interaction_depth = (
        build_trotter_step_naive(instance)
    )
    assert_nearest_neighbor_circuit(circuit)
    assert fp_depth == 0
    assert count_resources(circuit, 4, n_ancillas)["cnot_depth"] == (
        fp_depth + interaction_depth
    )
