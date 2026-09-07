"""Focused regressions for FFFT early-fault-tolerant rotation inventory."""

import cirq
import numpy as np
import pytest
from openfermion.circuits.primitives.ffft import F0

from common.fp_2d import GammaMethod
from exp2_ffft.ffft_baseline_1d import build_ffft_1d
from exp2_ffft.ffft_proper import build_ffft_core, build_ffft_proper
from exp2_ffft.ft_accounting import (
    classify_rotation_angle,
    count_ffft_magic_rotations,
)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, "clifford"),
        (np.pi / 2, "clifford"),
        (-np.pi, "clifford"),
        (np.pi / 4, "exact_t"),
        (-3 * np.pi / 4, "exact_t"),
        (np.pi / 8, "synthesized"),
    ],
)
def test_rotation_angle_classification(angle, expected):
    assert classify_rotation_angle(angle) == expected


def test_f0_is_clifford_z_after_two_exact_t_angle_pauli_rotations():
    q0, q1 = cirq.LineQubit.range(2)
    ryxxy = cirq.PhasedISwapPowGate(
        phase_exponent=0.25, exponent=0.5,
    ).on(q0, q1)
    decomposition = cirq.Circuit(ryxxy, cirq.Z(q1))
    cirq.testing.assert_allclose_up_to_global_phase(
        decomposition.unitary(qubit_order=[q0, q1]),
        cirq.unitary(F0),
        atol=1e-10,
    )
    assert count_ffft_magic_rotations(cirq.Circuit(F0.on(q0, q1))) == {
        "n_exact_t": 2,
        "n_synth_rz": 0,
    }


def test_diagonal_and_givens_rotations_are_counted_by_actual_angle():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        (cirq.Z ** 0.5).on(q0),      # Clifford S
        (cirq.Z ** 0.25).on(q0),     # exact T
        (cirq.Z ** 0.125).on(q0),    # arbitrary synthesis
        cirq.PhasedISwapPowGate(
            phase_exponent=0.25, exponent=0.25,
        ).on(q0, q1),                # theta=pi/8, twice
    )
    assert count_ffft_magic_rotations(circuit) == {
        "n_exact_t": 1,
        "n_synth_rz": 3,
    }


def test_generic_givens_is_exactly_two_commuting_pauli_rotations():
    """Regress the unitary identity underlying the two-Rz inventory."""
    theta = 0.37
    gate = cirq.PhasedISwapPowGate(
        phase_exponent=0.25,
        exponent=2.0 * theta / np.pi,
    )
    x = cirq.unitary(cirq.X)
    y = cirq.unitary(cirq.Y)
    yx = np.kron(y, x)
    xy = np.kron(x, y)

    assert np.allclose(yx @ xy, xy @ yx, rtol=0.0, atol=1e-12)

    def pauli_rotation(pauli, angle):
        # P^2=I, hence exp(-i angle P/2) has this closed form.
        return (
            np.cos(angle / 2.0) * np.eye(4)
            - 1j * np.sin(angle / 2.0) * pauli
        )

    decomposition = (
        pauli_rotation(yx, theta)
        @ pauli_rotation(xy, -theta)
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(gate), decomposition, atol=1e-10
    )
    assert count_ffft_magic_rotations(
        cirq.Circuit(gate.on(*cirq.LineQubit.range(2)))
    ) == {"n_exact_t": 0, "n_synth_rz": 2}


def test_unknown_nonclifford_primitive_fails_closed():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit((cirq.CZ ** 0.5).on(q0, q1))
    with pytest.raises(ValueError, match="unsupported non-Clifford"):
        count_ffft_magic_rotations(circuit)


@pytest.mark.parametrize(
    ("L", "expected_1d", "expected_2d"),
    [
        (2, {"n_exact_t": 8, "n_synth_rz": 0},
         {"n_exact_t": 8, "n_synth_rz": 0}),
        (3, {"n_exact_t": 24, "n_synth_rz": 16},
         {"n_exact_t": 24, "n_synth_rz": 16}),
        (4, {"n_exact_t": 70, "n_synth_rz": 4},
         {"n_exact_t": 68, "n_synth_rz": 4}),
    ],
)
def test_generated_ffft_rotation_inventories(L, expected_1d, expected_2d):
    one_d = count_ffft_magic_rotations(build_ffft_1d(L).circuit)
    core = count_ffft_magic_rotations(
        build_ffft_core(L, GammaMethod.FOLDED).circuit
    )
    proper = count_ffft_magic_rotations(
        build_ffft_proper(L, GammaMethod.FOLDED).circuit
    )
    assert one_d == expected_1d
    assert core == expected_2d
    # Gamma and the mode-restoring FP are Clifford, so they add no magic.
    assert proper == core


@pytest.mark.parametrize(
    ("L", "expected_inventory"),
    [
        (2, {"n_exact_t": 8, "n_synth_rz": 0}),
        (3, {"n_exact_t": 24, "n_synth_rz": 16}),
    ],
)
def test_collected_ancilla_proper_resources_use_explicit_ancilla_and_magic_counts(
    L, expected_inventory,
):
    """Audit the exact ancilla-Gamma circuit represented by its CSV row."""
    from exp2_ffft.collect import count_cnot_resources

    result = build_ffft_proper(L, GammaMethod.ANCILLA)
    inventory = count_ffft_magic_rotations(result.circuit)
    resources = count_cnot_resources(
        result.circuit, L, len(result.anc_qubits),
    )

    assert len(result.anc_qubits) == L
    assert inventory == expected_inventory
    assert resources["n_exact_t"] == inventory["n_exact_t"]
    assert resources["n_synth_rz"] == inventory["n_synth_rz"]
    assert resources["total_qubits"] == L * L + L
    spacetime_volume = resources["total_qubits"] * resources["cnot_depth"]
    assert resources["total_idle_slots_layered"] == (
        spacetime_volume - 2 * resources["total_cnot_equiv"]
    )
