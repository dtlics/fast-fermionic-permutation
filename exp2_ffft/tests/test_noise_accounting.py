"""Regression tests for Experiment 2 CNOT-equivalent noise accounting."""

import cirq
import pytest
from cirq.contrib.acquaintance.permutation import SwapPermutationGate
from openfermion.circuits.gates import FSWAP
from openfermion.circuits.primitives.ffft import F0

from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from common.gamma_folded import (
    FOLDED_GAMMA_SCHEDULE_MODEL,
    GAMMA_SCHEDULE_NOT_APPLICABLE,
)
from exp2_ffft.collect import (
    ACCOUNTING_MODEL,
    HALL_NOT_APPLICABLE,
    _append_rows,
    _gate_cnot_cost,
    count_cnot_resources,
    native_fidelity,
)
from exp2_ffft.plot import _ancilla_fidelity_from_resources


@pytest.mark.parametrize(
    "gate",
    [
        FSWAP,
        F0,
        cirq.PhasedISwapPowGate(phase_exponent=0.25, exponent=0.5),
        SwapPermutationGate(FSWAP),
    ],
    ids=["fswap", "f0_beam_splitter", "phased_iswap_givens", "swap_permutation"],
)
def test_all_ffft_two_axis_gate_classes_cost_two_native_layers(gate):
    assert _gate_cnot_cost(gate) == 2


def test_unknown_two_qubit_gate_fails_closed():
    with pytest.raises(ValueError, match="unrecognized two-qubit gate"):
        _gate_cnot_cost(cirq.SWAP)


def test_multi_qubit_primitive_fails_closed():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CCNOT.on(*qubits[:3]))

    with pytest.raises(ValueError, match="multi-qubit primitive"):
        count_cnot_resources(circuit, L=2)


def test_pure_fswap_uses_two_gate_and_time_layers():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(FSWAP.on(qubits[0], qubits[1]))

    resources = count_cnot_resources(circuit, L=2)

    assert resources["total_2q_gates"] == 1
    assert resources["total_cnot_equiv"] == 2
    assert resources["cnot_depth"] == 2
    assert resources["total_idle_slots_layered"] == 2 * (4 - 2)


def test_mixed_fswap_cnot_moment_charges_second_layer_idles():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.Moment(
            [
                FSWAP.on(qubits[0], qubits[1]),
                cirq.CNOT.on(qubits[2], qubits[3]),
            ]
        )
    )

    resources = count_cnot_resources(circuit, L=2)

    assert resources["total_cnot_equiv"] == 3
    assert resources["cnot_depth"] == 2
    assert resources["total_idle_slots"] == 0  # legacy raw-moment count
    assert resources["total_idle_slots_layered"] == 2
    assert resources["total_idle_slots_layered"] == (
        resources["total_qubits"] * resources["cnot_depth"]
        - 2 * resources["total_cnot_equiv"]
    )


def test_only_exact_pauli_z_frame_updates_are_free():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.Moment(
            [
                cirq.Z.on(qubits[0]),
                cirq.S.on(qubits[1]),
                (cirq.Z ** 0.25).on(qubits[2]),
                cirq.H.on(qubits[3]),
            ]
        )
    )

    resources = count_cnot_resources(circuit, L=2)

    assert resources["n_1q_nonz"] == 3


def _resource_fixture():
    # Raw counts deliberately differ from the CNOT-equivalent counts.  A
    # regression to total_2q_gates/total_idle_slots therefore fails loudly.
    return {
        "total_qubits": 4,
        "cnot_depth": 2,
        "total_2q_gates": 1,
        "total_idle_slots": 0,
        "total_cnot_equiv": 2,
        "total_idle_slots_layered": 4,
        "n_1q_nonz": 1,
        "n_exact_t": 0,
        "n_synth_rz": 0,
    }


def test_native_fidelity_never_uses_raw_gate_or_idle_counts():
    resources = _resource_fixture()
    p_2q = 0.1
    p_idle = 0.01
    expected = (1 - p_2q) ** 2 * (1 - p_idle) ** 4

    observed = native_fidelity(resources, p_2q=p_2q, p_idle=p_idle)

    assert observed == pytest.approx(expected)
    assert observed != pytest.approx((1 - p_2q) ** 1)


def test_collected_fidelity_and_provenance_use_native_model():
    resources = _resource_fixture()
    rows = []

    _append_rows(
        rows,
        L=2,
        N=4,
        method="test",
        res=resources,
        p_values=(0.1,),
        p_idle_factor=1.0,
        verify_err=0.0,
    )

    row = rows[0]
    expected = (1 - 0.1) ** 2 * (1 - 0.1) ** 4
    assert row["mult_fidelity"] == pytest.approx(expected)
    assert row["mult_fidelity_legacy"] == pytest.approx(1 - 0.1)
    assert row["p_idle"] == pytest.approx(0.1)
    assert row["p_1q"] == pytest.approx(0.1)
    assert row["accounting_model"] == ACCOUNTING_MODEL
    assert row["hall_decomposition_model"] == HALL_NOT_APPLICABLE
    assert row["gamma_schedule_model"] == GAMMA_SCHEDULE_NOT_APPLICABLE

    proper_rows = []
    _append_rows(
        proper_rows,
        L=2,
        N=4,
        method="gamma_2d_proper",
        res=resources,
        p_values=(0.1,),
        p_idle_factor=1.0,
        verify_err=0.0,
        gamma_method="folded",
    )
    assert proper_rows[0]["hall_decomposition_model"] == HALL_DECOMPOSITION_MODEL
    assert proper_rows[0]["gamma_schedule_model"] == FOLDED_GAMMA_SCHEDULE_MODEL


def test_on_the_fly_ancilla_curve_uses_uniform_nonentangling_noise():
    resources = _resource_fixture()
    p_2q = 0.1
    expected = (1 - p_2q) ** 2 * (1 - p_2q) ** 4

    observed = _ancilla_fidelity_from_resources(
        resources, p_2q=p_2q, p_idle_factor=1.0
    )

    assert observed == pytest.approx(expected)


def test_explicit_one_qubit_inventory_is_not_double_counted():
    resources = _resource_fixture()
    base = native_fidelity(resources, p_2q=0.1, p_idle=0.1, p_1q=0.1)
    changed_metadata = native_fidelity(
        resources, p_2q=0.1, p_idle=0.1, p_1q=0.9
    )

    assert changed_metadata == pytest.approx(base)
