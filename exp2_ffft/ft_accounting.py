"""Early-fault-tolerant rotation inventory for the OpenFermion FFFT.

The collected CNOT-equivalent depth already charges the two-CNOT Clifford
skeleton of each fermionic beam splitter/Givens gate.  This module counts the
additional injected rotations without confusing arbitrary rotations with
Cliffords or with exact T/T-dagger gates.

For OpenFermion's number-conserving Givens primitive,

    Ryxxy(theta) = exp[-i theta (Y X - X Y) / 2],

the Pauli products YX and XY commute.  The gate is therefore two Pauli
rotations of angle theta, each reducible by Clifford basis changes to one
single-qubit Rz(theta).  The special two-mode Fourier gate satisfies

    F0 = (I tensor Z) Ryxxy(pi/4),

so it consumes two exact T/T-dagger states.  Diagonal FFFT twiddles consume
zero states at Clifford angles, one state at exact T angles, and one
arbitrary-angle synthesis otherwise.
"""

from __future__ import annotations

from typing import Dict

import cirq
import numpy as np
from openfermion.circuits.primitives.ffft import F0


ROTATION_INVENTORY_MODEL = "ffft_commuting_pauli_rotations_v1"
ANGLE_ATOL = 1e-8


def classify_rotation_angle(angle_rads: float, *, atol: float = ANGLE_ATOL) -> str:
    """Classify a Pauli rotation as Clifford, exact-T, or synthesized.

    Global phase is immaterial.  Multiples of pi/2 are Clifford; odd
    multiples of pi/4 are exact T/T-dagger rotations; every other concrete
    angle is assigned one arbitrary-angle synthesis.
    """
    angle = float(angle_rads)
    if not np.isfinite(angle):
        raise ValueError(f"non-finite FFFT rotation angle {angle!r}")

    quarter_turns = 4.0 * angle / np.pi
    nearest_quarter = round(quarter_turns)
    if np.isclose(quarter_turns, nearest_quarter, rtol=0.0, atol=atol):
        return "clifford" if nearest_quarter % 2 == 0 else "exact_t"
    return "synthesized"


def _add_rotation(inventory: Dict[str, int], angle_rads: float, count: int = 1) -> None:
    kind = classify_rotation_angle(angle_rads)
    if kind == "exact_t":
        inventory["n_exact_t"] += int(count)
    elif kind == "synthesized":
        inventory["n_synth_rz"] += int(count)


def _diagonal_rotation_angle(gate: cirq.Gate) -> float:
    """Return the relative |1>/<0> phase of a one-qubit diagonal gate."""
    try:
        unitary = np.asarray(cirq.unitary(gate), dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cannot determine magic cost of one-qubit gate {gate!r}"
        ) from exc
    if unitary.shape != (2, 2) or not np.allclose(
        unitary - np.diag(np.diag(unitary)), 0.0, rtol=0.0, atol=ANGLE_ATOL
    ):
        raise ValueError(
            "unsupported non-Clifford one-qubit FFFT gate; expected a "
            f"diagonal rotation, found {gate!r}"
        )
    if abs(unitary[0, 0]) < ANGLE_ATOL:
        raise ValueError(f"invalid diagonal one-qubit unitary {gate!r}")
    return float(np.angle(unitary[1, 1] / unitary[0, 0]))


def count_ffft_magic_rotations(circuit: cirq.Circuit) -> Dict[str, int]:
    """Count exact-T and arbitrary synthesized rotations in an FFFT circuit.

    The inventory is fail-closed: an unfamiliar non-Clifford primitive raises
    instead of being silently treated as free.  All currently generated
    OpenFermion FFFT circuits contain only Clifford gates, F0, Ryxxy encoded as
    ``PhasedISwapPowGate(phase_exponent=1/4)``, and diagonal one-qubit
    rotations.
    """
    inventory = {"n_exact_t": 0, "n_synth_rz": 0}

    for op in circuit.all_operations():
        gate = op.gate
        if gate is None or len(op.qubits) == 0:
            continue
        if cirq.has_stabilizer_effect(gate):
            continue

        if len(op.qubits) == 1:
            _add_rotation(inventory, _diagonal_rotation_angle(gate))
            continue

        if len(op.qubits) == 2 and isinstance(gate, type(F0)):
            # F0 = (I tensor Z) Ryxxy(pi/4): two exact T-angle Pauli
            # rotations plus a free Clifford Z.
            inventory["n_exact_t"] += 2
            continue

        if len(op.qubits) == 2 and isinstance(gate, cirq.PhasedISwapPowGate):
            phase_turns = float(gate.phase_exponent)
            if not np.isclose(
                phase_turns, 0.25, rtol=0.0, atol=ANGLE_ATOL
            ):
                raise ValueError(
                    "unsupported phased-iSWAP basis in FFFT rotation "
                    f"inventory: phase_exponent={phase_turns!r}"
                )
            theta = np.pi * float(gate.exponent) / 2.0
            _add_rotation(inventory, theta, count=2)
            continue

        raise ValueError(
            "unsupported non-Clifford FFFT primitive in rotation inventory: "
            f"{gate!r}"
        )

    return inventory
