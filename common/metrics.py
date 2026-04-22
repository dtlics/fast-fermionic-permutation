"""Metrics for fermionic permutation circuit evaluation.

Flow: decompose composite 2-qubit gates (FSWAP etc.) -> optimize with passes -> run ``qp.specs`` on a Clifford device to read out depth and 2-qubit gate count.

Single-qubit gates are treated as free.

Metrics:

1. **CNOT depth**: ``resources.depth`` after decompose + optimize.
2. **Spacetime volume**: ``total_qubits * cnot_depth`` (includes ancillas).
3. **Idle qubit-moments**: ``spacetime - 2 * total_2q_gates``.
4. **Multiplicative fidelity**: ``(1-p_2q)^G * (1-p_idle)^I``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import pennylane as qp

if TYPE_CHECKING:
    from common.fp_2d import FPResult


def tape_to_qfunc(tape):
    for op in tape.operations:
        qp.apply(op)
    for meas in tape.measurements:
        qp.apply(meas)


def get_resources(circ):
    @qp.qnode(qp.device("default.clifford", tableau=False))
    def qfunc():
        tape_to_qfunc(circ)

    return qp.specs(qfunc)().resources


def decompose(circ):
    [circ], _ = qp.transforms.decompose(circ)
    return circ


def optimize(circ):
    [circ], _ = qp.transforms.commute_controlled(circ)
    [circ], _ = qp.transforms.merge_rotations(circ)
    [circ], _ = qp.transforms.cancel_inverses(circ, recursive=True)
    return circ


# ---------------------------------------------------------------------------
# Core resource counter: decompose -> optimize -> specs
# ---------------------------------------------------------------------------

def count_resources(circuit: qp.tape.qscript.QuantumScript, L: int, n_ancillas: int = 0) -> Dict:
    N = L * L
    total_qubits = N + n_ancillas

    circuit = decompose(circuit)
    circuit = optimize(circuit)
    resources = get_resources(circuit)

    depth = resources.depth
    total_2q_gates = resources.gate_sizes.get(2, 0)
    spacetime = total_qubits * depth

    return {
        "L": L,
        "N": N,
        "n_ancillas": n_ancillas,
        "total_qubits": total_qubits,
        "cnot_depth": depth,
        "total_2q_gates": total_2q_gates,
        "spacetime_volume": spacetime,
        "total_idle_slots": spacetime - 2 * total_2q_gates,
    }


# ---------------------------------------------------------------------------
# Metric 1: Spacetime volume
# ---------------------------------------------------------------------------

def spacetime_volume(total_qubits: int, cnot_depth: int) -> int:
    """Spacetime volume: S = total_qubits * cnot_depth."""
    return total_qubits * cnot_depth


# ---------------------------------------------------------------------------
# Metric 2: Multiplicative fidelity estimate
# ---------------------------------------------------------------------------

def multiplicative_fidelity(
    total_2q_gates: int,
    total_idle_slots: int,
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
) -> float:
    """Fidelity estimate: F = (1-p_2q)^G * (1-p_idle)^I."""
    return (1.0 - p_2q) ** total_2q_gates * (1.0 - p_idle) ** total_idle_slots


# ---------------------------------------------------------------------------
# Convenience: evaluate an FPResult
# ---------------------------------------------------------------------------

def evaluate_fp(
    result: "FPResult",
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
) -> Dict:
    """Compute all metrics for a fermionic permutation circuit result."""
    resources = count_resources(
        result.circuit, result.L, len(result.anc_qubits)
    )
    resources["fidelity"] = multiplicative_fidelity(
        resources["total_2q_gates"],
        resources["total_idle_slots"],
        p_2q=p_2q,
        p_idle=p_idle,
    )
    resources["gamma_method"] = result.gamma_method
    return resources
