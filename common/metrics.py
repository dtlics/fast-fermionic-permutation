"""Metrics for fermionic permutation circuit evaluation.

Two metrics (both ignore single-qubit gate layers):

1. **Spacetime volume**: S = total_qubits * two_q_gate_depth
   - total_qubits = data + ancilla (all qubits in the circuit)
   - two_q_gate_depth = number of moments with at least one 2-qubit gate

2. **Counting + union bound fidelity**:
   - At each 2-qubit-gate moment, count both 2-qubit gates AND idle qubits
   - idle = total_qubits - 2 * (number of 2q ops in that moment)
   - F_est = (1 - p_2q)^G * (1 - p_idle)^I
   - where G = total 2q gates, I = total idle-qubit-moments
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import cirq
import numpy as np

from common.fswap import FSWAP_CNOT_COST, is_fswap

if TYPE_CHECKING:
    from common.fp_2d import FPResult


# ---------------------------------------------------------------------------
# Core resource counter
# ---------------------------------------------------------------------------

def count_resources(circuit: cirq.Circuit, L: int, n_ancillas: int = 0) -> Dict:
    """Count gate resources for a fermionic permutation circuit.

    Scans every moment.  Only moments containing at least one 2-qubit gate
    contribute to depth and idle-slot counts (single-qubit-only moments are
    ignored -- single-qubit gates are "free").

    Args:
        circuit: the cirq.Circuit to analyse
        L: grid side length (N = L^2 data qubits)
        n_ancillas: number of ancilla qubits (0 for ancilla-free methods)

    Returns:
        dict with keys:
            L, N, n_ancillas, total_qubits,
            two_q_depth,    -- moments with >= 1 two-qubit gate
            total_2q_gates, -- total number of 2-qubit gate applications
            total_cnots,    -- CNOT-equivalent count (FSWAP=2, CZ/CNOT=1)
            total_idle_slots, -- sum over 2q moments of idle qubit count
    """
    N = L * L
    total_qubits = N + n_ancillas

    two_q_depth = 0
    total_2q_gates = 0
    total_cnots = 0
    total_idle_slots = 0

    for moment in circuit:
        n_2q_in_moment = 0
        for op in moment:
            if len(op.qubits) >= 2:
                n_2q_in_moment += 1
                if is_fswap(op.gate):
                    total_cnots += FSWAP_CNOT_COST
                else:
                    total_cnots += 1

        if n_2q_in_moment > 0:
            two_q_depth += 1
            total_2q_gates += n_2q_in_moment
            # Each 2q gate uses 2 qubits; the rest are idle
            active_qubits = 2 * n_2q_in_moment
            total_idle_slots += max(0, total_qubits - active_qubits)

    return {
        "L": L,
        "N": N,
        "n_ancillas": n_ancillas,
        "total_qubits": total_qubits,
        "two_q_depth": two_q_depth,
        "total_2q_gates": total_2q_gates,
        "total_cnots": total_cnots,
        "total_idle_slots": total_idle_slots,
    }


# ---------------------------------------------------------------------------
# Metric 1: Spacetime volume
# ---------------------------------------------------------------------------

def spacetime_volume(total_qubits: int, two_q_depth: int) -> int:
    """Spacetime volume: S = total_qubits * two_q_gate_depth.

    Args:
        total_qubits: data + ancilla qubit count
        two_q_depth: number of moments containing >= 1 two-qubit gate
    """
    return total_qubits * two_q_depth


# ---------------------------------------------------------------------------
# Metric 2: Counting + union bound fidelity
# ---------------------------------------------------------------------------

def counting_union_bound_fidelity(
    total_2q_gates: int,
    total_idle_slots: int,
    p_2q: float = 1e-3,
    p_idle: float = 1e-5,
) -> float:
    """Fidelity estimate via counting + union bound.

    F_est = (1 - p_2q)^G * (1 - p_idle)^I

    Args:
        total_2q_gates: G -- total two-qubit gate applications
        total_idle_slots: I -- total idle-qubit-moments
        p_2q: error rate per two-qubit gate
        p_idle: error rate per idle qubit per time step

    Returns:
        estimated circuit fidelity (float in [0, 1])
    """
    return (1.0 - p_2q) ** total_2q_gates * (1.0 - p_idle) ** total_idle_slots


# ---------------------------------------------------------------------------
# Convenience: evaluate an FPResult
# ---------------------------------------------------------------------------

def evaluate_fp(
    result: "FPResult",
    p_2q: float = 1e-3,
    p_idle: float = 1e-5,
) -> Dict:
    """Compute all metrics for a fermionic permutation circuit result.

    Args:
        result: FPResult from build_fp_1d or build_fp_2d
        p_2q: error rate per two-qubit gate
        p_idle: error rate per idle qubit per time step

    Returns:
        dict with all count_resources fields plus:
            spacetime_volume, fidelity, gamma_method
    """
    resources = count_resources(
        result.circuit, result.L, len(result.anc_qubits)
    )
    resources["spacetime_volume"] = spacetime_volume(
        resources["total_qubits"], resources["two_q_depth"]
    )
    resources["fidelity"] = counting_union_bound_fidelity(
        resources["total_2q_gates"],
        resources["total_idle_slots"],
        p_2q=p_2q,
        p_idle=p_idle,
    )
    resources["gamma_method"] = result.gamma_method
    return resources
