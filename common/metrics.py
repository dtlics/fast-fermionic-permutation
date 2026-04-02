"""Metrics for fermionic permutation circuit evaluation.

Three metrics (all ignore single-qubit gate layers):

1. **CNOT depth**: Number of CNOT/CZ-equivalent depth layers.
   FSWAP moments contribute 2 layers (FSWAP decomposes into 2 entangling
   layers); CNOT/CZ-only moments contribute 1 layer.

2. **Spacetime volume**: S = total_qubits * cnot_depth

3. **Multiplicative fidelity estimate**: F = (1-p_2q)^G * (1-p_idle)^I
   where G = total 2q gates, I = total idle-qubit-moments
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import pennylane as qp

from common.fswap import FSWAP_CNOT_COST, is_fswap

if TYPE_CHECKING:
    from common.fp_2d import FPResult


# ---------------------------------------------------------------------------
# Core resource counter
# ---------------------------------------------------------------------------

def count_resources(circuit: qp.tape.qscript.QuantumScript, L: int, n_ancillas: int = 0) -> Dict:
    """Count gate resources for a fermionic permutation circuit.

    Scans every moment.  Only moments containing at least one 2-qubit gate
    contribute to depth and idle-slot counts (single-qubit-only moments are
    ignored -- single-qubit gates are "free").

    Returns:
        dict with keys:
            L, N, n_ancillas, total_qubits,
            cnot_depth,     -- CNOT/CZ-equivalent depth (FSWAP moments = 2)
            two_q_depth,    -- moments with >= 1 two-qubit gate
            total_2q_gates, -- total number of 2-qubit gate applications
            total_cnots,    -- CNOT-equivalent count (FSWAP=2, CZ/CNOT=1)
            total_idle_slots, -- sum over 2q moments of idle qubit count
    """
    N = L * L
    total_qubits = N + n_ancillas

    two_q_depth = 0
    cnot_depth = 0
    total_2q_gates = 0
    total_cnots = 0
    total_idle_slots = 0

    for op in circuit:
        n_2q_in_moment = 0
        has_fswap = False
        if len(op.wires) >= 2:
            n_2q_in_moment += 1
            if is_fswap(op):
                total_cnots += FSWAP_CNOT_COST
                has_fswap = True
            else:
                total_cnots += 1

        if n_2q_in_moment > 0:
            two_q_depth += 1
            # FSWAP decomposes into 2 CNOT-depth layers; CNOT/CZ = 1 layer
            cnot_depth += 2 if has_fswap else 1
            total_2q_gates += n_2q_in_moment
            active_qubits = 2 * n_2q_in_moment
            total_idle_slots += max(0, total_qubits - active_qubits)

    return {
        "L": L,
        "N": N,
        "n_ancillas": n_ancillas,
        "total_qubits": total_qubits,
        "cnot_depth": cnot_depth,
        "two_q_depth": two_q_depth,
        "total_2q_gates": total_2q_gates,
        "total_cnots": total_cnots,
        "total_idle_slots": total_idle_slots,
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
    """Fidelity estimate: F = (1-p_2q)^G * (1-p_idle)^I.

    Args:
        total_2q_gates: G -- total two-qubit gate applications
        total_idle_slots: I -- total idle-qubit-moments
        p_2q: error rate per two-qubit gate
        p_idle: error rate per idle qubit per time step

    Returns:
        estimated fidelity in [0, 1]
    """
    return (1.0 - p_2q) ** total_2q_gates * (1.0 - p_idle) ** total_idle_slots


# ---------------------------------------------------------------------------
# Convenience: evaluate an FPResult
# ---------------------------------------------------------------------------

def evaluate_fp(
    result: "FPResult",
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
) -> Dict:
    """Compute all metrics for a fermionic permutation circuit result.

    Returns:
        dict with all count_resources fields plus:
            spacetime_volume, fidelity, gamma_method
    """
    resources = count_resources(
        result.circuit, result.L, len(result.anc_qubits)
    )
    resources["spacetime_volume"] = spacetime_volume(
        resources["total_qubits"], resources["cnot_depth"]
    )
    resources["fidelity"] = multiplicative_fidelity(
        resources["total_2q_gates"],
        resources["total_idle_slots"],
        p_2q=p_2q,
        p_idle=p_idle,
    )
    resources["gamma_method"] = result.gamma_method
    return resources
