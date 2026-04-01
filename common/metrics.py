"""Metrics for fermionic permutation circuit evaluation.

Three metrics per the applications spec:
1. Spacetime volume: S = (2-qubit gate depth) x N
2. Counting + union bound fidelity: F_est = (1-p)^G for total 2-qubit gate count G
3. Noisy Clifford simulation (future, Experiment 1 only)
"""

from typing import Dict

import cirq
import numpy as np

from common.fswap import is_fswap, FSWAP_CNOT_COST


def count_resources(circuit: cirq.Circuit, L: int, n_ancillas: int = 0) -> Dict:
    """Count gate depth and CNOT-equivalent resources.

    Gate depth: moments containing any 2-qubit gate.
    CNOT depth: FSWAP moments count as 2 (since FSWAP = 2 CNOTs on same pair),
                CZ/CNOT moments count as 1.

    Returns dict with: L, N, ancillas, gate_depth, cnot_depth,
                        total_2q_gates, total_cnots
    """
    gate_depth = 0
    cnot_depth = 0
    total_2q = 0
    total_cnots = 0

    for moment in circuit:
        has_fswap = False
        has_other_2q = False
        for op in moment:
            if len(op.qubits) >= 2:
                total_2q += 1
                if is_fswap(op.gate):
                    has_fswap = True
                    total_cnots += FSWAP_CNOT_COST
                else:
                    has_other_2q = True
                    total_cnots += 1

        if has_fswap or has_other_2q:
            gate_depth += 1
        if has_fswap:
            cnot_depth += FSWAP_CNOT_COST
        elif has_other_2q:
            cnot_depth += 1

    return {
        "L": L,
        "N": L * L,
        "ancillas": n_ancillas,
        "gate_depth": gate_depth,
        "cnot_depth": cnot_depth,
        "total_2q_gates": total_2q,
        "total_cnots": total_cnots,
    }


def spacetime_volume(two_q_gate_depth: int, N: int) -> int:
    """Spacetime volume: S = (2-qubit gate depth) x N.

    Only counts CNOT/CZ depth layers. Single-qubit gates are free.
    Idle time does not count.
    """
    return two_q_gate_depth * N


def counting_union_bound_fidelity(total_2q_gates: int, p: float) -> float:
    """Counting + union bound fidelity estimate: F_est = (1-p)^G.

    Args:
        total_2q_gates: total number of 2-qubit gates G
        p: per-gate error rate (e.g. 1e-2, 1e-3, 1e-4)

    Returns:
        estimated fidelity
    """
    return (1.0 - p) ** total_2q_gates
