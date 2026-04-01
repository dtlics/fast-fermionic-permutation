"""Baseline 2: Gamma operator with ancillas (Jiang et al.).

CNOT depth: 7L - 3
Ancillas: L (one per row)
Gate count: O(N)

The ancilla-based construction uses one ancilla qubit per row to accumulate
column parities, enabling CZ interactions without explicit column-parity
cascade sweeps on the system qubits.
"""

from typing import Dict, List, Tuple

import cirq

from common.grid import (
    column_parity_cascade_ops,
    make_ancilla_qubits,
    make_system_qubits,
)


def build_stage_B_ops(
    sq: Dict[Tuple[int, int], cirq.GridQubit],
    aq: Dict[int, cirq.NamedQubit],
    L: int,
) -> List[cirq.Operation]:
    """Stage B: parity-basis CZ sweep (R->L). Depth 3L."""
    ops = []
    for p in range(L - 1, -1, -1):
        for r in range(0, L, 2):
            if r + 2 <= L - 1:
                ops.append(cirq.CZ(sq[(r, p)], aq[r + 2]))
            if r >= 2:
                ops.append(cirq.CZ(sq[(r, p)], aq[r]))
        for r in range(L):
            ops.append(cirq.CNOT(sq[(r, p)], aq[r]))
    return ops


def ancilla_column_cascade_ops(
    aq: Dict[int, cirq.NamedQubit], L: int, inverse: bool = False
) -> List[cirq.Operation]:
    """CNOT cascade on ancilla column. Depth L-1."""
    if not inverse:
        return [cirq.CNOT(aq[r + 1], aq[r]) for r in range(L - 2, -1, -1)]
    else:
        return [cirq.CNOT(aq[r + 1], aq[r]) for r in range(L - 1)]


def build_stage_D_ops(
    sq: Dict[Tuple[int, int], cirq.GridQubit],
    aq: Dict[int, cirq.NamedQubit],
    L: int,
) -> List[cirq.Operation]:
    """Stage D: original-basis CZ sweep (L->R). Depth 2L+1."""
    ops = []
    for p in range(L):
        for r in range(L):
            ops.append(cirq.CNOT(sq[(r, p)], aq[r]))
        for r in range(0, L, 2):
            if r + 1 < L:
                ops.append(cirq.CZ(sq[(r, p)], aq[r]))
                ops.append(cirq.CZ(sq[(r, p)], aq[r + 1]))
            else:
                ops.append(cirq.CZ(sq[(r, p)], aq[r]))
    return ops


def build_gamma_with_ancillas(L: int):
    """Build Gamma with ancillas (Baseline 2).

    Returns:
        (circuit, sys_list, anc_list) where circuit has CNOT depth 7L-3.
    """
    sq = make_system_qubits(L)
    aq = make_ancilla_qubits(L)
    ops = []
    ops.extend(column_parity_cascade_ops(sq, L, inverse=False))   # Stage A
    ops.extend(build_stage_B_ops(sq, aq, L))                      # Stage B
    ops.extend(column_parity_cascade_ops(sq, L, inverse=True))    # Stage C (sys)
    ops.extend(ancilla_column_cascade_ops(aq, L, inverse=True))   # Stage C (anc)
    ops.extend(build_stage_D_ops(sq, aq, L))                      # Stage D
    circuit = cirq.Circuit(ops)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]
    anc_list = [aq[r] for r in range(L)]
    return circuit, sys_list, anc_list
