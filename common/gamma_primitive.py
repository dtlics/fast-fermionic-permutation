"""Baseline 3: Ancilla-free Gamma using sequential T(x,y) primitives.

CNOT depth: 9L + 12 (exact for L >= 5)
Ancillas: 0
Gate count: O(N)

Implements Gamma = D_D * C^{-1} * D_B * C using three circuit primitives:
    Primitive 1: same-row T(x,x) -- depth 2L
    Primitive 2: cross-row adjacent T(x,y) -- depth 2L
    Primitive 3: skip-row T(x,y) via middle row -- depth 2L+4

The four phases are:
    Phase 1: Column parity cascade forward (L-1)
    Phase 2: Parity-basis interactions f_B (same-row + skip-row on even rows)
    Phase 3: Column parity cascade inverse (L-1)
    Phase 4: Original-basis interactions f_D (same-row + cross-row)
"""

from typing import Dict, List, Tuple

import cirq

from common.grid import column_parity_cascade_ops, make_system_qubits


# ---------------------------------------------------------------------------
# Cascade primitives (also used by gamma_pipeline.py)
# ---------------------------------------------------------------------------

def suffix_cascade_ops(sq: Dict, r: int, L: int) -> List[cirq.Operation]:
    """Suffix CNOT cascade: right-to-left on row r."""
    return [cirq.CNOT(sq[(r, c + 1)], sq[(r, c)]) for c in range(L - 2, -1, -1)]


def undo_suffix_cascade_ops(sq: Dict, r: int, L: int) -> List[cirq.Operation]:
    """Undo suffix cascade: left-to-right on row r."""
    return [cirq.CNOT(sq[(r, c + 1)], sq[(r, c)]) for c in range(L - 1)]


def prefix_cascade_ops(sq: Dict, r: int, L: int) -> List[cirq.Operation]:
    """Prefix CNOT cascade: left-to-right on row r."""
    return [cirq.CNOT(sq[(r, c - 1)], sq[(r, c)]) for c in range(1, L)]


def undo_prefix_cascade_ops(sq: Dict, r: int, L: int) -> List[cirq.Operation]:
    """Undo prefix cascade: right-to-left on row r."""
    return [cirq.CNOT(sq[(r, c - 1)], sq[(r, c)]) for c in range(L - 1, 0, -1)]


# ---------------------------------------------------------------------------
# Three T(x,y) primitives
# ---------------------------------------------------------------------------

def same_row_T_ops(sq: Dict, r: int, L: int) -> List[cirq.Operation]:
    """Primitive 1: T(x,x) for row r using suffix cascade. Depth 2L."""
    ops = []
    ops.extend(suffix_cascade_ops(sq, r, L))
    for c in range(L - 1):
        ops.append(cirq.CZ(sq[(r, c)], sq[(r, c + 1)]))
    ops.extend(undo_suffix_cascade_ops(sq, r, L))
    for c in range(1, L, 2):
        ops.append(cirq.Z(sq[(r, c)]))
    return ops


def cross_row_adjacent_T_ops(sq: Dict, r1: int, r2: int, L: int) -> List[cirq.Operation]:
    """Primitive 2: T(x,y) for adjacent rows r1, r2. Depth 2L."""
    ops = []
    ops.extend(prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r1, c)], sq[(r2, c)]))
    ops.extend(undo_prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r1, c)], sq[(r2, c)]))
    return ops


def skip_row_T_ops(sq: Dict, r1: int, r2: int, r_mid: int, L: int) -> List[cirq.Operation]:
    """Primitive 3: T(x,y) for rows 2 apart, routing through r_mid. Depth 2L+4."""
    ops = []
    ops.extend(prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
    ops.extend(undo_prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
    return ops


# ---------------------------------------------------------------------------
# Full Gamma circuit (Baseline 3)
# ---------------------------------------------------------------------------

def build_gamma_ancilla_free(L: int):
    """Build ancilla-free Gamma using sequential primitives (Baseline 3).

    Returns:
        (circuit, sys_list) where circuit has CNOT depth 9L+12 (exact for L >= 5).
    """
    sq = make_system_qubits(L)
    ops = []

    # Phase 1: column parity cascade forward
    ops.extend(column_parity_cascade_ops(sq, L, inverse=False))

    # Phase 2a: f_B same-row terms for even rows r >= 2
    for r in range(2, L, 2):
        ops.extend(same_row_T_ops(sq, r, L))

    # Phase 2b: f_B skip-row terms (2-round scheduling)
    skip_rows = [r for r in range(0, L, 2) if r + 2 <= L - 1]
    for r in skip_rows[0::2]:
        ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))
    for r in skip_rows[1::2]:
        ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))

    # Phase 3: column parity cascade inverse
    ops.extend(column_parity_cascade_ops(sq, L, inverse=True))

    # Phase 4a: f_D same-row terms for all even rows
    for r in range(0, L, 2):
        ops.extend(same_row_T_ops(sq, r, L))

    # Phase 4b: f_D cross-row terms for right-closed pairs
    for r in range(0, L - 1, 2):
        ops.extend(cross_row_adjacent_T_ops(sq, r, r + 1, L))

    circuit = cirq.Circuit(ops)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]
    return circuit, sys_list
