"""Baseline 3: Ancilla-free Gamma using sequential T(x,y) primitives.

Ancillas: 0
Gate count: O(N)

Implements Gamma = D_D * C^{-1} * D_B * C using three circuit primitives:
    Primitive 1: same-row T(x,x) -- depth 2L
    Primitive 2: cross-row adjacent T(x,y) -- depth 2L
    Primitive 3: skip-row T(x,y) via middle row -- depth 2L+4

Depth scaling
-------------
Phase-separated construction (this implementation, phases joined with +):
    Phase 1  (col parity fwd):     L - 1
    Phase 2a (f_B same-row T):     2L + 1
    Phase 2b (f_B skip-row, x2):   2 * (2L + 4)
    Phase 3  (col parity inv):     L - 1
    Phase 4a (f_D same-row T):     2L + 1
    Phase 4b (f_D cross-row T):    2L
    Total:   12L + 8   (exact for L >= 5)

With cirq greedy pipelining (all ops in one cirq.Circuit, NOT used here):
    Total:   9L + 12   (odd L) or 9L + 13 (even L), exact for L >= 5
    Main savings:
      P2a + P2b1: same-row T on even rows overlaps with skip-row T
        batch 1 on disjoint row triples.  ~6 moments saved.
      P4a + P4b:  same-row T on even rows overlaps with cross-row T
        on disjoint even-odd pairs.  ~6 moments saved.
    NOTE: greedy pipelining is intentionally NOT applied to this baseline.
    Baseline 4 (gamma_pipeline.py) achieves 8L+9/10 via manual pipelining,
    which is strictly better than the ~9L greedy result here.
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

def build_gamma_ancilla_free(L: int, sq=None):
    """Build ancilla-free Gamma using sequential primitives (Baseline 3).

    Each phase is built as a separate cirq.Circuit and concatenated with ``+``
    to prevent cross-phase moment merging.  Within each phase, operations on
    disjoint qubits (e.g. same-row T on different rows) are still parallelised
    by cirq's greedy scheduler.

    This gives depth **12L + 8** (exact for L >= 5), matching the theoretical
    sequential-phase analysis:

        Phase 1  (col parity fwd):    L - 1
        Phase 2a (f_B same-row T):    2L + 1
        Phase 2b (f_B skip-row, x2):  2 * (2L + 4)
        Phase 3  (col parity inv):    L - 1
        Phase 4a (f_D same-row T):    2L + 1
        Phase 4b (f_D cross-row T):   2L

    Args:
        L: grid side length
        sq: optional system qubit dict {(r,c): Qid}.  Created internally if None.

    Returns:
        (circuit, sys_list)
    """
    if sq is None:
        sq = make_system_qubits(L)

    # Phase 1: column parity cascade forward
    phase1_ops = list(column_parity_cascade_ops(sq, L, inverse=False))

    # Phase 2a: f_B same-row terms for even rows r >= 2
    phase2a_ops = []
    for r in range(2, L, 2):
        phase2a_ops.extend(same_row_T_ops(sq, r, L))

    # Phase 2b: f_B skip-row terms (2 batches to avoid row overlap)
    skip_rows = [r for r in range(0, L, 2) if r + 2 <= L - 1]
    phase2b1_ops = []
    for r in skip_rows[0::2]:
        phase2b1_ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))
    phase2b2_ops = []
    for r in skip_rows[1::2]:
        phase2b2_ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))

    # Phase 3: column parity cascade inverse
    phase3_ops = list(column_parity_cascade_ops(sq, L, inverse=True))

    # Phase 4a: f_D same-row terms for all even rows
    phase4a_ops = []
    for r in range(0, L, 2):
        phase4a_ops.extend(same_row_T_ops(sq, r, L))

    # Phase 4b: f_D cross-row terms for right-closed pairs
    phase4b_ops = []
    for r in range(0, L - 1, 2):
        phase4b_ops.extend(cross_row_adjacent_T_ops(sq, r, r + 1, L))

    # Concatenate phases -- '+' preserves moment boundaries across phases
    circuit = (
        cirq.Circuit(phase1_ops)
        + cirq.Circuit(phase2a_ops)
        + cirq.Circuit(phase2b1_ops)
        + cirq.Circuit(phase2b2_ops)
        + cirq.Circuit(phase3_ops)
        + cirq.Circuit(phase4a_ops)
        + cirq.Circuit(phase4b_ops)
    )
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]
    return circuit, sys_list
