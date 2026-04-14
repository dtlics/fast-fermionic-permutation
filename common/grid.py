"""Grid topology, snake JW indexing, and column parity cascade.

Provides the L x L qubit grid with snake (Jordan-Wigner) ordering,
coordinate conversions, and the column parity cascade used by all
three Gamma constructions.
"""

from typing import Dict, List, Sequence, Tuple

import pennylane as qp
import numpy as np


# ---------------------------------------------------------------------------
# Snake (Jordan-Wigner) ordering
# ---------------------------------------------------------------------------

def snake_order_indices(L: int) -> List[int]:
    """Return raster indices in snake order."""
    order = []
    for r in range(L):
        row = [r * L + c for c in range(L)]
        if r % 2 == 1:
            row.reverse()
        order.extend(row)
    return order


def rc_to_snake(r: int, c: int, L: int) -> int:
    """Convert grid (row, col) to snake-order index."""
    if r % 2 == 0:
        return r * L + c
    else:
        return r * L + (L - 1 - c)


def snake_to_rc(idx: int, L: int) -> Tuple[int, int]:
    """Convert snake-order index to grid (row, col)."""
    r = idx // L
    pos_in_row = idx % L
    if r % 2 == 0:
        c = pos_in_row
    else:
        c = L - 1 - pos_in_row
    return r, c


def is_L_row(r: int) -> bool:
    """Check if row r is even (left-to-right in snake order)."""
    return r % 2 == 0


def sites_between(r1: int, c: int, r2: int, L: int) -> List[int]:
    """Snake-order indices strictly between (r1,c) and (r2,c) for vertical hop."""
    j = rc_to_snake(r1, c, L)
    k = rc_to_snake(r2, c, L)
    lo, hi = min(j, k), max(j, k)
    return list(range(lo + 1, hi))


# ---------------------------------------------------------------------------
# Qubit creation
# ---------------------------------------------------------------------------

def make_system_qubits(L: int) -> Dict[Tuple[int, int], qp.wires.Wires]:
    """Create L x L grid of system qubits."""
    return {(r, c): qp.wires.Wires([f"({r}, {c})"]) for r in range(L) for c in range(L)}


def make_ancilla_qubits(L: int) -> Dict[int, qp.wires.Wires]:
    """Create L ancilla qubits at column L (one per row), used by Baseline 2.

    Ancillas are placed one column to the right of the data grid,
    physically adjacent to the rightmost data column (L-1).
    """
    return {r: qp.wires.Wires([f"({r}, {L})"]) for r in range(L)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_permutation(perm: Sequence[int], n: int) -> None:
    """Validate that perm is a permutation of 0..n-1."""
    if len(perm) != n or set(perm) != set(range(n)):
        raise ValueError(
            f"Expected permutation of 0..{n-1}, got length {len(perm)}"
        )


# ---------------------------------------------------------------------------
# Raster <-> snake conversion
# ---------------------------------------------------------------------------

def raster_to_snake_index_map(L: int) -> Dict[int, int]:
    """Map raster index -> snake-order position."""
    snake = snake_order_indices(L)
    return {r_idx: s_pos for s_pos, r_idx in enumerate(snake)}


def raster_perm_to_snake(L: int, perm_raster: Sequence[int]) -> List[int]:
    """Convert a permutation in raster coordinates to snake coordinates."""
    validate_permutation(list(perm_raster), L * L)
    r2s = raster_to_snake_index_map(L)
    perm_snake = [0] * (L * L)
    for src_r, dst_r in enumerate(perm_raster):
        perm_snake[r2s[src_r]] = r2s[dst_r]
    return perm_snake


# ---------------------------------------------------------------------------
# Column parity cascade (shared by all three Gamma constructions)
# ---------------------------------------------------------------------------

def column_parity_cascade_ops(
    sq: Dict[Tuple[int, int], qp.wires.Wires],
    L: int,
    inverse: bool = False,
) -> List[qp.ops.Operation]:
    """Column parity CNOT cascade.

    Forward: bottom-to-top, CNOT(r+1,c -> r,c) for r from L-2 down to 0.
    After forward cascade, qubit (r,c) holds XOR of s_{r,c}, s_{r+1,c}, ..., s_{L-1,c}.

    Inverse: top-to-bottom, undoes the forward cascade.
    """
    ops = []
    if not inverse:
        for r in range(L - 2, -1, -1):
            for c in range(L):
                ops.append(qp.CNOT(sq[(r + 1, c)] + sq[(r, c)]))
    else:
        for r in range(L - 1):
            for c in range(L):
                ops.append(qp.CNOT(sq[(r + 1, c)] + sq[(r, c)]))
    return ops
