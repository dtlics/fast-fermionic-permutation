"""Bare column FFTs for the 2D FFFT circuit.

Generates the FFFT gate sequence for each column (top-to-bottom), applied
as bare local operations.  The enclosing Gamma sandwich provides JW parity
correction.  All L columns execute in parallel.
"""

from typing import Dict, Tuple

import cirq
from openfermion.circuits import ffft


def build_bare_column_fffts(
    L: int,
    sq: Dict[Tuple[int, int], cirq.GridQubit],
) -> cirq.Circuit:
    """Build bare column FFFT circuits for all L columns.

    Each column c uses qubits [sq[(0,c)], sq[(1,c)], ..., sq[(L-1,c)]]
    in top-to-bottom order.  The openfermion ffft gate sequence is applied
    directly (bare) — the Gamma sandwich handles parity.

    Args:
        L: grid side length.
        sq: system qubit dict {(r, c): GridQubit}.

    Returns:
        cirq.Circuit with all L columns' FFT ops merged (Cirq schedules
        independent columns in parallel).
    """
    all_ops = []
    for c in range(L):
        col_qubits = [sq[(r, c)] for r in range(L)]
        col_optree = ffft(col_qubits)
        all_ops.append(col_optree)
    raw = cirq.Circuit(all_ops)
    # Decompose multi-qubit gates (>2q) to 2q primitives.
    # Keep 2q openfermion gates (F0, FSWAP) intact — full decompose breaks them.
    return cirq.Circuit(
        cirq.decompose(raw, keep=lambda op: len(op.qubits) <= 2))


def build_row_fffts(
    L: int,
    sq: Dict[Tuple[int, int], cirq.GridQubit],
) -> cirq.Circuit:
    """Build row FFFT circuits for all L rows (JW-local, no Gamma needed).

    Row qubits are passed to ffft in raster order (left-to-right for all
    rows).  This is consistent with the Cooley-Tukey factorization in
    raster coordinates.  All gates are between horizontally adjacent
    qubits (which are JW-adjacent regardless of row parity).

    Args:
        L: grid side length.
        sq: system qubit dict {(r, c): GridQubit}.

    Returns:
        cirq.Circuit with all L rows' FFT ops merged in parallel.
    """
    all_ops = []
    for r in range(L):
        row_qubits = [sq[(r, c)] for c in range(L)]
        row_optree = ffft(row_qubits)
        all_ops.append(row_optree)
    raw = cirq.Circuit(all_ops)
    return cirq.Circuit(
        cirq.decompose(raw, keep=lambda op: len(op.qubits) <= 2))
