"""Twiddle factor circuit for Cooley-Tukey FFFT decomposition.

After column DFTs, grid position (r, c) holds the column-DFT output with
index k1 = r.  The twiddle factor is:

    omega_N^{c * r} = exp(-2 pi i * c * r / N)

Applied as ZPowGate(exponent = -2*c*r / N) where N = L^2.
All gates are single-qubit Z rotations -> depth O(1).
"""

from typing import Dict, List, Tuple

import cirq


def build_twiddle_circuit(
    L: int,
    sq: Dict[Tuple[int, int], cirq.GridQubit],
) -> cirq.Circuit:
    """Build the twiddle factor circuit.

    Args:
        L: grid side length (N = L^2 modes).
        sq: system qubit dict {(r, c): GridQubit}.

    Returns:
        cirq.Circuit with ZPowGate at each position where c*r != 0.
    """
    N = L * L
    ops: List[cirq.Operation] = []
    for r in range(L):
        for c in range(L):
            if c * r == 0:
                continue
            exponent = -2.0 * c * r / N
            ops.append(cirq.ZPowGate(exponent=exponent).on(sq[(r, c)]))
    return cirq.Circuit(ops)
