"""1D FFFT baseline using OpenFermion on all N = L^2 qubits.

Applies openfermion.circuits.ffft to all N qubits in snake JW order.
Depth: O(N) = O(L^2).  Zero ancillas.
"""

import cirq
from openfermion.circuits import ffft

from common.fp_2d import FPResult
from common.grid import make_system_qubits, snake_to_rc


def build_ffft_1d(L: int) -> FPResult:
    """Build the 1D FFFT baseline circuit.

    Args:
        L: grid side length (N = L^2 modes).

    Returns:
        FPResult with the 1D FFFT circuit on snake-ordered qubits.
    """
    N = L * L
    sq = make_system_qubits(L)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]

    # Qubits in snake JW order
    snake_qubits = [sq[snake_to_rc(i, L)] for i in range(N)]

    # Build the 1D FFFT (decompose multi-qubit gates to 2q primitives)
    raw = cirq.Circuit(ffft(snake_qubits))
    circuit = cirq.Circuit(
        cirq.decompose(raw, keep=lambda op: len(op.qubits) <= 2))

    return FPResult(
        circuit=circuit,
        sys_qubits=sys_list,
        anc_qubits=[],
        gamma_method=None,
        L=L,
    )
