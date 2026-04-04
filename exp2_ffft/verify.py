"""Verification of FFFT circuits against the expected DFT matrix.

Statevector method (L <= 4): extracts the N x N single-particle
transformation matrix M where M[k, j] = <e_k| Circuit |e_j>,
then compares to the DFT matrix F_N.
"""

from typing import List, Tuple

import numpy as np
import cirq


# ---------------------------------------------------------------------------
# DFT reference
# ---------------------------------------------------------------------------

def dft_matrix(N: int) -> np.ndarray:
    """Standard DFT matrix: F[k,n] = (1/sqrt(N)) exp(-2 pi i kn/N)."""
    return np.fft.fft(np.eye(N), axis=0, norm="ortho")


# ---------------------------------------------------------------------------
# Statevector extraction
# ---------------------------------------------------------------------------

def extract_single_particle_matrix_statevector(
    circuit: cirq.Circuit,
    qubit_order: List[cirq.Qid],
) -> np.ndarray:
    """Extract the N x N single-particle matrix via full statevector sim.

    For each input |e_j> (single excitation on qubit j), simulate the full
    circuit and read off amplitudes at all single-excitation outputs |e_k>.

    Args:
        circuit: the Cirq circuit to simulate.
        qubit_order: qubit ordering (determines which qubit is "mode j").

    Returns:
        N x N complex matrix M where M[k, j] = <e_k| circuit |e_j>.
    """
    N = len(qubit_order)
    sim = cirq.Simulator()
    M = np.zeros((N, N), dtype=complex)

    for j in range(N):
        init = np.zeros(2**N, dtype=complex)
        init[1 << (N - 1 - j)] = 1.0
        result = sim.simulate(circuit, initial_state=init,
                              qubit_order=qubit_order)
        state = result.final_state_vector
        for k in range(N):
            M[k, j] = state[1 << (N - 1 - k)]

    return M


# ---------------------------------------------------------------------------
# High-level verification helpers
# ---------------------------------------------------------------------------

def verify_ffft_1d(
    circuit: cirq.Circuit,
    sys_qubits: List[cirq.Qid],
    L: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Verify a 1D FFFT circuit (snake-ordered) against F_N.

    Returns (frobenius_error, M_circuit, F_N).
    """
    from common.grid import snake_to_rc
    N = L * L

    sq_dict = {(q.row, q.col): q for q in sys_qubits}
    snake_qubits = [sq_dict[snake_to_rc(i, L)] for i in range(N)]

    M = extract_single_particle_matrix_statevector(circuit, snake_qubits)
    F = dft_matrix(N)
    return float(np.linalg.norm(M - F)), M, F


def verify_ffft_proper(
    circuit: cirq.Circuit,
    sys_qubits: List[cirq.Qid],
    L: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Verify a proper FFFT circuit (mode k at qubit k) against F_N.

    Returns (frobenius_error, M_circuit, F_N).
    """
    from common.grid import snake_to_rc
    N = L * L

    sq_dict = {(q.row, q.col): q for q in sys_qubits}
    snake_qubits = [sq_dict[snake_to_rc(i, L)] for i in range(N)]

    M = extract_single_particle_matrix_statevector(circuit, snake_qubits)
    F = dft_matrix(N)
    return float(np.linalg.norm(M - F)), M, F
