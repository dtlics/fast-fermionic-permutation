"""Verification of FFFT circuits against the expected DFT matrix.

Statevector method (L <= 4): extracts the N x N single-particle
transformation matrix M where M[k, j] = <e_k| Circuit |e_j>,
then compares to the DFT matrix F_N.

For small full-Fock checks, :func:`second_quantized_unitary` lifts a
single-particle unitary to its exact number-conserving action on all ``2**N``
occupation states.  This catches the many-body fermionic signs that are
invisible in the single-particle sector.
"""

from itertools import combinations
from typing import List, Tuple

import numpy as np
import cirq


# ---------------------------------------------------------------------------
# DFT reference
# ---------------------------------------------------------------------------

def dft_matrix(N: int) -> np.ndarray:
    """Standard DFT matrix: F[k,n] = (1/sqrt(N)) exp(-2 pi i kn/N)."""
    return np.fft.fft(np.eye(N), axis=0, norm="ortho")


def second_quantized_unitary(single_particle: np.ndarray) -> np.ndarray:
    """Lift a number-conserving mode unitary to the full occupation basis.

    For occupied input modes ``S`` and output modes ``T``, the many-body
    matrix element is ``det(single_particle[T, S])``.  Qubit/mode zero is the
    most-significant computational-basis bit, matching Cirq's statevector
    convention for an explicit qubit order.

    This helper is exponential and intended only for small verification
    instances (currently ``L=2``, hence four modes).
    """
    single_particle = np.asarray(single_particle, dtype=complex)
    if (
        single_particle.ndim != 2
        or single_particle.shape[0] != single_particle.shape[1]
    ):
        raise ValueError("single_particle must be a square matrix")

    n_modes = single_particle.shape[0]
    dimension = 1 << n_modes
    result = np.zeros((dimension, dimension), dtype=complex)

    for weight in range(n_modes + 1):
        sectors = list(combinations(range(n_modes), weight))
        basis_indices = [
            sum(1 << (n_modes - 1 - mode) for mode in occupied)
            for occupied in sectors
        ]
        for input_modes, input_index in zip(sectors, basis_indices):
            for output_modes, output_index in zip(sectors, basis_indices):
                if weight == 0:
                    amplitude = 1.0
                else:
                    amplitude = np.linalg.det(
                        single_particle[np.ix_(output_modes, input_modes)]
                    )
                result[output_index, input_index] = amplitude

    return result


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
    # The publication audit stores this residual.  Cirq's default complex64
    # statevector leaves an avoidable O(1e-7) roundoff floor at N <= 16.
    sim = cirq.Simulator(dtype=np.complex128)
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
