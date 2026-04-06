"""Tests for the proper 2D FFFT (mode k stays at qubit k)."""

import numpy as np
import pytest
import cirq

from common.fp_2d import GammaMethod
from common.grid import make_system_qubits, snake_to_rc
from exp2_ffft.ffft_proper import (
    build_ffft_proper,
    build_ffft_fp_sandwich_2d,
    build_ffft_fp_sandwich_1d,
)
from exp2_ffft.ffft_baseline_1d import build_ffft_1d
from exp2_ffft.verify import (
    extract_single_particle_matrix_statevector,
    dft_matrix,
)


def _snake_qubits(L):
    sq = make_system_qubits(L)
    return [sq[snake_to_rc(i, L)] for i in range(L * L)]


class TestProperFFFTCorrectness:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_dft_matrix(self, L):
        """Proper 2D FFFT single-particle matrix equals F_N in mode basis."""
        result = build_ffft_proper(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M = extract_single_particle_matrix_statevector(result.circuit, snake_qs)
        F = dft_matrix(L * L)
        error = np.linalg.norm(M - F)
        assert error < 1e-5, f"L={L}: ||M - F_N|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_1d_baseline(self, L):
        """Proper 2D FFFT matches 1D OpenFermion baseline in mode basis."""
        r2d = build_ffft_proper(L, GammaMethod.PIPELINED)
        r1d = build_ffft_1d(L)
        snake_qs = _snake_qubits(L)
        M_2d = extract_single_particle_matrix_statevector(r2d.circuit, snake_qs)
        M_1d = extract_single_particle_matrix_statevector(r1d.circuit, snake_qs)
        error = np.linalg.norm(M_2d - M_1d)
        assert error < 1e-5, f"L={L}: ||M_2d - M_1d|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_output_is_unitary(self, L):
        """Single-particle matrix is unitary."""
        result = build_ffft_proper(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M = extract_single_particle_matrix_statevector(result.circuit, snake_qs)
        err = np.linalg.norm(M @ M.conj().T - np.eye(L * L))
        assert err < 1e-5, f"L={L}: unitarity error {err:.2e}"


class TestProperFFFTStructure:
    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_nn_compliance(self, L):
        """All 2-qubit gates are between NN after decomposition."""
        result = build_ffft_proper(L, GammaMethod.PIPELINED)
        decomposed = cirq.Circuit(cirq.decompose(
            result.circuit, keep=lambda op: len(op.qubits) <= 2))
        for moment in decomposed:
            for op in moment:
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    dist = abs(q1.row - q2.row) + abs(q1.col - q2.col)
                    assert dist == 1, f"Non-NN gate between {q1} and {q2}"

    @pytest.mark.parametrize("L", [3, 4, 5])
    def test_zero_ancillas(self, L):
        """Pipelined Gamma uses zero ancillas."""
        result = build_ffft_proper(L, GammaMethod.PIPELINED)
        assert len(result.anc_qubits) == 0

    @pytest.mark.parametrize("L", [4, 6, 8])
    def test_depth_is_O_L(self, L):
        """Depth should be O(L), with constant ~50."""
        from common.metrics import count_resources
        result = build_ffft_proper(L, GammaMethod.PIPELINED)
        res = count_resources(result.circuit, L, 0)
        depth = res["cnot_depth"]
        # ~46L expected (2*8L Gamma_DFT + ~22L FP_transpose + ~6L rest)
        assert depth < 60 * L, f"L={L}: depth {depth} >= 60L={60*L}"


# ---------------------------------------------------------------------------
# FP-sandwich 2D (transpose + 2D FP)
# ---------------------------------------------------------------------------

class TestFPSandwich2DCorrectness:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_dft_matrix(self, L):
        """FP-sandwich 2D single-particle matrix equals F_N."""
        result = build_ffft_fp_sandwich_2d(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M = extract_single_particle_matrix_statevector(result.circuit, snake_qs)
        F = dft_matrix(L * L)
        error = np.linalg.norm(M - F)
        assert error < 1e-5, f"L={L}: ||M - F_N|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_proper(self, L):
        """FP-sandwich 2D matches Gamma-sandwich proper FFFT."""
        r_fp = build_ffft_fp_sandwich_2d(L, GammaMethod.PIPELINED)
        r_g = build_ffft_proper(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M_fp = extract_single_particle_matrix_statevector(r_fp.circuit, snake_qs)
        M_g = extract_single_particle_matrix_statevector(r_g.circuit, snake_qs)
        error = np.linalg.norm(M_fp - M_g)
        assert error < 1e-5, f"L={L}: ||M_fp - M_gamma|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_output_is_unitary(self, L):
        """Single-particle matrix is unitary."""
        result = build_ffft_fp_sandwich_2d(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M = extract_single_particle_matrix_statevector(result.circuit, snake_qs)
        err = np.linalg.norm(M @ M.conj().T - np.eye(L * L))
        assert err < 1e-5, f"L={L}: unitarity error {err:.2e}"


class TestFPSandwich2DStructure:
    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_nn_compliance(self, L):
        """All 2-qubit gates are between NN after decomposition."""
        result = build_ffft_fp_sandwich_2d(L, GammaMethod.PIPELINED)
        decomposed = cirq.Circuit(cirq.decompose(
            result.circuit, keep=lambda op: len(op.qubits) <= 2))
        for moment in decomposed:
            for op in moment:
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    dist = abs(q1.row - q2.row) + abs(q1.col - q2.col)
                    assert dist == 1, f"Non-NN gate between {q1} and {q2}"

    @pytest.mark.parametrize("L", [3, 4, 5])
    def test_zero_ancillas(self, L):
        """Pipelined Gamma uses zero ancillas."""
        result = build_ffft_fp_sandwich_2d(L, GammaMethod.PIPELINED)
        assert len(result.anc_qubits) == 0


# ---------------------------------------------------------------------------
# FP-sandwich 2D with ancillas
# ---------------------------------------------------------------------------

def _extract_with_ancillas(circuit, snake_qs, anc_qs):
    """Extract single-particle matrix when circuit has ancilla qubits.

    Ancillas are initialised to |0> and traced out after simulation.
    """
    N = len(snake_qs)
    n_anc = len(anc_qs)
    total = N + n_anc
    # qubit order: system (snake) then ancillas
    qubit_order = list(snake_qs) + list(anc_qs)
    sim = cirq.Simulator()
    M = np.zeros((N, N), dtype=complex)

    for j in range(N):
        init = np.zeros(2**total, dtype=complex)
        # excite system qubit j; ancillas stay |0>
        init[1 << (total - 1 - j)] = 1.0
        res = sim.simulate(circuit, initial_state=init,
                           qubit_order=qubit_order)
        state = res.final_state_vector
        for k in range(N):
            # ancillas should be |0>, so only read system-qubit excitations
            M[k, j] = state[1 << (total - 1 - k)]
    return M


class TestFPSandwich2DAncillaCorrectness:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_dft_matrix(self, L):
        """FP-sandwich 2D (ancilla) single-particle matrix equals F_N."""
        result = build_ffft_fp_sandwich_2d(L, GammaMethod.ANCILLA)
        snake_qs = _snake_qubits(L)
        M = _extract_with_ancillas(result.circuit, snake_qs, result.anc_qubits)
        F = dft_matrix(L * L)
        error = np.linalg.norm(M - F)
        assert error < 1e-5, f"L={L}: ||M - F_N|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_proper(self, L):
        """FP-sandwich 2D (ancilla) matches Gamma-sandwich proper FFFT."""
        r_fp = build_ffft_fp_sandwich_2d(L, GammaMethod.ANCILLA)
        r_g = build_ffft_proper(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M_fp = _extract_with_ancillas(r_fp.circuit, snake_qs, r_fp.anc_qubits)
        M_g = extract_single_particle_matrix_statevector(r_g.circuit, snake_qs)
        error = np.linalg.norm(M_fp - M_g)
        assert error < 1e-5, f"L={L}: ||M_fp - M_gamma|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_output_is_unitary(self, L):
        """Single-particle matrix is unitary."""
        result = build_ffft_fp_sandwich_2d(L, GammaMethod.ANCILLA)
        snake_qs = _snake_qubits(L)
        M = _extract_with_ancillas(result.circuit, snake_qs, result.anc_qubits)
        err = np.linalg.norm(M @ M.conj().T - np.eye(L * L))
        assert err < 1e-5, f"L={L}: unitarity error {err:.2e}"

    @pytest.mark.parametrize("L", [3, 4])
    def test_has_ancillas(self, L):
        """Ancilla method uses L ancilla qubits."""
        result = build_ffft_fp_sandwich_2d(L, GammaMethod.ANCILLA)
        assert len(result.anc_qubits) == L


# ---------------------------------------------------------------------------
# FP-sandwich 1D (transpose + 1D snake FP)
# ---------------------------------------------------------------------------

class TestFPSandwich1DCorrectness:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_dft_matrix(self, L):
        """FP-sandwich 1D single-particle matrix equals F_N."""
        result = build_ffft_fp_sandwich_1d(L)
        snake_qs = _snake_qubits(L)
        M = extract_single_particle_matrix_statevector(result.circuit, snake_qs)
        F = dft_matrix(L * L)
        error = np.linalg.norm(M - F)
        assert error < 1e-5, f"L={L}: ||M - F_N|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_proper(self, L):
        """FP-sandwich 1D matches Gamma-sandwich proper FFFT."""
        r_fp = build_ffft_fp_sandwich_1d(L)
        r_g = build_ffft_proper(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M_fp = extract_single_particle_matrix_statevector(r_fp.circuit, snake_qs)
        M_g = extract_single_particle_matrix_statevector(r_g.circuit, snake_qs)
        error = np.linalg.norm(M_fp - M_g)
        assert error < 1e-5, f"L={L}: ||M_fp - M_gamma|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_output_is_unitary(self, L):
        """Single-particle matrix is unitary."""
        result = build_ffft_fp_sandwich_1d(L)
        snake_qs = _snake_qubits(L)
        M = extract_single_particle_matrix_statevector(result.circuit, snake_qs)
        err = np.linalg.norm(M @ M.conj().T - np.eye(L * L))
        assert err < 1e-5, f"L={L}: unitarity error {err:.2e}"
