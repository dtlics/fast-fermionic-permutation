"""Tests for the proper 2D FFFT (mode k stays at qubit k)."""

import numpy as np
import pytest
import cirq

from common.fp_2d import GammaMethod
from common.grid import make_system_qubits, snake_to_rc
from exp2_ffft.ffft_proper import (
    build_ffft_core,
    build_ffft_proper,
    build_ffft_fp_sandwich_2d,
    build_ffft_fp_sandwich_1d,
)
from exp2_ffft.ffft_baseline_1d import build_ffft_1d
from exp2_ffft.verify import (
    extract_single_particle_matrix_statevector,
    dft_matrix,
    second_quantized_unitary,
)


def _snake_qubits(L):
    sq = make_system_qubits(L)
    return [sq[snake_to_rc(i, L)] for i in range(L * L)]


def test_public_2d_builders_default_to_folded_gamma():
    """All public 2D FFFT entry points use the current Gamma by default."""
    assert build_ffft_core(2).gamma_method == GammaMethod.FOLDED.value
    assert build_ffft_proper(2).gamma_method == GammaMethod.FOLDED.value
    assert (
        build_ffft_fp_sandwich_2d(2).gamma_method
        == GammaMethod.FOLDED.value
    )


class TestProperFFFTCorrectness:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_matches_dft_matrix(self, L):
        """Proper 2D FFFT single-particle matrix equals F_N in mode basis."""
        result = build_ffft_proper(L, GammaMethod.PIPELINED)
        snake_qs = _snake_qubits(L)
        M = extract_single_particle_matrix_statevector(result.circuit, snake_qs)
        F = dft_matrix(L * L)
        error = np.linalg.norm(M - F)
        assert error < 1e-12, f"L={L}: ||M - F_N|| = {error:.2e}"

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

    def test_folded_matches_full_second_quantized_dft(self):
        """Folded Gamma gives the exact many-body FFFT, not only its 1p block."""
        L = 2
        result = build_ffft_proper(L, GammaMethod.FOLDED)
        snake_qs = _snake_qubits(L)
        actual = result.circuit.unitary(qubit_order=snake_qs)
        expected = second_quantized_unitary(dft_matrix(L * L))
        error = np.linalg.norm(actual - expected)
        assert error < 1e-10, f"L={L}: full-Fock ||U - F_N^wedge|| = {error:.2e}"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_folded_matches_1d_on_coherent_full_fock_state(self, L):
        """Compare all number sectors coherently, including their relative phases."""
        folded = build_ffft_proper(L, GammaMethod.FOLDED)
        baseline = build_ffft_1d(L)
        snake_qs = _snake_qubits(L)
        dimension = 1 << (L * L)
        rng = np.random.default_rng(20260811 + L)
        initial = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        initial /= np.linalg.norm(initial)

        simulator = cirq.Simulator(dtype=np.complex128)
        folded_state = simulator.simulate(
            folded.circuit,
            qubit_order=snake_qs,
            initial_state=initial,
        ).final_state_vector
        baseline_state = simulator.simulate(
            baseline.circuit,
            qubit_order=snake_qs,
            initial_state=initial,
        ).final_state_vector
        error = np.linalg.norm(folded_state - baseline_state)
        assert error < 1e-9, f"L={L}: coherent full-Fock error {error:.2e}"


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

    @pytest.mark.parametrize("L", [4, 8, 16])
    def test_folded_power_of_two_depth_bounds(self, L):
        """Check the FFFT KAK metric and constructive power-of-two bounds."""
        from exp2_ffft.collect import count_cnot_resources

        log_L = int(np.log2(L))
        core = build_ffft_core(L, GammaMethod.FOLDED)
        proper = build_ffft_proper(L, GammaMethod.FOLDED)
        core_depth = count_cnot_resources(core.circuit, L, 0)["cnot_depth"]
        proper_depth = count_cnot_resources(proper.circuit, L, 0)["cnot_depth"]

        if L == 4:
            # The explicit depth-seven boundary Gamma is substantially below
            # the asymptotic parity bound.
            assert core_depth == 38
            assert proper_depth == 82
        else:
            assert core_depth == 14 * L - 4 * log_L
            assert proper_depth <= 26 * L - 4 * log_L + 8

    @pytest.mark.parametrize("L", [3, 4, 5])
    def test_folded_is_nn_and_ancilla_free(self, L):
        result = build_ffft_proper(L, GammaMethod.FOLDED)
        assert result.anc_qubits == []
        decomposed = cirq.Circuit(
            cirq.decompose(result.circuit, keep=lambda op: len(op.qubits) <= 2)
        )
        for moment in decomposed:
            for op in moment:
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    distance = abs(q1.row - q2.row) + abs(q1.col - q2.col)
                    assert distance == 1, f"Non-NN gate between {q1} and {q2}"


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

def _extract_with_ancillas(circuit, snake_qs, anc_qs, *, return_leakage=False):
    """Extract single-particle matrix when circuit has ancilla qubits.

    Ancillas are initialised to |0> and traced out after simulation.
    """
    N = len(snake_qs)
    n_anc = len(anc_qs)
    total = N + n_anc
    # qubit order: system (snake) then ancillas
    qubit_order = list(snake_qs) + list(anc_qs)
    sim = cirq.Simulator(dtype=np.complex128)
    M = np.zeros((N, N), dtype=complex)
    max_leakage = 0.0

    for j in range(N):
        init = np.zeros(2**total, dtype=complex)
        # excite system qubit j; ancillas stay |0>
        init[1 << (total - 1 - j)] = 1.0
        res = sim.simulate(circuit, initial_state=init,
                           qubit_order=qubit_order)
        state = res.final_state_vector
        ancilla_stride = 1 << n_anc
        ancilla_nonzero = state.reshape(-1, ancilla_stride)[:, 1:]
        max_leakage = max(max_leakage, float(np.linalg.norm(ancilla_nonzero)))
        for k in range(N):
            # ancillas should be |0>, so only read system-qubit excitations
            M[k, j] = state[1 << (total - 1 - k)]
    if return_leakage:
        return M, max_leakage
    return M


def _full_fock_system_block_with_ancillas(circuit, snake_qs, anc_qs):
    """Return the induced system operator and worst ancilla leakage."""
    n_system = len(snake_qs)
    n_anc = len(anc_qs)
    qubit_order = list(snake_qs) + list(anc_qs)
    system_dimension = 1 << n_system
    ancilla_stride = 1 << n_anc
    block = np.zeros((system_dimension, system_dimension), dtype=complex)
    max_leakage = 0.0
    simulator = cirq.Simulator(dtype=np.complex128)

    for input_index in range(system_dimension):
        result = simulator.simulate(
            circuit,
            initial_state=input_index * ancilla_stride,
            qubit_order=qubit_order,
        )
        state = result.final_state_vector
        ancilla_zero = state[::ancilla_stride]
        block[:, input_index] = ancilla_zero
        ancilla_nonzero = state.reshape(-1, ancilla_stride)[:, 1:]
        max_leakage = max(
            max_leakage, float(np.linalg.norm(ancilla_nonzero))
        )

    return block, max_leakage


class TestProperFFFTAncillaCorrectness:
    @pytest.mark.parametrize("L", [2, 3])
    def test_collected_ancilla_circuit_matches_dft_and_cleans_ancillas(self, L):
        """Directly verify the ancilla-Gamma circuit collected by Exp2."""
        result = build_ffft_proper(L, GammaMethod.ANCILLA)
        snake_qs = _snake_qubits(L)
        matrix, leakage = _extract_with_ancillas(
            result.circuit,
            snake_qs,
            result.anc_qubits,
            return_leakage=True,
        )

        assert len(result.anc_qubits) == L
        assert set(result.anc_qubits).isdisjoint(snake_qs)
        assert np.linalg.norm(matrix - dft_matrix(L * L)) < 1e-9
        assert leakage < 1e-9

    def test_collected_ancilla_circuit_matches_full_fock_dft_at_L2(self):
        """Catch fermionic phases that a one-particle test cannot observe."""
        L = 2
        result = build_ffft_proper(L, GammaMethod.ANCILLA)
        actual, leakage = _full_fock_system_block_with_ancillas(
            result.circuit, _snake_qubits(L), result.anc_qubits,
        )
        expected = second_quantized_unitary(dft_matrix(L * L))

        assert np.linalg.norm(actual - expected) < 1e-9
        assert leakage < 1e-9


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
