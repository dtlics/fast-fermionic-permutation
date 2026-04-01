"""Tests for fermionic permutation circuit builders.

Verifies:
    1. OET sort correctness
    2. Hall RCR decomposition correctness
    3. End-to-end: all 4 baselines produce the same unitary for small L
    4. CNOT depth scaling claims
"""

import numpy as np
import pytest
import cirq
from openfermion.circuits.gates import FSWAP

from common.grid import validate_permutation
from common.hall_decomposition import decompose_permutation_rcr
from common.oet_sort import fswap_odd_even_sort_ops
from common.fp_1d import build_fp_1d, structured_permutations
from common.fp_2d import build_fp_2d, GammaMethod
from common.metrics import count_resources


# ---------------------------------------------------------------------------
# OET Sort
# ---------------------------------------------------------------------------

def test_fswap_oet_sort():
    """FSWAP odd-even transposition sort produces correct permutation."""
    rng = np.random.default_rng(42)
    for n in [3, 4, 5, 6]:
        for _ in range(10):
            perm = rng.permutation(n).tolist()
            qubits = cirq.LineQubit.range(n)
            ops = fswap_odd_even_sort_ops(list(qubits), perm)
            check = list(perm)
            for op in ops:
                q0, q1 = op.qubits
                i, j = list(qubits).index(q0), list(qubits).index(q1)
                assert abs(i - j) == 1
                check[i], check[j] = check[j], check[i]
            assert check == list(range(n)), f"Sort failed for perm={perm}"


# ---------------------------------------------------------------------------
# Hall RCR Decomposition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [3, 4, 5])
def test_hall_decomposition(L):
    """Hall RCR decomposition recovers the original permutation."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        perm = rng.permutation(L * L).tolist()
        s1, s2, s3 = decompose_permutation_rcr(L, perm)
        for r in range(L):
            for c in range(L):
                c_after_rowA = s1[r][c]
                r_after_col = s2[c_after_rowA][r]
                c_final = s3[r_after_col][c_after_rowA]
                actual = r_after_col * L + c_final
                expected = perm[r * L + c]
                assert actual == expected, (
                    f"Mismatch at ({r},{c}): got {actual}, expected {expected}"
                )


# ---------------------------------------------------------------------------
# End-to-end verification: all 4 baselines agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [3])
def test_end_to_end_all_baselines(L):
    """All 4 baselines produce the same density matrix for L=3."""
    rng = np.random.default_rng(42)
    N = L * L
    sim = cirq.Simulator()
    canonical_qubits = [cirq.GridQubit(r, c) for r in range(L) for c in range(L)]

    perms = list(structured_permutations(L).items())
    perms.append(("random_0", rng.permutation(N).tolist()))

    for perm_name, perm in perms:
        init_bits = rng.integers(0, 2, size=N).tolist()
        init_state = np.zeros(2**N, dtype=complex)
        idx = sum(init_bits[i] << (N - 1 - i) for i in range(N))
        init_state[idx] = 1.0

        # Baseline 1: 1D
        circ_1d, _ = build_fp_1d(L, perm)
        circ_1d_full = circ_1d + cirq.Circuit([cirq.I(q) for q in canonical_qubits])
        result_1d = sim.simulate(circ_1d_full, qubit_order=canonical_qubits,
                                 initial_state=init_state.copy())
        dm_1d = np.outer(result_1d.final_state_vector,
                         np.conj(result_1d.final_state_vector))

        # Baseline 3: Ancilla-free primitive
        circ_3, sys_3, _ = build_fp_2d(L, perm, GammaMethod.PRIMITIVE)
        circ_3_full = circ_3 + cirq.Circuit([cirq.I(q) for q in canonical_qubits])
        result_3 = sim.simulate(circ_3_full, qubit_order=canonical_qubits,
                                initial_state=init_state.copy())
        dm_3 = np.outer(result_3.final_state_vector,
                        np.conj(result_3.final_state_vector))

        # Baseline 4: Pipelined
        circ_4, sys_4, _ = build_fp_2d(L, perm, GammaMethod.PIPELINED)
        circ_4_full = circ_4 + cirq.Circuit([cirq.I(q) for q in canonical_qubits])
        result_4 = sim.simulate(circ_4_full, qubit_order=canonical_qubits,
                                initial_state=init_state.copy())
        dm_4 = np.outer(result_4.final_state_vector,
                        np.conj(result_4.final_state_vector))

        # Baseline 2: Ancilla (need partial trace)
        circ_2, sys_2, anc_2 = build_fp_2d(L, perm, GammaMethod.ANCILLA)
        anc_sorted = sorted(anc_2, key=str)
        all_qubits_2 = canonical_qubits + anc_sorted
        n_total = len(all_qubits_2)
        bits_2 = init_bits + [0] * len(anc_2)
        idx_2 = sum(bits_2[i] << (n_total - 1 - i) for i in range(n_total))
        init_state_2 = np.zeros(2**n_total, dtype=complex)
        init_state_2[idx_2] = 1.0
        circ_2_full = circ_2 + cirq.Circuit([cirq.I(q) for q in all_qubits_2])
        result_2 = sim.simulate(circ_2_full, qubit_order=all_qubits_2,
                                initial_state=init_state_2)
        state_2_full = result_2.final_state_vector
        dm_2_full = np.outer(state_2_full, np.conj(state_2_full))
        dm_2 = cirq.partial_trace(
            dm_2_full.reshape([2] * n_total * 2),
            keep_indices=list(range(N))
        ).reshape(2**N, 2**N)

        # Check all pairs agree
        for name_a, dm_a, name_b, dm_b in [
            ("1D", dm_1d, "primitive", dm_3),
            ("1D", dm_1d, "pipelined", dm_4),
            ("1D", dm_1d, "ancilla", dm_2),
            ("primitive", dm_3, "pipelined", dm_4),
        ]:
            fid = np.real(np.trace(dm_a @ dm_b))
            assert abs(fid - 1.0) < 1e-6, (
                f"L={L}, perm={perm_name}: {name_a} vs {name_b} fidelity = {fid}"
            )


# ---------------------------------------------------------------------------
# Depth scaling claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [15, 20])
def test_2d_depth_better_than_1d(L):
    """2D methods (Baselines 2-4) have lower CNOT depth than 1D for large L.

    Uses reverse permutation (worst-case displacement for 1D OET sort).
    The crossover happens because 2D has O(sqrt(N)) = O(L) depth vs O(N) = O(L^2).
    """
    perm = structured_permutations(L)["reverse"]

    circ_1d, _ = build_fp_1d(L, perm)
    circ_2d, _, _ = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    r_1d = count_resources(circ_1d, L, 0)
    r_2d = count_resources(circ_2d, L, 0)

    assert r_2d["cnot_depth"] < r_1d["cnot_depth"], (
        f"L={L}: 2D cnot_depth={r_2d['cnot_depth']} >= 1D cnot_depth={r_1d['cnot_depth']}"
    )


@pytest.mark.parametrize("L", [5, 7, 9])
def test_pipelined_depth_better_than_primitive(L):
    """Pipelined Gamma (Baseline 4) has lower depth than primitive (Baseline 3)."""
    perm = structured_permutations(L)["transpose"]

    circ_3, _, _ = build_fp_2d(L, perm, GammaMethod.PRIMITIVE)
    circ_4, _, _ = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    r_3 = count_resources(circ_3, L, 0)
    r_4 = count_resources(circ_4, L, 0)

    assert r_4["cnot_depth"] < r_3["cnot_depth"], (
        f"L={L}: pipelined cnot_depth={r_4['cnot_depth']} >= primitive cnot_depth={r_3['cnot_depth']}"
    )
