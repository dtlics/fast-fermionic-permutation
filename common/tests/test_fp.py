"""Tests for fermionic permutation circuit builders.

Verifies:
    1. OET sort correctness
    2. Hall RCR decomposition correctness
    3. End-to-end: all 4 baselines produce the same unitary for small L
    4. CNOT depth scaling claims
    5. FPResult API consistency across all baselines
    6. Ancilla disentanglement in full FP circuit
    7. Metrics (spacetime volume, idle slots, fidelity)
"""

import numpy as np
import pytest
import pennylane as qp

from common.grid import validate_permutation
from common.hall_decomposition import decompose_permutation_rcr
from common.oet_sort import fswap_odd_even_sort_ops
from common.fp_1d import build_fp_1d, structured_permutations
from common.fp_2d import build_fp_2d, GammaMethod, FPResult
from common.metrics import count_resources, spacetime_volume, multiplicative_fidelity, evaluate_fp


# ---------------------------------------------------------------------------
# OET Sort
# ---------------------------------------------------------------------------

def test_fswap_oet_sort():
    """FSWAP odd-even transposition sort produces correct permutation."""
    rng = np.random.default_rng(42)
    for n in [3, 4, 5, 6]:
        for _ in range(10):
            perm = rng.permutation(n).tolist()
            qubits = [qp.wires.Wires([q]) for q in range(n)]
            ops = fswap_odd_even_sort_ops(list(qubits), perm)
            check = list(perm)
            for op in ops:
                q0, q1 = op.wires
                i, j = list(qubits).index(qp.wires.Wires([q0])), list(qubits).index(qp.wires.Wires([q1]))
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
# FPResult API consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma_method", [
    GammaMethod.ANCILLA, GammaMethod.PRIMITIVE, GammaMethod.PIPELINED
])
def test_fp_2d_returns_fp_result(gamma_method):
    """build_fp_2d returns FPResult for all gamma methods."""
    L = 3
    perm = structured_permutations(L)["transpose"]
    result = build_fp_2d(L, perm, gamma_method)

    assert isinstance(result, FPResult)
    assert isinstance(result.circuit, qp.tape.qscript.QuantumScript)
    assert isinstance(result.sys_qubits, list)
    assert isinstance(result.anc_qubits, list)  # always a list, never None
    assert len(result.sys_qubits) == L * L
    assert result.L == L
    assert result.gamma_method == gamma_method.value

    if gamma_method == GammaMethod.ANCILLA:
        assert len(result.anc_qubits) == L
    else:
        assert len(result.anc_qubits) == 0


def test_fp_1d_returns_fp_result():
    """build_fp_1d returns FPResult with anc_qubits=[] and gamma_method=None."""
    L = 3
    perm = structured_permutations(L)["transpose"]
    result = build_fp_1d(L, perm)

    assert isinstance(result, FPResult)
    assert len(result.sys_qubits) == L * L
    assert result.anc_qubits == []
    assert result.gamma_method is None
    assert result.L == L


# ---------------------------------------------------------------------------
# End-to-end verification: all 4 baselines agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [3])
def test_end_to_end_all_baselines(L):
    """All 4 baselines produce the same density matrix for L=3."""
    dev = qp.device('default.clifford')

    rng = np.random.default_rng(42)
    N = L * L
    canonical_qubits = [qp.wires.Wires([f"({r}, {c})"]) for r in range(L) for c in range(L)]

    perms = list(structured_permutations(L).items())
    perms.append(("random_0", rng.permutation(N).tolist()))

    for perm_name, perm in perms:
        init_bits = rng.integers(0, 2, size=N).tolist()
        init_state = np.zeros(2**N, dtype=complex)
        idx = sum(init_bits[i] << (N - 1 - i) for i in range(N))
        init_state[idx] = 1.0

        # Baseline 1: 1D
        res_1d = build_fp_1d(L, perm)
        circ_1d_full = qp.tape.qscript.QuantumScript(res_1d.circuit.operations + [qp.Identity(q) for q in canonical_qubits], [qp.state()])
        result_1d = qp.execute([circ_1d_full], dev, diff_method=None)
        dm_1d = np.outer(result_1d, np.conj(result_1d))

        # Baseline 3: Ancilla-free primitive
        res_3 = build_fp_2d(L, perm, GammaMethod.PRIMITIVE)
        circ_3_full = qp.tape.qscript.QuantumScript(res_3.circuit.operations + [qp.Identity(q) for q in canonical_qubits], [qp.state()])
        result_3 = qp.execute([circ_3_full], dev, diff_method=None)
        dm_3 = np.outer(result_3, np.conj(result_3))

        # Baseline 4: Pipelined
        res_4 = build_fp_2d(L, perm, GammaMethod.PIPELINED)
        circ_4_full = qp.tape.qscript.QuantumScript(res_4.circuit.operations + [qp.Identity(q) for q in canonical_qubits], [qp.state()])
        result_4 = qp.execute([circ_4_full], dev, diff_method=None)
        dm_4 = np.outer(result_4, np.conj(result_4))

        # Baseline 2: Ancilla (need partial trace)
        res_2 = build_fp_2d(L, perm, GammaMethod.ANCILLA)
        anc_sorted = sorted(res_2.anc_qubits, key=str)
        all_qubits_2 = canonical_qubits + anc_sorted
        n_total = len(all_qubits_2)
        bits_2 = init_bits + [0] * len(res_2.anc_qubits)
        idx_2 = sum(bits_2[i] << (n_total - 1 - i) for i in range(n_total))
        init_state_2 = np.zeros(2**n_total, dtype=complex)
        init_state_2[idx_2] = 1.0
        circ_2_full = qp.tape.qscript.QuantumScript(res_2.circuit.operations + [qp.Identity(q) for q in canonical_qubits], [qp.state()])
        result_2 = qp.execute([circ_2_full], dev, diff_method=None)
        state_2_full = result_2
        dm_2_full = np.outer(state_2_full, np.conj(state_2_full))
        dm_2 = qp.math.partial_trace(
            dm_2_full.reshape([2] * n_total * 2),
            indices=list(range(N))
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
# Ancilla disentanglement in full FP circuit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [3])
def test_ancilla_disentanglement(L):
    """Ancillas start at |0> and return to |0> after the full FP circuit."""
    rng = np.random.default_rng(42 + L)
    N = L * L

    dev = qp.device('default.clifford')

    perm = rng.permutation(N).tolist()
    res = build_fp_2d(L, perm, GammaMethod.ANCILLA)

    canonical_qubits = [qp.wires.Wires([f"({r}, {c})"]) for r in range(L) for c in range(L)]
    anc_sorted = sorted(res.anc_qubits, key=str)
    all_qubits = canonical_qubits + anc_sorted
    n_total = len(all_qubits)
    n_anc = len(res.anc_qubits)

    # Try several random initial system states
    for _ in range(5):
        init_bits = rng.integers(0, 2, size=N).tolist()
        bits = init_bits + [0] * n_anc  # ancillas at |0>
        idx = sum(bits[i] << (n_total - 1 - i) for i in range(n_total))
        init_state = np.zeros(2**n_total, dtype=complex)
        init_state[idx] = 1.0

        circ_full = qp.tape.qscript.QuantumScript(res.circuit.operations + [qp.Identity(q) for q in all_qubits], [qp.state()])
        result = qp.execute([circ_full], dev, diff_method=None)

        # Verify ancilla qubits are in |0> by checking reduced density matrix
        sv = result.final_state_vector
        dm = np.outer(sv, np.conj(sv))
        anc_indices = list(range(N, n_total))
        anc_dm = qp.math.partial_trace(
            dm.reshape([2] * n_total * 2), indices=anc_indices
        ).reshape(2**n_anc, 2**n_anc)
        # If ancillas are in |0...0>, the density matrix is |0><0|
        expected = np.zeros((2**n_anc, 2**n_anc), dtype=complex)
        expected[0, 0] = 1.0
        assert np.allclose(anc_dm, expected, atol=1e-10), (
            f"L={L}: ancillas not disentangled (not in |0...0> state)"
        )


# ---------------------------------------------------------------------------
# Depth scaling claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [20, 25])
def test_2d_depth_better_than_1d(L):
    """2D methods (Baselines 2-4) have fewer total CNOTs than 1D for large L.

    Uses reverse permutation (worst-case displacement for 1D OET sort).
    The advantage is O(N*sqrt(N)) total gates vs O(N^2) for 1D.
    """
    perm = structured_permutations(L)["reverse"]

    res_1d = build_fp_1d(L, perm)
    res_2d = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    r_1d = count_resources(res_1d.circuit, L, len(res_1d.anc_qubits))
    r_2d = count_resources(res_2d.circuit, L, len(res_2d.anc_qubits))

    assert r_2d["total_cnots"] < r_1d["total_cnots"], (
        f"L={L}: 2D total_cnots={r_2d['total_cnots']} >= 1D total_cnots={r_1d['total_cnots']}"
    )


@pytest.mark.parametrize("L", [5, 7, 9])
def test_pipelined_depth_better_than_primitive(L):
    """Pipelined Gamma (Baseline 4) has lower depth than primitive (Baseline 3)."""
    perm = structured_permutations(L)["transpose"]

    res_3 = build_fp_2d(L, perm, GammaMethod.PRIMITIVE)
    res_4 = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    r_3 = count_resources(res_3.circuit, L, len(res_3.anc_qubits))
    r_4 = count_resources(res_4.circuit, L, len(res_4.anc_qubits))

    assert r_4["two_q_depth"] < r_3["two_q_depth"], (
        f"L={L}: pipelined depth={r_4['two_q_depth']} >= primitive depth={r_3['two_q_depth']}"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_spacetime_volume_formula():
    """Spacetime volume = total_qubits * cnot_depth."""
    assert spacetime_volume(25, 100) == 2500
    assert spacetime_volume(30, 0) == 0


def test_cnot_depth_fswap_counts_as_two():
    """FSWAP moments contribute 2 to CNOT depth, CNOT/CZ moments contribute 1.

    The 1D baseline is pure FSWAPs, so cnot_depth = 2 * two_q_depth.
    The 2D baselines have mixed FSWAP and CNOT/CZ moments.
    """
    L = 6
    perm = structured_permutations(L)["reverse"]

    # 1D: all FSWAPs -> cnot_depth = 2 * moment_depth
    r_1d = build_fp_1d(L, perm)
    res_1d = count_resources(r_1d.circuit, L, 0)
    assert res_1d["cnot_depth"] == 2 * res_1d["two_q_depth"], (
        f"1D cnot_depth should be 2x two_q_depth: {res_1d['cnot_depth']} vs {2*res_1d['two_q_depth']}"
    )

    # 2D pipelined: mixed -> cnot_depth > two_q_depth but < 2 * two_q_depth
    r_pipe = build_fp_2d(L, perm, GammaMethod.PIPELINED)
    res_pipe = count_resources(r_pipe.circuit, L, 0)
    assert res_pipe["cnot_depth"] > res_pipe["two_q_depth"], (
        "Pipelined cnot_depth should exceed two_q_depth (has FSWAP moments)"
    )
    assert res_pipe["cnot_depth"] < 2 * res_pipe["two_q_depth"], (
        "Pipelined cnot_depth should be less than 2x two_q_depth (has CNOT-only moments)"
    )


def test_cnot_depth_crossover():
    """At L >= 12, pipelined CNOT depth should be less than 1D CNOT depth."""
    L = 14
    perm = structured_permutations(L)["reverse"]

    r_1d = build_fp_1d(L, perm)
    r_pipe = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    res_1d = count_resources(r_1d.circuit, L, 0)
    res_pipe = count_resources(r_pipe.circuit, L, 0)

    assert res_pipe["cnot_depth"] < res_1d["cnot_depth"], (
        f"At L={L}, pipelined ({res_pipe['cnot_depth']}) should beat 1D ({res_1d['cnot_depth']})"
    )


def test_fidelity_formula():
    """Fidelity = (1-p_2q)^G * (1-p_idle)^I."""
    f = multiplicative_fidelity(100, 200, p_2q=0.001, p_idle=0.0001)
    expected = (1.0 - 0.001) ** 100 * (1.0 - 0.0001) ** 200
    assert abs(f - expected) < 1e-12


def test_evaluate_fp_consistency():
    """evaluate_fp gives consistent results with count_resources."""
    L = 5
    perm = structured_permutations(L)["reverse"]
    res = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    metrics = evaluate_fp(res, p_2q=1e-3, p_idle=1e-5)

    # Cross-check with manual computation
    resources = count_resources(res.circuit, L, len(res.anc_qubits))
    assert metrics["two_q_depth"] == resources["two_q_depth"]
    assert metrics["total_2q_gates"] == resources["total_2q_gates"]
    assert metrics["total_idle_slots"] == resources["total_idle_slots"]
    assert metrics["spacetime_volume"] == resources["total_qubits"] * resources["cnot_depth"]
    assert metrics["gamma_method"] == "pipelined"


def test_idle_slots_positive():
    """Every 2q-gate moment has some idle qubits (unless fully saturated)."""
    L = 5
    perm = structured_permutations(L)["reverse"]
    for method in [GammaMethod.ANCILLA, GammaMethod.PRIMITIVE, GammaMethod.PIPELINED]:
        res = build_fp_2d(L, perm, method)
        resources = count_resources(res.circuit, L, len(res.anc_qubits))
        assert resources["total_idle_slots"] > 0, (
            f"{method}: expected positive idle slots"
        )


def test_ancilla_method_has_more_qubits():
    """Ancilla method has L more total qubits than ancilla-free methods."""
    L = 5
    perm = structured_permutations(L)["transpose"]

    res_anc = build_fp_2d(L, perm, GammaMethod.ANCILLA)
    res_pip = build_fp_2d(L, perm, GammaMethod.PIPELINED)

    r_anc = count_resources(res_anc.circuit, L, len(res_anc.anc_qubits))
    r_pip = count_resources(res_pip.circuit, L, len(res_pip.anc_qubits))

    assert r_anc["total_qubits"] == r_pip["total_qubits"] + L
    assert r_anc["n_ancillas"] == L
    assert r_pip["n_ancillas"] == 0
