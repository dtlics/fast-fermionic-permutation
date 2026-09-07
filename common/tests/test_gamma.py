"""Tests for the ancilla, primitive, pipelined, and folded Gamma constructions.

Verifies:
    1. Depth formulas for all four families, including the all-L>=2 folded bound
    2. NN compliance: every 2q gate is between adjacent GridQubits
    3. No qubit conflicts: no qubit appears in two gates within one moment
    4. The legacy constructions produce identical phases on basis states
    5. Gamma is diagonal: preserves bit values, only applies phases
    6. Ancilla disentanglement: ancillas return to |0> after Gamma
    7. Property (*) holds: Gamma * FSWAP_bare * Gamma = FSWAP_full
"""

import numpy as np
import pytest
import cirq

from common.grid import make_system_qubits, sites_between, snake_to_rc
from common.gamma_ancilla import build_gamma_with_ancillas
from common.gamma_primitive import build_gamma_ancilla_free
from common.gamma_pipeline import build_gamma_pipelined
from common.gamma_folded import (
    FOLDED_GAMMA_BOUNDARY_SCHEDULE_MODEL,
    FOLDED_GAMMA_SCHEDULE_MODEL,
    build_gamma_folded,
    folded_gamma_schedule_model,
)


# ---------------------------------------------------------------------------
# Classical Clifford simulator (Gamma is entirely Clifford: CNOT, CZ, Z)
# ---------------------------------------------------------------------------

def classical_sim_phase(ops_list, qubit_to_idx, n_qubits, basis_state_bits):
    """Simulate Clifford circuit on a computational basis state.
    Returns (phase, final_bits) where phase is +1 or -1."""
    bits = list(basis_state_bits)
    phase = 0
    for op in ops_list:
        gate = op.gate
        qubits = op.qubits
        if isinstance(gate, cirq.ops.common_gates.CNotPowGate) and gate.exponent == 1:
            ctrl_idx = qubit_to_idx[qubits[0]]
            tgt_idx = qubit_to_idx[qubits[1]]
            bits[tgt_idx] ^= bits[ctrl_idx]
        elif isinstance(gate, cirq.ops.common_gates.CZPowGate) and gate.exponent == 1:
            a_idx = qubit_to_idx[qubits[0]]
            b_idx = qubit_to_idx[qubits[1]]
            phase ^= (bits[a_idx] & bits[b_idx])
        elif isinstance(gate, cirq.ops.common_gates.ZPowGate) and gate.exponent == 1:
            idx = qubit_to_idx[qubits[0]]
            phase ^= bits[idx]
        else:
            raise ValueError(f"Unsupported gate in Gamma: {gate}")
    return (-1) ** phase, bits


def get_phase_ancilla_free(circuit, qubit_order, basis_state_int):
    """Get phase from ancilla-free Gamma circuit on basis state."""
    n = len(qubit_order)
    q2i = {q: i for i, q in enumerate(qubit_order)}
    bits = [(basis_state_int >> (n - 1 - i)) & 1 for i in range(n)]
    all_ops = [op for moment in circuit for op in moment]
    phase, _ = classical_sim_phase(all_ops, q2i, n, bits)
    return phase


def get_phase_with_ancillas(circuit, sys_qubits, anc_qubits, basis_state_int):
    """Get phase from ancilla Gamma circuit on basis state."""
    n_sys = len(sys_qubits)
    n_anc = len(anc_qubits)
    all_qubits = sys_qubits + anc_qubits
    q2i = {q: i for i, q in enumerate(all_qubits)}
    bits = [(basis_state_int >> (n_sys - 1 - i)) & 1 for i in range(n_sys)] + [0] * n_anc
    all_ops = [op for moment in circuit for op in moment]
    phase, final_bits = classical_sim_phase(all_ops, q2i, n_sys + n_anc, bits)
    for i in range(n_anc):
        assert final_bits[n_sys + i] == 0, f"Ancilla {i} not disentangled!"
    return phase


# ---------------------------------------------------------------------------
# Helper: verify NN compliance and no conflicts for any circuit
# ---------------------------------------------------------------------------

def _assert_nn_and_no_conflicts(circ, L, label=""):
    """Assert every 2q gate is NN GridQubit and no qubit conflicts per moment."""
    for i, moment in enumerate(circ):
        used = set()
        for op in moment:
            for q in op.qubits:
                assert q not in used, (
                    f"{label} L={L} moment {i}: qubit {q} used by two gates"
                )
                used.add(q)
            if len(op.qubits) == 2:
                q0, q1 = op.qubits
                assert isinstance(q0, cirq.GridQubit) and isinstance(q1, cirq.GridQubit), (
                    f"{label} L={L} moment {i}: non-GridQubit gate {op}"
                )
                dist = abs(q0.row - q1.row) + abs(q0.col - q1.col)
                assert dist == 1, (
                    f"{label} L={L} moment {i}: non-NN gate {op.gate} "
                    f"on {q0},{q1} (manhattan dist={dist})"
                )


# ===================================================================
# 1. Depth formula tests
# ===================================================================

@pytest.mark.parametrize("L", [7, 9, 11, 15, 20])
def test_gamma_ancilla_depth(L):
    """Ancilla Gamma depth = 13L + 4 (exact for L >= 7).

    Achieved by cirq's greedy cross-step pipelining:
      Stage B: 10/step -> ~8/step (tail of step p overlaps head of step p-1)
      Stage D: 6/step -> ~3/step (odd advance overlaps next even advance)
    Without pipelining: 18L - 1.
    """
    circ, _, _ = build_gamma_with_ancillas(L)
    expected = 13 * L + 4
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


@pytest.mark.parametrize("L", [5, 7, 9, 11, 15, 20])
def test_gamma_primitive_depth(L):
    """Ancilla-free primitive Gamma depth = 12L + 8 (exact for L >= 5).

    Phase-separated construction (phases joined with +) prevents
    cross-phase moment merging.  Without separation: ~9L + 12.
    """
    circ, _ = build_gamma_ancilla_free(L)
    expected = 12 * L + 8
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


@pytest.mark.parametrize("L", [5, 7, 9, 11, 15, 20])
def test_gamma_pipelined_depth(L):
    """Pipelined Gamma depth = 8L+9 (odd) or 8L+10 (even).

    Manual pipelining already saturates all parallelism -- cirq's greedy
    scheduler achieves zero additional savings.
    """
    circ, _ = build_gamma_pipelined(L)
    if L % 2 == 1:
        expected = 8 * L + 9
    else:
        expected = 8 * L + 10
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


@pytest.mark.parametrize("L", [2, 3, 4, 5, 6, 7, 8, 11, 20])
def test_gamma_folded_two_qubit_depth(L):
    """Check the exact folded-Gamma depth, including boundary witnesses."""
    circ, _ = build_gamma_folded(L)
    depth = sum(any(len(op.qubits) == 2 for op in moment) for moment in circ)
    boundary_depth = {2: 5, 3: 6, 4: 7, 5: 10, 6: 13}
    expected = boundary_depth.get(L, 2 * L + (4 if L % 2 == 0 else 6))
    assert depth == expected, f"L={L}: got {depth}, expected {expected}"


def test_folded_gamma_schedule_provenance_changes_only_at_new_boundaries():
    """Keep old L=2/asymptotic rows valid and invalidate changed L=3..6 rows."""
    assert folded_gamma_schedule_model(2) == FOLDED_GAMMA_SCHEDULE_MODEL
    assert {
        folded_gamma_schedule_model(L) for L in range(3, 7)
    } == {FOLDED_GAMMA_BOUNDARY_SCHEDULE_MODEL}
    assert {
        folded_gamma_schedule_model(L) for L in (7, 8, 11, 20)
    } == {FOLDED_GAMMA_SCHEDULE_MODEL}
    with pytest.raises(ValueError, match="at least 2"):
        folded_gamma_schedule_model(1)


# ===================================================================
# 2. NN compliance and no-conflict tests
# ===================================================================

@pytest.mark.parametrize("L", [3, 5, 7, 9])
def test_ancilla_gamma_nn_compliance(L):
    """Every gate in ancilla Gamma is between NN GridQubits, no conflicts."""
    circ, _, _ = build_gamma_with_ancillas(L)
    _assert_nn_and_no_conflicts(circ, L, label="ancilla")


@pytest.mark.parametrize("L", [3, 5, 7, 9])
def test_primitive_gamma_nn_compliance(L):
    """Every gate in primitive Gamma is between NN GridQubits, no conflicts."""
    circ, _ = build_gamma_ancilla_free(L)
    _assert_nn_and_no_conflicts(circ, L, label="primitive")


@pytest.mark.parametrize("L", [3, 5, 7, 9])
def test_pipelined_gamma_nn_compliance(L):
    """Every gate in pipelined Gamma is between NN GridQubits, no conflicts."""
    circ, _ = build_gamma_pipelined(L)
    _assert_nn_and_no_conflicts(circ, L, label="pipelined")


@pytest.mark.parametrize("L", [2, 3, 4, 5, 8, 11])
def test_folded_gamma_nn_compliance(L):
    circ, _ = build_gamma_folded(L)
    _assert_nn_and_no_conflicts(circ, L, label="folded")


# ===================================================================
# 3. Cross-construction equivalence
# ===================================================================

@pytest.mark.parametrize("L", [3, 4, 5, 7])
def test_gamma_equivalence(L):
    """The three legacy Gamma constructions agree on sampled basis states."""
    c1, sys1, anc1 = build_gamma_with_ancillas(L)
    c2, sys2 = build_gamma_ancilla_free(L)
    c3, sys3 = build_gamma_pipelined(L)
    N = L * L

    rng = np.random.default_rng(42 + L)
    for _ in range(200):
        bits = rng.integers(0, 2, size=N)
        s = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))

        p1 = get_phase_with_ancillas(c1, sys1, anc1, s)
        p2 = get_phase_ancilla_free(c2, sys2, s)
        p3 = get_phase_ancilla_free(c3, sys3, s)

        assert abs(p1 - p2) < 1e-6, f"L={L}: ancilla vs primitive mismatch on state {s}"
        assert abs(p1 - p3) < 1e-6, f"L={L}: ancilla vs pipelined mismatch on state {s}"


# ===================================================================
# 4. Diagonal property: Gamma preserves bit values
# ===================================================================

@pytest.mark.parametrize("L", [3, 5, 7])
def test_gamma_is_diagonal(L):
    """All Gamma constructions preserve bit values (only apply phases).

    For the ancilla variant this also checks ancillas return to |0>.
    """
    builders = [
        ("ancilla", lambda: build_gamma_with_ancillas(L)),
        ("primitive", lambda: build_gamma_ancilla_free(L)),
        ("pipelined", lambda: build_gamma_pipelined(L)),
    ]
    N = L * L
    rng = np.random.default_rng(42 + L)

    for name, builder in builders:
        result = builder()
        if name == "ancilla":
            circ, sys_list, anc_list = result
            all_qubits = sys_list + anc_list
            q2i = {q: i for i, q in enumerate(all_qubits)}
            n = len(all_qubits)
        else:
            circ, sys_list = result
            q2i = {q: i for i, q in enumerate(sys_list)}
            n = N

        all_ops = [op for moment in circ for op in moment]
        for _ in range(200):
            bits_sys = rng.integers(0, 2, size=N).tolist()
            if name == "ancilla":
                bits = bits_sys + [0] * len(anc_list)
            else:
                bits = bits_sys
            _, final_bits = classical_sim_phase(all_ops, q2i, n, bits)
            assert bits == final_bits, (
                f"L={L}, {name}: bits changed from {bits} to {final_bits}"
            )


# ===================================================================
# 5. Ancilla disentanglement (dedicated, more thorough)
# ===================================================================

@pytest.mark.parametrize("L", [3, 5, 7, 9])
def test_ancilla_disentanglement(L):
    """Ancillas start at |0> and return to |0> for many random basis states.

    Tests 500 states per L (more than the 200 in test_gamma_is_diagonal).
    """
    circ, sys_list, anc_list = build_gamma_with_ancillas(L)
    N = L * L
    n_anc = len(anc_list)
    all_qubits = sys_list + anc_list
    q2i = {q: i for i, q in enumerate(all_qubits)}
    n = len(all_qubits)
    all_ops = [op for moment in circ for op in moment]

    rng = np.random.default_rng(42 + L)
    for _ in range(500):
        bits_sys = rng.integers(0, 2, size=N).tolist()
        bits = bits_sys + [0] * n_anc
        _, final_bits = classical_sim_phase(all_ops, q2i, n, bits)
        for i in range(n_anc):
            assert final_bits[N + i] == 0, (
                f"L={L}: ancilla {i} not |0> after Gamma "
                f"(input sys bits={bits_sys})"
            )


# ===================================================================
# 6. Property (*) verification
# ===================================================================

def verify_property_star(phase_fn, L, num_samples=200, seed=42):
    """Verify property (*) on sampled basis states.

    For every vertical grid-neighbor pair (r,c)<->(r+1,c) with JWT indices j<k:
        gamma_s * gamma_s' = (-1)^{sum of bits between j and k}
    where s,s' differ only at j,k with s_j + s_k = 1.
    """
    rng = np.random.default_rng(seed)
    N = L * L
    checked, passed = 0, 0
    for _ in range(num_samples):
        bits = rng.integers(0, 2, size=N)
        s_idx = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))
        gamma_s = phase_fn(s_idx)
        for r in range(L - 1):
            for c in range(L):
                if bits[r * L + c] == bits[(r + 1) * L + c]:
                    continue
                s_prime_idx = s_idx ^ (1 << (N - 1 - r * L - c)) ^ (1 << (N - 1 - (r + 1) * L - c))
                gamma_s_prime = phase_fn(s_prime_idx)
                between = sites_between(r, c, r + 1, L)
                P = 0
                for site_snake in between:
                    sr, sc = snake_to_rc(site_snake, L)
                    P ^= int(bits[sr * L + sc])
                expected = (-1) ** P
                actual = gamma_s * gamma_s_prime
                checked += 1
                if abs(actual - expected) < 1e-6:
                    passed += 1
    return checked, passed


@pytest.mark.parametrize("L", [3, 5, 7])
def test_property_star_all_constructions(L):
    """Property (*) holds for every Gamma construction, including coset siblings."""
    c1, sys1, anc1 = build_gamma_with_ancillas(L)
    c2, sys2 = build_gamma_ancilla_free(L)
    c3, sys3 = build_gamma_pipelined(L)
    c4, sys4 = build_gamma_folded(L)

    for name, phase_fn in [
        ("ancilla", lambda s: get_phase_with_ancillas(c1, sys1, anc1, s)),
        ("primitive", lambda s: get_phase_ancilla_free(c2, sys2, s)),
        ("pipelined", lambda s: get_phase_ancilla_free(c3, sys3, s)),
        ("folded", lambda s: get_phase_ancilla_free(c4, sys4, s)),
    ]:
        checked, passed = verify_property_star(phase_fn, L, num_samples=300, seed=42 + L)
        assert checked > 0, f"L={L}, {name}: no states checked"
        assert passed == checked, f"L={L}, {name}: {passed}/{checked} passed"


def _quadratic_signature(circuit, qubit_order):
    """Extract the exact GF(2) frame and Boolean phase polynomial."""
    n = len(qubit_order)
    q2i = {q: i for i, q in enumerate(qubit_order)}
    frame = [1 << i for i in range(n)]
    edges = set()
    linear = 0

    def indices(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    for moment in circuit:
        for op in moment:
            gate = op.gate
            if isinstance(gate, cirq.ops.common_gates.CNotPowGate):
                a, b = (q2i[q] for q in op.qubits)
                frame[b] ^= frame[a]
            elif isinstance(gate, cirq.ops.common_gates.CZPowGate):
                a, b = (q2i[q] for q in op.qubits)
                for i in indices(frame[a]):
                    for j in indices(frame[b]):
                        if i == j:
                            linear ^= 1 << i
                        else:
                            edge = (i, j) if i < j else (j, i)
                            if edge in edges:
                                edges.remove(edge)
                            else:
                                edges.add(edge)
            elif isinstance(gate, cirq.ops.common_gates.ZPowGate):
                linear ^= frame[q2i[op.qubits[0]]]
            else:
                raise AssertionError(f"unsupported folded-Gamma gate: {gate}")
    return frame, edges, linear


@pytest.mark.parametrize("L", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_folded_property_star_all_residue_classes(L):
    """Prove coefficient-exact property (*) for every production branch."""
    circuit, sys_qubits = build_gamma_folded(L)
    frame, edges, linear = _quadratic_signature(circuit, sys_qubits)
    assert frame == [1 << i for i in range(L * L)]
    assert linear == 0

    for r in range(L - 1):
        for c in range(L):
            a, b = r * L + c, (r + 1) * L + c
            expected = {
                sr * L + sc
                for snake_index in sites_between(r, c, r + 1, L)
                for sr, sc in [snake_to_rc(snake_index, L)]
            }
            for k in range(L * L):
                if k in (a, b):
                    continue
                ak = (a, k) if a < k else (k, a)
                bk = (b, k) if b < k else (k, b)
                actual = (ak in edges) ^ (bk in edges)
                assert actual == (k in expected), (L, r, c, k)


# ===================================================================
# 7. Gate type inventory: only CNOT, CZ, Z gates in Gamma circuits
# ===================================================================

@pytest.mark.parametrize("L", [5, 7])
def test_gamma_only_clifford_gates(L):
    """Gamma circuits contain only CNOT, CZ, and Z gates."""
    allowed = (
        cirq.ops.common_gates.CNotPowGate,
        cirq.ops.common_gates.CZPowGate,
        cirq.ops.common_gates.ZPowGate,
    )
    for name, builder in [
        ("ancilla", lambda: build_gamma_with_ancillas(L)),
        ("primitive", lambda: build_gamma_ancilla_free(L)),
        ("pipelined", lambda: build_gamma_pipelined(L)),
        ("folded", lambda: build_gamma_folded(L)),
    ]:
        result = builder()
        circ = result[0]
        for moment in circ:
            for op in moment:
                assert isinstance(op.gate, allowed), (
                    f"{name} L={L}: unexpected gate type {type(op.gate).__name__}: {op}"
                )
                if hasattr(op.gate, "exponent"):
                    assert op.gate.exponent == 1, (
                        f"{name} L={L}: non-unit exponent {op.gate.exponent}: {op}"
                    )


# ===================================================================
# 8. Larger L depth checks (spot-check exact formulas at scale)
# ===================================================================

@pytest.mark.parametrize("L", [25, 30])
def test_gamma_ancilla_depth_large(L):
    """Spot-check ancilla Gamma depth = 13L + 4 at larger L."""
    circ, _, _ = build_gamma_with_ancillas(L)
    expected = 13 * L + 4
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


@pytest.mark.parametrize("L", [25, 30])
def test_gamma_primitive_depth_large(L):
    """Spot-check primitive Gamma depth = 12L + 8 at larger L."""
    circ, _ = build_gamma_ancilla_free(L)
    expected = 12 * L + 8
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


@pytest.mark.parametrize("L", [25, 30])
def test_gamma_pipelined_depth_large(L):
    """Spot-check pipelined Gamma depth at larger L."""
    circ, _ = build_gamma_pipelined(L)
    if L % 2 == 1:
        expected = 8 * L + 9
    else:
        expected = 8 * L + 10
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"
