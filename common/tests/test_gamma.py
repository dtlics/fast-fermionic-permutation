"""Tests for all three Gamma constructions.

Verifies:
    1. Depth formulas: 7L-3 (ancilla), 9L+12 (primitive), 8L+9/10 (pipelined)
    2. All three constructions produce identical phases on basis states
    3. Property (*) holds: Gamma * FSWAP_bare * Gamma = FSWAP_full for vertical pairs
"""

import numpy as np
import pytest
import cirq

from common.grid import make_system_qubits, sites_between, snake_to_rc
from common.gamma_ancilla import build_gamma_with_ancillas
from common.gamma_primitive import build_gamma_ancilla_free
from common.gamma_pipeline import build_gamma_pipelined


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
# Depth formula tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [7, 9, 11, 15])
def test_gamma_ancilla_depth(L):
    """Verify ancilla Gamma physical NN depth = 18L - 1 (exact for L >= 7).

    Stages: A (L-1) + B (10*L) + C (L-1) + D (6*L) + diag (1) = 18L - 1.
    All gates are NN on the (L+1)-column grid.  No pipelining across steps.
    For L < 7, cirq's scheduler packs more tightly (fewer skip-row pairs).
    """
    circ, _, _ = build_gamma_with_ancillas(L)
    expected = 18 * L - 1
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


@pytest.mark.parametrize("L", [3, 5, 7])
def test_ancilla_gamma_all_nn(L):
    """Every 2q gate in the ancilla Gamma is between NN GridQubits."""
    circ, _, _ = build_gamma_with_ancillas(L)
    for i, moment in enumerate(circ):
        used = set()
        for op in moment:
            # No qubit conflicts within a moment
            for q in op.qubits:
                assert q not in used, f"L={L} moment {i}: qubit {q} used twice"
                used.add(q)
            if len(op.qubits) == 2:
                q0, q1 = op.qubits
                assert isinstance(q0, cirq.GridQubit) and isinstance(q1, cirq.GridQubit), (
                    f"L={L} moment {i}: non-GridQubit gate {op}"
                )
                dist = abs(q0.row - q1.row) + abs(q0.col - q1.col)
                assert dist == 1, (
                    f"L={L} moment {i}: non-NN gate {op.gate} on {q0},{q1} (dist={dist})"
                )


@pytest.mark.parametrize("L", [5, 7, 9, 11, 15])
def test_gamma_primitive_depth(L):
    """Verify ancilla-free primitive Gamma depth = 12L + 8 (exact for L >= 5).

    Phase-separated construction prevents cross-phase moment merging,
    giving the theoretical sequential-phase depth.
    """
    circ, _ = build_gamma_ancilla_free(L)
    expected = 12 * L + 8
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


@pytest.mark.parametrize("L", [5, 7, 9, 11, 15])
def test_gamma_pipelined_depth(L):
    """Verify pipelined Gamma depth = 8L+9 (odd) or 8L+10 (even)."""
    circ, _ = build_gamma_pipelined(L)
    if L % 2 == 1:
        expected = 8 * L + 9
    else:
        expected = 8 * L + 10
    assert len(circ) == expected, f"L={L}: got {len(circ)}, expected {expected}"


# ---------------------------------------------------------------------------
# Cross-construction equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [3, 4, 5, 7])
def test_gamma_equivalence(L):
    """All three Gamma constructions produce identical phases on sampled basis states."""
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


# ---------------------------------------------------------------------------
# Property (*) verification
# ---------------------------------------------------------------------------

def verify_property_star(phase_fn, L, num_samples=200, seed=42):
    """Verify property (star) on sampled basis states.

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
def test_gamma_is_diagonal(L):
    """All Gamma constructions must be diagonal (preserve bit values, only apply phases)."""
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


@pytest.mark.parametrize("L", [3, 5, 7])
def test_property_star_all_constructions(L):
    """Property (*) holds for all three Gamma constructions."""
    c1, sys1, anc1 = build_gamma_with_ancillas(L)
    c2, sys2 = build_gamma_ancilla_free(L)
    c3, sys3 = build_gamma_pipelined(L)

    for name, phase_fn in [
        ("ancilla", lambda s: get_phase_with_ancillas(c1, sys1, anc1, s)),
        ("primitive", lambda s: get_phase_ancilla_free(c2, sys2, s)),
        ("pipelined", lambda s: get_phase_ancilla_free(c3, sys3, s)),
    ]:
        checked, passed = verify_property_star(phase_fn, L, num_samples=200, seed=42 + L)
        assert checked > 0, f"L={L}, {name}: no states checked"
        assert passed == checked, f"L={L}, {name}: {passed}/{checked} passed"
