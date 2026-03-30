"""
Pipelined Gamma circuit: Construction A + B from gamma.tex.

Reduces ancilla-free Gamma depth from 9L+12 to 8L+O(1) by fusing
same-row T(x,x) with cross-row/skip-row T(x,y) into shared cascade sweeps.
"""

import cirq
import numpy as np
from typing import Dict, List, Tuple

# =============================================================================
# Utilities (copied from Gamma.ipynb for self-contained verification)
# =============================================================================

def snake_order_indices(L: int) -> List[int]:
    order = []
    for r in range(L):
        row = [r * L + c for c in range(L)]
        if r % 2 == 1:
            row.reverse()
        order.extend(row)
    return order

def rc_to_snake(r: int, c: int, L: int) -> int:
    if r % 2 == 0:
        return r * L + c
    else:
        return r * L + (L - 1 - c)

def snake_to_rc(idx: int, L: int) -> Tuple[int, int]:
    r = idx // L
    pos_in_row = idx % L
    if r % 2 == 0:
        c = pos_in_row
    else:
        c = L - 1 - pos_in_row
    return r, c

def sites_between(r1: int, c: int, r2: int, L: int) -> List[int]:
    j = rc_to_snake(r1, c, L)
    k = rc_to_snake(r2, c, L)
    lo, hi = min(j, k), max(j, k)
    return list(range(lo + 1, hi))

def make_system_qubits(L: int) -> Dict[Tuple[int, int], cirq.GridQubit]:
    return {(r, c): cirq.GridQubit(r, c) for r in range(L) for c in range(L)}

def make_ancilla_qubits(L: int) -> Dict[int, cirq.NamedQubit]:
    return {r: cirq.NamedQubit(f"anc_r{r}") for r in range(L)}

# =============================================================================
# Classical Clifford simulation
# =============================================================================

def classical_sim_phase(ops_list, qubit_to_idx, n_qubits, basis_state_bits):
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
            raise ValueError(f"Unsupported gate: {gate}")
    return (-1)**phase, bits

def get_phase_from_ops(ops_list, qubit_order, basis_state):
    n = len(qubit_order)
    q2i = {q: i for i, q in enumerate(qubit_order)}
    bits = [(basis_state >> (n - 1 - i)) & 1 for i in range(n)]
    phase, _ = classical_sim_phase(ops_list, q2i, n, bits)
    return phase

def get_phase_classical(circuit, qubit_order, basis_state):
    all_ops = [op for moment in circuit for op in moment]
    return get_phase_from_ops(all_ops, qubit_order, basis_state)

# =============================================================================
# Existing primitives (from Gamma.ipynb cell 7)
# =============================================================================

def column_parity_cascade_ops(sq, L, inverse=False):
    ops = []
    if not inverse:
        for r in range(L - 2, -1, -1):
            for c in range(L):
                ops.append(cirq.CNOT(sq[(r + 1, c)], sq[(r, c)]))
    else:
        for r in range(L - 1):
            for c in range(L):
                ops.append(cirq.CNOT(sq[(r + 1, c)], sq[(r, c)]))
    return ops

def suffix_cascade_ops(sq, r, L):
    return [cirq.CNOT(sq[(r, c + 1)], sq[(r, c)]) for c in range(L - 2, -1, -1)]

def undo_suffix_cascade_ops(sq, r, L):
    return [cirq.CNOT(sq[(r, c + 1)], sq[(r, c)]) for c in range(L - 1)]

def prefix_cascade_ops(sq, r, L):
    return [cirq.CNOT(sq[(r, c - 1)], sq[(r, c)]) for c in range(1, L)]

def undo_prefix_cascade_ops(sq, r, L):
    return [cirq.CNOT(sq[(r, c - 1)], sq[(r, c)]) for c in range(L - 1, 0, -1)]

# Existing same-row T using SUFFIX cascade
def same_row_T_suffix_ops(sq, r, L):
    ops = []
    ops.extend(suffix_cascade_ops(sq, r, L))
    for c in range(L - 1):
        ops.append(cirq.CZ(sq[(r, c)], sq[(r, c + 1)]))
    ops.extend(undo_suffix_cascade_ops(sq, r, L))
    for c in range(1, L, 2):
        ops.append(cirq.Z(sq[(r, c)]))
    return ops

# Existing cross-row T using PREFIX cascade
def cross_row_adjacent_T_ops(sq, r1, r2, L):
    ops = []
    ops.extend(prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r1, c)], sq[(r2, c)]))
    ops.extend(undo_prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r1, c)], sq[(r2, c)]))
    return ops

# Existing skip-row T using PREFIX cascade
def skip_row_T_ops(sq, r1, r2, r_mid, L):
    ops = []
    ops.extend(prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
    ops.extend(undo_prefix_cascade_ops(sq, r1, L))
    for c in range(L):
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
        ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        ops.append(cirq.CNOT(sq[(r1, c)], sq[(r_mid, c)]))
    return ops

# Existing ancilla-free Gamma (9L+12)
def build_gamma_existing(L):
    sq = make_system_qubits(L)
    ops = []
    ops.extend(column_parity_cascade_ops(sq, L, inverse=False))
    for r in range(2, L, 2):
        ops.extend(same_row_T_suffix_ops(sq, r, L))
    skip_rows = [r for r in range(0, L, 2) if r + 2 <= L - 1]
    for r in skip_rows[0::2]:
        ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))
    for r in skip_rows[1::2]:
        ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))
    ops.extend(column_parity_cascade_ops(sq, L, inverse=True))
    for r in range(0, L, 2):
        ops.extend(same_row_T_suffix_ops(sq, r, L))
    for r in range(0, L - 1, 2):
        ops.extend(cross_row_adjacent_T_ops(sq, r, r + 1, L))
    circuit = cirq.Circuit(ops)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]
    return circuit, sys_list

# =============================================================================
# Step 1: Same-row T using PREFIX cascade
# =============================================================================

def same_row_T_prefix_ops(sq, r, L):
    """Same-row T(x,x) using prefix cascade instead of suffix.

    After prefix cascade, position c holds x̂_c = XOR_{c'<=c} x_{c'}.
    CZ(c, c+1) applies (-1)^{x̂_c · x̂_{c+1}}.

    x̂_c · x̂_{c+1} = x̂_c · (x̂_c ⊕ x_{c+1}) = x̂_c + x̂_c·x_{c+1}  (GF(2))

    Summing degree-2 parts: Σ x̂_c · x_{c+1} = Σ_{p<c'} x_p·x_{c'} = T(x,x)
    Degree-1 residual: Σ_{c=0}^{L-2} x̂_c = Σ_p (L-1-p) x_p

    Z correction: Z on column p where (L-1-p) is odd.
    - If L is even: (L-1-p) odd when p even → Z on even columns 0,2,...,L-2
    - If L is odd:  (L-1-p) odd when p odd  → Z on odd columns 1,3,...,L-2

    But wait — x̂_c is the content after cascade, not x_c. The Z gate acts on
    the current qubit value. After the undo cascade, positions hold original x_c.
    So Z corrections must be applied AFTER undoing the cascade.
    """
    ops = []
    # Prefix cascade: left-to-right
    ops.extend(prefix_cascade_ops(sq, r, L))
    # Adjacent CZ gates
    for c in range(L - 1):
        ops.append(cirq.CZ(sq[(r, c)], sq[(r, c + 1)]))
    # Undo prefix cascade
    ops.extend(undo_prefix_cascade_ops(sq, r, L))
    # Z corrections: need Z on column p where (L-1-p) mod 2 == 1
    # Only p from 0..L-2 contribute (from the CZ sum range)
    for p in range(L - 1):
        if (L - 1 - p) % 2 == 1:
            ops.append(cirq.Z(sq[(r, p)]))
    return ops


def verify_same_row_T_prefix(L_values=range(3, 8)):
    """Compare prefix-based vs suffix-based same-row T for multiple L."""
    print("\n=== Step 1: Verify prefix-based same-row T ===")
    for L in L_values:
        sq = make_system_qubits(L)
        qubit_order = [sq[(r, c)] for r in range(L) for c in range(L)]

        # Test on a single row (row 0)
        r = 0
        ops_suffix = same_row_T_suffix_ops(sq, r, L)
        ops_prefix = same_row_T_prefix_ops(sq, r, L)

        N = L * L
        n_samples = 500
        rng = np.random.default_rng(42)
        mismatches = 0

        for _ in range(n_samples):
            bits = rng.integers(0, 2, size=N)
            s = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))
            p_suf = get_phase_from_ops(ops_suffix, qubit_order, s)
            p_pre = get_phase_from_ops(ops_prefix, qubit_order, s)
            if abs(p_suf - p_pre) > 1e-6:
                mismatches += 1

        status = "PASS" if mismatches == 0 else f"FAIL ({mismatches} mismatches)"
        print(f"  L={L}: {status}")


# =============================================================================
# Step 2: Construction B — PipelineSameCross
# =============================================================================

def pipeline_same_cross_ops(sq, r, L):
    """Construction B: Fused same-row T(x,x) + cross-row T(x, y) in one sweep.

    Row r (even) holds x, row r+1 holds y.
    Implements (-1)^{T(x,x) ⊕ T(x,y)}.

    Pipelined: trailing interactions follow the cascade wavefront at fixed offsets.

    Forward sweep (τ = 0 to L-2, plus drain):
      Offset  0:    Cascade CNOT (r,τ) -> (r,τ+1)
      Offset -2:    Cross-row CZ: CZ((r,τ-2), (r+1,τ-2))
      Offset -3/-4: Same-row CZ:  CZ((r,τ-4), (r,τ-3))

    Undo sweep (reverse, plus drain):
      Undo cascade
      Offset +2: Cross-row CZ correction
      Offset +3: Same-row Z correction

    Emitted in time-step order for optimal scheduling.
    """
    ops = []
    r2 = r + 1

    # Forward sweep: cascade advances, trailing ops follow
    # Total forward time steps: 0 to L-2 (cascade) + drain up to L+2
    max_fwd = L - 2 + 4  # enough to drain all trailing ops
    for tau in range(max_fwd + 1):
        # Cascade CNOT (only during τ = 0 to L-2)
        if 0 <= tau <= L - 2:
            ops.append(cirq.CNOT(sq[(r, tau)], sq[(r, tau + 1)]))
        # Cross-row CZ at column τ-2
        c = tau - 2
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r, c)], sq[(r2, c)]))
        # Same-row CZ between columns τ-4 and τ-3
        c_lo, c_hi = tau - 4, tau - 3
        if 0 <= c_lo and c_hi < L:
            ops.append(cirq.CZ(sq[(r, c_lo)], sq[(r, c_hi)]))

    # Undo sweep: reverse cascade with trailing corrections
    max_undo_extra = 3  # drain after undo
    for tau in range(L - 2, -1 - max_undo_extra - 1, -1):
        # Undo cascade CNOT (only during τ = L-2 down to 0)
        if 0 <= tau <= L - 2:
            ops.append(cirq.CNOT(sq[(r, tau)], sq[(r, tau + 1)]))
        # Cross-row CZ correction at column τ+2
        c = tau + 2
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r, c)], sq[(r2, c)]))
        # Same-row Z correction at column τ+3
        c = tau + 3
        if 0 <= c < L and (L - 1 - c) % 2 == 1:
            ops.append(cirq.Z(sq[(r, c)]))

    return ops


def verify_construction_B(L_values=range(3, 10)):
    """Compare Construction B against separate same-row + cross-row T."""
    print("\n=== Step 2: Verify Construction B (PipelineSameCross) ===")
    for L in L_values:
        sq = make_system_qubits(L)
        qubit_order = [sq[(r, c)] for r in range(L) for c in range(L)]
        N = L * L

        r = 0  # test on row 0 (even), with row 1 as cross target

        # Reference: separate same-row T(x,x) + cross-row T(x,y)
        ops_ref = same_row_T_prefix_ops(sq, r, L) + cross_row_adjacent_T_ops(sq, r, r + 1, L)

        # Pipelined
        ops_pipe = pipeline_same_cross_ops(sq, r, L)

        n_samples = 500
        rng = np.random.default_rng(42)
        mismatches = 0

        for _ in range(n_samples):
            bits = rng.integers(0, 2, size=N)
            s = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))
            p_ref = get_phase_from_ops(ops_ref, qubit_order, s)
            p_pipe = get_phase_from_ops(ops_pipe, qubit_order, s)
            if abs(p_ref - p_pipe) > 1e-6:
                mismatches += 1
                if mismatches <= 3:
                    bits = [(s >> (N - 1 - i)) & 1 for i in range(N)]
                    print(f"    L={L} MISMATCH: state={s}, ref={p_ref}, pipe={p_pipe}")

        status = "PASS" if mismatches == 0 else f"FAIL ({mismatches}/{n_samples})"
        print(f"  L={L}: {status}")


# =============================================================================
# Step 3: Construction A — PipelineSameSkip
# =============================================================================

def pipeline_same_skip_ops(sq, r, L):
    """Construction A: Fused same-row T(x̃,x̃) + skip-row T(x̃, ỹ) in one sweep.

    Row r (even) holds x̃ (source), row r+1 (odd) is intermediary, row r+2 holds ỹ.
    Implements (-1)^{T(x̃,x̃) ⊕ T(x̃,ỹ)}.

    Pipelined: trailing interactions follow the cascade wavefront.

    Forward sweep:
      Offset  0:     Cascade CNOT (r,τ) -> (r,τ+1)
      Offset -1:     CZ (r+1,τ-1), (r+2,τ-1)   [pre-cancel]
      Offset -2:     CNOT (r,τ-2) -> (r+1,τ-2)  [copy prefix]
      Offset -3:     CZ (r+1,τ-3), (r+2,τ-3)    [interact]
      Offset -4:     CNOT (r,τ-4) -> (r+1,τ-4)  [restore]
      Offset -5/-6:  CZ (r,τ-6), (r,τ-5)         [same-row]

    Undo sweep (mirrored):
      Undo cascade
      Offset +1..+4: Skip-row correction gadget
      Offset +5:     Same-row Z correction
    """
    ops = []
    r_mid = r + 1
    r2 = r + 2

    # Forward sweep
    max_fwd = L - 2 + 6  # drain trailing ops
    for tau in range(max_fwd + 1):
        if 0 <= tau <= L - 2:
            ops.append(cirq.CNOT(sq[(r, tau)], sq[(r, tau + 1)]))
        # Skip-row gadget offsets -1 to -4
        c = tau - 1
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau - 2
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))
        c = tau - 3
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau - 4
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))
        # Same-row CZ at offsets -5/-6
        c_lo, c_hi = tau - 6, tau - 5
        if 0 <= c_lo and c_hi < L:
            ops.append(cirq.CZ(sq[(r, c_lo)], sq[(r, c_hi)]))

    # Undo sweep
    max_undo_extra = 5
    for tau in range(L - 2, -1 - max_undo_extra - 1, -1):
        if 0 <= tau <= L - 2:
            ops.append(cirq.CNOT(sq[(r, tau)], sq[(r, tau + 1)]))
        # Skip-row correction gadget at offsets +1 to +4
        c = tau + 1
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau + 2
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))
        c = tau + 3
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau + 4
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))
        # Same-row Z correction at offset +5
        c = tau + 5
        if 0 <= c < L and (L - 1 - c) % 2 == 1:
            ops.append(cirq.Z(sq[(r, c)]))

    return ops


def pipeline_skip_only_ops(sq, r, L):
    """Pipelined skip-row T(x̃, ỹ) only (no same-row T). For row 0 in parity basis.

    Same pipeline as Construction A but without same-row CZ (offsets -5/-6)
    and without Z corrections (offset +5).
    """
    ops = []
    r_mid = r + 1
    r2 = r + 2

    # Forward sweep
    max_fwd = L - 2 + 4  # skip gadget uses offsets -1 to -4
    for tau in range(max_fwd + 1):
        if 0 <= tau <= L - 2:
            ops.append(cirq.CNOT(sq[(r, tau)], sq[(r, tau + 1)]))
        c = tau - 1
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau - 2
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))
        c = tau - 3
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau - 4
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))

    # Undo sweep
    max_undo_extra = 4
    for tau in range(L - 2, -1 - max_undo_extra - 1, -1):
        if 0 <= tau <= L - 2:
            ops.append(cirq.CNOT(sq[(r, tau)], sq[(r, tau + 1)]))
        c = tau + 1
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau + 2
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))
        c = tau + 3
        if 0 <= c < L:
            ops.append(cirq.CZ(sq[(r_mid, c)], sq[(r2, c)]))
        c = tau + 4
        if 0 <= c < L:
            ops.append(cirq.CNOT(sq[(r, c)], sq[(r_mid, c)]))

    return ops


def verify_pipeline_skip_only(L_values=range(3, 10)):
    """Verify pipelined skip-only matches non-pipelined skip_row_T_ops."""
    print("\n=== Verify pipeline_skip_only ===")
    for L in L_values:
        if L < 3:
            continue
        sq = make_system_qubits(L)
        qubit_order = [sq[(r, c)] for r in range(L) for c in range(L)]
        N = L * L
        r = 0
        if r + 2 >= L:
            continue
        ops_ref = skip_row_T_ops(sq, r, r + 2, r + 1, L)
        ops_pipe = pipeline_skip_only_ops(sq, r, L)
        rng = np.random.default_rng(42)
        mismatches = 0
        for _ in range(500):
            bits = rng.integers(0, 2, size=N)
            s = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))
            p_ref = get_phase_from_ops(ops_ref, qubit_order, s)
            p_pipe = get_phase_from_ops(ops_pipe, qubit_order, s)
            if abs(p_ref - p_pipe) > 1e-6:
                mismatches += 1
        status = "PASS" if mismatches == 0 else f"FAIL ({mismatches}/500)"
        d_ref = len(cirq.Circuit(ops_ref))
        d_pipe = len(cirq.Circuit(ops_pipe))
        print(f"  L={L}: {status}, depth_ref={d_ref}, depth_pipe={d_pipe}")


def verify_construction_A(L_values=range(3, 10)):
    """Compare Construction A against separate same-row + skip-row T."""
    print("\n=== Step 3: Verify Construction A (PipelineSameSkip) ===")
    for L in L_values:
        if L < 3:
            continue
        sq = make_system_qubits(L)
        qubit_order = [sq[(r, c)] for r in range(L) for c in range(L)]
        N = L * L

        r = 0  # test on rows 0, 1, 2
        if r + 2 >= L:
            print(f"  L={L}: SKIP (need r+2 < L)")
            continue

        # Reference: separate same-row T(x̃,x̃) + skip-row T(x̃,ỹ)
        ops_ref = same_row_T_prefix_ops(sq, r, L) + skip_row_T_ops(sq, r, r + 2, r + 1, L)

        # Pipelined
        ops_pipe = pipeline_same_skip_ops(sq, r, L)

        n_samples = 500
        rng = np.random.default_rng(42)
        mismatches = 0

        for _ in range(n_samples):
            bits = rng.integers(0, 2, size=N)
            s = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))
            p_ref = get_phase_from_ops(ops_ref, qubit_order, s)
            p_pipe = get_phase_from_ops(ops_pipe, qubit_order, s)
            if abs(p_ref - p_pipe) > 1e-6:
                mismatches += 1
                if mismatches <= 3:
                    print(f"    L={L} MISMATCH: state={s}, ref={p_ref}, pipe={p_pipe}")

        status = "PASS" if mismatches == 0 else f"FAIL ({mismatches}/{n_samples})"
        print(f"  L={L}: {status}")


# =============================================================================
# Step 4: Full Pipelined Gamma
# =============================================================================

def build_gamma_pipelined(L):
    """Build the pipelined Gamma circuit.

    Phase 1: Column parity cascade forward (depth L-1)
    Phase 2: Parity-basis interactions (f_B), 2 batches
    Phase 3: Column parity cascade inverse (depth L-1)
    Phase 4: Original-basis interactions (f_D), single pass

    f_B terms (parity basis):
      - Skip-row T(x̃_r, x̃_{r+2}): even r where r+2 <= L-1
      - Same-row T(x̃_r, x̃_r):    even r >= 2

    f_D terms (original basis):
      - Same-row T(s_r, s_r):     ALL even r
      - Cross-row T(s_r, s_{r+1}): ALL even r with r+1 < L

    Construction A fuses same-row + skip-row → only for even r >= 2 with r+2 <= L-1.
    Row 0 gets skip-row only (no same-row in parity basis per formula).
    Construction B fuses same-row + cross-row → for all even-odd pairs.
    """
    sq = make_system_qubits(L)
    ops = []

    # Phase 1: Enter column-parity basis
    ops.extend(column_parity_cascade_ops(sq, L, inverse=False))

    # Phase 2: Parity-basis interactions
    # Categorize rows:
    fused_rows = [r for r in range(2, L, 2) if r + 2 <= L - 1]   # same + skip
    skip_only = [0] if 2 <= L - 1 else []                         # row 0: skip only
    same_only_parity = [r for r in range(2, L, 2) if r + 2 > L - 1]  # same only

    # Batch by r mod 4 for conflict-free parallel execution
    # Batch 1: r ≡ 0 (mod 4)
    batch1_fused = [r for r in fused_rows if r % 4 == 0]
    batch1_skip_only = [r for r in skip_only if r % 4 == 0]

    for r in batch1_fused:
        ops.extend(pipeline_same_skip_ops(sq, r, L))
    for r in batch1_skip_only:
        ops.extend(pipeline_skip_only_ops(sq, r, L))

    # same_only_parity rows that don't conflict with batch 1
    batch1_rows_used = set()
    for r in batch1_fused:
        batch1_rows_used.update([r, r + 1, r + 2])
    for r in batch1_skip_only:
        batch1_rows_used.update([r, r + 1, r + 2])
    batch1_same_done = []
    for r in same_only_parity:
        if r not in batch1_rows_used:
            ops.extend(same_row_T_prefix_ops(sq, r, L))
            batch1_same_done.append(r)

    # Batch 2: r ≡ 2 (mod 4)
    batch2_fused = [r for r in fused_rows if r % 4 == 2]
    batch2_skip_only = [r for r in skip_only if r % 4 == 2]

    for r in batch2_fused:
        ops.extend(pipeline_same_skip_ops(sq, r, L))
    for r in batch2_skip_only:
        ops.extend(pipeline_skip_only_ops(sq, r, L))

    # Remaining same_only_parity rows
    batch2_rows_used = set()
    for r in batch2_fused:
        batch2_rows_used.update([r, r + 1, r + 2])
    for r in batch2_skip_only:
        batch2_rows_used.update([r, r + 1, r + 2])
    for r in same_only_parity:
        if r not in batch1_same_done and r not in batch2_rows_used:
            ops.extend(same_row_T_prefix_ops(sq, r, L))

    # Phase 3: Exit column-parity basis
    ops.extend(column_parity_cascade_ops(sq, L, inverse=True))

    # Phase 4: Original-basis interactions (f_D)
    # Construction B fuses same-row + cross-row for each even-odd pair
    for r in range(0, L - 1, 2):
        ops.extend(pipeline_same_cross_ops(sq, r, L))

    # Last even row if L is odd (no cross-row partner, same-row only)
    if L % 2 == 1:
        r_last = L - 1
        ops.extend(same_row_T_prefix_ops(sq, r_last, L))

    circuit = cirq.Circuit(ops)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]
    return circuit, sys_list


def verify_property_star(phase_fn, L, num_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    N = L * L
    checked = 0
    passed = 0
    for _ in range(num_samples):
        bits = rng.integers(0, 2, size=N)
        s_idx = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))
        gamma_s = phase_fn(s_idx)
        for r_hop in range(L - 1):
            for c in range(L):
                i_rc = r_hop * L + c
                i_rc1 = (r_hop + 1) * L + c
                if bits[i_rc] == bits[i_rc1]:
                    continue
                s_prime_idx = s_idx ^ (1 << (N - 1 - i_rc)) ^ (1 << (N - 1 - i_rc1))
                gamma_s_prime = phase_fn(s_prime_idx)
                between = sites_between(r_hop, c, r_hop + 1, L)
                P = 0
                for site_snake in between:
                    sr, sc = snake_to_rc(site_snake, L)
                    raster_idx = sr * L + sc
                    P ^= int(bits[raster_idx])
                expected = (-1) ** P
                actual = gamma_s * gamma_s_prime
                checked += 1
                if abs(actual - expected) < 1e-6:
                    passed += 1
    return checked, passed


def verify_full_gamma(L_values=range(3, 10)):
    """Compare pipelined Gamma against existing Gamma."""
    print("\n=== Step 4: Verify Full Pipelined Gamma ===")
    for L in L_values:
        sq_dummy = make_system_qubits(L)
        qubit_order = [sq_dummy[(r, c)] for r in range(L) for c in range(L)]
        N = L * L

        # Existing
        c_exist, sys_exist = build_gamma_existing(L)
        # Pipelined
        c_pipe, sys_pipe = build_gamma_pipelined(L)

        n_samples = 500
        rng = np.random.default_rng(42)
        mismatches = 0

        for _ in range(n_samples):
            bits = rng.integers(0, 2, size=N)
            s = sum(int(b) << (N - 1 - i) for i, b in enumerate(bits))
            p_exist = get_phase_classical(c_exist, sys_exist, s)
            p_pipe = get_phase_classical(c_pipe, sys_pipe, s)
            if abs(p_exist - p_pipe) > 1e-6:
                mismatches += 1
                if mismatches <= 5:
                    print(f"    L={L} MISMATCH: state={s}, exist={p_exist}, pipe={p_pipe}")

        # Property star
        phase_fn = lambda s, circ=c_pipe, ql=sys_pipe: get_phase_classical(circ, ql, s)
        checked, passed_star = verify_property_star(phase_fn, L, num_samples=min(200, n_samples))

        depth = len(c_pipe)
        depth_exist = len(c_exist)

        match_str = "PASS" if mismatches == 0 else f"FAIL ({mismatches}/{n_samples})"
        star_str = f"{passed_star}/{checked}"
        print(f"  L={L}: match={match_str}, prop*={star_str}, depth_pipe={depth}, depth_exist={depth_exist}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    verify_same_row_T_prefix()
    verify_construction_B()
    verify_pipeline_skip_only()
    verify_construction_A()
    verify_full_gamma()

    print("\n=== Depth Comparison ===")
    print(f"{'L':>3} | {'Existing':>10} | {'Pipelined':>10} | {'8L+?':>6}")
    for L in range(3, 16):
        c_e, _ = build_gamma_existing(L)
        c_p, _ = build_gamma_pipelined(L)
        d_e = len(c_e)
        d_p = len(c_p)
        print(f"{L:3d} | {d_e:10d} | {d_p:10d} | {d_p - 8*L:+6d}")

    # Per-phase depth breakdown
    print("\n=== Per-Phase Depth Breakdown ===")
    for L in [5, 7, 9, 11, 15]:
        sq = make_system_qubits(L)
        phase1 = cirq.Circuit(column_parity_cascade_ops(sq, L, inverse=False))
        phase3 = cirq.Circuit(column_parity_cascade_ops(sq, L, inverse=True))

        # Phase 2 ops
        fused_rows = [r for r in range(2, L, 2) if r + 2 <= L - 1]
        skip_only = [0] if 2 <= L - 1 else []
        same_only_parity = [r for r in range(2, L, 2) if r + 2 > L - 1]

        # Batch 1
        b1_ops = []
        for r in [r for r in fused_rows if r % 4 == 0]:
            b1_ops.extend(pipeline_same_skip_ops(sq, r, L))
        for r in [r for r in skip_only if r % 4 == 0]:
            b1_ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))
        b1_rows_used = set()
        for r in [r for r in fused_rows if r % 4 == 0]:
            b1_rows_used.update([r, r + 1, r + 2])
        for r in [r for r in skip_only if r % 4 == 0]:
            b1_rows_used.update([r, r + 1, r + 2])
        b1_same_done = []
        for r in same_only_parity:
            if r not in b1_rows_used:
                b1_ops.extend(same_row_T_prefix_ops(sq, r, L))
                b1_same_done.append(r)
        phase2_b1 = cirq.Circuit(b1_ops)

        b2_ops = []
        for r in [r for r in fused_rows if r % 4 == 2]:
            b2_ops.extend(pipeline_same_skip_ops(sq, r, L))
        for r in [r for r in skip_only if r % 4 == 2]:
            b2_ops.extend(skip_row_T_ops(sq, r, r + 2, r + 1, L))
        b2_rows_used = set()
        for r in [r for r in fused_rows if r % 4 == 2]:
            b2_rows_used.update([r, r + 1, r + 2])
        for r in [r for r in skip_only if r % 4 == 2]:
            b2_rows_used.update([r, r + 1, r + 2])
        for r in same_only_parity:
            if r not in b1_same_done and r not in b2_rows_used:
                b2_ops.extend(same_row_T_prefix_ops(sq, r, L))
        phase2_b2 = cirq.Circuit(b2_ops)

        # Phase 4 ops
        p4_ops = []
        for r in range(0, L - 1, 2):
            p4_ops.extend(pipeline_same_cross_ops(sq, r, L))
        if L % 2 == 1:
            p4_ops.extend(same_row_T_prefix_ops(sq, L - 1, L))
        phase4 = cirq.Circuit(p4_ops)

        d1 = len(phase1)
        d2b1 = len(phase2_b1)
        d2b2 = len(phase2_b2)
        d3 = len(phase3)
        d4 = len(phase4)
        total_naive = d1 + d2b1 + d2b2 + d3 + d4
        c_p, _ = build_gamma_pipelined(L)
        d_actual = len(c_p)

        print(f"  L={L}: P1={d1}, P2b1={d2b1}, P2b2={d2b2}, P3={d3}, P4={d4}, "
              f"naive_sum={total_naive}, actual={d_actual}, overlap={total_naive-d_actual}")
        print(f"    Expected: P1={L-1}, P2_each~2L+O(1), P3={L-1}, P4~2L+O(1)")
