"""Baseline 4: Pipelined ancilla-free Gamma (best construction).

Fuses same-row T(x,x) with cross-row/skip-row T(x,y) into shared prefix
cascade sweeps.  This is the best known construction for ancilla-free Gamma.

Depth scaling
-------------
    8L + 9   for odd  L >= 5
    8L + 10  for even L >= 6
    (verified for L = 3..50)

Ancillas: 0
Gate count: O(N)

Cirq greedy scheduling achieves zero additional savings -- the manual
pipelining already saturates all available parallelism.  (Verified: feeding
all ops to one cirq.Circuit() produces identical depth.)

Comparison with Baseline 3 (sequential primitives, 12L + 8):
    The 4L saving comes from fusing same-row T with skip/cross-row T into
    shared prefix cascade sweeps (Constructions A and B below).  The
    interaction gates trail the cascade wavefront at fixed column offsets,
    so multiple T(x,y) terms sharing the same source row fold into a single
    forward-and-back sweep.

Two pipelined constructions:
    Construction A (PipelineSameSkip): same-row T + skip-row T in parity basis
    Construction B (PipelineSameCross): same-row T + cross-row T in original basis
"""

from typing import Dict, List, Tuple

import pennylane as qp

from common.grid import column_parity_cascade_ops, make_system_qubits
from common.gamma_primitive import prefix_cascade_ops, undo_prefix_cascade_ops


# ---------------------------------------------------------------------------
# Prefix-based same-row T(x,x)
# ---------------------------------------------------------------------------

def same_row_T_prefix_ops(sq: Dict, r: int, L: int) -> List[qp.ops.Operation]:
    """Same-row T(x,x) using prefix cascade (instead of suffix).

    After prefix cascade, position c holds x_hat_c = XOR_{c'<=c} x_{c'}.
    CZ(c, c+1) contributes x_hat_c * x_hat_{c+1} = x_hat_c * x_{c+1} + x_hat_c.
    Degree-2 sum = T(x,x).  Degree-1 residual = sum_p (L-1-p) x_p.
    Z correction: column p where (L-1-p) is odd.
    """
    ops = []
    ops.extend(prefix_cascade_ops(sq, r, L))
    for c in range(L - 1):
        ops.append(qp.CZ(sq[(r, c)] + sq[(r, c + 1)]))
    ops.extend(undo_prefix_cascade_ops(sq, r, L))
    for p in range(L - 1):
        if (L - 1 - p) % 2 == 1:
            ops.append(qp.Z(sq[(r, p)]))
    return ops


# ---------------------------------------------------------------------------
# Construction B: PipelineSameCross (same-row T + cross-row T)
# ---------------------------------------------------------------------------

def pipeline_same_cross_ops(sq: Dict, r: int, L: int) -> List[qp.ops.Operation]:
    """Fused same-row T(x,x) + cross-row T(x,y) for even row r, odd row r+1.

    Forward: cascade CNOT at offset 0, cross-row CZ at -2, same-row CZ at -4/-3.
    Undo:    cross-row CZ correction at +2, Z correction at +3.
    """
    ops = []
    r2 = r + 1
    max_fwd = L - 2 + 4
    for tau in range(max_fwd + 1):
        if 0 <= tau <= L - 2:
            ops.append(qp.CNOT(sq[(r, tau)] + sq[(r, tau + 1)]))
        c = tau - 2
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r, c)] + sq[(r2, c)]))
        c_lo, c_hi = tau - 4, tau - 3
        if 0 <= c_lo and c_hi < L:
            ops.append(qp.CZ(sq[(r, c_lo)] + sq[(r, c_hi)]))
    max_undo_extra = 3
    for tau in range(L - 2, -1 - max_undo_extra - 1, -1):
        if 0 <= tau <= L - 2:
            ops.append(qp.CNOT(sq[(r, tau)] + sq[(r, tau + 1)]))
        c = tau + 2
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r, c)] + sq[(r2, c)]))
        c = tau + 3
        if 0 <= c < L and (L - 1 - c) % 2 == 1:
            ops.append(qp.Z(sq[(r, c)]))
    return ops


# ---------------------------------------------------------------------------
# Construction A: PipelineSameSkip (same-row T + skip-row T)
# ---------------------------------------------------------------------------

def pipeline_same_skip_ops(sq: Dict, r: int, L: int) -> List[qp.ops.Operation]:
    """Fused same-row T(x~,x~) + skip-row T(x~, y~) for rows r, r+1, r+2.

    Forward: cascade at 0; skip gadget at -1..-4; same-row CZ at -6/-5.
    Undo:    skip correction at +1..+4; Z correction at +5.
    """
    ops = []
    r_mid, r2 = r + 1, r + 2
    max_fwd = L - 2 + 6
    for tau in range(max_fwd + 1):
        if 0 <= tau <= L - 2:
            ops.append(qp.CNOT(sq[(r, tau)] + sq[(r, tau + 1)]))
        c = tau - 1
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau - 2
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
        c = tau - 3
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau - 4
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
        c_lo, c_hi = tau - 6, tau - 5
        if 0 <= c_lo and c_hi < L:
            ops.append(qp.CZ(sq[(r, c_lo)] + sq[(r, c_hi)]))
    max_undo_extra = 5
    for tau in range(L - 2, -1 - max_undo_extra - 1, -1):
        if 0 <= tau <= L - 2:
            ops.append(qp.CNOT(sq[(r, tau)] + sq[(r, tau + 1)]))
        c = tau + 1
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau + 2
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
        c = tau + 3
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau + 4
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
        c = tau + 5
        if 0 <= c < L and (L - 1 - c) % 2 == 1:
            ops.append(qp.Z(sq[(r, c)]))
    return ops


# ---------------------------------------------------------------------------
# Skip-only pipeline (row 0 in parity basis -- no same-row T per f_B formula)
# ---------------------------------------------------------------------------

def pipeline_skip_only_ops(sq: Dict, r: int, L: int) -> List[qp.ops.Operation]:
    """Skip-row T only (no same-row T). For row 0 in parity basis.

    f_B excludes same-row T for r=0.  Same pipeline structure as
    Construction A but without same-row CZ (-6/-5) and Z correction (+5).
    """
    ops = []
    r_mid, r2 = r + 1, r + 2
    # max_fwd must reach (L-1) + 4 to cover the 4th gadget step for the last column
    max_fwd = L - 1 + 4
    for tau in range(max_fwd + 1):
        if 0 <= tau <= L - 2:
            ops.append(qp.CNOT(sq[(r, tau)] + sq[(r, tau + 1)]))
        c = tau - 1
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau - 2
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
        c = tau - 3
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau - 4
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
    max_undo_extra = 4
    for tau in range(L - 2, -1 - max_undo_extra - 1, -1):
        if 0 <= tau <= L - 2:
            ops.append(qp.CNOT(sq[(r, tau)] + sq[(r, tau + 1)]))
        c = tau + 1
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau + 2
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
        c = tau + 3
        if 0 <= c < L:
            ops.append(qp.CZ(sq[(r_mid, c)] + sq[(r2, c)]))
        c = tau + 4
        if 0 <= c < L:
            ops.append(qp.CNOT(sq[(r, c)] + sq[(r_mid, c)]))
    return ops


# ---------------------------------------------------------------------------
# Full Pipelined Gamma (Baseline 4)
# ---------------------------------------------------------------------------

def build_gamma_pipelined(L: int, sq=None):
    """Build the pipelined Gamma circuit (Baseline 4, best construction).

    Depth: 8L + 9 (odd L >= 5) or 8L + 10 (even L >= 6), 0 ancillas.

    Four phases:
        Phase 1: Column parity cascade forward          (L-1)
        Phase 2: Parity-basis interactions f_B           (~4L + O(1), 2 batches)
        Phase 3: Column parity cascade inverse           (L-1)
        Phase 4: Original-basis interactions f_D          (~2L + O(1))

    Args:
        L: grid side length
        sq: optional system qubit dict {(r,c): Qid}.  Created internally if None.

    Returns:
        (circuit, sys_list)
    """
    if sq is None:
        sq = make_system_qubits(L)
    ops = []

    # Phase 1
    ops.extend(column_parity_cascade_ops(sq, L, inverse=False))

    # Phase 2
    fused_rows = [r for r in range(2, L, 2) if r + 2 <= L - 1]
    skip_only = [0] if 2 <= L - 1 else []
    same_only_parity = [r for r in range(2, L, 2) if r + 2 > L - 1]

    # Batch 1: r % 4 == 0
    batch1_fused = [r for r in fused_rows if r % 4 == 0]
    batch1_skip = [r for r in skip_only if r % 4 == 0]
    for r in batch1_fused:
        ops.extend(pipeline_same_skip_ops(sq, r, L))
    for r in batch1_skip:
        ops.extend(pipeline_skip_only_ops(sq, r, L))
    b1_rows = set()
    for r in batch1_fused:
        b1_rows.update([r, r + 1, r + 2])
    for r in batch1_skip:
        b1_rows.update([r, r + 1, r + 2])
    b1_same_done = []
    for r in same_only_parity:
        if r not in b1_rows:
            ops.extend(same_row_T_prefix_ops(sq, r, L))
            b1_same_done.append(r)

    # Batch 2: r % 4 == 2
    batch2_fused = [r for r in fused_rows if r % 4 == 2]
    batch2_skip = [r for r in skip_only if r % 4 == 2]
    for r in batch2_fused:
        ops.extend(pipeline_same_skip_ops(sq, r, L))
    for r in batch2_skip:
        ops.extend(pipeline_skip_only_ops(sq, r, L))
    b2_rows = set()
    for r in batch2_fused:
        b2_rows.update([r, r + 1, r + 2])
    for r in batch2_skip:
        b2_rows.update([r, r + 1, r + 2])
    for r in same_only_parity:
        if r not in b1_same_done and r not in b2_rows:
            ops.extend(same_row_T_prefix_ops(sq, r, L))

    # Phase 3
    ops.extend(column_parity_cascade_ops(sq, L, inverse=True))

    # Phase 4
    for r in range(0, L - 1, 2):
        ops.extend(pipeline_same_cross_ops(sq, r, L))
    if L % 2 == 1:
        ops.extend(same_row_T_prefix_ops(sq, L - 1, L))

    circuit = qp.tape.qscript.QuantumScript(ops)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]
    return circuit, sys_list
