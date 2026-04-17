"""Baseline 2: Gamma operator with ancillas on a physical NN grid.

Ancillas at Wires(r, L) sweep left (Stage B) and right (Stage D)
via advance primitives.  All gates are nearest-neighbor on the
(L+1)-column physical grid.

Depth scaling
-------------
Without cross-step pipelining (step-separated construction):
    Stage A:  L - 1     (column parity cascade)
    Stage B:  10L       (leftward sweep, 10 per step)
    Stage C:  L - 1     (undo column parity + ancilla cascade)
    Stage D:  6L + 1    (rightward sweep, 6 per step, + diagonal correction)
    Total:    18L - 1   (exact for L >= 7)
"""

from typing import List

import pennylane as qp

from common.grid import make_ancilla_qubits, make_system_qubits


# ---------------------------------------------------------------------------
# Advance primitives
# ---------------------------------------------------------------------------

def _left_advance_ops(r: int, p: int) -> List[qp.operation.Operation]:
    """Left-advance (2 CNOTs): SWAP + parity accumulation fused.

    Ancilla moves from (r, p+1) to (r, p), accumulating parity.
    """
    return [
        qp.CNOT(qp.wires.Wires([f"({r}, {p + 1})", f"({r}, {p})"])),
        qp.CNOT(qp.wires.Wires([f"({r}, {p})", f"({r}, {p + 1})"])),
    ]


def _right_advance_ops(r: int, p: int) -> List[qp.operation.Operation]:
    """Right-advance (2 CNOTs): SWAP + parity accumulation fused.

    Ancilla moves from (r, p) to (r, p+1), accumulating parity.
    """
    return [
        qp.CNOT(qp.wires.Wires([f"({r}, {p})", f"({r}, {p + 1})"])),
        qp.CNOT(qp.wires.Wires([f"({r}, {p + 1})", f"({r}, {p})"])),
    ]


def _bare_swap_ops(r: int, p: int) -> List[qp.operation.Operation]:
    """Bare SWAP (3 CNOTs): move ancilla without parity accumulation.

    Ancilla moves from (r, p+1) to (r, p). Value unchanged.
    """
    return [
        qp.CNOT(qp.wires.Wires([f"({r}, {p + 1})", f"({r}, {p})"])),
        qp.CNOT(qp.wires.Wires([f"({r}, {p})", f"({r}, {p + 1})"])),
        qp.CNOT(qp.wires.Wires([f"({r}, {p + 1})", f"({r}, {p})"])),
    ]


# ---------------------------------------------------------------------------
# Skip-row CZ gadget: distance-2 CZ via 4 NN gates through intermediary
# ---------------------------------------------------------------------------

def _skip_cz_gadget_ops(r: int, r2: int, col: int) -> List[qp.operation.Operation]:
    """CZ between rows r and r2 (distance 2) at column col.

    Routes through intermediary at (r+1, col).  All 4 gates are vertical NN.
    Net phase: A*C (where A = value at (r,col), C = value at (r2,col)).
    Intermediary fully restored.
    """
    mid = r + 1
    A = qp.wires.Wires([f"({r}, {col})"])
    B = qp.wires.Wires([f"({mid}, {col})"])
    C = qp.wires.Wires([f"({r2}, {col})"])
    return [
        qp.CNOT(C + B),   # B <- B XOR C
        qp.CZ(A + B),     # phase += A * (B XOR C)
        qp.CNOT(C + B),   # B restored
        qp.CZ(A + B),     # phase += A * B
        # Net: A*(B XOR C) + A*B = A*C
    ]


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def _build_stage_A(L: int) -> List[qp.operation.Operation]:
    """Stage A: enter column parity basis.  Data at columns 0..L-1."""
    ops = []
    for r in range(L - 2, -1, -1):
        for c in range(L):
            ops.append(qp.CNOT(qp.wires.Wires([f"({r + 1}, {c})", f"({r}, {c})"])))
    return ops


def _build_stage_B_step(p: int, L: int) -> List[qp.operation.Operation]:
    """One column step of Stage B (leftward sweep).

    At the start of this step, all ancillas are at column p+1.
    After this step, all ancillas are at column p, holding accumulated
    parity through column p.

    Three substeps per the spec:

    Substep 1 -- Batch B bare SWAP (3 CNOTs):
        Batch B (r % 4 == 2) ancillas move to column p WITHOUT parity
        accumulation.  After: batch B ancilla at (r', p), sys at (r', p+1).
        Batch A and odd rows unchanged.

    Substep 2 -- All CZ gates (no ordering, compiler schedules):
        - Skip-row A->B: CZ(sys_r at (r, p), a_{r+2} at (r+2, p))
          for r % 4 == 0, r+2 <= L-1.  Same column p, distance 2.
        - Skip-row B->A: CZ(sys_{r'} at (r', p+1), a_{r'+2} at (r'+2, p+1))
          for r' % 4 == 2, r'+2 <= L-1.  Same column p+1, distance 2.
        - Same-row A: CZ(sys_r at (r, p), a_r at (r, p+1))
          for each even r >= 2 with r % 4 == 0.  Horizontal NN.
        - Same-row B: CZ(sys_{r'} at (r', p+1), a_{r'} at (r', p))
          for each even r' >= 2 with r' % 4 == 2.  Horizontal NN.

    Substep 3 -- Wrap-up:
        - Batch A + odd rows: standard 2-CNOT left-advance.
        - Batch B: 1-CNOT parity accumulation:
          CNOT(ctrl=(r', p+1), tgt=(r', p))
    """
    ops = []

    # --- Substep 1: batch B bare SWAP ---
    for r in range(2, L, 4):
        ops.extend(_bare_swap_ops(r, p))

    # --- Substep 2: all CZ gates ---

    # Skip-row A->B at column p (distance 2, gadget through odd row)
    for r in range(0, L, 4):
        if r + 2 <= L - 1:
            ops.extend(_skip_cz_gadget_ops(r, r + 2, p))

    # Skip-row B->A at column p+1 (distance 2, gadget through odd row)
    for r in range(2, L, 4):
        if r + 2 <= L - 1:
            ops.extend(_skip_cz_gadget_ops(r, r + 2, p + 1))

    # Same-row A: CZ(sys at (r, p), anc at (r, p+1))
    for r in range(0, L, 4):
        if r >= 2:
            ops.append(qp.CZ(qp.wires.Wires([f"({r}, {p})", f"({r}, {p + 1})"])))

    # Same-row B: CZ(sys at (r', p+1), anc at (r', p))
    for r in range(2, L, 4):
        if r >= 2:
            ops.append(qp.CZ(qp.wires.Wires([f"({r}, {p + 1})", f"({r}, {p})"])))

    # --- Substep 3: wrap-up ---

    # Batch A + odd rows: 2-CNOT left-advance
    for r in range(L):
        if r % 4 == 0 or r % 2 == 1:
            ops.extend(_left_advance_ops(r, p))

    # Batch B: 1-CNOT parity accumulation
    for r in range(2, L, 4):
        ops.append(qp.CNOT(qp.wires.Wires([f"({r}, {p + 1})", f"({r}, {p})"])))

    return ops


def _build_stage_C(L: int) -> List[qp.operation.Operation]:
    """Stage C: exit column parity basis.

    After Stage B, data is displaced to columns 1..L and ancillas at column 0.
    System cascade: top-down on columns 1..L.
    Ancilla cascade: top-down on column 0.
    Both are on disjoint qubits and can be scheduled in parallel by qp.
    """
    ops = []
    # System cascade (top-down) on displaced columns 1..L
    for r in range(L - 1):
        for c in range(1, L + 1):
            ops.append(qp.CNOT(qp.wires.Wires([f"({r + 1}, {c})", f"({r}, {c})"])))
    # Ancilla cascade (top-down) on column 0
    for r in range(L - 1):
        ops.append(qp.CNOT(qp.wires.Wires([f"({r + 1}, 0)", f"({r}, 0)"])))
    return ops


def _build_stage_D_step(p: int, L: int) -> List[qp.operation.Operation]:
    """One column step of Stage D (rightward sweep).

    At the start of this step, all ancillas are at column p.
    After this step, all ancillas are at column p+1.

    Three substeps:

    Substep 1 -- Even rows right-advance (2 CNOTs):
        After: a_r at (r, p+1), sys_r at (r, p).
        Odd rows unchanged: a_{r+1} at (r+1, p).

    Substep 2 -- CZ gates (compiler schedules):
        - Same-row: CZ(sys_r at (r, p), a_r at (r, p+1))  horizontal NN
        - Cross-row: CZ(sys_r at (r, p), a_{r+1} at (r+1, p))  vertical NN
        for each even r with r+1 <= L-1.

    Substep 3 -- Odd rows right-advance (2 CNOTs).
    """
    ops = []

    # Substep 1: even rows right-advance
    for r in range(0, L, 2):
        ops.extend(_right_advance_ops(r, p))

    # Substep 2: CZ gates
    for r in range(0, L, 2):
        # Same-row CZ
        ops.append(qp.CZ(qp.wires.Wires([f"({r}, {p})", f"({r}, {p + 1})"])))
        # Cross-row CZ
        if r + 1 <= L - 1:
            ops.append(qp.CZ(qp.wires.Wires([f"({r}, {p})", f"({r + 1}, {p})"])))

    # Substep 3: odd rows right-advance
    for r in range(1, L, 2):
        ops.extend(_right_advance_ops(r, p))

    return ops


# ---------------------------------------------------------------------------
# Full builder
# ---------------------------------------------------------------------------

def build_gamma_with_ancillas(L: int, sq=None, aq=None):
    """Build Gamma with ancillas on a physical NN grid (Baseline 2).

    System qubits at Wires(r, 0..L-1), ancillas at Wires(r, L).
    All gates are nearest-neighbor.  Ancillas physically sweep left then
    right via advance primitives.

    Args:
        L: grid side length
        sq: optional system qubit dict (unused in physical construction,
            kept for API compatibility).
        aq: optional ancilla qubit dict (unused in physical construction,
            kept for API compatibility).

    Returns:
        (circuit, sys_list, anc_list) where:
          sys_list = [Wires(r,c) for r,c in raster order]
          anc_list = [Wires(r,L) for r in range(L)]
          Ancillas must be initialised to |0> and return to |0>.
    """
    # Stage A: column parity cascade (no cross-stage merging with B)
    stage_a = _build_stage_A(L)

    # Stage B: leftward sweep.  All steps fed to one qp.tape.qscript.QuantumScript() so
    # the greedy scheduler can pipeline across step boundaries.
    # Savings: substep 3 of step p (batch A + odd advance) overlaps with
    # substep 1 of step p-1 (batch B SWAP) on disjoint rows.  ~2/step.
    stage_b_ops = []
    for p in range(L - 1, -1, -1):
        stage_b_ops.extend(_build_stage_B_step(p, L))
    stage_b = stage_b_ops

    # Stage C: undo column parity (no cross-stage merging)
    stage_c = _build_stage_C(L)

    # Stage D: rightward sweep.  All steps fed to one qp.Circuit() so
    # the greedy scheduler can pipeline across step boundaries.
    # Savings: substep 3 of step p (odd advance) fully overlaps with
    # substep 1 of step p+1 (even advance) on disjoint rows.  ~3/step.
    stage_d_ops = []
    for p in range(L):
        stage_d_ops.extend(_build_stage_D_step(p, L))

    # Diagonal correction: Stage D's cross-row CZ produces T(s_{r+1}, s_r)
    # instead of T(s_r, s_{r+1}).  The difference is the diagonal:
    # ⊕_c s_{r,c} * s_{r+1,c}.  Correct with vertical CZ per column.
    # After Stage D, data is back at columns 0..L-1, so these are all NN.
    # Appended to stage_d_ops so the greedy scheduler can absorb it into
    # the last step's wrap-up moments.
    for r in range(0, L - 1, 2):
        for c in range(L):
            stage_d_ops.append(qp.CZ(qp.wires.Wires([f"({r}, {c})", f"({r + 1}, {c})"])))
    stage_d = stage_d_ops

    circuit = qp.tape.qscript.QuantumScript(stage_a + stage_b + stage_c + stage_d)

    sys_list = [qp.wires.Wires(f"({r}, {c})") for r in range(L) for c in range(L)]
    anc_list = [qp.wires.Wires(f"({r}, {L})") for r in range(L)]
    return circuit, sys_list, anc_list
