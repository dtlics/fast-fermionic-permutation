"""Trotter step circuit builders for sparse SYK simulation.

Two approaches:
1. FP-based: color groups -> FP(pack) -> local Pauli rotations -> FP(unpack)
2. Naive Pauli: directly implement each chi_i chi_j chi_k chi_l as a
   (potentially long-range) Pauli exponential via CNOT staircase.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cirq
import numpy as np

from common.fp_1d import build_fp_1d
from common.fp_2d import FPResult, GammaMethod, build_fp_2d
from common.grid import snake_order_indices, snake_to_rc
from common.metrics import count_resources

from exp3_syk.syk_instance import SYKInstance, invert_permutation


# ---------------------------------------------------------------------------
# Pauli algebra helpers
# ---------------------------------------------------------------------------

# Pauli multiplication table: (result_pauli, phase_factor)
# Convention: P1 * P2 = phase * result
_PAULI_MUL = {
    ("I", "I"): ("I", 1),  ("I", "X"): ("X", 1),  ("I", "Y"): ("Y", 1),  ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1),  ("X", "X"): ("I", 1),   ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1),  ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1),  ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1),  ("Z", "X"): ("Y", 1j),  ("Z", "Y"): ("X", -1j),("Z", "Z"): ("I", 1),
}


def _multiply_paulis(p1: str, p2: str) -> Tuple[str, complex]:
    return _PAULI_MUL[(p1, p2)]


def majorana_to_pauli(
    majorana_indices: Tuple[int, ...],
    mode_to_jw_pos: Dict[int, int],
    L: int,
) -> Tuple[Dict[cirq.GridQubit, str], complex]:
    """Compute the Pauli string for a product of Majorana operators under JW.

    Under JW in snake ordering:
        chi_{2t} = (prod_{s: jw_pos(s) < jw_pos(t)} Z_s) X_{qubit(t)}
        chi_{2t+1} = (prod_{s: jw_pos(s) < jw_pos(t)} Z_s) Y_{qubit(t)}

    Args:
        majorana_indices: tuple of Majorana indices (e.g., (i,j,k,l))
        mode_to_jw_pos: mapping from mode index to JW position (snake position)
        L: grid side length

    Returns:
        (qubit_pauli_map, phase) where qubit_pauli_map maps GridQubit -> "X"/"Y"/"Z"
        and phase is the overall complex coefficient.
    """
    N = L * L
    # Accumulate Pauli on each JW position
    pauli_at_pos: Dict[int, str] = {}  # jw_pos -> Pauli
    overall_phase: complex = 1.0

    for m in majorana_indices:
        t = m // 2
        jw_t = mode_to_jw_pos[t]
        local_op = "X" if m % 2 == 0 else "Y"

        # Z tail: all positions with jw_pos < jw_t
        for mode, jw_pos in mode_to_jw_pos.items():
            if jw_pos < jw_t:
                if jw_pos in pauli_at_pos:
                    result, phase = _multiply_paulis(pauli_at_pos[jw_pos], "Z")
                    overall_phase *= phase
                    pauli_at_pos[jw_pos] = result
                else:
                    pauli_at_pos[jw_pos] = "Z"

        # Local operator on jw_t
        if jw_t in pauli_at_pos:
            result, phase = _multiply_paulis(pauli_at_pos[jw_t], local_op)
            overall_phase *= phase
            pauli_at_pos[jw_t] = result
        else:
            pauli_at_pos[jw_t] = local_op

    # Convert jw positions to GridQubits, drop identities
    snake = snake_order_indices(L)
    qubit_pauli: Dict[cirq.GridQubit, str] = {}
    for jw_pos, pauli in pauli_at_pos.items():
        if pauli != "I":
            raster_idx = snake[jw_pos]
            r, c = raster_idx // L, raster_idx % L
            qubit_pauli[cirq.GridQubit(r, c)] = pauli

    return qubit_pauli, overall_phase


def build_pauli_exp_circuit(
    qubit_pauli: Dict[cirq.GridQubit, str],
    angle: float,
) -> cirq.Circuit:
    """Build exp(-i * angle * P) circuit using CNOT staircase.

    Standard decomposition:
    1. Basis change (H for X, S†H for Y, nothing for Z)
    2. CNOT staircase
    3. Rz(2*angle)
    4. Reverse CNOT staircase
    5. Undo basis change

    Args:
        qubit_pauli: mapping from GridQubit to "X"/"Y"/"Z"
        angle: rotation angle (implements exp(-i * angle * P))

    Returns:
        cirq.Circuit implementing the Pauli exponential
    """
    if not qubit_pauli:
        return cirq.Circuit()

    # Sort qubits for deterministic CNOT ordering
    qubits = sorted(qubit_pauli.keys())
    ops: List[cirq.Operation] = []

    # 1. Basis change
    for q in qubits:
        p = qubit_pauli[q]
        if p == "X":
            ops.append(cirq.H(q))
        elif p == "Y":
            # Ry basis: S†H maps Y eigenstates to Z eigenstates
            ops.append(cirq.inverse(cirq.S(q)))
            ops.append(cirq.H(q))
        # Z: no change

    # 2. CNOT staircase (compute parity into last qubit)
    for i in range(len(qubits) - 1):
        ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    # 3. Rz rotation
    ops.append(cirq.rz(2 * angle)(qubits[-1]))

    # 4. Reverse CNOT staircase
    for i in range(len(qubits) - 2, -1, -1):
        ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    # 5. Undo basis change
    for q in qubits:
        p = qubit_pauli[q]
        if p == "X":
            ops.append(cirq.H(q))
        elif p == "Y":
            ops.append(cirq.H(q))
            ops.append(cirq.S(q))

    return cirq.Circuit(ops)


# ---------------------------------------------------------------------------
# FP-based Trotter step
# ---------------------------------------------------------------------------

def _build_fp(
    L: int,
    perm: List[int],
    baseline_name: str,
    gamma_method: Optional[GammaMethod],
) -> FPResult:
    if baseline_name == "1d":
        return build_fp_1d(L, perm)
    return build_fp_2d(L, perm, gamma_method)


def build_trotter_step_fp(
    instance: SYKInstance,
    baseline_name: str,
    gamma_method: Optional[GammaMethod],
) -> Tuple[cirq.Circuit, int, int, int]:
    """Build one Trotter step using FP-based approach.

    For each color group: FP(pack) + local rotations + FP(unpack).

    Args:
        instance: SYKInstance with coloring and packing permutations
        baseline_name: "1d", "ancilla", "primitive", or "pipelined"
        gamma_method: GammaMethod enum (None for 1d)

    Returns:
        (circuit, n_ancillas, fp_cnot_depth, interaction_cnot_depth)
    """
    L = instance.L
    N = L * L
    snake = snake_order_indices(L)

    total_circuit = cirq.Circuit()
    fp_cnot_depth = 0
    int_cnot_depth = 0
    n_ancillas = 0

    for alpha, group in enumerate(instance.color_groups):
        if not group:
            continue

        perm_fwd = instance.packing_perms[alpha]
        perm_inv = invert_permutation(perm_fwd)

        # Build mode_to_jw_pos AFTER packing: mode t -> snake position of perm_fwd[t]
        raster_to_snake = {snake[s]: s for s in range(N)}
        mode_to_jw_pos = {t: raster_to_snake[perm_fwd[t]] for t in range(N)}

        # Forward FP
        fp_fwd = _build_fp(L, perm_fwd, baseline_name, gamma_method)
        n_ancillas = len(fp_fwd.anc_qubits)  # same for all groups
        fp_res = count_resources(fp_fwd.circuit, L, n_ancillas)
        fp_cnot_depth += fp_res["cnot_depth"]

        # Local Pauli rotations for this color group.
        # All quartets have disjoint qubits, so their rotations can run in
        # parallel.  Collect individual circuits and zip them together.
        rot_circuits = []
        for qi in group:
            quartet = instance.quartets[qi]
            coupling = instance.couplings[qi]
            qp, phase = majorana_to_pauli(quartet, mode_to_jw_pos, L)
            effective_angle = coupling * phase.real if phase.imag == 0 else coupling
            rot_circuits.append(
                build_pauli_exp_circuit(qp, float(np.real(effective_angle)))
            )
        # Zip into parallel moments (disjoint qubits → safe)
        rotation_circuit = cirq.Circuit()
        if rot_circuits:
            max_len = max(len(c) for c in rot_circuits)
            for mi in range(max_len):
                ops = []
                for c in rot_circuits:
                    if mi < len(c):
                        ops.extend(c[mi].operations)
                rotation_circuit.append(cirq.Moment(ops))

        int_res = count_resources(rotation_circuit, L, n_ancillas)
        int_cnot_depth += int_res["cnot_depth"]

        # Inverse FP
        fp_inv = _build_fp(L, perm_inv, baseline_name, gamma_method)
        fp_inv_res = count_resources(fp_inv.circuit, L, n_ancillas)
        fp_cnot_depth += fp_inv_res["cnot_depth"]

        # Assemble this color group
        total_circuit += fp_fwd.circuit
        total_circuit += rotation_circuit
        total_circuit += fp_inv.circuit

    return total_circuit, n_ancillas, fp_cnot_depth, int_cnot_depth


# ---------------------------------------------------------------------------
# Naive Pauli baseline
# ---------------------------------------------------------------------------

def build_trotter_step_naive(
    instance: SYKInstance,
) -> Tuple[cirq.Circuit, int, int, int]:
    """Build one Trotter step using naive Pauli strings (no FP, no coloring).

    Each chi_i chi_j chi_k chi_l is directly converted to a JW Pauli string
    and implemented with a CNOT staircase.

    Returns:
        (circuit, n_ancillas=0, fp_cnot_depth=0, interaction_cnot_depth)
    """
    L = instance.L
    N = L * L
    snake = snake_order_indices(L)

    # In the original (unpermuted) frame, mode t is at snake position t
    # (identity permutation)
    raster_to_snake = {snake[s]: s for s in range(N)}
    mode_to_jw_pos = {t: raster_to_snake[t] for t in range(N)}

    total_circuit = cirq.Circuit()
    for qi, quartet in enumerate(instance.quartets):
        coupling = instance.couplings[qi]
        qp, phase = majorana_to_pauli(quartet, mode_to_jw_pos, L)
        effective_angle = coupling * phase.real if phase.imag == 0 else coupling
        rot_circ = build_pauli_exp_circuit(qp, float(np.real(effective_angle)))
        total_circuit += rot_circ

    int_res = count_resources(total_circuit, L, 0)
    return total_circuit, 0, 0, int_res["cnot_depth"]
