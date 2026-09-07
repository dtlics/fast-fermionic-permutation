"""Trotter step circuit builders for sparse SYK simulation.

Two approaches:
1. FP-based: color groups -> FP(pack) -> local Pauli rotations -> FP(unpack)
2. Naive Pauli: directly implement each chi_i chi_j chi_k chi_l as a
   nearest-neighbor Pauli exponential along the Jordan--Wigner snake.
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


PAULI_ROTATION_COMPILER_MODEL = "snake_nn_dirty_bridge_sweep_v1"


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


def _snake_qubits(L: int) -> List[cirq.GridQubit]:
    """Return the physical grid qubits in Jordan--Wigner snake order."""

    return [
        cirq.GridQubit(*divmod(raster_index, L))
        for raster_index in snake_order_indices(L)
    ]


def assert_nearest_neighbor_circuit(circuit: cirq.Circuit) -> None:
    """Fail closed unless every two-qubit operation is square-grid local."""

    for moment_index, moment in enumerate(circuit):
        for operation in moment:
            if len(operation.qubits) > 2:
                raise ValueError(
                    "undecomposed multi-qubit SYK operation: "
                    f"moment={moment_index}, operation={operation!r}"
                )
            if len(operation.qubits) < 2:
                continue
            a, b = operation.qubits
            if not isinstance(a, cirq.GridQubit) or not isinstance(
                b, cirq.GridQubit
            ):
                raise ValueError(
                    "two-qubit SYK operation is not on GridQubits: "
                    f"moment={moment_index}, operation={operation!r}"
                )
            distance = abs(a.row - b.row) + abs(a.col - b.col)
            if distance != 1:
                raise ValueError(
                    "non-nearest-neighbor SYK operation: "
                    f"moment={moment_index}, distance={distance}, "
                    f"operation={operation!r}"
                )


def build_pauli_exp_circuit(
    qubit_pauli: Dict[cirq.GridQubit, str],
    angle: float,
    L: int,
) -> cirq.Circuit:
    """Build ``exp(-i * angle * P)`` using only nearest-neighbor gates.

    Standard decomposition:
    1. Basis change (H for X, S†H for Y, nothing for Z)
    2. Sweep the support parity through dirty bridge qubits to one endpoint
    3. Rz(2*angle)
    4. Reverse the complete parity network
    5. Undo basis change

    If the support has ``k`` sites and its snake interval contains ``b``
    unsupported bridge sites, the compute network uses ``k - 1 + 2b`` CNOTs;
    compute plus uncompute therefore uses ``2(k - 1 + 2b)``.  Bridge qubits
    need no initialization and are restored exactly.

    Args:
        qubit_pauli: mapping from GridQubit to "X"/"Y"/"Z"
        angle: rotation angle (implements exp(-i * angle * P))
        L: side length of the square grid

    Returns:
        cirq.Circuit implementing the Pauli exponential
    """
    if not qubit_pauli:
        return cirq.Circuit()

    if L < 1:
        raise ValueError("L must be positive")
    snake = _snake_qubits(L)
    position = {qubit: index for index, qubit in enumerate(snake)}
    unknown = sorted(set(qubit_pauli) - set(position))
    if unknown:
        raise ValueError(f"Pauli support lies outside the {L}x{L} grid: {unknown}")
    for pauli in qubit_pauli.values():
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError(f"unsupported Pauli label {pauli!r}")

    # Sorting GridQubits lexicographically is not the physical JW path on odd
    # rows and can create diagonal or long-range CNOTs.  Use the actual snake.
    qubits = sorted(qubit_pauli, key=position.__getitem__)

    # 1. Basis change
    basis_in = cirq.Circuit()
    basis_out = cirq.Circuit()
    for q in qubits:
        p = qubit_pauli[q]
        if p == "X":
            basis_in.append(cirq.H(q))
            basis_out.append(cirq.H(q))
        elif p == "Y":
            # Ry basis: S†H maps Y eigenstates to Z eigenstates
            basis_in.append(cirq.inverse(cirq.S(q)))
            basis_in.append(cirq.H(q))
            basis_out.append(cirq.H(q))
            basis_out.append(cirq.S(q))
        # Z: no change

    # 2. Reduce the support indicator from right to left.  When the site just
    # left of the moving frontier is absent from the Pauli string, the first
    # CNOT introduces it as a dirty bridge; the second CNOT cancels the old
    # frontier.  In the Heisenberg picture this leaves Z on exactly the
    # requested support while all bridge data are restored by uncomputation.
    left = position[qubits[0]]
    right = position[qubits[-1]]
    active = {position[qubit] for qubit in qubits}
    compute_ops: List[cirq.Operation] = []
    for index in range(right, left, -1):
        if index - 1 not in active:
            compute_ops.append(cirq.CNOT(snake[index - 1], snake[index]))
            active.add(index - 1)
        compute_ops.append(cirq.CNOT(snake[index], snake[index - 1]))
        active.discard(index)
    if active != {left}:
        raise AssertionError("dirty-bridge parity sweep did not reach one root")
    compute = cirq.Circuit(
        compute_ops, strategy=cirq.InsertStrategy.NEW
    )
    global_target = snake[left]

    # 4. Rotate the accumulated parity and restore every data qubit.
    circuit = (
        basis_in
        + compute
        + cirq.Circuit(cirq.rz(2 * angle)(global_target))
        + cirq.inverse(compute)
        + basis_out
    )

    assert_nearest_neighbor_circuit(circuit)
    return circuit


def _assert_disjoint_packed_rotation_spans(
    rotations: List[Tuple[int, Dict[cirq.GridQubit, str], cirq.Circuit]],
    L: int,
    color_index: int,
) -> None:
    """Require the complete dirty-bridge footprints to be pairwise disjoint.

    Disjoint fermionic quartets alone are not sufficient for moment-wise
    zipping: a nearest-neighbor Pauli compiler can also touch every bridge site
    between the leftmost and rightmost support qubits.  This assertion checks
    both that each compiled footprint is exactly that full snake interval and
    that no two intervals in the packed color group overlap.
    """

    snake = _snake_qubits(L)
    position = {qubit: index for index, qubit in enumerate(snake)}
    occupied_by: Dict[int, int] = {}

    for quartet_index, pauli, circuit in rotations:
        support_positions = sorted(position[qubit] for qubit in pauli)
        expected_span = (
            set(range(support_positions[0], support_positions[-1] + 1))
            if support_positions
            else set()
        )
        compiled_positions = {position[qubit] for qubit in circuit.all_qubits()}
        if compiled_positions != expected_span:
            raise AssertionError(
                "packed Pauli compiler footprint differs from its complete "
                f"dirty-bridge snake span: color={color_index}, "
                f"quartet={quartet_index}"
            )

        overlap = sorted(expected_span.intersection(occupied_by))
        if overlap:
            conflicting_quartets = sorted(
                {occupied_by[index] for index in overlap}
            )
            raise ValueError(
                "packed Pauli dirty-bridge spans overlap before moment-wise "
                f"zipping: color={color_index}, quartet={quartet_index}, "
                f"conflicts={conflicting_quartets}, snake_positions={overlap}"
            )
        for index in expected_span:
            occupied_by[index] = quartet_index


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
        baseline_name: "1d", "ancilla", "primitive", "pipelined", or
            "folded".  ``folded`` is the current ancilla-free construction.
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
        rotations = []
        for qi in group:
            quartet = instance.quartets[qi]
            coupling = instance.couplings[qi]
            qp, phase = majorana_to_pauli(quartet, mode_to_jw_pos, L)
            effective_angle = coupling * phase.real if phase.imag == 0 else coupling
            rotation = build_pauli_exp_circuit(
                qp, float(np.real(effective_angle)), L
            )
            rotations.append((qi, qp, rotation))

        # Coloring makes quartets mode-disjoint, while packing is intended to
        # make the larger compiler footprints disjoint as well.  Check the
        # latter explicitly before combining equal-index moments.
        _assert_disjoint_packed_rotation_spans(rotations, L, alpha)
        rot_circuits = [rotation for _, _, rotation in rotations]
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

    assert_nearest_neighbor_circuit(total_circuit)
    return total_circuit, n_ancillas, fp_cnot_depth, int_cnot_depth


# ---------------------------------------------------------------------------
# Naive Pauli baseline
# ---------------------------------------------------------------------------

def build_trotter_step_naive(
    instance: SYKInstance,
) -> Tuple[cirq.Circuit, int, int, int]:
    """Build one Trotter step using naive Pauli strings (no FP, no coloring).

    Each chi_i chi_j chi_k chi_l is directly converted to a JW Pauli string
    and implemented by the exact nearest-neighbor dirty-bridge sweep used by
    the FP-based methods.

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
        rot_circ = build_pauli_exp_circuit(
            qp, float(np.real(effective_angle)), L
        )
        total_circuit += rot_circ

    assert_nearest_neighbor_circuit(total_circuit)
    int_res = count_resources(total_circuit, L, 0)
    return total_circuit, 0, 0, int_res["cnot_depth"]
