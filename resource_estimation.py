"""Resource estimation helpers for fermionic permutation circuits.

This module compares two implementations:
1) OpenFermion baseline on a 1D snake ordering.
2) Custom compressed method with row-stage CNOTs compiled through middle ancillas.

Counting is execution-driven:
build circuits -> decompose/synthesize -> count primitive events.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

# Keep matplotlib cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cirq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openfermion.circuits.gates import FSWAP, FSwapPowGate

from visualize_fermionic import CompressedFermionicPermutation, GridTopology, decompose_permutation

LinearPermutationGate = cirq.contrib.acquaintance.permutation.LinearPermutationGate

MethodName = Literal["baseline_openfermion_snake", "custom_ancilla_row_cnot"]
PermutationKind = Literal["random", "transpose", "reverse", "identity"]
PlotMetric = Literal["success_probability", "error_sum_total", "weighted_count_metric"]

_FSWAP_SYNTH_TEMPLATE_OPS: Optional[Tuple[cirq.Operation, ...]] = None


def validate_permutation(permutation: Sequence[int], n: int) -> None:
    if len(permutation) != n:
        raise ValueError(f"Permutation length {len(permutation)} != {n}.")
    if sorted(permutation) != list(range(n)):
        raise ValueError("Permutation must be a reordering of 0..N-1.")


def snake_order_indices(L: int) -> List[int]:
    order: List[int] = []
    for r in range(L):
        row = [r * L + c for c in range(L)]
        if r % 2 == 1:
            row.reverse()
        order.extend(row)
    return order


def raster_to_snake_index_map(L: int) -> Dict[int, int]:
    snake = snake_order_indices(L)
    return {r_idx: s_pos for s_pos, r_idx in enumerate(snake)}


def raster_perm_to_snake(L: int, perm_raster: Sequence[int]) -> List[int]:
    validate_permutation(perm_raster, L * L)
    r2s = raster_to_snake_index_map(L)
    perm_snake = [0] * (L * L)
    for src_r, dst_r in enumerate(perm_raster):
        perm_snake[r2s[src_r]] = r2s[dst_r]
    return perm_snake


def build_benchmark_permutation(
    L: int,
    kind: PermutationKind = "random",
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    n = L * L
    if kind == "identity":
        return list(range(n))
    if kind == "reverse":
        return list(range(n - 1, -1, -1))
    if kind == "transpose":
        return [c * L + r for r in range(L) for c in range(L)]
    if kind == "random":
        local_rng = rng if rng is not None else np.random.default_rng(0)
        return local_rng.permutation(n).tolist()
    raise ValueError(f"Unsupported permutation kind: {kind}")


def build_openfermion_baseline_circuit(L: int, perm_raster: Sequence[int]) -> cirq.Circuit:
    n = L * L
    validate_permutation(perm_raster, n)
    perm_snake = raster_perm_to_snake(L, perm_raster)
    q = cirq.LineQubit.range(n)
    op = LinearPermutationGate(n, {i: perm_snake[i] for i in range(n)}, FSWAP).on(*q)
    return cirq.Circuit(op)


def _middle_row_ancilla(program: CompressedFermionicPermutation, q0: cirq.Qid, q1: cirq.Qid) -> cirq.Qid:
    pos = {q: (r, c) for (r, c), q in program.topo.data_map.items()}
    if q0 not in pos or q1 not in pos:
        raise ValueError("Row-stage ancilla routing expects data qubits.")
    r0, c0 = pos[q0]
    r1, c1 = pos[q1]
    if r0 != r1:
        raise ValueError("Row-stage ancilla routing expects qubits from the same row.")
    if abs(c0 - c1) != 1:
        raise ValueError(
            f"Row-stage ancilla routing expects neighboring data qubits, got columns {c0} and {c1}."
        )
    return program.topo.ancilla_map[(r0, min(c0, c1))]


def _compile_cnot_via_middle_ancilla(
    program: CompressedFermionicPermutation, control: cirq.Qid, target: cirq.Qid
) -> List[cirq.Operation]:
    anc = _middle_row_ancilla(program, control, target)
    return [
        cirq.reset(anc),
        cirq.CNOT(control, anc),
        cirq.CNOT(anc, target),
        cirq.CNOT(control, anc),
    ]


def _canonical_pair(
    q0: cirq.Qid, q1: cirq.Qid, data_index: Dict[cirq.Qid, int]
) -> Tuple[cirq.Qid, cirq.Qid]:
    i0 = data_index.get(q0)
    i1 = data_index.get(q1)
    if i0 is not None and i1 is not None:
        return (q0, q1) if i0 < i1 else (q1, q0)
    return (q0, q1) if str(q0) < str(q1) else (q1, q0)


def _split_cz_edges_two_rounds(
    edges: Sequence[Tuple[cirq.Qid, cirq.Qid]],
) -> Optional[List[List[Tuple[cirq.Qid, cirq.Qid]]]]:
    if not edges:
        return [[], []]

    incident: Dict[cirq.Qid, List[int]] = {}
    for idx, (q0, q1) in enumerate(edges):
        incident.setdefault(q0, []).append(idx)
        incident.setdefault(q1, []).append(idx)

    adjacency = {idx: set() for idx in range(len(edges))}
    feasible = True
    for edge_indices in incident.values():
        if len(edge_indices) > 2:
            feasible = False
        for i in range(len(edge_indices)):
            for j in range(i + 1, len(edge_indices)):
                a = edge_indices[i]
                b = edge_indices[j]
                adjacency[a].add(b)
                adjacency[b].add(a)

    color: Dict[int, int] = {}
    for start in range(len(edges)):
        if start in color:
            continue
        stack = [start]
        color[start] = 0
        while stack:
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if nxt not in color:
                    color[nxt] = 1 - color[cur]
                    stack.append(nxt)
                elif color[nxt] == color[cur]:
                    feasible = False

    if not feasible:
        return None

    rounds: List[List[Tuple[cirq.Qid, cirq.Qid]]] = [[], []]
    for idx, edge in enumerate(edges):
        rounds[color[idx]].append(edge)
    return rounds


def _compile_horizontal_cz_via_middle_ancilla(
    program: CompressedFermionicPermutation,
    q0: cirq.Qid,
    q1: cirq.Qid,
    data_pos: Dict[cirq.Qid, Tuple[int, int]],
) -> Tuple[cirq.Operation, cirq.Operation, cirq.Operation, cirq.Qid]:
    r0, c0 = data_pos[q0]
    r1, c1 = data_pos[q1]
    if r0 != r1 or abs(c0 - c1) != 1:
        raise ValueError("Horizontal CZ routing expects nearest-neighbor data qubits in one row.")

    if c0 < c1:
        left, right = q0, q1
    else:
        left, right = q1, q0
    anc = program.topo.ancilla_map[(r0, min(c0, c1))]
    return cirq.CNOT(left, anc), cirq.CZ(anc, right), cirq.CNOT(left, anc), anc


def _build_phase_correction_moments(
    program: CompressedFermionicPermutation, active_swaps: Dict[int, List[int]], round_parity: int
) -> List[cirq.Moment]:
    data_pos = {q: (r, c) for (r, c), q in program.topo.data_map.items()}
    data_index = {q: i for i, q in enumerate(program.topo.data_qubits)}

    cz_counts: Dict[Tuple[cirq.Qid, cirq.Qid], int] = {}
    z_counts: Dict[cirq.Qid, int] = {}

    def add_cz(a: cirq.Qid, b: cirq.Qid) -> None:
        key = _canonical_pair(a, b, data_index)
        cz_counts[key] = cz_counts.get(key, 0) + 1

    def add_z(q: cirq.Qid) -> None:
        z_counts[q] = z_counts.get(q, 0) + 1

    for c in range(program.L):
        for r in active_swaps[c]:
            row_qs = program.topo.get_data_row(r)
            row_qs_next = program.topo.get_data_row(r + 1)
            tl = row_qs[c]
            bl = row_qs_next[c]

            tr = br = None
            if round_parity == 0 and c < program.L - 1:
                tr = row_qs[c + 1]
                br = row_qs_next[c + 1]
            elif round_parity == 1 and c > 0:
                tr = row_qs[c - 1]
                br = row_qs_next[c - 1]

            add_cz(tl, bl)
            if tr is not None:
                add_cz(tl, tr)
                add_z(tr)
            if br is not None:
                add_cz(bl, br)
                add_z(br)
            if tr is not None and br is not None:
                add_cz(tr, br)

    z_ops = [cirq.Z(q) for q, count in z_counts.items() if count % 2 == 1]
    cz_edges = [edge for edge, count in cz_counts.items() if count % 2 == 1]

    rounds = _split_cz_edges_two_rounds(cz_edges)
    if rounds is None:
        raise ValueError("Phase CZ edges were expected to be 2-round schedulable, but coloring failed.")

    used_horizontal_ancillas = []
    for q0, q1 in cz_edges:
        r0, c0 = data_pos[q0]
        r1, c1 = data_pos[q1]
        if r0 == r1 and abs(c0 - c1) == 1:
            used_horizontal_ancillas.append(program.topo.ancilla_map[(r0, min(c0, c1))])

    moments: List[cirq.Moment] = []
    if used_horizontal_ancillas:
        moments.append(cirq.Moment([cirq.reset(a) for a in used_horizontal_ancillas]))
    if z_ops:
        moments.append(cirq.Moment(z_ops))

    for edges_in_round in rounds:
        direct_vertical_ops: List[cirq.Operation] = []
        h_cnot_1_ops: List[cirq.Operation] = []
        h_mid_ops: List[cirq.Operation] = []
        h_cnot_2_ops: List[cirq.Operation] = []
        direct_other_ops: List[cirq.Operation] = []

        for q0, q1 in edges_in_round:
            p0 = data_pos.get(q0)
            p1 = data_pos.get(q1)
            if p0 is None or p1 is None:
                direct_other_ops.append(cirq.CZ(q0, q1))
                continue
            r0, c0 = p0
            r1, c1 = p1
            if c0 == c1:
                direct_vertical_ops.append(cirq.CZ(q0, q1))
            elif r0 == r1 and abs(c0 - c1) == 1:
                cnot_1, mid, cnot_2, _ = _compile_horizontal_cz_via_middle_ancilla(program, q0, q1, data_pos)
                h_cnot_1_ops.append(cnot_1)
                h_mid_ops.append(mid)
                h_cnot_2_ops.append(cnot_2)
            else:
                direct_other_ops.append(cirq.CZ(q0, q1))

        if h_cnot_1_ops:
            moments.append(cirq.Moment(h_cnot_1_ops))

        mid_ops = direct_vertical_ops + h_mid_ops + direct_other_ops
        if mid_ops:
            moments.append(cirq.Moment(mid_ops))

        if h_cnot_2_ops:
            moments.append(cirq.Moment(h_cnot_2_ops))

    return moments


def _exp_mod_2(exponent: float) -> float:
    value = float(exponent) % 2.0
    if np.isclose(value, 2.0, atol=1e-9):
        return 0.0
    return value


def _is_identity_exponent(exponent: float) -> bool:
    return np.isclose(_exp_mod_2(exponent), 0.0, atol=1e-9)


def _is_odd_exponent(exponent: float) -> bool:
    return np.isclose(_exp_mod_2(exponent), 1.0, atol=1e-9)


def _strip_operation_wrappers(op: cirq.Operation) -> cirq.Operation:
    unwrapped = op
    while True:
        if isinstance(unwrapped, cirq.TaggedOperation):
            unwrapped = unwrapped.sub_operation
            continue
        if isinstance(unwrapped, cirq.ClassicallyControlledOperation):
            unwrapped = unwrapped.without_classical_controls()
            continue
        return unwrapped


def _synthesize_fswap_template_ops() -> Tuple[cirq.Operation, ...]:
    global _FSWAP_SYNTH_TEMPLATE_OPS
    if _FSWAP_SYNTH_TEMPLATE_OPS is not None:
        return _FSWAP_SYNTH_TEMPLATE_OPS

    q0, q1 = cirq.LineQubit.range(2)
    # Direct 2-CNOT Clifford decomposition (no CZ intermediary).
    template_ops: List[cirq.Operation] = [
        cirq.H(q1),
        cirq.CNOT(q1, q0),
        cirq.H(q0),
        cirq.H(q1),
        cirq.CNOT(q1, q0),
        cirq.H(q1),
    ]

    target_u = cirq.unitary(FSWAP(q0, q1))
    synth_u = cirq.unitary(cirq.Circuit(template_ops))
    if not cirq.linalg.allclose_up_to_global_phase(target_u, synth_u, atol=1e-8):
        raise ValueError("Synthesized FSWAP decomposition is not unitary-equivalent.")

    _FSWAP_SYNTH_TEMPLATE_OPS = tuple(template_ops)
    return _FSWAP_SYNTH_TEMPLATE_OPS


def _lower_fswap_to_common(op: cirq.Operation) -> List[cirq.Operation]:
    if len(op.qubits) != 2:
        raise ValueError("FSWAP must act on exactly two qubits.")
    a, b = op.qubits
    q_map = {cirq.LineQubit(0): a, cirq.LineQubit(1): b}

    lowered: List[cirq.Operation] = []
    for template_op in _synthesize_fswap_template_ops():
        mapped = template_op.with_qubits(*(q_map[q] for q in template_op.qubits))
        lowered.extend(_lower_operation_to_common(mapped))
    return lowered


def _lower_swap_to_common(op: cirq.Operation) -> List[cirq.Operation]:
    if len(op.qubits) != 2:
        raise ValueError("SWAP must act on exactly two qubits.")
    a, b = op.qubits
    return [cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.CNOT(a, b)]


def _lower_cz_to_common(op: cirq.Operation) -> List[cirq.Operation]:
    if len(op.qubits) != 2:
        raise ValueError("CZ must act on exactly two qubits.")
    control, target = op.qubits
    return [cirq.H(target), cirq.CNOT(control, target), cirq.H(target)]


def _lower_operation_to_common(op: cirq.Operation) -> List[cirq.Operation]:
    op = _strip_operation_wrappers(op)
    gate = op.gate
    if gate is None:
        return []

    if isinstance(gate, (cirq.MeasurementGate, cirq.ResetChannel)):
        return [op]

    if isinstance(gate, (cirq.HPowGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate)):
        return [] if _is_identity_exponent(gate.exponent) else [op]

    if isinstance(gate, cirq.CNotPowGate):
        return [] if _is_identity_exponent(gate.exponent) else [op]

    if isinstance(gate, cirq.CZPowGate):
        if _is_identity_exponent(gate.exponent):
            return []
        if not _is_odd_exponent(gate.exponent):
            raise ValueError(f"Unsupported CZ exponent for counting: {gate.exponent}")
        lowered: List[cirq.Operation] = []
        for sub_op in _lower_cz_to_common(op):
            lowered.extend(_lower_operation_to_common(sub_op))
        return lowered

    if isinstance(gate, cirq.SwapPowGate):
        if _is_identity_exponent(gate.exponent):
            return []
        if not _is_odd_exponent(gate.exponent):
            raise ValueError(f"Unsupported SWAP exponent for counting: {gate.exponent}")
        return _lower_swap_to_common(op)

    if isinstance(gate, FSwapPowGate):
        if _is_identity_exponent(gate.exponent):
            return []
        if not _is_odd_exponent(gate.exponent):
            raise ValueError(f"Unsupported FSWAP exponent for counting: {gate.exponent}")
        return _lower_fswap_to_common(op)

    if isinstance(gate, LinearPermutationGate):
        lowered: List[cirq.Operation] = []
        for sub_op in cirq.decompose(op):
            lowered.extend(_lower_operation_to_common(sub_op))
        return lowered

    decomposed = cirq.decompose_once(op, default=None)
    if decomposed is None:
        raise ValueError(f"Unable to lower operation to common gateset: {op!r}")

    lowered = []
    for sub_op in decomposed:
        lowered.extend(_lower_operation_to_common(sub_op))
    return lowered


def lower_circuit_to_common_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    lowered_ops: List[cirq.Operation] = []
    for op in circuit.all_operations():
        lowered_ops.extend(_lower_operation_to_common(op))
    return cirq.Circuit(lowered_ops)


def _build_custom_circuit_with_row_routing(L: int, perm_raster: Sequence[int]) -> cirq.Circuit:
    validate_permutation(perm_raster, L * L)
    program = CompressedFermionicPermutation(L, topology=GridTopology(L))
    s1, s2, s3 = decompose_permutation(L, perm_raster)
    moments: List[cirq.Moment] = []

    def append_col_stage(schedule: Dict[int, List[int]]) -> None:
        cur_perms = {c: list(schedule[c]) for c in range(L)}
        for t in range(L):
            active_swaps = {c: [] for c in range(L)}
            overall_active = False
            round_parity = t % 2
            start_idx = 1 if round_parity == 1 else 0

            for c in range(L):
                arr = cur_perms[c]
                for i in range(start_idx, L - 1, 2):
                    if arr[i] > arr[i + 1]:
                        active_swaps[c].append(i)
                        overall_active = True

            if not overall_active:
                if all(cur_perms[c] == sorted(cur_perms[c]) for c in range(L)):
                    break
                continue

            moments.extend(program.build_prefix_xor_moments(round_parity, t))
            moments.extend(_build_phase_correction_moments(program, active_swaps, round_parity))
            moments.extend(program.unbuild_prefix_xor_moments(round_parity, t))
            moments.extend(program.get_swaps_moments(active_swaps))

            for c in range(L):
                arr = cur_perms[c]
                for r in active_swaps[c]:
                    arr[r], arr[r + 1] = arr[r + 1], arr[r]

    append_col_stage(s1)

    for r in range(L):
        row_qs = program.topo.get_data_row(r)
        row_perm = {i: s2[r][i] for i in range(L)}
        row_op = LinearPermutationGate(L, row_perm, FSWAP).on(*row_qs)

        routed_row_ops: List[cirq.Operation] = []
        for fswap_op in cirq.decompose(row_op):
            primitive_ops = _lower_operation_to_common(fswap_op)
            for primitive in primitive_ops:
                primitive_gate = primitive.gate
                if isinstance(primitive_gate, cirq.CNotPowGate):
                    routed_row_ops.extend(_compile_cnot_via_middle_ancilla(program, *primitive.qubits))
                else:
                    routed_row_ops.append(primitive)
        moments.extend(cirq.Circuit(routed_row_ops).moments)

    append_col_stage(s3)
    return cirq.Circuit(moments)


def count_common_gates(circuit: cirq.Circuit) -> Dict[str, int]:
    counts = {
        "H": 0,
        "S": 0,
        "X": 0,
        "Y": 0,
        "Z": 0,
        "CNOT": 0,
        "CZ": 0,
        "MEASURE": 0,
        "RESET": 0,
        "OTHER_1Q": 0,
        "OTHER_2Q": 0,
        "OTHER_NQ": 0,
    }

    for raw_op in circuit.all_operations():
        op = _strip_operation_wrappers(raw_op)
        gate = op.gate
        if gate is None:
            continue

        n_qubits = len(op.qubits)

        if isinstance(gate, cirq.MeasurementGate):
            counts["MEASURE"] += n_qubits
            continue

        if isinstance(gate, cirq.ResetChannel):
            counts["RESET"] += n_qubits
            continue

        if isinstance(gate, cirq.CNotPowGate):
            if _is_identity_exponent(gate.exponent):
                continue
            if _is_odd_exponent(gate.exponent):
                counts["CNOT"] += 1
            else:
                counts["OTHER_2Q"] += 1
            continue

        if isinstance(gate, cirq.CZPowGate):
            if _is_identity_exponent(gate.exponent):
                continue
            if _is_odd_exponent(gate.exponent):
                counts["CZ"] += 1
            else:
                counts["OTHER_2Q"] += 1
            continue

        if isinstance(gate, cirq.HPowGate):
            if _is_identity_exponent(gate.exponent):
                continue
            if _is_odd_exponent(gate.exponent):
                counts["H"] += n_qubits
            else:
                counts["OTHER_1Q"] += n_qubits
            continue

        if isinstance(gate, cirq.XPowGate):
            if _is_identity_exponent(gate.exponent):
                continue
            if _is_odd_exponent(gate.exponent):
                counts["X"] += n_qubits
            else:
                counts["OTHER_1Q"] += n_qubits
            continue

        if isinstance(gate, cirq.YPowGate):
            if _is_identity_exponent(gate.exponent):
                continue
            if _is_odd_exponent(gate.exponent):
                counts["Y"] += n_qubits
            else:
                counts["OTHER_1Q"] += n_qubits
            continue

        if isinstance(gate, cirq.ZPowGate):
            mod = _exp_mod_2(gate.exponent)
            if np.isclose(mod, 0.0, atol=1e-9):
                continue
            if np.isclose(mod, 1.0, atol=1e-9):
                counts["Z"] += n_qubits
            elif np.isclose(mod, 0.5, atol=1e-9) or np.isclose(mod, 1.5, atol=1e-9):
                counts["S"] += n_qubits
            else:
                counts["OTHER_1Q"] += n_qubits
            continue

        if n_qubits == 1:
            counts["OTHER_1Q"] += 1
        elif n_qubits == 2:
            counts["OTHER_2Q"] += 1
        else:
            counts["OTHER_NQ"] += 1

    counts["SINGLE_TOTAL"] = (
        counts["H"] + counts["S"] + counts["X"] + counts["Y"] + counts["Z"] + counts["OTHER_1Q"]
    )
    counts["TWO_QUBIT_TOTAL"] = counts["CNOT"] + counts["CZ"] + counts["OTHER_2Q"]
    counts["READOUT_TOTAL"] = counts["MEASURE"] + counts["RESET"]
    counts["TOTAL_EVENTS"] = (
        counts["SINGLE_TOTAL"] + counts["TWO_QUBIT_TOTAL"] + counts["READOUT_TOTAL"] + counts["OTHER_NQ"]
    )
    return counts


def _measurement_event_count(counts: Dict[str, int], include_reset_as_measurement: bool) -> int:
    return counts["MEASURE"] + (counts["RESET"] if include_reset_as_measurement else 0)


def _single_qubit_estimation_count(counts: Dict[str, int], include_single_qubit_in_estimation: bool) -> int:
    return counts["SINGLE_TOTAL"] if include_single_qubit_in_estimation else 0


def cnot_equivalent_two_qubit_count(counts: Dict[str, int], other_2q_cnot_equivalent: float = 1.0) -> float:
    return float(counts["CNOT"] + counts["CZ"] + other_2q_cnot_equivalent * counts["OTHER_2Q"])


def estimate_error_models(
    counts: Dict[str, int],
    single_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-4,
    measurement_error: float = 1e-4,
    idle_error: float = 1e-6,
    idle_multiplier: float = 0.01,
    include_reset_as_measurement: bool = True,
    include_single_qubit_in_estimation: bool = False,
    measurement_duration_mu: float = 3.65,
    other_2q_cnot_equivalent: float = 1.0,
) -> Dict[str, float]:
    single_count = _single_qubit_estimation_count(counts, include_single_qubit_in_estimation)
    two_count = counts["TWO_QUBIT_TOTAL"]
    measurement_events = _measurement_event_count(counts, include_reset_as_measurement)
    idle_cnot_time = float(measurement_events) * measurement_duration_mu
    cnot_eq_twoq = cnot_equivalent_two_qubit_count(counts, other_2q_cnot_equivalent=other_2q_cnot_equivalent)

    log_success = (
        single_count * np.log1p(-single_qubit_error)
        + two_count * np.log1p(-two_qubit_error)
        + measurement_events * np.log1p(-measurement_error)
        + idle_cnot_time * np.log1p(-idle_error)
    )
    tiny_log = np.log(np.finfo(float).tiny)
    success_prob = 0.0 if log_success < tiny_log else float(np.exp(log_success))

    error_sum_single = float(single_count) * single_qubit_error
    error_sum_two_qubit = float(two_count) * two_qubit_error
    error_sum_measurement = float(measurement_events) * measurement_error
    error_sum_idle = idle_cnot_time * idle_error
    error_sum_total = error_sum_single + error_sum_two_qubit + error_sum_measurement + error_sum_idle
    weighted_cnot_component = cnot_eq_twoq
    weighted_meas_component = float(measurement_events)
    weighted_idle_component = idle_multiplier * idle_cnot_time
    weighted_count_metric = weighted_cnot_component + weighted_meas_component + weighted_idle_component

    return {
        "single_qubit_estimation_count": float(single_count),
        "measurement_event_count": float(measurement_events),
        "idle_cnot_time": idle_cnot_time,
        "CNOT_EQ_TWOQ": cnot_eq_twoq,
        "weighted_cnot_component": weighted_cnot_component,
        "weighted_meas_component": weighted_meas_component,
        "weighted_idle_component": weighted_idle_component,
        "weighted_count_metric": weighted_count_metric,
        "success_probability": success_prob,
        "log_success": float(log_success),
        "log10_success": float(log_success / np.log(10)),
        "error_sum_single": error_sum_single,
        "error_sum_two_qubit": error_sum_two_qubit,
        "error_sum_measurement": error_sum_measurement,
        "error_sum_idle": error_sum_idle,
        "error_sum_total": error_sum_total,
    }


def estimate_success_probability(
    counts: Dict[str, int],
    single_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-4,
    measurement_error: float = 1e-4,
    idle_error: float = 1e-6,
    idle_multiplier: float = 0.01,
    include_reset_as_measurement: bool = True,
    include_single_qubit_in_estimation: bool = False,
    measurement_duration_mu: float = 3.65,
) -> Tuple[float, float]:
    model = estimate_error_models(
        counts,
        single_qubit_error=single_qubit_error,
        two_qubit_error=two_qubit_error,
        measurement_error=measurement_error,
        idle_error=idle_error,
        idle_multiplier=idle_multiplier,
        include_reset_as_measurement=include_reset_as_measurement,
        include_single_qubit_in_estimation=include_single_qubit_in_estimation,
        measurement_duration_mu=measurement_duration_mu,
    )
    return model["success_probability"], model["log_success"]


def _estimate_single_method(
    method: MethodName,
    L: int,
    perm_raster: Sequence[int],
    single_qubit_error: float,
    two_qubit_error: float,
    measurement_error: float,
    idle_error: float,
    idle_multiplier: float,
    include_reset_as_measurement: bool,
    include_single_qubit_in_estimation: bool,
    measurement_duration_mu: float,
    other_2q_cnot_equivalent: float,
) -> Dict[str, float]:
    if method == "baseline_openfermion_snake":
        raw = build_openfermion_baseline_circuit(L, perm_raster)
    elif method == "custom_ancilla_row_cnot":
        raw = _build_custom_circuit_with_row_routing(L, perm_raster)
    else:
        raise ValueError(f"Unknown method: {method}")

    common = lower_circuit_to_common_gates(raw)
    counts = count_common_gates(common)
    model = estimate_error_models(
        counts,
        single_qubit_error=single_qubit_error,
        two_qubit_error=two_qubit_error,
        measurement_error=measurement_error,
        idle_error=idle_error,
        idle_multiplier=idle_multiplier,
        include_reset_as_measurement=include_reset_as_measurement,
        include_single_qubit_in_estimation=include_single_qubit_in_estimation,
        measurement_duration_mu=measurement_duration_mu,
        other_2q_cnot_equivalent=other_2q_cnot_equivalent,
    )

    row: Dict[str, float] = {"method": method, "L": L}
    row.update(counts)
    row.update(model)
    row["rough_expected_baseline_cnot_eq"] = float(L**4)
    row["rough_expected_custom_cnot_eq"] = float(16 * (L**3))
    row["rough_expected_custom_measure"] = float(4 * (L**3))
    row["rough_expected_custom_idle_cnot_time"] = float(4 * measurement_duration_mu * (L**3))

    if method == "baseline_openfermion_snake":
        row["baseline_ratio_cnot_eq_over_L4"] = float(row["CNOT_EQ_TWOQ"] / (L**4))
        row["custom_ratio_cnot_eq_over_L3"] = np.nan
        row["custom_ratio_measure_over_L3"] = np.nan
        row["custom_ratio_idle_cnot_time_over_L3"] = np.nan
    else:
        row["baseline_ratio_cnot_eq_over_L4"] = np.nan
        row["custom_ratio_cnot_eq_over_L3"] = float(row["CNOT_EQ_TWOQ"] / (L**3))
        row["custom_ratio_measure_over_L3"] = float(counts["MEASURE"] / (L**3))
        row["custom_ratio_idle_cnot_time_over_L3"] = float(row["idle_cnot_time"] / (L**3))
    return row


def run_resource_scaling_study(
    L_values: Sequence[int] = tuple(range(10, 21, 2)),
    permutation_kind: PermutationKind = "random",
    seed: int = 0,
    single_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-4,
    measurement_error: float = 1e-4,
    idle_error: float = 1e-6,
    idle_multiplier: float = 0.01,
    include_single_qubit_in_estimation: bool = False,
    measurement_duration_mu: float = 3.65,
    other_2q_cnot_equivalent: float = 1.0,
    include_reset_as_measurement: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for L in L_values:
        rng = np.random.default_rng(seed + 7919 * L)
        perm_raster = build_benchmark_permutation(L, kind=permutation_kind, rng=rng)

        rows.append(
            _estimate_single_method(
                method="baseline_openfermion_snake",
                L=L,
                perm_raster=perm_raster,
                single_qubit_error=single_qubit_error,
                two_qubit_error=two_qubit_error,
                measurement_error=measurement_error,
                idle_error=idle_error,
                idle_multiplier=idle_multiplier,
                include_reset_as_measurement=include_reset_as_measurement,
                include_single_qubit_in_estimation=include_single_qubit_in_estimation,
                measurement_duration_mu=measurement_duration_mu,
                other_2q_cnot_equivalent=other_2q_cnot_equivalent,
            )
        )
        rows.append(
            _estimate_single_method(
                method="custom_ancilla_row_cnot",
                L=L,
                perm_raster=perm_raster,
                single_qubit_error=single_qubit_error,
                two_qubit_error=two_qubit_error,
                measurement_error=measurement_error,
                idle_error=idle_error,
                idle_multiplier=idle_multiplier,
                include_reset_as_measurement=include_reset_as_measurement,
                include_single_qubit_in_estimation=include_single_qubit_in_estimation,
                measurement_duration_mu=measurement_duration_mu,
                other_2q_cnot_equivalent=other_2q_cnot_equivalent,
            )
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["L", "method"]).reset_index(drop=True)


def _cache_value_mask(series: pd.Series, value: object) -> pd.Series:
    if isinstance(value, bool):
        normalized = series.astype(str).str.strip().str.lower()
        target = "true" if value else "false"
        numeric_alt = "1" if value else "0"
        return normalized.isin([target, numeric_alt])

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = pd.to_numeric(series, errors="coerce")
        return np.isclose(numeric, float(value), atol=1e-12, rtol=0.0)

    return series.astype(str) == str(value)


def run_weighted_metric_study_with_cache(
    L_values: Sequence[int] = tuple(range(20, 51, 5)),
    permutation_kind: PermutationKind = "random",
    seed: int = 0,
    include_reset_as_measurement: bool = True,
    measurement_duration_mu: float = 3.65,
    idle_multiplier: float = 0.01,
    other_2q_cnot_equivalent: float = 1.0,
    max_runtime_minutes: float = 90.0,
    cache_csv_path: str = (
        "/Users/dantongli/Desktop/efficient-classical-shadow/fast fermionic permutation/resource_estimation_cache.csv"
    ),
    single_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-4,
    measurement_error: float = 1e-4,
    idle_error: float = 1e-6,
    include_single_qubit_in_estimation: bool = False,
) -> pd.DataFrame:
    methods: Tuple[MethodName, MethodName] = ("baseline_openfermion_snake", "custom_ancilla_row_cnot")
    l_values = tuple(int(L) for L in L_values)
    requested_pairs = [(L, method) for L in l_values for method in methods]

    cache_config = {
        "permutation_kind": permutation_kind,
        "seed": int(seed),
        "include_reset_as_measurement": bool(include_reset_as_measurement),
        "measurement_duration_mu": float(measurement_duration_mu),
        "idle_multiplier": float(idle_multiplier),
        "other_2q_cnot_equivalent": float(other_2q_cnot_equivalent),
    }
    cache_key_cols = ["method", "L"] + list(cache_config.keys())
    cache_path = Path(cache_csv_path)

    if cache_path.exists():
        try:
            cache_df = pd.read_csv(cache_path)
        except Exception:
            cache_df = pd.DataFrame()
    else:
        cache_df = pd.DataFrame()

    cached_subset = pd.DataFrame()
    if not cache_df.empty:
        mask = pd.Series(True, index=cache_df.index)
        for key, value in cache_config.items():
            if key not in cache_df.columns:
                mask = pd.Series(False, index=cache_df.index)
                break
            mask &= _cache_value_mask(cache_df[key], value)

        if "L" in cache_df.columns:
            mask &= pd.to_numeric(cache_df["L"], errors="coerce").isin(l_values)
        else:
            mask = pd.Series(False, index=cache_df.index)

        if "method" in cache_df.columns:
            mask &= cache_df["method"].isin(methods)
        else:
            mask = pd.Series(False, index=cache_df.index)

        cached_subset = cache_df[mask].copy()
        if not cached_subset.empty:
            required_payload_cols = [
                "weighted_count_metric",
                "weighted_cnot_component",
                "weighted_meas_component",
                "weighted_idle_component",
                "CNOT_EQ_TWOQ",
                "measurement_event_count",
                "idle_cnot_time",
            ]
            for col in required_payload_cols:
                if col not in cached_subset.columns:
                    cached_subset[col] = np.nan
            cached_subset = cached_subset[cached_subset["weighted_count_metric"].notna()].copy()
        if not cached_subset.empty:
            if "result_origin" not in cached_subset.columns:
                cached_subset["result_origin"] = "exact"
            cached_subset["_origin_priority"] = cached_subset["result_origin"].map(
                {"exact": 0, "extrapolated": 1}
            ).fillna(2)
            cached_subset = cached_subset.sort_values(["method", "L", "_origin_priority"])
            cached_subset = cached_subset.drop_duplicates(["method", "L"], keep="first")
            cached_subset = cached_subset.drop(columns=["_origin_priority"])

    cached_pairs = set()
    if not cached_subset.empty:
        cached_pairs = {
            (int(L), str(method))
            for L, method in zip(cached_subset["L"], cached_subset["method"])
        }
    pending_pairs = [pair for pair in requested_pairs if pair not in cached_pairs]

    exact_counts = {method: 0 for method in methods}
    if not cached_subset.empty and "result_origin" in cached_subset.columns:
        for method in methods:
            mask = (cached_subset["method"] == method) & (cached_subset["result_origin"] == "exact")
            exact_counts[method] = int(mask.sum())

    start_wall = time.time()
    runtime_limit_seconds = max(0.0, 60.0 * float(max_runtime_minutes))
    new_exact_rows: List[Dict[str, float]] = []
    extrapolate_pairs: List[Tuple[int, MethodName]] = []

    permutations_by_L = {
        L: build_benchmark_permutation(
            L,
            kind=permutation_kind,
            rng=np.random.default_rng(seed + 7919 * L),
        )
        for L in l_values
    }

    for L, method in pending_pairs:
        need_min_exact = exact_counts[method] < 2
        elapsed_total = time.time() - start_wall
        if elapsed_total >= runtime_limit_seconds and not need_min_exact:
            extrapolate_pairs.append((L, method))
            continue

        step_start = time.time()
        row = _estimate_single_method(
            method=method,
            L=L,
            perm_raster=permutations_by_L[L],
            single_qubit_error=single_qubit_error,
            two_qubit_error=two_qubit_error,
            measurement_error=measurement_error,
            idle_error=idle_error,
            idle_multiplier=idle_multiplier,
            include_reset_as_measurement=include_reset_as_measurement,
            include_single_qubit_in_estimation=include_single_qubit_in_estimation,
            measurement_duration_mu=measurement_duration_mu,
            other_2q_cnot_equivalent=other_2q_cnot_equivalent,
        )
        row["result_origin"] = "exact"
        row["elapsed_seconds"] = float(time.time() - step_start)
        for key, value in cache_config.items():
            row[key] = value
        new_exact_rows.append(row)
        exact_counts[method] += 1

    exact_sources = []
    if not cached_subset.empty:
        exact_sources.append(cached_subset[cached_subset["result_origin"] == "exact"])
    if new_exact_rows:
        exact_sources.append(pd.DataFrame(new_exact_rows))

    exact_df = pd.concat(exact_sources, ignore_index=True) if exact_sources else pd.DataFrame()

    extrap_rows: List[Dict[str, float]] = []
    if extrapolate_pairs:
        for method in methods:
            num_exact = 0 if exact_df.empty else int((exact_df["method"] == method).sum())
            if num_exact < 2:
                raise ValueError(
                    f"Need at least 2 exact points for {method} before extrapolation; got {num_exact}."
                )

        def fit_component(method: MethodName, col: str, exponent: int) -> float:
            method_rows = exact_df[exact_df["method"] == method]
            values = pd.to_numeric(method_rows[col], errors="coerce").to_numpy(dtype=float)
            ls = pd.to_numeric(method_rows["L"], errors="coerce").to_numpy(dtype=float)
            ratios = values / np.power(ls, exponent)
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size == 0:
                return float("nan")
            return float(np.median(ratios))

        for L, method in extrapolate_pairs:
            exponent = 4 if method == "baseline_openfermion_snake" else 3
            scale = float(L**exponent)

            coeff_cnot_eq = fit_component(method, "CNOT_EQ_TWOQ", exponent)
            coeff_meas_events = fit_component(method, "measurement_event_count", exponent)
            coeff_cnot = fit_component(method, "CNOT", exponent)
            coeff_meas = fit_component(method, "MEASURE", exponent)
            coeff_reset = fit_component(method, "RESET", exponent)

            cnot_eq = coeff_cnot_eq * scale
            meas_events = coeff_meas_events * scale
            idle_cnot_time = meas_events * measurement_duration_mu

            row: Dict[str, float] = {
                "method": method,
                "L": float(L),
                "CNOT": coeff_cnot * scale,
                "MEASURE": coeff_meas * scale,
                "RESET": coeff_reset * scale,
                "CNOT_EQ_TWOQ": cnot_eq,
                "measurement_event_count": meas_events,
                "idle_cnot_time": idle_cnot_time,
                "weighted_cnot_component": cnot_eq,
                "weighted_meas_component": meas_events,
                "weighted_idle_component": idle_multiplier * idle_cnot_time,
                "weighted_count_metric": cnot_eq + meas_events + idle_multiplier * idle_cnot_time,
                "result_origin": "extrapolated",
                "elapsed_seconds": 0.0,
                "rough_expected_baseline_cnot_eq": float(L**4),
                "rough_expected_custom_cnot_eq": float(16 * (L**3)),
                "rough_expected_custom_measure": float(4 * (L**3)),
                "rough_expected_custom_idle_cnot_time": float(4 * measurement_duration_mu * (L**3)),
                "baseline_ratio_cnot_eq_over_L4": np.nan,
                "custom_ratio_cnot_eq_over_L3": np.nan,
                "custom_ratio_measure_over_L3": np.nan,
                "custom_ratio_idle_cnot_time_over_L3": np.nan,
            }

            if method == "baseline_openfermion_snake":
                row["baseline_ratio_cnot_eq_over_L4"] = float(cnot_eq / (L**4))
            else:
                row["custom_ratio_cnot_eq_over_L3"] = float(cnot_eq / (L**3))
                row["custom_ratio_measure_over_L3"] = float((coeff_meas * scale) / (L**3))
                row["custom_ratio_idle_cnot_time_over_L3"] = float(idle_cnot_time / (L**3))

            for key, value in cache_config.items():
                row[key] = value
            extrap_rows.append(row)

    new_rows_df = pd.concat(
        [pd.DataFrame(new_exact_rows), pd.DataFrame(extrap_rows)],
        ignore_index=True,
        sort=False,
    ) if (new_exact_rows or extrap_rows) else pd.DataFrame()

    cache_updates = []
    if not new_rows_df.empty:
        cache_updates.append(new_rows_df)

    if cache_updates:
        updated_cache = pd.concat([cache_df] + cache_updates, ignore_index=True, sort=False)
        for col in cache_key_cols:
            if col not in updated_cache.columns:
                updated_cache[col] = np.nan
        if "result_origin" not in updated_cache.columns:
            updated_cache["result_origin"] = "exact"
        updated_cache["_origin_priority"] = updated_cache["result_origin"].map(
            {"exact": 0, "extrapolated": 1}
        ).fillna(2)
        updated_cache = updated_cache.sort_values(cache_key_cols + ["_origin_priority"])
        updated_cache = updated_cache.drop_duplicates(cache_key_cols, keep="first")
        updated_cache = updated_cache.drop(columns=["_origin_priority"])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        updated_cache.to_csv(cache_path, index=False)
        cache_df = updated_cache

    result_frames = []
    if not cached_subset.empty:
        result_frames.append(cached_subset)
    if not new_rows_df.empty:
        result_frames.append(new_rows_df)
    if not result_frames:
        return pd.DataFrame()

    result = pd.concat(result_frames, ignore_index=True, sort=False)
    if "result_origin" not in result.columns:
        result["result_origin"] = "exact"
    result["_origin_priority"] = result["result_origin"].map({"exact": 0, "extrapolated": 1}).fillna(2)
    result = result.sort_values(["L", "method", "_origin_priority"])
    result = result.drop_duplicates(["L", "method"], keep="first").drop(columns=["_origin_priority"])
    result = result[result["L"].astype(int).isin(l_values)]
    result = result[result["method"].isin(methods)]
    result["L"] = result["L"].astype(int)
    return result.sort_values(["L", "method"]).reset_index(drop=True)


def build_scaling_consistency_report(df: pd.DataFrame, drift_tolerance: float = 0.2) -> pd.DataFrame:
    checks = [
        ("baseline_openfermion_snake", "baseline_ratio_cnot_eq_over_L4"),
        ("custom_ancilla_row_cnot", "custom_ratio_cnot_eq_over_L3"),
        ("custom_ancilla_row_cnot", "custom_ratio_measure_over_L3"),
        ("custom_ancilla_row_cnot", "custom_ratio_idle_cnot_time_over_L3"),
    ]
    rows: List[Dict[str, float]] = []
    for method, col in checks:
        values = (
            df[df["method"] == method][col]
            .dropna()
            .astype(float)
            .to_numpy()
        )
        if values.size == 0:
            continue
        median = float(np.median(values))
        if np.isclose(median, 0.0, atol=1e-12):
            max_relative_deviation = float(np.inf) if np.any(np.abs(values) > 0.0) else 0.0
        else:
            max_relative_deviation = float(np.max(np.abs(values - median) / abs(median)))
        rows.append(
            {
                "method": method,
                "ratio_column": col,
                "median_ratio": median,
                "max_relative_deviation": max_relative_deviation,
                "drift_flag": max_relative_deviation > drift_tolerance,
            }
        )
    return pd.DataFrame(rows)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "method",
        "L",
        "H",
        "S",
        "X",
        "Y",
        "Z",
        "CNOT",
        "CZ",
        "MEASURE",
        "RESET",
        "SINGLE_TOTAL",
        "TWO_QUBIT_TOTAL",
        "READOUT_TOTAL",
        "TOTAL_EVENTS",
        "single_qubit_estimation_count",
        "measurement_event_count",
        "idle_cnot_time",
        "CNOT_EQ_TWOQ",
        "weighted_cnot_component",
        "weighted_meas_component",
        "weighted_idle_component",
        "weighted_count_metric",
        "success_probability",
        "log10_success",
        "error_sum_single",
        "error_sum_two_qubit",
        "error_sum_measurement",
        "error_sum_idle",
        "error_sum_total",
        "rough_expected_baseline_cnot_eq",
        "rough_expected_custom_cnot_eq",
        "rough_expected_custom_measure",
        "rough_expected_custom_idle_cnot_time",
        "baseline_ratio_cnot_eq_over_L4",
        "custom_ratio_cnot_eq_over_L3",
        "custom_ratio_measure_over_L3",
        "custom_ratio_idle_cnot_time_over_L3",
    ]
    return df[cols].copy()


def plot_fidelity_scaling(df: pd.DataFrame, metric: PlotMetric = "success_probability") -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    style = {
        "baseline_openfermion_snake": {"label": "Baseline (OpenFermion snake)", "marker": "o"},
        "custom_ancilla_row_cnot": {"label": "Custom (row CNOT via ancilla)", "marker": "s"},
    }

    for method, cfg in style.items():
        sub = df[df["method"] == method].sort_values("L")
        y = sub[metric].clip(lower=1e-300)
        ax.plot(sub["L"], y, marker=cfg["marker"], linewidth=2, label=cfg["label"])

    if metric == "success_probability":
        ax.set_yscale("log")
        ylabel = "Estimated success probability"
        title = "Fidelity scaling vs L (product model)"
    elif metric == "error_sum_total":
        ax.set_yscale("log")
        ylabel = "Estimated total error budget"
        title = "Error-budget scaling vs L (summation model)"
    elif metric == "weighted_count_metric":
        ax.set_yscale("log")
        ylabel = "Weighted count metric"
        title = "Lambda-free weighted count scaling vs L"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    ax.set_xlabel("L")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def run_and_plot_resource_estimation(
    L_values: Sequence[int] = tuple(range(10, 21, 2)),
    permutation_kind: PermutationKind = "random",
    seed: int = 0,
    single_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-4,
    measurement_error: float = 1e-4,
    idle_error: float = 1e-6,
    idle_multiplier: float = 0.01,
    include_single_qubit_in_estimation: bool = False,
    measurement_duration_mu: float = 3.65,
    other_2q_cnot_equivalent: float = 1.0,
    include_reset_as_measurement: bool = True,
    plot_metric: PlotMetric = "success_probability",
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    df = run_resource_scaling_study(
        L_values=L_values,
        permutation_kind=permutation_kind,
        seed=seed,
        single_qubit_error=single_qubit_error,
        two_qubit_error=two_qubit_error,
        measurement_error=measurement_error,
        idle_error=idle_error,
        idle_multiplier=idle_multiplier,
        include_single_qubit_in_estimation=include_single_qubit_in_estimation,
        measurement_duration_mu=measurement_duration_mu,
        other_2q_cnot_equivalent=other_2q_cnot_equivalent,
        include_reset_as_measurement=include_reset_as_measurement,
    )
    fig, ax = plot_fidelity_scaling(df, metric=plot_metric)
    return build_summary_table(df), fig, ax
