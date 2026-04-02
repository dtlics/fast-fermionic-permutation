"""
Fermionic Permutation Visualization Module

This module provides tools to visualize the constant-depth fermionic permutation algorithm
on a 2D grid layout, with specific focus on Hybrid Schematic views for classical feedforward phases.
"""

import pennylane as qp
import numpy as np
import networkx as nx
import sympy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os
import math
import re
from typing import Dict, List, Sequence, Optional, Literal, Any, Tuple

# --- Utilities ---

def infer_grid_size(num_data_qubits: int) -> int:
    L = int(np.sqrt(num_data_qubits))
    if L * L != num_data_qubits:
        raise ValueError(f"Custom grid algorithm requires a square number of qubits, got {num_data_qubits}.")
    return L

def validate_permutation(permutation: Sequence[int], n: int) -> None:
    if len(permutation) != n:
        raise ValueError(f"Permutation length {len(permutation)} != {n}.")
    if sorted(permutation) != list(range(n)):
        raise ValueError("Permutation must be a reordering of 0..N-1.")

# --- Grid Topology ---

class GridTopology:
    def __init__(self, L: int, data_qubits: Optional[Sequence[str]] = None,
                 ancilla_qubits: Optional[Sequence[str]] = None):
        self.L = L
        self.data_map: Dict[tuple, str] = {}
        self.ancilla_map: Dict[tuple, str] = {}

        if data_qubits is None:
            for r in range(L):
                for c in range(L):
                    self.data_map[(r, c)] = qp.wires.Wires(r, 2 * c)
        else:
            if len(data_qubits) != L * L:
                raise ValueError("data_qubits must have length L*L.")
            data_qubits = list(data_qubits)
            for r in range(L):
                for c in range(L):
                    self.data_map[(r, c)] = data_qubits[r * L + c]

        if ancilla_qubits is None:
            if data_qubits is None:
                for r in range(L):
                    for k in range(L - 1):
                        self.ancilla_map[(r, k)] = qp.wires.Wires(r, 2 * k + 1)
            else:
                for r in range(L):
                    for k in range(L - 1):
                        self.ancilla_map[(r, k)] = qp.wires.Wires(f"anc_{r}_{k}")
        else:
            if len(ancilla_qubits) != L * (L - 1):
                raise ValueError("ancilla_qubits must have length L*(L-1).")
            ancilla_qubits = list(ancilla_qubits)
            idx = 0
            for r in range(L):
                for k in range(L - 1):
                    self.ancilla_map[(r, k)] = ancilla_qubits[idx]
                    idx += 1

        self.data_qubits = [self.data_map[(r, c)] for r in range(L) for c in range(L)]
        self.ancilla_qubits = [self.ancilla_map[(r, k)] for r in range(L) for k in range(L - 1)]
        self.all_qubits = self.data_qubits + self.ancilla_qubits

    def get_data_row(self, r: int) -> List[str]:
        return [self.data_map[(r, c)] for c in range(self.L)]

    def get_ancilla_row(self, r: int) -> List[str]:
        return [self.ancilla_map[(r, k)] for k in range(self.L - 1)]

    def get_data_col(self, c: int) -> List[str]:
        return [self.data_map[(r, c)] for r in range(self.L)]

# --- Decomposition ---

def decompose_permutation(L: int, logical_perm: Sequence[int]):
    G = nx.MultiGraph()
    G.add_nodes_from([f"C{i}" for i in range(L)], bipartite=0)
    G.add_nodes_from([f"T{i}" for i in range(L)], bipartite=1)

    edge_keys = []
    for r in range(L):
        for c in range(L):
            curr_idx = r * L + c
            dst_idx = logical_perm[curr_idx]
            target_c, target_r = dst_idx % L, dst_idx // L
            key = G.add_edge(f"C{c}", f"T{target_c}")
            edge_keys.append((f"C{c}", f"T{target_c}", key, curr_idx, target_r))

    matchings, H = [], G.copy()
    for _ in range(L):
        simp_H = nx.Graph(H.edges())
        matching_dict = nx.bipartite.maximum_matching(simp_H, top_nodes=[f"C{i}" for i in range(L)])
        matched_edges = []
        seen = set()
        for u, v in matching_dict.items():
            if (u, v) in seen or (v, u) in seen:
                continue
            seen.add((u, v))
            key = list(H[u][v].keys())[0]
            matched_edges.append((u, v, key))
            H.remove_edge(u, v, key=key)
        matchings.append(matched_edges)

    item_lookup = {(u, v, k): (idx, tr) for u, v, k, idx, tr in edge_keys}
    item_lookup.update({(v, u, k): (idx, tr) for u, v, k, idx, tr in edge_keys})

    temp_s1 = {c: {} for c in range(L)}
    post_s1_loc = {}
    for k, matching in enumerate(matchings):
        for u, v, key in matching:
            info_key = (u, v, key) if u.startswith("C") else (v, u, key)
            item_idx, final_r = item_lookup[info_key]
            c_curr, r_curr = item_idx % L, item_idx // L
            temp_s1[c_curr][r_curr] = k
            post_s1_loc[(k, c_curr)] = (item_idx, final_r)

    s1, s2, s3 = {c: [0] * L for c in range(L)}, {r: [0] * L for r in range(L)}, {c: [0] * L for c in range(L)}
    post_s2_loc = {}

    for c in range(L):
        for r, target in temp_s1[c].items():
            s1[c][r] = target

    for r in range(L):
        for c in range(L):
            if (r, c) in post_s1_loc:
                item_idx, final_r = post_s1_loc[(r, c)]
                real_target = logical_perm[item_idx]
                real_c = real_target % L
                s2[r][c] = real_c
                post_s2_loc[(r, real_c)] = final_r

    for c in range(L):
        for r in range(L):
             if (r, c) in post_s2_loc:
                 s3[c][r] = post_s2_loc[(r, c)]

    return s1, s2, s3

# --- Circuit Construction ---

class CompressedFermionicPermutation:
    def __init__(self, L: int, data_qubits: Optional[Sequence[str]] = None,
                 ancilla_qubits: Optional[Sequence[str]] = None,
                 topology: Optional[GridTopology] = None):
        self.L = L
        self.topo = topology or GridTopology(L, data_qubits=data_qubits, ancilla_qubits=ancilla_qubits)
        self._data_pos = {q: (r, c) for (r, c), q in self.topo.data_map.items()}
        self._data_index = {q: i for i, q in enumerate(self.topo.data_qubits)}

    def _get_direction(self, round_parity: int, r: int):
        data = self.topo.get_data_row(r)
        anc = self.topo.get_ancilla_row(r)
        if round_parity == 0:  # R->L
            return list(reversed(data)), list(reversed(anc))
        return data, anc  # L->R

    def _build_key(self, t_idx: int, r: int) -> str:
        return f"m_build_t{t_idx}_r{r}"

    def _unbuild_key(self, t_idx: int, r: int) -> str:
        return f"m_unbuild_t{t_idx}_r{r}"

    def build_prefix_xor_moments(self, round_parity: int, t_idx: int) -> List[qp.Moment]:
        moments = []
        moments.append(qp.Moment([qp.reset(a) for a in self.topo.ancilla_qubits]))
        moments.append(qp.Moment([qp.H(a) for a in self.topo.ancilla_qubits]))
        ops = []
        for r in range(self.L):
            Q, A = self._get_direction(round_parity, r)
            for k in range(self.L - 1):
                ops.append(qp.CNOT(A[k], Q[k + 1]))
        moments.append(qp.Moment(ops))
        ops = []
        for r in range(self.L):
            Q, A = self._get_direction(round_parity, r)
            for k in range(self.L - 1):
                ops.append(qp.CNOT(Q[k], A[k]))
        moments.append(qp.Moment(ops))
        ops = []
        for r in range(self.L):
            row_anc = self.topo.get_ancilla_row(r)
            ops.append(qp.measure(*row_anc, key=self._build_key(t_idx, r)))
        moments.append(qp.Moment(ops))
        ops = []
        for r in range(self.L):
            Q, A = self._get_direction(round_parity, r)
            row_anc = self.topo.get_ancilla_row(r)
            phys_index = {a: k for k, a in enumerate(row_anc)}
            key = self._build_key(t_idx, r)
            base = sympy.IndexedBase(key)
            prefix_expr = None
            for k in range(self.L - 1):
                idx = phys_index[A[k]]
                term = base[idx]
                prefix_expr = term if prefix_expr is None else sympy.Xor(prefix_expr, term)
                cond = qp.SympyCondition(prefix_expr)
                ops.append(qp.X(Q[k + 1]).with_classical_controls(cond))
        if ops:
            moments.append(qp.Moment(ops))
        return moments

    def unbuild_prefix_xor_moments(self, round_parity: int, t_idx: int) -> List[qp.Moment]:
        moments = []
        ops = []
        for r in range(self.L):
            ops.extend([qp.reset(a) for a in self.topo.get_ancilla_row(r)])
        moments.append(qp.Moment(ops))
        ops = []
        for r in range(self.L):
            Q, A = self._get_direction(round_parity, r)
            for k in range(self.L - 1):
                ops.append(qp.CNOT(Q[k], A[k]))
        moments.append(qp.Moment(ops))
        ops = []
        for r in range(self.L):
            Q, A = self._get_direction(round_parity, r)
            for k in range(self.L - 1):
                ops.append(qp.CNOT(A[k], Q[k + 1]))
        moments.append(qp.Moment(ops))
        ops_h = [qp.H(a) for a in self.topo.ancilla_qubits]
        ops_m = []
        for r in range(self.L):
            row_anc = self.topo.get_ancilla_row(r)
            ops_m.append(qp.measure(*row_anc, key=self._unbuild_key(t_idx, r)))
        moments.append(qp.Moment(ops_h))
        moments.append(qp.Moment(ops_m))
        ops = []
        for r in range(self.L):
            Q, A = self._get_direction(round_parity, r)
            row_anc = self.topo.get_ancilla_row(r)
            phys_index = {a: k for k, a in enumerate(row_anc)}
            key = self._unbuild_key(t_idx, r)
            base = sympy.IndexedBase(key)
            dir_indices = [phys_index[a] for a in A]
            suffix_expr = None
            for k in range(self.L - 2, -1, -1):
                idx = dir_indices[k]
                term = base[idx]
                suffix_expr = term if suffix_expr is None else sympy.Xor(suffix_expr, term)
                cond = qp.SympyCondition(suffix_expr)
                ops.append(qp.Z(Q[k]).with_classical_controls(cond))
        if ops:
            moments.append(qp.Moment(ops))
        return moments

    def _pair_key(self, q1: str, q2: str):
        i1 = self._data_index.get(q1)
        i2 = self._data_index.get(q2)
        if i1 is not None and i2 is not None:
            return (q1, q2) if i1 < i2 else (q2, q1)
        return (q1, q2) if str(q1) < str(q2) else (q2, q1)

    def get_phase_corrections_moments(self, active_swaps, round_parity: int) -> List[qp.Moment]:
        cz_counts: Dict[tuple, int] = {}
        z_counts: Dict[str, int] = {}
        def add_cz(a, b):
            key = self._pair_key(a, b)
            cz_counts[key] = cz_counts.get(key, 0) + 1
        def add_z(q):
            z_counts[q] = z_counts.get(q, 0) + 1
        for c in range(self.L):
            for r in active_swaps[c]:
                row_qs = self.topo.get_data_row(r)
                row_qs_next = self.topo.get_data_row(r + 1)
                TL = row_qs[c]
                BL = row_qs_next[c]
                TR = BR = None
                if round_parity == 0 and c < self.L - 1:
                    TR, BR = row_qs[c + 1], row_qs_next[c + 1]
                elif round_parity == 1 and c > 0:
                    TR, BR = row_qs[c - 1], row_qs_next[c - 1]
                add_cz(TL, BL)
                if TR:
                    add_cz(TL, TR)
                    add_z(TR)
                if BR:
                    add_cz(BL, BR)
                    add_z(BR)
                if TR and BR:
                    add_cz(TR, BR)
        z_ops = [qp.Z(q) for q, cnt in z_counts.items() if cnt % 2 == 1]
        cz_pairs = [pair for pair, cnt in cz_counts.items() if cnt % 2 == 1]
        vertical, horiz_even, horiz_odd, other = [], [], [], []
        for q1, q2 in cz_pairs:
            p1 = self._data_pos.get(q1)
            p2 = self._data_pos.get(q2)
            if p1 is None or p2 is None:
                other.append((q1, q2))
                continue
            r1, c1 = p1
            r2, c2 = p2
            if c1 == c2:
                vertical.append((q1, q2))
            elif r1 == r2:
                min_c = min(c1, c2)
                if min_c % 2 == 0:
                    horiz_even.append((q1, q2))
                else:
                    horiz_odd.append((q1, q2))
            else:
                other.append((q1, q2))
        moments = []
        if z_ops: moments.append(qp.Moment(z_ops))
        if vertical: moments.append(qp.Moment([qp.CZ(a, b) for a, b in vertical]))
        if horiz_even: moments.append(qp.Moment([qp.CZ(a, b) for a, b in horiz_even]))
        if horiz_odd: moments.append(qp.Moment([qp.CZ(a, b) for a, b in horiz_odd]))
        if other: moments.extend(list(qp.tape.qscript.QuantumScript([qp.CZ(a, b) for a, b in other]).moments))
        return moments

    def get_swaps_moments(self, active_swaps) -> List[qp.Moment]:
        ops = []
        for c in range(self.L):
            col_qs = self.topo.get_data_col(c)
            for r in active_swaps[c]:
                ops.append(qp.SWAP(col_qs[r], col_qs[r + 1]))
        return [qp.Moment(ops)]

# --- Grouped Circuit Generation ---

def build_grouped_circuit_structure(qubits: Sequence[str], permutation: Sequence[int],
                                    **kwargs) -> List[Dict[str, Any]]:
    L = infer_grid_size(len(qubits))
    program = CompressedFermionicPermutation(L, data_qubits=qubits, **kwargs)
    s1, s2, s3 = decompose_permutation(L, permutation)
    
    groups = []

    def append_group(stage_name, moments, round_parity=0, is_unbuild=False):
        if moments:
            groups.append({'name': stage_name, 'moments': moments, 'parity': round_parity, 'is_unbuild': is_unbuild})

    def process_col_stage(stage_prefix, schedule):
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

            m_prefix = program.build_prefix_xor_moments(round_parity, t)
            append_group(f"{stage_prefix} - Round {t} - Build Prefix", m_prefix, round_parity, is_unbuild=False)

            m_phase = program.get_phase_corrections_moments(active_swaps, round_parity)
            append_group(f"{stage_prefix} - Round {t} - Phase Correction", m_phase, round_parity)
            
            m_postfix = program.unbuild_prefix_xor_moments(round_parity, t)
            append_group(f"{stage_prefix} - Round {t} - Unbuild Prefix", m_postfix, round_parity, is_unbuild=True)
            
            m_swap = program.get_swaps_moments(active_swaps)
            append_group(f"{stage_prefix} - Round {t} - Swap", m_swap, round_parity)

            for c in range(L):
                arr = cur_perms[c]
                for r in active_swaps[c]:
                    arr[r], arr[r + 1] = arr[r + 1], arr[r]

    process_col_stage("ColStage1", s1)
    m_row = []
    for r in range(L):
        row_qs = program.topo.get_data_row(r)
        m_row.append(qp.Moment([qp.I(row_qs[0]).with_tags(f"Row {r}")]))
    append_group("RowStage", m_row)
    process_col_stage("ColStage2", s3)
    return groups

# --- Visualization (Grouped & Schematic & Hybrid) ---

class GridVisualizer:
    def __init__(self, topology: 'GridTopology', title_suffix: str = ""):
        self.topo = topology
        self.L = topology.L
        self.title_suffix = title_suffix
        
        self.q_coords = {}
        self.row_spacing = 1.5
        
        for (r, c), q in self.topo.data_map.items():
            self.q_coords[q] = (2 * c, -r * self.row_spacing)
            
        for (r, k), q in self.topo.ancilla_map.items():
            self.q_coords[q] = (2 * k + 1, -r * self.row_spacing)

    def _draw_qubit_grid(self, ax):
        for q, (x, y) in self.q_coords.items():
            is_data = q in self.topo.data_qubits
            color = 'lightgrey' if is_data else 'white'
            size = 0.8 if is_data else 0.6
            ofs = 0.4 if is_data else 0.3
            rect = patches.Rectangle((x - ofs, y - ofs), size, size, 
                                     linewidth=1, edgecolor='black', facecolor=color, zorder=1)
            ax.add_patch(rect)

    def _extract_control_sources(self, op: qp.ops.Operation) -> List[Tuple[float, float]]:
        if not isinstance(op, qp.ClassicallyControlledOperation):
            return []
        sources = []
        conditions = op.classical_controls
        for cond in conditions:
            if hasattr(cond, 'expr'): # SympyCondition
                symbols = cond.expr.free_symbols
            else:
                 symbols = []
            
            for s in symbols:
                s_str = str(s)
                match = re.search(r'_r(\d+)', s_str)
                if match:
                    r_idx = int(match.group(1))
                    y = -r_idx * self.row_spacing
                    sources.append((1.0, float(y))) 
        return sources

    def _draw_on_axis(self, ax, moment, step_label):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-1, 2 * self.L)
        ax.set_ylim(-self.L * self.row_spacing, 1)
        self._draw_qubit_grid(ax)

        for op in moment:
            control_sources = self._extract_control_sources(op)

            real_op = op
            is_controlled = False
            while isinstance(real_op, (qp.ClassicallyControlledOperation, qp.TaggedOperation)):
                 if isinstance(real_op, qp.ClassicallyControlledOperation):
                     is_controlled = True
                     real_op = real_op.without_classical_controls()
                 else:
                     real_op = real_op.sub_operation

            qubits = real_op.qubits
            gate = real_op.gate
            
            pts = [self.q_coords.get(q) for q in qubits if q in self.q_coords]
            if len(pts) != len(qubits): continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            
            label = ""
            font_weight = 'bold'
            if gate is None: continue
            
            if isinstance(gate, qp.MeasurementGate): label = "M"
            elif isinstance(gate, qp.ResetChannel): label = "|0>"; font_weight = 'normal'
            elif isinstance(gate, qp.HPowGate) and gate.exponent == 1.0: label = "H"
            elif isinstance(gate, qp.XPowGate) and gate.exponent == 1.0: label = "X"
            elif isinstance(gate, qp.ZPowGate) and gate.exponent == 1.0: label = "Z"
            elif isinstance(gate, (qp.CNotPowGate, qp.CZPowGate, qp.SwapPowGate)): pass
            else: label = str(gate)[:2]

            if control_sources:
                 target_x, target_y = xs[0], ys[0]
                 for src_x, src_y in control_sources:
                     ax.plot([src_x, target_x], [src_y, target_y], 
                             color='blue', linestyle='--', linewidth=1, alpha=0.7, zorder=10)

            if len(qubits) == 2:
                x1, y1 = xs[0], ys[0]; x2, y2 = xs[1], ys[1]
                style = '-' if not is_controlled else '--'
                ax.plot([x1, x2], [y1, y2], color='black', linewidth=2, linestyle=style, zorder=2)
                if isinstance(gate, qp.CNotPowGate):
                    ax.add_patch(patches.Circle((x1, y1), 0.15, color='black', zorder=3))
                    ax.add_patch(patches.Circle((x2, y2), 0.15, facecolor='white', edgecolor='black', zorder=3))
                    ax.text(x2, y2, '+', ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)
                elif isinstance(gate, qp.CZPowGate):
                    for tx, ty in zip(xs, ys): ax.add_patch(patches.Circle((tx, ty), 0.15, color='black', zorder=3))
                elif isinstance(gate, qp.SwapPowGate):
                    for tx, ty in zip(xs, ys): ax.text(tx, ty, 'x', ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)
            
            if isinstance(gate, (qp.MeasurementGate, qp.ResetChannel, qp.HPowGate, qp.XPowGate, qp.ZPowGate)):
                for tx, ty in zip(xs, ys):
                    final_label = label
                    if is_controlled and not control_sources: final_label = "C-" + label
                    ax.text(tx, ty, final_label, ha='center', va='center', fontsize=8, fontweight=font_weight, zorder=10, color='black')
        
        ax.set_title(step_label, fontsize=8)

    def _draw_schematic_on_axis(self, ax, moments_list, step_label="Schematic", parity=0, is_unbuild=False):
        # V8: Direction Inversion for Unbuild
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-1, 2 * self.L)
        ax.set_ylim(-self.L * self.row_spacing, 1)
        self._draw_qubit_grid(ax)

        corrections = []
        measurements = []

        # 1. Parse Ops
        for moment in moments_list:
            for op in moment:
                real_op = op
                is_controlled = False
                while isinstance(real_op, (qp.ClassicallyControlledOperation, qp.TaggedOperation)):
                     if isinstance(real_op, qp.ClassicallyControlledOperation):
                         is_controlled = True
                         real_op = real_op.without_classical_controls()
                     else:
                         real_op = real_op.sub_operation
                
                gate = real_op.gate
                qubits = real_op.qubits
                
                if isinstance(gate, qp.MeasurementGate):
                    for q in qubits: measurements.append(q)
                
                if is_controlled:
                    q = qubits[0]
                    g_name = "X" if isinstance(gate, qp.XPowGate) else "Z"
                    corrections.append((q, g_name))

        # 2. Draw per row
        for r in range(self.L):
            row_meas = [q for q in measurements if self.q_coords[q][1] == -r * self.row_spacing]
            row_corr = [q for q in corrections if self.q_coords[q[0]][1] == -r * self.row_spacing]
            
            if not row_meas and not row_corr: continue
            
            meas_xs = [self.q_coords[q][0] for q in row_meas]
            corr_xs = [self.q_coords[c_q][0] for c_q, _ in row_corr]
            
            all_xs = meas_xs + corr_xs
            
            if not all_xs: continue
            
            min_x = min(all_xs)
            max_x = max(all_xs)
            
            if parity == 0:
                 if not is_unbuild:
                     # R->L
                     start_x = max(meas_xs) if meas_xs else max_x
                     end_x = min_x
                     direction_sign = -1
                 else:
                     # L->R (Backwards scan)
                     start_x = min(meas_xs) if meas_xs else min_x
                     end_x = max_x
                     direction_sign = 1
            else: # Parity 1
                 if not is_unbuild:
                     # L->R
                     start_x = min(meas_xs) if meas_xs else min_x
                     end_x = max_x
                     direction_sign = 1
                 else:
                     # R->L
                     start_x = max(meas_xs) if meas_xs else max_x
                     end_x = min_x
                     direction_sign = -1
                
            y = -r * self.row_spacing
            y_bus = y + 0.8
            offset = 0.08

            # Draw Bus Line (Horizontal)
            ax.plot([start_x, end_x], [y_bus + offset, y_bus + offset], color='blue', linewidth=2)
            ax.plot([start_x, end_x], [y_bus - offset, y_bus - offset], color='blue', linewidth=2)

            # Draw Measurements
            for q in row_meas:
                x, _ = self.q_coords[q]
                
                # M Box
                ax.add_patch(patches.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, facecolor='white', edgecolor='black', zorder=5))
                ax.text(x, y, "M", ha='center', va='center', fontweight='bold', zorder=6)
                
                is_start = (x == start_x)
                
                if is_start:
                    if direction_sign == 1: # L->R
                         # Trace 1: (x-o, y+0.25) goes up to (x-o, y_bus+o), then right.
                         ax.plot([x-offset, x-offset], [y+0.25, y_bus+offset], color='blue', linewidth=2) 
                         # Trace 2: (x+o, y+0.25) goes up to (x+o, y_bus-offset), then right.
                         ax.plot([x+offset, x+offset], [y+0.25, y_bus-offset], color='blue', linewidth=2) 
                    else: # R->L
                         # Trace 1 (Right/Outer): (x+o, y+0.25) -> (x+o, y_bus+o) -> Left.
                         ax.plot([x+offset, x+offset], [y+0.25, y_bus+offset], color='blue', linewidth=2)
                         # Trace 2 (Left/Inner): (x-o, y+0.25) -> (x-o, y_bus-offset) -> Left.
                         ax.plot([x-offset, x-offset], [y+0.25, y_bus-offset], color='blue', linewidth=2)
                         
                else: # Intermediate Join
                    ax.plot([x - offset, x - offset], [y + 0.25, y_bus - offset], color='blue', linewidth=1)
                    ax.plot([x + offset, x + offset], [y + 0.25, y_bus - offset], color='blue', linewidth=1)
                    
                    # XOR on Bus
                    ax.add_patch(patches.Circle((x, y_bus), 0.2, facecolor='white', edgecolor='blue', linewidth=2, zorder=15))
                    ax.text(x, y_bus, '+', ha='center', va='center', color='blue', fontsize=12, fontweight='bold', zorder=16)

            # Draw Corrections
            for c_q, g_type in row_corr:
                tx, ty = self.q_coords[c_q]
                y_bus = ty + 0.8
                
                ax.add_patch(patches.Rectangle((tx - 0.25, ty - 0.25), 0.5, 0.5, facecolor='none', edgecolor='blue', linestyle='--', zorder=5))
                ax.text(tx, ty, g_type, ha='center', va='center', color='blue', fontweight='bold', zorder=6)
                
                ax.add_patch(patches.Circle((tx, y_bus), 0.1, color='blue', zorder=7))
                offset = 0.05
                ax.plot([tx - offset, tx - offset], [y_bus, ty + 0.25], color='blue', linewidth=1)
                ax.plot([tx + offset, tx + offset], [y_bus, ty + 0.25], color='blue', linewidth=1)

        ax.set_title(step_label, fontsize=8)

    def draw_hybrid_group(self, group: Dict[str, Any], save_path: str):
        name = group['name']
        moments = group['moments']
        parity = group.get('parity', 0)
        is_unbuild = group.get('is_unbuild', False)
        
        if not moments: return
        
        schematic_indices = []
        standard_indices = []
        
        for i, m in enumerate(moments):
            is_schematic = False
            for op in m:
                real_op = op
                is_controlled = False
                while isinstance(real_op, (qp.ClassicallyControlledOperation, qp.TaggedOperation)):
                     if isinstance(real_op, qp.ClassicallyControlledOperation): 
                        is_controlled = True
                        real_op = real_op.without_classical_controls()
                     else: real_op = real_op.sub_operation
                
                gate = real_op.gate
                if isinstance(gate, qp.MeasurementGate) or is_controlled:
                    is_schematic = True
                    break
            
            if is_schematic: schematic_indices.append(i)
            else: standard_indices.append(i)
        
        if not schematic_indices:
            self.draw_group(group, save_path)
            return
            
        num_standard = len(standard_indices)
        num_plots = num_standard + 1
        
        cols, rows = num_plots, 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, 4.5), constrained_layout=True)
        fig.suptitle(f"{self.title_suffix}: {name}", fontsize=14)
        
        if isinstance(axes, np.ndarray): ax_list = axes.flatten()
        else: ax_list = [axes]
        
        for i, idx in enumerate(standard_indices):
            ax = ax_list[i]
            moment = moments[idx]
            ops = list(moment.operations)
            step_title = "Empty"
            if ops:
                first_op = ops[0]
                real_op = first_op
                while isinstance(real_op, (qp.ClassicallyControlledOperation, qp.TaggedOperation)):
                     if isinstance(real_op, qp.ClassicallyControlledOperation): real_op = real_op.without_classical_controls()
                     else: real_op = real_op.sub_operation
                gate = real_op.gate
                if gate:
                    step_title = gate.__class__.__name__.replace("PowGate", "").replace("Gate", "")
                    if isinstance(gate, qp.MeasurementGate): step_title = "Measure"
                    if isinstance(gate, qp.ResetChannel): step_title = "Reset"
            
            self._draw_on_axis(ax, moment, f"Step {idx+1}: {step_title}")
            
        ax_schematic = ax_list[num_standard]
        schematic_moments = [moments[i] for i in schematic_indices]
        self._draw_schematic_on_axis(ax_schematic, schematic_moments, step_label="Schematic", parity=parity, is_unbuild=is_unbuild)

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def draw_group(self, group: Dict[str, Any], save_path: str):
        moments = group['moments']
        name = group['name']
        N = len(moments)
        if N == 0: return

        cols, rows = N, 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, 4.5), constrained_layout=True)
        fig.suptitle(f"{self.title_suffix}: {name}", fontsize=14)
        
        if isinstance(axes, np.ndarray): ax_list = axes.flatten()
        else: ax_list = [axes]
        
        for i, moment in enumerate(moments):
            ax = ax_list[i]
            ops = list(moment.operations)
            step_title = "Empty"
            if ops:
                first_op = ops[0]
                real_op = first_op
                while isinstance(real_op, (qp.ClassicallyControlledOperation, qp.TaggedOperation)):
                     if isinstance(real_op, qp.ClassicallyControlledOperation): real_op = real_op.without_classical_controls()
                     else: real_op = real_op.sub_operation
                gate = real_op.gate
                if gate:
                    step_title = gate.__class__.__name__.replace("PowGate", "").replace("Gate", "")
                    if isinstance(gate, qp.MeasurementGate): step_title = "Measure"
                    if isinstance(gate, qp.ResetChannel): step_title = "Reset"
            self._draw_on_axis(ax, moment, f"Step {i+1}: {step_title}")

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

# --- Public API ---

def visualize_permutation(L: int, permutation: Sequence[int], output_dir: str):
    """
    Visualizes the steps of a fermionic permutation algorithm on an LxL grid.
    
    Args:
        L: Grid dimension (L x L).
        permutation: List of length L*L, where index i is mapped to permutation[i].
        output_dir: Directory where visualization frames will be saved.
    """
    print(f"Visualizing {L}x{L} permutation in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    validate_permutation(permutation, L*L)
    
    # 1. Setup Grid & Circuit
    data_qubits = [qp.wires.Wires(r, 2 * c) for r in range(L) for c in range(L)]
    program = CompressedFermionicPermutation(L, data_qubits=data_qubits)
    
    # 2. Build Groups
    groups = build_grouped_circuit_structure(data_qubits, permutation)
    
    # 3. Visualize
    vis = GridVisualizer(program.topo, title_suffix=f"Permutation {L}x{L}")
    
    for i, group in enumerate(groups):
        safe_name = group['name'].replace(" ", "_").replace("-", "").replace("__", "_")
        filename = f"{i:02d}_{safe_name}.png"
        path = os.path.join(output_dir, filename)
        
        if "Build_Prefix" in safe_name or "Unbuild_Prefix" in safe_name:
             vis.draw_hybrid_group(group, path)
        else:
             vis.draw_group(group, path)
             
        # Optional: Print progress sparingly
        if i % 5 == 0:
            print(f"Generated frame {i}/{len(groups)}")
            
    print(f"Visualization complete! Frames saved to {output_dir}")

