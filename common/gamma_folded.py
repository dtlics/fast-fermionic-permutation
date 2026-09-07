"""Folded-congruence Gamma with CNOT-equivalent two-qubit depth 2L + O(1).

This ancilla-free construction implements an admissible Gamma polynomial.
The boundary and asymptotic schedules may use different legal coset siblings;
all have the same Gamma--Col--Gamma semantics required by
:mod:`common.fp_2d`.

For every L >= 2 it obeys the uniform upper bound

    2L + 4  (even L),
    2L + 6  (odd L).

The explicit boundary depths for ``L=2,3,4,5,6`` are respectively
``5, 6, 7, 10, 13``.  The canonical direct construction is retained only for
``L=2``.  The shorter schedules use legal noncanonical Gamma-coset siblings.

A final parallel Z correction is appended.  It adds at most one ordinary
Cirq moment but no CNOT-equivalent two-qubit depth.

The all-L algebra and residue-class placement proof are Appendix A of the
paper ("Folded Gamma Construction and Proof", Theorem on the zero-extra-layer
residual).  The analytic ``L>=7`` production
path evaluates those formulas directly; the boundary path dispatches the
certified explicit tables.  Neither path invokes an optimizer.
"""

from typing import Dict, List, Set, Tuple

import cirq

from common.grid import make_system_qubits
from common.gamma_small import small_gamma_layers, small_gamma_z_qubits


Edge = Tuple[int, int]


# Stable experiment provenance is deliberately branch-specific.  The
# asymptotic schedule is unchanged, so its published rows keep their original
# identifier; only the new L=3..6 boundary witnesses invalidate old rows.
FOLDED_GAMMA_SCHEDULE_MODEL = "folded_zero_extra_ge7_v1"
FOLDED_GAMMA_BOUNDARY_SCHEDULE_MODEL = "folded_boundary_l3_l6_v1"
GAMMA_SCHEDULE_NOT_APPLICABLE = "not_applicable"


def folded_gamma_schedule_model(L: int) -> str:
    """Return the exact folded-Gamma scheduler provenance used at size ``L``."""

    if L < 2:
        raise ValueError("L must be at least 2")
    if 3 <= L <= 6:
        return FOLDED_GAMMA_BOUNDARY_SCHEDULE_MODEL
    # L=2 retains the unchanged canonical fallback; L>=7 retains the unchanged
    # zero-extra schedule.  Both were already covered by the original model.
    return FOLDED_GAMMA_SCHEDULE_MODEL


def _qid(r: int, c: int, L: int) -> int:
    return r * L + c


def _toggle(edges: Set[Edge], a: int, b: int) -> None:
    if a == b:
        return
    edge = (a, b) if a < b else (b, a)
    if edge in edges:
        edges.remove(edge)
    else:
        edges.add(edge)


def _edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _h_edge(L: int, row: int, cell: int) -> Edge:
    return _edge(_qid(row, cell, L), _qid(row, cell + 1, L))


def _v_edge(L: int, band: int, column: int) -> Edge:
    return _edge(_qid(band, column, L), _qid(band + 1, column, L))


def _row_slope(r: int, L: int) -> int:
    """Return +1 for a backslash diagonal and -1 for a slash diagonal."""

    m = (L + 1) // 2
    if r < m - 1:
        return 1 if r % 2 == 0 else -1
    if r == m - 1:
        return -1
    return 1 if r % 2 == 1 else -1


def _local_phase_edges(L: int) -> Set[Edge]:
    """Edges of polar((D.T S D) tensor (D.T U D)) from closed forms."""

    m = (L + 1) // 2

    # R=D.T S D: one directed edge on every row link and one loop.
    row_entries: List[Tuple[int, int]] = []
    for r in range(L - 1):
        row_entries.append((r, r + 1) if _row_slope(r, L) == 1 else (r + 1, r))
    p = m if m % 2 == 1 else m - 1
    row_entries.append((p, p))

    # C=D.T U D: every superdiagonal entry, plus all diagonal entries
    # except the two fold sites m-1,m.
    col_entries = [(c, c + 1) for c in range(L - 1)]
    col_entries.extend((c, c) for c in range(L) if c < m - 1 or c > m)

    edges: Set[Edge] = set()
    for r, s in row_entries:
        for c, d in col_entries:
            _toggle(edges, _qid(r, c, L), _qid(s, d, L))
    return edges


def _gather_layers(L: int) -> List[List[Tuple[str, int, int]]]:
    """Folded vertical prefixes/suffixes followed by horizontal ones."""

    m = (L + 1) // 2
    half_depth = max(m - 1, L - m - 1)
    layers: List[List[Tuple[str, int, int]]] = []

    for t in range(half_depth):
        layer: List[Tuple[str, int, int]] = []
        if t < m - 1:
            layer.extend(("cx", _qid(t, c, L), _qid(t + 1, c, L)) for c in range(L))
        if t < L - m - 1:
            layer.extend(
                ("cx", _qid(L - 1 - t, c, L), _qid(L - 2 - t, c, L))
                for c in range(L)
            )
        if layer:
            layers.append(layer)

    for t in range(half_depth):
        layer = []
        if t < m - 1:
            layer.extend(("cx", _qid(r, t, L), _qid(r, t + 1, L)) for r in range(L))
        if t < L - m - 1:
            layer.extend(
                ("cx", _qid(r, L - 1 - t, L), _qid(r, L - 2 - t, L))
                for r in range(L)
            )
        if layer:
            layers.append(layer)
    return layers


def _diagonal_base_layers(
    L: int,
) -> Tuple[List[List[Tuple[str, int, int]]], Set[Edge], Set[Edge]]:
    """Eight layers producing all plaquette diagonals plus vertical junk."""

    layers: List[List[Tuple[str, int, int]]] = []
    diagonal: Set[Edge] = set()
    junk: Set[Edge] = set()
    for cell_parity in range(2):
        directions: Dict[int, int] = {}
        forward: List[Tuple[str, int, int]] = []
        for row in range(1, L, 2):
            direction = (
                _row_slope(row, L)
                if row < L - 1
                else -_row_slope(row - 1, L)
            )
            directions[row] = direction
            for c in range(cell_parity, L - 1, 2):
                source_c, target_c = (c, c + 1) if direction == 1 else (c + 1, c)
                forward.append(("cx", _qid(row, source_c, L), _qid(row, target_c, L)))
        layers.append(forward)

        for band_parity in range(2):
            middle: List[Tuple[str, int, int]] = []
            for band in range(band_parity, L - 1, 2):
                odd_row = band if band % 2 else band + 1
                direction = directions[odd_row]
                for c in range(cell_parity, L - 1, 2):
                    source_c, target_c = (c, c + 1) if direction == 1 else (c + 1, c)
                    other_row = band + 1 if odd_row == band else band
                    middle.append(
                        ("cz", _qid(band, target_c, L), _qid(band + 1, target_c, L))
                    )
                    _toggle(
                        diagonal,
                        _qid(odd_row, source_c, L),
                        _qid(other_row, target_c, L),
                    )
                    _toggle(junk, _qid(band, target_c, L), _qid(band + 1, target_c, L))
            layers.append(middle)
        layers.append(list(forward))
    return layers, diagonal, junk


def _nonidentity_frames_before(
    layers: List[List[Tuple[str, int, int]]],
) -> List[Set[int]]:
    """Track the nonidentity CX targets without large bitset frame integers."""

    changed: Set[int] = set()
    result: List[Set[int]] = []
    for layer in layers:
        result.append(set(changed))
        for kind, a, b in layer:
            if kind == "cx":
                if a in changed:
                    raise AssertionError("odd-row base unexpectedly modified a CX control")
                if b in changed:
                    changed.remove(b)
                else:
                    changed.add(b)
    if changed:
        raise AssertionError("odd-row diagonal base did not restore its CX frame")
    return result


def _explicit_overlays(L: int) -> List[Tuple[Edge, int]]:
    """Literal correction CZs placed in free identity-frame base slots."""

    if L == 3:
        return [
            ((_qid(0, 1, L), _qid(1, 1, L)), 1),
            ((_qid(1, 1, L), _qid(2, 1, L)), 2),
        ]
    if L == 4:
        return [
            ((_qid(0, 1, L), _qid(1, 1, L)), 1),
            ((_qid(1, 1, L), _qid(2, 1, L)), 2),
            ((_qid(0, 2, L), _qid(1, 2, L)), 5),
            ((_qid(1, 2, L), _qid(2, 2, L)), 6),
            ((_qid(0, 3, L), _qid(1, 3, L)), 1),
        ]
    if L < 5:
        raise ValueError("explicit eight-layer overlays require L >= 3")

    m = (L + 1) // 2
    p = m if m % 2 else m - 1
    overlays: List[Tuple[Edge, int]] = [
        ((_qid(p - 1, m - 1, L), _qid(p, m - 1, L)), 1),
        ((_qid(p, m - 1, L), _qid(p + 1, m - 1, L)), 2),
        ((_qid(p - 1, m, L), _qid(p, m, L)), 5),
        ((_qid(p, m, L), _qid(p + 1, m, L)), 6),
    ]
    if m % 2:
        overlays.append(((_qid(p, 0, L), _qid(p, 1, L)), 5))
    elif L % 2:
        overlays.append(((_qid(p, L - 2, L), _qid(p, L - 1, L)), 1))
    else:
        overlays.append(((_qid(p, L - 2, L), _qid(p, L - 1, L)), 5))
    return overlays


def _two_colour_degree_two(edges: Set[Edge]) -> List[List[Tuple[str, int, int]]]:
    """Edge-colour a bipartite graph of maximum degree two without recursion."""

    if not edges:
        return []
    edge_list = sorted(edges)
    incident: Dict[int, List[int]] = {}
    for eid, (a, b) in enumerate(edge_list):
        incident.setdefault(a, []).append(eid)
        incident.setdefault(b, []).append(eid)
    if max(map(len, incident.values())) > 2:
        raise AssertionError("residual midpoint correction exceeds degree two")

    colour: Dict[int, int] = {}
    for seed in range(len(edge_list)):
        if seed in colour:
            continue
        colour[seed] = 0
        stack = [seed]
        while stack:
            eid = stack.pop()
            for q in edge_list[eid]:
                for other in incident[q]:
                    if other == eid:
                        continue
                    wanted = colour[eid] ^ 1
                    if other in colour and colour[other] != wanted:
                        raise AssertionError("midpoint correction contains an odd cycle")
                    if other not in colour:
                        colour[other] = wanted
                        stack.append(other)
    layers: List[List[Tuple[str, int, int]]] = [[], []]
    for eid, (a, b) in enumerate(edge_list):
        layers[colour[eid]].append(("cz", a, b))
    return [layer for layer in layers if layer]


def _midpoint_layers(L: int, phase_edges: Set[Edge]) -> List[List[Tuple[str, int, int]]]:
    """Route the folded local phase in ten layers (five when L=2)."""

    if L == 2:
        return [
            [("cx", _qid(1, 0, L), _qid(1, 1, L))],
            [("cz", _qid(0, 1, L), _qid(1, 1, L))],
            [("cx", _qid(1, 0, L), _qid(1, 1, L))],
            [("cz", _qid(0, 1, L), _qid(1, 1, L))],
            [("cz", _qid(1, 0, L), _qid(1, 1, L))],
        ]

    nearest: Set[Edge] = set()
    expected_diagonal: Set[Edge] = set()
    for edge in phase_edges:
        a, b = edge
        ra, ca = divmod(a, L)
        rb, cb = divmod(b, L)
        if abs(ra - rb) + abs(ca - cb) == 1:
            nearest.add(edge)
        else:
            expected_diagonal.add(edge)

    layers, made_diagonal, junk = _diagonal_base_layers(L)
    if made_diagonal != expected_diagonal:
        raise AssertionError("eight-layer base did not make the folded diagonals")
    correction = nearest ^ junk
    if any(
        abs(divmod(a, L)[0] - divmod(b, L)[0])
        + abs(divmod(a, L)[1] - divmod(b, L)[1])
        != 1
        for a, b in correction
    ):
        raise AssertionError("midpoint correction is not nearest-neighbour")

    nonidentity_before = _nonidentity_frames_before(layers)
    used = [{q for _, a, b in layer for q in (a, b)} for layer in layers]
    selected: Set[Edge] = set()
    for edge, t in _explicit_overlays(L):
        a, b = edge
        if edge not in correction:
            raise AssertionError(f"overlay is not a midpoint correction: {edge}")
        if a in used[t] or b in used[t]:
            raise AssertionError(f"midpoint overlay collision in layer {t}")
        if a in nonidentity_before[t] or b in nonidentity_before[t]:
            raise AssertionError(f"midpoint overlay is not in an identity frame: layer {t}")
        layers[t].append(("cz", a, b))
        used[t].update((a, b))
        selected.add(edge)

    result = layers + _two_colour_degree_two(correction - selected)
    if len(result) > 10:
        raise AssertionError("folded midpoint exceeded ten layers")
    return result


def _zero_extra_base_layers(L: int) -> List[List[Tuple[str, int, int]]]:
    """Eight-slot midpoint with the one central parity gadget reversed."""

    work, _, _ = _diagonal_base_layers(L)
    m = (L + 1) // 2
    row = m if m % 2 else m - 1
    cell = m if m % 2 == 0 else m - 2
    parity_start = 4 * (cell % 2)

    for layer_index in (parity_start, parity_start + 3):
        changed: List[Tuple[str, int, int]] = []
        for kind, a, b in work[layer_index]:
            ra, ca = divmod(a, L)
            _, cb = divmod(b, L)
            if kind == "cx" and ra == row and min(ca, cb) == cell:
                changed.append((kind, b, a))
            else:
                changed.append((kind, a, b))
        work[layer_index] = changed

    old_direction = (
        _row_slope(row, L)
        if row < L - 1
        else -_row_slope(row - 1, L)
    )
    old_target = cell + (1 if old_direction == 1 else 0)
    new_target = cell + (0 if old_direction == 1 else 1)
    for band in (row - 1, row):
        layer_index = parity_start + 1 + band % 2
        old_edge = _v_edge(L, band, old_target)
        new_edge = _v_edge(L, band, new_target)
        replaced: List[Tuple[str, int, int]] = []
        hit = False
        for kind, a, b in work[layer_index]:
            if kind == "cz" and _edge(a, b) == old_edge:
                replaced.append(("cz", *new_edge))
                hit = True
            else:
                replaced.append((kind, a, b))
        if not hit:
            raise AssertionError("central folded-Gamma coupler was not found")
        work[layer_index] = replaced
    return work


def _zero_extra_telescope_patches(L: int) -> Tuple[Tuple[int, int], ...]:
    """Logical K2,2 patches, represented by ``(row band, column cell)``."""

    m = (L + 1) // 2
    if L % 4 == 3:
        return ((m - 2, m), (m - 1, m - 3), (m - 1, m - 2), (m - 1, m))
    if L % 4 == 0:
        return (
            (m - 2, m),
            (m - 1, m - 3),
            (m - 1, m - 2),
            (m - 1, m + 1),
        )
    if L % 4 == 1:
        return ((m - 1, m - 3), (m, m - 2))
    return (
        (m - 1, m - 3),
        (m - 1, m),
        (m - 1, m + 1),
        (m, m - 2),
    )


def _zero_extra_rail_edges(L: int) -> Set[Edge]:
    """The three vertical rails and split horizontal path (4L-4 edges)."""

    m = (L + 1) // 2
    edges: Set[Edge] = set()

    for band in range(m - 1):
        for column in (m - 1, m, L - 1):
            edges.add(_v_edge(L, band, column))
    for band in range(m, L - 1):
        for column in (0, m - 1, m):
            edges.add(_v_edge(L, band, column))

    boundary = L - 1 if m % 2 == 0 else 0
    reflected = m if L % 2 else m + 2
    for column in (boundary, m - 3, reflected):
        edges.add(_v_edge(L, m - 1, column))

    if L % 4 == 3:
        upper_cells = set(range(0, m - 3)) | set(range(m - 1, L - 1))
    elif L % 4 == 0:
        upper_cells = (
            set(range(0, m - 3)) | {m - 1} | set(range(m + 2, L - 1))
        )
    elif L % 4 == 1:
        upper_cells = {m - 3, m - 2}
    else:
        upper_cells = {m - 3, m - 2, m, m + 1}
    for cell in range(L - 1):
        edges.add(_h_edge(L, m - 1 if cell in upper_cells else m, cell))

    if len(edges) != 4 * L - 4:
        raise AssertionError("folded-Gamma rail formula has the wrong size")
    return edges


def _zero_extra_central_vertical_table(L: int) -> Tuple[Tuple[int, int], ...]:
    """Return ``(column, base slot)``; ``-1`` denotes the last gather layer."""

    m = (L + 1) // 2
    if L % 4 == 3:
        return ((m - 3, 2), (m, 6), (L - 1, 0))
    if L % 4 == 0:
        return ((m - 3, 2), (m + 2, 6), (L - 1, 2))
    if L % 4 == 1:
        return ((0, 1), (m - 3, 1), (m, -1))
    return ((0, 1), (m - 3, 1), (m + 2, 5))


def _zero_extra_central_horizontal_table(
    L: int,
) -> Tuple[Tuple[int, int, int], ...]:
    """Return ``(row, cell, base slot)``; slot 8 is inverse-H slot zero."""

    m = (L + 1) // 2
    if L % 4 == 3:
        return (
            (m - 1, m - 1, 2),
            (m - 1, m, 8),
            (m, m - 3, 0),
            (m, m - 2, 3),
        )
    if L % 4 == 0:
        return (
            (m - 1, m - 1, 2),
            (m, m - 3, 0),
            (m, m - 2, 3),
            (m, m, 0),
            (m, m + 1, 3),
        )
    if L % 4 == 1:
        return (
            (m - 1, m - 3, 0),
            (m - 1, m - 2, 3),
            (m, m - 1, 5),
            (m, m, 8),
        )
    return (
        (m - 1, m - 3, 0),
        (m - 1, m - 2, 3),
        (m - 1, m, 0),
        (m - 1, m + 1, 3),
        (m, m - 1, 5),
    )


def _append_zero_extra_cz(
    L: int,
    layers: List[List[Tuple[str, int, int]]],
    occupied: List[Set[int]],
    tag: int,
    edge: Edge,
) -> None:
    """Append a formula gate while enforcing its local matching certificate."""

    if not 0 <= tag < len(layers):
        raise AssertionError("folded-Gamma layer index is out of range")
    a, b = edge
    ra, ca = divmod(a, L)
    rb, cb = divmod(b, L)
    if abs(ra - rb) + abs(ca - cb) != 1:
        raise AssertionError("folded-Gamma formula emitted a non-NN CZ")
    if a in occupied[tag] or b in occupied[tag]:
        raise AssertionError("folded-Gamma formula has a layer collision")
    layers[tag].append(("cz", a, b))
    occupied[tag].update((a, b))


def _zero_extra_layers(L: int) -> List[List[Tuple[str, int, int]]]:
    """Explicit no-new-layer Gamma schedule for every ``L >= 7``."""

    gather = _gather_layers(L)
    g = len(gather)
    h = g // 2
    if g != 2 * ((L + 1) // 2 - 1):
        raise AssertionError("unexpected folded gather shape")
    base = _zero_extra_base_layers(L)
    layers = (
        [list(layer) for layer in gather]
        + [list(layer) for layer in base]
        + [list(layer) for layer in reversed(gather)]
    )
    occupied = [{q for _, a, b in layer for q in (a, b)} for layer in layers]
    m = (L + 1) // 2

    # Before the column gather, a vertical CZ on a difference coordinate
    # deposits all four products of the corresponding 2x2 plaquette.
    for band, cell in _zero_extra_telescope_patches(L):
        physical_column = cell + 1 if cell < m - 1 else cell
        tag = h
        if L % 4 == 3 and (band, cell) == (m - 1, m):
            tag = g + 8 + h - 1
        _append_zero_extra_cz(
            L, layers, occupied, tag, _v_edge(L, band, physical_column)
        )

    rails = _zero_extra_rail_edges(L)
    pair_a = (1, 2) if m % 2 == 0 else (5, 6)
    pair_b = (5, 6) if m % 2 == 0 else (1, 2)

    # All noncentral vertical rail edges alternate by row-band parity.
    for a, b in sorted(rails):
        ra, ca = divmod(a, L)
        rb, cb = divmod(b, L)
        if ca != cb:
            continue
        band, column = min(ra, rb), ca
        if band == m - 1:
            continue
        if band < m - 1:
            pair = pair_a if column == m - 1 else pair_b
        else:
            pair = pair_b if column == m - 1 else pair_a
        _append_zero_extra_cz(
            L, layers, occupied, g + pair[band % 2], (a, b)
        )

    for column, slot in _zero_extra_central_vertical_table(L):
        tag = g - 1 if slot == -1 else g + slot
        _append_zero_extra_cz(
            L, layers, occupied, tag, _v_edge(L, m - 1, column)
        )

    central_horizontal = {
        _h_edge(L, row, cell)
        for row, cell, _ in _zero_extra_central_horizontal_table(L)
    }
    for a, b in sorted(rails - central_horizontal):
        ra, ca = divmod(a, L)
        rb, cb = divmod(b, L)
        if ra != rb:
            continue
        cell = min(ca, cb)
        if cell <= m - 2:
            local_step = cell + 2
        elif cell >= m:
            local_step = L - cell
        else:
            raise AssertionError("unlisted central folded-Gamma edge")
        if not 0 <= local_step < h:
            raise AssertionError("horizontal folded-Gamma step is out of range")
        _append_zero_extra_cz(
            L, layers, occupied, h + local_step, (a, b)
        )

    for row, cell, slot in _zero_extra_central_horizontal_table(L):
        tag = g + 8 if slot == 8 else g + slot
        _append_zero_extra_cz(
            L, layers, occupied, tag, _h_edge(L, row, cell)
        )

    if len(layers) != 2 * g + 8 or any(not layer for layer in layers):
        raise AssertionError("folded-Gamma zero-extra depth accounting failed")
    return layers


def _zero_extra_z_correction(L: int) -> List[int]:
    """Closed-form linear repair for the ``L >= 7`` streamed schedule."""

    m = (L + 1) // 2
    if L % 4 in (0, 3):
        left = set(range(0, m - 2, 2))
        reflection_axis = L - 1 if L % 4 == 0 else L - 2
        top_columns = left | {reflection_axis - c for c in left}
        bottom_columns = {m - 2, m + 1} if L % 4 == 0 else {m - 2}
    elif L % 4 == 1:
        top_columns = {m - 2}
        left = set(range(1, m - 2, 2))
        bottom_columns = left | {L - c for c in range(1, m, 2)}
    else:
        top_columns = {m - 2, m + 1}
        left = set(range(1, m - 2, 2))
        bottom_columns = left | {L - 1 - c for c in left}
    return [
        _qid(row, column, L)
        for row in range(L)
        for column in sorted(top_columns if row < m else bottom_columns)
    ]


def _folded_interval(i: int, L: int) -> Tuple[int, int]:
    m = (L + 1) // 2
    return (0, i) if i < m else (i, L - 1)


def _z_correction(L: int, phase_edges: Set[Edge]) -> List[int]:
    """Linear terms from parity products, accumulated by 2-D XOR rectangles."""

    delta = [[0] * (L + 1) for _ in range(L + 1)]
    for a, b in phase_edges:
        ra, ca = divmod(a, L)
        rb, cb = divmod(b, L)
        r0a, r1a = _folded_interval(ra, L)
        r0b, r1b = _folded_interval(rb, L)
        c0a, c1a = _folded_interval(ca, L)
        c0b, c1b = _folded_interval(cb, L)
        r0, r1 = max(r0a, r0b), min(r1a, r1b)
        c0, c1 = max(c0a, c0b), min(c1a, c1b)
        if r0 <= r1 and c0 <= c1:
            delta[r0][c0] ^= 1
            delta[r1 + 1][c0] ^= 1
            delta[r0][c1 + 1] ^= 1
            delta[r1 + 1][c1 + 1] ^= 1

    correction: List[int] = []
    for r in range(L):
        for c in range(L):
            if r:
                delta[r][c] ^= delta[r - 1][c]
            if c:
                delta[r][c] ^= delta[r][c - 1]
            if r and c:
                delta[r][c] ^= delta[r - 1][c - 1]
            if delta[r][c]:
                correction.append(_qid(r, c, L))
    return correction


def _to_moment(
    layer: List[Tuple[str, int, int]], sys_list: List[cirq.Qid]
) -> cirq.Moment:
    operations = []
    for kind, a, b in layer:
        gate = cirq.CNOT if kind == "cx" else cirq.CZ
        operations.append(gate(sys_list[a], sys_list[b]))
    return cirq.Moment(operations)


def build_gamma_folded(L: int, sq: Dict | None = None):
    """Build the complete folded-congruence Gamma circuit.

    Args:
        L: grid side length, at least 2.
        sq: optional ``{(row, column): cirq.Qid}`` system-qubit map.

    Returns:
        ``(circuit, sys_list)`` in the same form as the other ancilla-free
        Gamma builders.
    """

    if L < 2:
        raise ValueError("L must be at least 2")
    if sq is None:
        sq = make_system_qubits(L)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]

    if L >= 7:
        layers = _zero_extra_layers(L)
        z_qubits = _zero_extra_z_correction(L)
    elif L >= 3:
        layers = small_gamma_layers(L)
        z_qubits = small_gamma_z_qubits(L)
    else:
        phase_edges = _local_phase_edges(L)
        gather = _gather_layers(L)
        middle = _midpoint_layers(L, phase_edges)
        layers = gather + middle + list(reversed(gather))
        z_qubits = _z_correction(L, phase_edges)
    moments = [_to_moment(layer, sys_list) for layer in layers]

    if z_qubits:
        moments.append(cirq.Moment(cirq.Z(sys_list[q]) for q in z_qubits))
    return cirq.Circuit(moments), sys_list
