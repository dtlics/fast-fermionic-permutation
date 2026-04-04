"""2D FFFT on an L x L nearest-neighbor qubit grid.

Two public builders
-------------------

build_ffft_core(L)
    The minimal Cooley-Tukey DFT circuit.  Lowest depth, but the output
    layout differs from the input layout (see below).

build_ffft_proper(L)
    Wraps the core with input/output reordering so that mode k stays at
    qubit k.  Higher depth, but drop-in compatible with any code that
    expects row-major-snake layout.


Layout conventions
------------------
"Label" always means *fermion mode index* = JW index = 0 .. N-1.

Row-major snake (the standard JW layout on a 2D grid):
    grid (r, c) holds label = rc_to_snake(r, c, L).
    Even rows go left-to-right, odd rows go right-to-left.

Raster:
    grid (r, c) holds label = r*L + c.  All rows left-to-right.

Column-major raster:
    grid (r, c) holds label = r + c*L.  Reading down columns gives
    0, 1, ..., N-1.


build_ffft_core: output layout
------------------------------
Assumes input is in **row-major snake** layout (the standard JW layout).

Circuit:  Gamma -> col F_L -> Gamma -> twiddle -> row F_L

After the circuit, grid position (r, c) holds DFT output mode

    k = r + c * L                                   (column-major raster)

To convert mode label k to a grid position after the core circuit:

    row = k % L
    col = k // L

Or equivalently, to find which mode label sits at grid position (r, c):

    label_at(r, c) = r + c * L

This is a *known, fixed* relabeling that depends only on L.  Downstream
code that operates in the DFT-output basis can use this mapping directly
without appending the FP transpose — saving ~22L depth.

Example for L = 3 (N = 9):

    Input (row-major snake):      Output (col-major raster):
      0  1  2                       0  3  6
      5  4  3                       1  4  7
      6  7  8                       2  5  8

If a downstream task needs to apply an operator to DFT mode k, it should
target the qubit at grid position (k % L, k // L).


build_ffft_proper: output layout
--------------------------------
Wraps the core with:
  - Odd-row reversal before  (snake -> raster, ~2L depth)
  - FP transpose after       (col-major-raster -> row-major-snake, ~22L depth)

After the circuit, grid (r, c) holds rc_to_snake(r, c, L) — same as
input.  The single-particle unitary equals F_N exactly.  Use this when
you need mode k at qubit k with no further bookkeeping.
"""

from typing import Dict, List, Tuple

import cirq
from openfermion.circuits.gates import FSWAP

from common.fp_2d import FPResult, GammaMethod, build_fp_2d
from common.grid import make_system_qubits, rc_to_snake, snake_to_rc
from common.gamma_pipeline import build_gamma_pipelined
from common.gamma_primitive import build_gamma_ancilla_free
from common.gamma_ancilla import build_gamma_with_ancillas

from exp2_ffft.twiddle import build_twiddle_circuit
from exp2_ffft.col_ffft_bare import build_bare_column_fffts, build_row_fffts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_odd_row_reversal(L: int, sq: dict) -> cirq.Circuit:
    """Reverse qubits within every odd row using FSWAPs.

    Converts row-major snake <-> raster (self-inverse).
    All odd rows execute in parallel.  Depth: O(L).
    """
    from common.oet_sort import fswap_odd_even_sort_ops

    ops: List[cirq.Operation] = []
    rev_perm = list(range(L - 1, -1, -1))
    for r in range(L):
        if r % 2 == 1:
            row_qs = [sq[(r, c)] for c in range(L)]
            ops.extend(fswap_odd_even_sort_ops(row_qs, rev_perm))
    return cirq.Circuit(ops)


def _col_major_raster_to_row_major_snake_perm(L: int) -> List[int]:
    """Raster-order FP permutation: col-major-raster -> row-major-snake.

    Returns perm where perm[src_raster] = dst_raster.
    """
    N = L * L
    perm = [0] * N
    for k in range(N):
        cur_r, cur_c = k % L, k // L
        tgt_r, tgt_c = snake_to_rc(k, L)
        perm[cur_r * L + cur_c] = tgt_r * L + tgt_c
    return perm


def _build_gamma(L, sq, gamma_method):
    """Build Gamma circuit, return (circuit, anc_list)."""
    if gamma_method == GammaMethod.ANCILLA:
        from common.grid import make_ancilla_qubits
        aq = make_ancilla_qubits(L)
        circ, _, _ = build_gamma_with_ancillas(L, sq=sq, aq=aq)
        return circ, [aq[r] for r in range(L)]
    elif gamma_method == GammaMethod.PRIMITIVE:
        circ, _ = build_gamma_ancilla_free(L, sq=sq)
        return circ, []
    elif gamma_method == GammaMethod.PIPELINED:
        circ, _ = build_gamma_pipelined(L, sq=sq)
        return circ, []
    else:
        raise ValueError(f"Unknown gamma method: {gamma_method}")


# ---------------------------------------------------------------------------
# Output-layout helpers (for downstream code using the core circuit)
# ---------------------------------------------------------------------------

def core_output_label_at(r: int, c: int, L: int) -> int:
    """After build_ffft_core, the DFT output mode at grid (r, c).

    Returns k = r + c * L  (column-major raster).
    """
    return r + c * L


def core_output_grid_of(k: int, L: int) -> Tuple[int, int]:
    """After build_ffft_core, the grid position of DFT output mode k.

    Returns (row, col) = (k % L, k // L).
    """
    return k % L, k // L


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_ffft_core(
    L: int,
    gamma_method: GammaMethod = GammaMethod.PIPELINED,
) -> FPResult:
    """Build the core 2D FFFT circuit (no input/output reordering).

    Circuit:  Gamma -> bare col F_L -> Gamma -> twiddle -> row F_L

    Assumes input in row-major snake layout.  After the circuit, grid
    position (r, c) holds DFT output mode k = r + c*L (column-major
    raster).  See module docstring for the full mapping and examples.

    Use core_output_label_at(r, c, L) or core_output_grid_of(k, L) to
    convert between grid positions and mode labels in downstream code.

    Args:
        L: grid side length (N = L^2 modes).
        gamma_method: which Gamma construction to use.

    Returns:
        FPResult with the core circuit.
    """
    sq = make_system_qubits(L)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]

    gamma_circ, anc_list = _build_gamma(L, sq, gamma_method)
    col_circ = build_bare_column_fffts(L, sq)
    twiddle_circ = build_twiddle_circuit(L, sq)
    row_circ = build_row_fffts(L, sq)

    circuit = (
        gamma_circ
        + col_circ
        + gamma_circ
        + twiddle_circ
        + row_circ
    )

    return FPResult(
        circuit=circuit,
        sys_qubits=sys_list,
        anc_qubits=anc_list,
        gamma_method=gamma_method.value,
        L=L,
    )


# ---------------------------------------------------------------------------
# Proper builder (mode-preserving)
# ---------------------------------------------------------------------------

def build_ffft_proper(
    L: int,
    gamma_method: GammaMethod = GammaMethod.PIPELINED,
) -> FPResult:
    """Build the proper 2D FFFT circuit (mode k stays at qubit k).

    Circuit:
        odd-row-rev -> Gamma -> col F_L -> Gamma -> twiddle -> row F_L -> FP

    After this circuit, mode k is back at qubit k (row-major snake).
    The single-particle unitary equals F_N (the N-point DFT matrix).

    Args:
        L: grid side length (N = L^2 modes).
        gamma_method: which Gamma construction to use.

    Returns:
        FPResult with the assembled circuit.
    """
    sq = make_system_qubits(L)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]

    gamma_circ, anc_list = _build_gamma(L, sq, gamma_method)

    rev_circ = _build_odd_row_reversal(L, sq)
    col_circ = build_bare_column_fffts(L, sq)
    twiddle_circ = build_twiddle_circuit(L, sq)
    row_circ = build_row_fffts(L, sq)

    fp_perm = _col_major_raster_to_row_major_snake_perm(L)
    fp_circ = build_fp_2d(L, fp_perm, gamma_method).circuit

    circuit = (
        rev_circ
        + gamma_circ
        + col_circ
        + gamma_circ
        + twiddle_circ
        + row_circ
        + fp_circ
    )

    return FPResult(
        circuit=circuit,
        sys_qubits=sys_list,
        anc_qubits=anc_list,
        gamma_method=gamma_method.value,
        L=L,
    )
