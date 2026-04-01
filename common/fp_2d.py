"""Baselines 2-4: 2D fermionic permutation via Hall Row-Col-Row + Gamma.

All three baselines share the same Row-Col-Row decomposition structure:
    RowA (FSWAP sort) -> Gamma -> Col (bare FSWAP sort) -> Gamma -> RowB (FSWAP sort)

They differ only in the Gamma operator used:
    Baseline 2 (ANCILLA):   7L-3 depth, L ancillas
    Baseline 3 (PRIMITIVE):  9L+12 depth, 0 ancillas
    Baseline 4 (PIPELINED):  8L+O(1) depth, 0 ancillas  <-- best
"""

from enum import Enum
from typing import List, Optional, Sequence, Tuple

import cirq

from common.grid import make_ancilla_qubits, make_system_qubits, validate_permutation
from common.hall_decomposition import decompose_permutation_rcr
from common.oet_sort import fswap_odd_even_sort_ops
from common.gamma_ancilla import (
    ancilla_column_cascade_ops,
    build_stage_B_ops,
    build_stage_D_ops,
    build_gamma_with_ancillas,
)
from common.gamma_primitive import build_gamma_ancilla_free
from common.gamma_pipeline import build_gamma_pipelined
from common.grid import column_parity_cascade_ops


class GammaMethod(Enum):
    """Which Gamma construction to use."""
    ANCILLA = "ancilla"        # Baseline 2
    PRIMITIVE = "primitive"    # Baseline 3
    PIPELINED = "pipelined"    # Baseline 4


def build_fp_2d(
    L: int,
    perm: Sequence[int],
    gamma_method: GammaMethod = GammaMethod.PIPELINED,
) -> Tuple[cirq.Circuit, List[cirq.Qid], Optional[List[cirq.Qid]]]:
    """Build 2D fermionic permutation circuit using Hall RCR + Gamma.

    Args:
        L: grid side length
        perm: permutation in raster order (perm[src] = dst)
        gamma_method: which Gamma construction to use

    Returns:
        (circuit, sys_qubits, anc_qubits_or_None)
    """
    N = L * L
    validate_permutation(list(perm), N)
    s1, s2, s3 = decompose_permutation_rcr(L, perm)

    sq = make_system_qubits(L)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]

    # Build row/col sort ops
    rowA_ops = []
    for r in range(L):
        row_qs = [sq[(r, c)] for c in range(L)]
        rowA_ops.extend(fswap_odd_even_sort_ops(row_qs, s1[r]))

    col_ops = []
    for c in range(L):
        col_qs = [sq[(r, c)] for r in range(L)]
        col_ops.extend(fswap_odd_even_sort_ops(col_qs, s2[c]))

    rowB_ops = []
    for r in range(L):
        row_qs = [sq[(r, c)] for c in range(L)]
        rowB_ops.extend(fswap_odd_even_sort_ops(row_qs, s3[r]))

    # Build Gamma and assemble
    if gamma_method == GammaMethod.ANCILLA:
        aq = make_ancilla_qubits(L)
        anc_list = [aq[r] for r in range(L)]

        # Build gamma ops inline (not as separate circuit) to share qubit references
        gamma_ops = []
        gamma_ops.extend(column_parity_cascade_ops(sq, L, inverse=False))
        gamma_ops.extend(build_stage_B_ops(sq, aq, L))
        gamma_ops.extend(column_parity_cascade_ops(sq, L, inverse=True))
        gamma_ops.extend(ancilla_column_cascade_ops(aq, L, inverse=True))
        gamma_ops.extend(build_stage_D_ops(sq, aq, L))

        circuit = (
            cirq.Circuit(rowA_ops)
            + cirq.Circuit(gamma_ops)
            + cirq.Circuit(col_ops)
            + cirq.Circuit(gamma_ops)
            + cirq.Circuit(rowB_ops)
        )
        return circuit, sys_list, anc_list

    elif gamma_method == GammaMethod.PRIMITIVE:
        gamma_circ, _ = build_gamma_ancilla_free(L)
        circuit = (
            cirq.Circuit(rowA_ops)
            + gamma_circ
            + cirq.Circuit(col_ops)
            + gamma_circ
            + cirq.Circuit(rowB_ops)
        )
        return circuit, sys_list, None

    elif gamma_method == GammaMethod.PIPELINED:
        gamma_circ, _ = build_gamma_pipelined(L)
        circuit = (
            cirq.Circuit(rowA_ops)
            + gamma_circ
            + cirq.Circuit(col_ops)
            + gamma_circ
            + cirq.Circuit(rowB_ops)
        )
        return circuit, sys_list, None

    else:
        raise ValueError(f"Unknown gamma method: {gamma_method}")
