"""Baselines 2-4: 2D fermionic permutation via Hall Row-Col-Row + Gamma.

All three baselines share the same Row-Col-Row decomposition structure:
    RowA (FSWAP sort) -> Gamma -> Col (bare FSWAP sort) -> Gamma -> RowB (FSWAP sort)

They differ only in the Gamma operator used:
    Baseline 2 (ANCILLA):   7L-3 depth, L ancillas
    Baseline 3 (PRIMITIVE):  12L+8 depth, 0 ancillas
    Baseline 4 (PIPELINED):  8L+O(1) depth, 0 ancillas  <-- best
"""

from enum import Enum
from typing import List, NamedTuple, Optional, Sequence

import pennylane as qp

from common.grid import make_ancilla_qubits, make_system_qubits, validate_permutation
from common.hall_decomposition import decompose_permutation_rcr
from common.oet_sort import fswap_odd_even_sort_ops
from common.gamma_ancilla import build_gamma_with_ancillas
from common.gamma_primitive import build_gamma_ancilla_free
from common.gamma_pipeline import build_gamma_pipelined


class GammaMethod(Enum):
    """Which Gamma construction to use."""
    ANCILLA = "ancilla"        # Baseline 2
    PRIMITIVE = "primitive"    # Baseline 3
    PIPELINED = "pipelined"    # Baseline 4


class FPResult(NamedTuple):
    """Result of building a fermionic permutation circuit.

    Attributes:
        circuit: the qp.tape.qscript.QuantumScript implementing F_pi
        sys_qubits: list of system (data) qubits in raster order
        anc_qubits: list of ancilla qubits (empty [] if ancilla-free).
                    Ancillas are assumed initialised to |0> and guaranteed
                    to return to |0> after the circuit.
        gamma_method: which Gamma was used ("ancilla", "primitive", "pipelined",
                      or None for 1D baseline)
        L: grid side length (N = L^2 data qubits)
    """
    circuit: qp.tape.qscript.QuantumScript
    sys_qubits: List[qp.wires.Wires]
    anc_qubits: List[qp.wires.Wires]
    gamma_method: Optional[str]
    L: int


def build_fp_2d(
    L: int,
    perm: Sequence[int],
    gamma_method: GammaMethod = GammaMethod.PIPELINED,
) -> FPResult:
    """Build 2D fermionic permutation circuit using Hall RCR + Gamma.

    Args:
        L: grid side length
        perm: permutation in raster order (perm[src] = dst)
        gamma_method: which Gamma construction to use

    Returns:
        FPResult with circuit, sys_qubits, anc_qubits ([] if ancilla-free),
        gamma_method name, and L.
    """
    N = L * L
    validate_permutation(list(perm), N)
    s1, s2, s3 = decompose_permutation_rcr(L, perm)

    sq = make_system_qubits(L)
    sys_list = [sq[(r, c)] for r in range(L) for c in range(L)]

    # Build row/col sort ops using the same sq dict
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

    # Build Gamma with the SAME sq (and aq) to guarantee qubit identity
    if gamma_method == GammaMethod.ANCILLA:
        aq = make_ancilla_qubits(L)
        gamma_circ, _, _ = build_gamma_with_ancillas(L, sq=sq, aq=aq)
        anc_list = [aq[r] for r in range(L)]
    elif gamma_method == GammaMethod.PRIMITIVE:
        gamma_circ, _ = build_gamma_ancilla_free(L, sq=sq)
        anc_list = []
    elif gamma_method == GammaMethod.PIPELINED:
        gamma_circ, _ = build_gamma_pipelined(L, sq=sq)
        anc_list = []
    else:
        raise ValueError(f"Unknown gamma method: {gamma_method}")

    # Assemble: RowA + Gamma + Col + Gamma + RowB
    circuit = qp.tape.qscript.QuantumScript(
        rowA_ops
        + gamma_circ.operations
        + col_ops
        + gamma_circ.operations
        + rowB_ops
    )

    return FPResult(
        circuit=circuit,
        sys_qubits=sys_list,
        anc_qubits=anc_list,
        gamma_method=gamma_method.value,
        L=L,
    )
