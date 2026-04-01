"""Baseline 1: 1D snake-order FSWAP odd-even transposition sort.

Treats the L x L grid as a 1D chain in snake (Jordan-Wigner) order
and sorts with odd-even transposition using FSWAPs.

Depth: O(N) = O(L^2), specifically ~2L^2 CNOT depth
Ancillas: 0

This is the simplest baseline -- no Hall decomposition, no Gamma.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import cirq
import numpy as np

from common.grid import (
    raster_to_snake_index_map,
    snake_order_indices,
    snake_to_rc,
    validate_permutation,
)
from common.oet_sort import fswap_odd_even_sort_ops


def build_fp_1d(L: int, perm_raster: Sequence[int]):
    """Build Baseline 1: 1D snake ordering + full FSWAP odd-even sort.

    Args:
        L: grid side length
        perm_raster: permutation in raster order (perm[src] = dst)

    Returns:
        (circuit, qubit_list) where qubits are in snake order.
    """
    N = L * L
    validate_permutation(list(perm_raster), N)

    snake = snake_order_indices(L)
    r2s = {r_idx: s_pos for s_pos, r_idx in enumerate(snake)}

    perm_snake = [0] * N
    for src_r in range(N):
        dst_r = perm_raster[src_r]
        perm_snake[r2s[src_r]] = r2s[dst_r]

    qubits = [cirq.GridQubit(*snake_to_rc(i, L)) for i in range(N)]
    ops = fswap_odd_even_sort_ops(qubits, perm_snake)

    circuit = cirq.Circuit(ops)
    return circuit, qubits


# ---------------------------------------------------------------------------
# Benchmark permutation generators
# ---------------------------------------------------------------------------

def build_benchmark_permutation(
    L: int,
    kind: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Generate a benchmark permutation for an L x L grid.

    Args:
        L: grid side length
        kind: one of "identity", "reverse", "transpose", "random",
              "cyclic_row_shift", "reflection_2d"

    Returns:
        permutation as list in raster order
    """
    N = L * L
    if kind == "identity":
        return list(range(N))
    if kind == "reverse":
        return list(range(N - 1, -1, -1))
    if kind == "transpose":
        return [c * L + r for r in range(L) for c in range(L)]
    if kind == "cyclic_row_shift":
        # pi(r,c) = ((r+1) mod L, c)
        return [((r + 1) % L) * L + c for r in range(L) for c in range(L)]
    if kind == "reflection_2d":
        # pi(r,c) = (c,r) -- same as transpose for square grid
        return [c * L + r for r in range(L) for c in range(L)]
    if kind == "random":
        local_rng = rng if rng is not None else np.random.default_rng(0)
        return local_rng.permutation(N).tolist()
    raise ValueError(f"Unsupported permutation kind: {kind}")


def structured_permutations(L: int) -> Dict[str, List[int]]:
    """Return named structured permutations for an L x L grid."""
    N = L * L
    return {
        "identity": list(range(N)),
        "transpose": [c * L + r for r in range(L) for c in range(L)],
        "reverse": list(range(N - 1, -1, -1)),
        "cyclic_row_shift": [((r + 1) % L) * L + c for r in range(L) for c in range(L)],
    }
