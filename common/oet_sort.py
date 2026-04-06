"""Odd-even transposition sort using FSWAP gates.

Sorts L elements in at most L rounds. Each FSWAP costs 2 CNOT depth.
Used for within-row and within-column sorting in the Row-Col-Row decomposition.
"""

import numpy as np

from typing import List

import pennylane as qp

from common.gates import FSWAP


def fswap_odd_even_sort_ops(qubits: List[qp.wires.Wires], perm: List[int]) -> List[qp.ops.Operation]:
    """Build FSWAP-based odd-even transposition sort.

    Args:
        qubits: list of qp qubits in order
        perm: permutation where perm[i] = destination of item at position i

    Returns:
        list of FSWAP operations implementing the sort
    """
    n = len(qubits)
    if n <= 1:
        return []

    ops = []
    current = list(perm)

    for round_num in range(n):
        parity = round_num % 2
        swapped = False
        for i in range(parity, n - 1, 2):
            if current[i] > current[i + 1]:
                ops.append(FSWAP(qubits[i] + qubits[i + 1]))
                current[i], current[i + 1] = current[i + 1], current[i]
                swapped = True
        if not swapped and round_num > 0:
            break

    return ops
