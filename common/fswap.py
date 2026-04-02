"""FSWAP gate re-export and decomposition.

The fermionic SWAP (FSWAP) acts as:
    FSWAP|s_j s_k> = (-1)^{s_j s_k} |s_k s_j>

It decomposes into exactly 2 entangling gates (2 CNOTs), making it
strictly better than naive SWAP (3 CNOTs) + CZ (1 CNOT).
"""

from typing import List

import pennylane as qp
from openfermion.circuits.gates import FSWAP, FSwapPowGate

# Cost of one FSWAP in CNOT-equivalent depth
FSWAP_CNOT_COST = 2


def is_fswap(gate) -> bool:
    """Check if a gate is an FSWAP gate."""
    return isinstance(gate, FSwapPowGate)
