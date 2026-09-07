"""Metrics for fermionic permutation circuit evaluation.

The canonical two-qubit unit is one CNOT-equivalent entangling layer.  An FSWAP
costs two such gates/layers; CNOT and CZ cost one.  See
``docs/ft_gate_accounting.md`` for the accounting convention.

1. **CNOT depth**: Number of CNOT/CZ-equivalent depth layers.
   FSWAP moments contribute 2 layers (FSWAP decomposes into 2 entangling
   layers); CNOT/CZ-only moments contribute 1 layer.

2. **Spacetime volume**: S = total_qubits * cnot_depth

3. **Independent-location no-fault estimate**:
   P_nf = (1-p_2q)^G * (1-p_idle)^I, where G is ``total_cnots`` and I is
   ``total_idle_slots_layered``.  This is a probability that no modeled
   stochastic location fails, not a state or channel fidelity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import cirq

from common.fswap import FSWAP_CNOT_COST, is_fswap

if TYPE_CHECKING:
    from common.fp_2d import FPResult


# ---------------------------------------------------------------------------
# Single-qubit gate classification
# ---------------------------------------------------------------------------

def classify_single_qubit_gate(gate) -> str:
    """Classify a one-qubit gate for the named noise models.

    The returned class is one of ``pauli_z``, ``s_sdag``, ``rz``,
    ``hadamard``, and ``other_1q``.  Arbitrary computational-basis diagonal
    gates are placed in ``rz``; their non-Clifford synthesis cost is handled
    separately from the Clifford noise model.
    """
    if isinstance(gate, cirq.ZPowGate):
        exponent = gate.exponent % 2
        if exponent == 1:
            return "pauli_z"
        if exponent in (0.5, 1.5):
            return "s_sdag"
        return "rz"
    if isinstance(gate, cirq.HPowGate) and gate.exponent % 2 == 1:
        return "hadamard"

    try:
        unitary = cirq.unitary(gate)
        if abs(unitary[0, 1]) < 1e-12 and abs(unitary[1, 0]) < 1e-12:
            return "rz"
    except (TypeError, ValueError):
        pass
    return "other_1q"


def count_single_qubit_gates(circuit: cirq.Circuit) -> Dict[str, int]:
    """Count one-qubit operations by noise class."""
    counts = {
        "pauli_z": 0,
        "s_sdag": 0,
        "rz": 0,
        "hadamard": 0,
        "other_1q": 0,
    }
    for moment in circuit:
        for op in moment:
            if len(op.qubits) == 1:
                counts[classify_single_qubit_gate(op.gate)] += 1
    counts["total_1q"] = sum(counts.values())
    return counts


# ---------------------------------------------------------------------------
# Core resource counter
# ---------------------------------------------------------------------------

def count_resources(circuit: cirq.Circuit, L: int, n_ancillas: int = 0) -> Dict:
    """Count gate resources for a fermionic permutation circuit.

    Scans every moment.  Only moments containing at least one 2-qubit gate
    contribute to depth and idle-slot counts (single-qubit-only moments are
    excluded from these two metrics and reported separately by gate class).

    Returns:
        dict with keys:
            L, N, n_ancillas, total_qubits,
            cnot_depth,     -- CNOT/CZ-equivalent depth (FSWAP moments = 2)
            two_q_depth,    -- moments with >= 1 two-qubit gate
            total_2q_gates, -- legacy raw 2q applications (FSWAP=1)
            total_cnots,    -- CNOT-equivalent count (FSWAP=2, CZ/CNOT=1)
            total_idle_slots, -- legacy per-moment idle count
            total_idle_slots_layered, -- canonical idle qubit-layer count

    For every two-qubit moment m, the canonical idle contribution is
    ``I_m = Q * D_m - 2 * G_m``.  This definition also handles a mixed
    FSWAP/CNOT moment: its CNOT pair idles during the second FSWAP layer.
    """
    N = L * L
    total_qubits = N + n_ancillas

    two_q_depth = 0
    cnot_depth = 0
    total_2q_gates = 0
    total_cnots = 0
    n_fswap = 0
    total_idle_slots = 0
    total_idle_slots_layered = 0

    for moment in circuit:
        n_2q_in_moment = 0
        cnots_in_moment = 0
        has_fswap = False
        for op in moment:
            if len(op.qubits) >= 2:
                n_2q_in_moment += 1
                if is_fswap(op.gate):
                    total_cnots += FSWAP_CNOT_COST
                    cnots_in_moment += FSWAP_CNOT_COST
                    n_fswap += 1
                    has_fswap = True
                else:
                    total_cnots += 1
                    cnots_in_moment += 1

        if n_2q_in_moment > 0:
            two_q_depth += 1
            moment_depth = FSWAP_CNOT_COST if has_fswap else 1
            cnot_depth += moment_depth
            total_2q_gates += n_2q_in_moment
            total_idle_slots += total_qubits - 2 * n_2q_in_moment
            total_idle_slots_layered += (
                total_qubits * moment_depth - 2 * cnots_in_moment
            )

    sq_counts = count_single_qubit_gates(circuit)

    return {
        "L": L,
        "N": N,
        "n_ancillas": n_ancillas,
        "total_qubits": total_qubits,
        "cnot_depth": cnot_depth,
        "two_q_depth": two_q_depth,
        "total_2q_gates": total_2q_gates,
        "total_cnots": total_cnots,
        "n_fswap": n_fswap,
        "n_2q_cnot_cz": total_2q_gates - n_fswap,
        "total_idle_slots": total_idle_slots,
        "total_idle_slots_layered": total_idle_slots_layered,
        "n_1q_pauli_z": sq_counts["pauli_z"],
        "n_1q_s_sdag": sq_counts["s_sdag"],
        "n_1q_rz": sq_counts["rz"],
        "n_1q_hadamard": sq_counts["hadamard"],
        "n_1q_other": sq_counts["other_1q"],
        "n_1q_total": sq_counts["total_1q"],
    }


# ---------------------------------------------------------------------------
# Metric 1: Spacetime volume
# ---------------------------------------------------------------------------

def spacetime_volume(total_qubits: int, cnot_depth: int) -> int:
    """Spacetime volume: S = total_qubits * cnot_depth."""
    return total_qubits * cnot_depth


# ---------------------------------------------------------------------------
# Metric 2: Independent-location no-fault estimate
# ---------------------------------------------------------------------------

def no_fault_probability(
    total_cnots: int,
    total_idle_slots: int,
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
) -> float:
    """Independent-location estimate ``P(no modeled fault)``.

    Args:
        total_cnots: G -- CNOT-equivalent count (FSWAP contributes two)
        total_idle_slots: I -- idle qubit-layers
        p_2q: error rate per CNOT-equivalent gate
        p_idle: error rate per idle qubit-layer

    Returns:
        independent-location no-fault probability in [0, 1]
    """
    return (1.0 - p_2q) ** total_cnots * (1.0 - p_idle) ** total_idle_slots


def multiplicative_fidelity(
    total_cnots: int,
    total_idle_slots: int,
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
) -> float:
    """Backward-compatible wrapper for :func:`no_fault_probability`.

    The return value is not a state or channel fidelity.  New code should use
    the name ``no_fault_probability`` to make that distinction explicit.
    """
    return no_fault_probability(
        total_cnots,
        total_idle_slots,
        p_2q=p_2q,
        p_idle=p_idle,
    )


# ---------------------------------------------------------------------------
# Convenience: evaluate an FPResult
# ---------------------------------------------------------------------------

def evaluate_fp(
    result: "FPResult",
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
) -> Dict:
    """Compute all metrics for a fermionic permutation circuit result.

    Returns:
        dict with all count_resources fields plus:
            spacetime_volume, fidelity, gamma_method.  ``fidelity`` is the
            legacy result key for the independent-location no-fault estimate.
    """
    resources = count_resources(
        result.circuit, result.L, len(result.anc_qubits)
    )
    resources["spacetime_volume"] = spacetime_volume(
        resources["total_qubits"], resources["cnot_depth"]
    )
    resources["fidelity"] = no_fault_probability(
        resources["total_cnots"],
        resources["total_idle_slots_layered"],
        p_2q=p_2q,
        p_idle=p_idle,
    )
    resources["gamma_method"] = result.gamma_method
    return resources
