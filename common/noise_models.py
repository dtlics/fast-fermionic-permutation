"""Named multiplicative noise models for the numerical experiments.

Both models use CNOT-equivalent accounting: ``total_cnots`` is the
two-qubit exponent and ``total_idle_slots_layered`` is the non-entangling
qubit-layer exponent.
Consequently an FSWAP contributes two two-qubit error opportunities and two
time layers.  See ``docs/ft_gate_accounting.md``.

``modelA`` is a hardware-oriented model with one-qubit and idle error rates
at one tenth of the two-qubit rate.  Computational-basis phase operations
are treated as virtual Z operations.

``modelB`` is the publication early-fault-tolerant location model.  Every
CNOT-equivalent gate and every non-entangling qubit-layer has the same error
rate.  The latter exponent already includes qubits undergoing explicit
one-qubit gates, so those gates are inventoried but are not charged a second
time.  Logical Pauli corrections remain free through Pauli-frame tracking.

``modelB_pauli`` is an ablation that charges even explicit Pauli-Z gates.
Non-Clifford phase synthesis is outside this Clifford-only model and is
accounted for separately by the experiment that introduces those rotations.
"""

from __future__ import annotations

from typing import Dict, Mapping


NOISE_MODELS: Dict[str, Dict[str, float | str]] = {
    "modelA": {
        "label": "hardware (p1q = p_idle = p/10; virtual Z free)",
        "idle": 0.1,
        "hadamard": 0.1,
        "s_sdag": 0.1,
        "pauli_z": 0.0,
        "rz": 0.0,
    },
    "modelB": {
        "label": "early FT (uniform CNOT-equivalent-location error; Paulis free)",
        "idle": 1.0,
        "hadamard": 1.0,
        "s_sdag": 1.0,
        "pauli_z": 0.0,
        "rz": 0.0,
    },
    "modelB_pauli": {
        "label": "early FT ablation (explicit Pauli-Z charged)",
        "idle": 1.0,
        "hadamard": 1.0,
        "s_sdag": 1.0,
        "pauli_z": 1.0,
        "rz": 0.0,
    },
}


def model_rates(model_name: str, p: float) -> Dict[str, float]:
    """Resolve absolute per-class error rates for base rate ``p``."""
    model = NOISE_MODELS[model_name]
    return {
        "p_2q": p,
        "p_idle": float(model["idle"]) * p,
        "p_hadamard": float(model["hadamard"]) * p,
        "p_s_sdag": float(model["s_sdag"]) * p,
        "p_pauli_z": float(model["pauli_z"]) * p,
        "p_rz": float(model["rz"]) * p,
    }


def clifford_idle_fidelity(
    counts: Mapping[str, int],
    model_name: str,
    p: float,
    *,
    p_idle: float | None = None,
) -> float:
    """Return the independent-location no-fault estimate under a named model.

    The function name is retained for compatibility.  Its product is the
    probability that none of the modeled locations fails; it is not a state
    or channel fidelity.

    ``counts`` must use the canonical keys ``total_cnots`` and
    ``total_idle_slots_layered``.  Optional one-qubit keys are ``hadamard``,
    ``other_1q``, ``s_sdag``, ``pauli_z``, and ``rz``.

    ``p_idle`` may override the named model's non-entangling-slot rate for an
    explicit sweep.  In the early-FT models, that exponent already covers
    explicit one-qubit gates scheduled inside the CNOT-depth window; charging
    their inventory again would double count the same qubit-layer.
    """
    rates = model_rates(model_name, p)

    total_cnots = counts["total_cnots"]
    idle_layers = counts["total_idle_slots_layered"]
    hadamards = counts.get("hadamard", 0) + counts.get("other_1q", 0)

    fidelity = (1.0 - rates["p_2q"]) ** total_cnots
    idle_rate = rates["p_idle"] if p_idle is None else p_idle
    fidelity *= (1.0 - idle_rate) ** idle_layers
    if model_name == "modelA":
        # Retained only for reproducing the historical hardware-oriented
        # estimate, where explicit one-qubit gate faults were modeled in
        # addition to memory faults.
        fidelity *= (1.0 - rates["p_hadamard"]) ** hadamards
        fidelity *= (1.0 - rates["p_s_sdag"]) ** counts.get("s_sdag", 0)
        fidelity *= (1.0 - rates["p_pauli_z"]) ** counts.get("pauli_z", 0)
        fidelity *= (1.0 - rates["p_rz"]) ** counts.get("rz", 0)
    elif model_name == "modelB_pauli":
        # Publication modelB treats all non-entangling activity as one slot.
        # This diagnostic ablation adds only the normally frame-tracked Pauli
        # corrections as an extra failure opportunity.
        fidelity *= (1.0 - rates["p_pauli_z"]) ** counts.get("pauli_z", 0)
    return fidelity
