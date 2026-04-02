"""Convert qp FP circuits to Stim and run noisy Clifford fidelity simulation.

Gate mapping (all gates in these circuits are Clifford):
    FSWAP  -> SWAP + CZ  (treated as one 2-qubit gate for noise)
    CNOT   -> CX
    CZ     -> CZ
    Z      -> Z

Noise model:
    - DEPOLARIZE2(p_2q) after each 2-qubit gate
    - DEPOLARIZE1(p_idle) on idle qubits in each moment containing 2-qubit gates

Performance: builds the Stim circuit as a string and parses once,
avoiding per-instruction append overhead (~3000x faster than append).
"""

from __future__ import annotations

from typing import Optional, Sequence

import pennylane as qp
import numpy as np
import stim


class CNotPowGate(qp.operation.Operator):
    num_wires = 2
    num_params = 1

    def __init__(self, t: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, wires=wires, id=None)

    def compute_matrix(self, t):
        g = qp.math.exp((1j * np.pi * t) / 2)
        s = qp.math.sin(np.pi * t / 2)
        c = qp.math.cos(np.pi * t / 2)
        return qp.math.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, g*c, -1j*g*s],
            [0, 0, -1j*g*s, g*c]
        ])


class ZPowGate(qp.operation.Operator):
    num_wires = 1
    num_params = 2

    def __init__(self, t: float, s: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, s, wires=wires, id=None)

    def compute_matrix(self, t, s):
        a = qp.math.exp(1j * np.pi * s * t)
        b = qp.math.exp(1j * np.pi * t)
        return a * qp.math.array([
            [1, 0],
            [0, b]
        ])


class CZPowGate(qp.operation.Operator):
    num_wires = 2
    num_params = 1

    def __init__(self, t: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, wires=wires, id=None)

    def compute_matrix(self, t):
        g = qp.math.exp(1j * np.pi * t)
        return qp.math.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, g]
        ])


class FSwapPowGate(qp.operation.Operator):
    num_wires = 2
    num_params = 2

    def __init__(self, t: float, s: float, wires: qp.wires.WiresLike):
        self.exponent = t
        super().__init__(t, s, wires=wires, id=None)

    def compute_matrix(self, t):
        p = qp.math.exp(1j * np.pi * t)
        g = qp.math.exp((1j * np.pi * t) / 2)
        s = qp.math.sin(np.pi * t / 2)
        c = qp.math.cos(np.pi * t / 2)
        return qp.math.array([
            [1, 0, 0, 0],
            [0, 0, g*c, -1j*g*s],
            [0, -1j*g*s, g*c, 0],
            [0, 0, 0, p]
        ])


# Cache type objects for fast comparison
_CNOT_TYPE = CNotPowGate
_Z_TYPE = ZPowGate
_CZ_TYPE = CZPowGate
_FSWAP_TYPE = FSwapPowGate


def qp_to_stim_circuit(
    circuit: qp.tape.qscript.QuantumScript,
    qubit_order: Sequence[str],
    p_2q: Optional[float] = None,
    p_idle: Optional[float] = None,
) -> stim.Circuit:
    """Convert a qp circuit to a Stim circuit, optionally with noise.

    Builds the circuit as a string for fast parsing.
    """
    qmap = {q: i for i, q in enumerate(qubit_order)}
    n_qubits = len(qubit_order)
    all_indices = set(range(n_qubits))
    add_2q_noise = p_2q is not None and p_2q > 0
    add_idle_noise = p_idle is not None and p_idle > 0

    lines = []

    # TODO: refactor to work with QScripts, maybe barriers?
    # TODO: Maybe add a function that maps a QScript with barriers to a bunch of "moments"?
    for moment in circuit:
        swap_t = []
        cz_t = []
        cx_t = []
        z_t = []
        noise_t = []
        active = set()
        has_2q = False

        for op in moment:
            gate = op.gate
            qubits = op.qubits
            nq = len(qubits)

            if nq == 2:
                i0 = qmap[qubits[0]]
                i1 = qmap[qubits[1]]
                has_2q = True
                active.add(i0)
                active.add(i1)
                noise_t.extend((i0, i1))

                gt = type(gate)
                if gt is _FSWAP_TYPE:
                    swap_t.extend((i0, i1))
                    cz_t.extend((i0, i1))
                elif gt is _CNOT_TYPE:
                    cx_t.extend((i0, i1))
                elif gt is _CZ_TYPE:
                    cz_t.extend((i0, i1))
                else:
                    raise ValueError(f"Unsupported 2-qubit gate: {gate}")

            elif nq == 1:
                i0 = qmap[qubits[0]]
                if type(gate) is _Z_TYPE and abs(gate.exponent) == 1:
                    z_t.append(i0)

        if swap_t:
            lines.append("SWAP " + " ".join(map(str, swap_t)))
        if cz_t:
            lines.append("CZ " + " ".join(map(str, cz_t)))
        if cx_t:
            lines.append("CX " + " ".join(map(str, cx_t)))
        if z_t:
            lines.append("Z " + " ".join(map(str, z_t)))

        if has_2q and add_2q_noise:
            lines.append(f"DEPOLARIZE2({p_2q}) " + " ".join(map(str, noise_t)))
        if has_2q and add_idle_noise:
            idle = sorted(all_indices - active)
            if idle:
                lines.append(f"DEPOLARIZE1({p_idle}) " + " ".join(map(str, idle)))

        lines.append("TICK")

    return stim.Circuit("\n".join(lines))


def simulate_clifford_fidelity(
    forward_circuit: qp.tape.qscript.QuantumScript,
    qubit_order: Sequence[str],
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
    shots: int = 1000,
) -> float:
    """Estimate fidelity via noisy Clifford simulation.

    Protocol: apply noisy forward circuit, then noiseless inverse circuit,
    measure all qubits. Fidelity = P(measuring all zeros).
    """
    noisy_fwd = qp_to_stim_circuit(forward_circuit, qubit_order, p_2q, p_idle)

    inverse_circuit = qp.inverse(forward_circuit)
    noiseless_inv = qp_to_stim_circuit(inverse_circuit, qubit_order)

    combined = noisy_fwd + noiseless_inv
    combined.append("M", list(range(len(qubit_order))))

    sampler = combined.compile_sampler()
    results = sampler.sample(shots)

    all_zero = np.all(results == 0, axis=1)
    return float(np.mean(all_zero))
