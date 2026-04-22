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

from functools import reduce
from typing import Optional, Sequence

import pennylane as qp
import numpy as np

from common.gates import CNotPowGate, ZPowGate, CZPowGate, FSwapPowGate, FSWAP
from common.scheduler import greedy_schedule

# Cache type objects for fast comparison
_CNOT_POW_TYPE = CNotPowGate
_CNOT_TYPE = qp.CNOT
_Z_TYPE = ZPowGate
_CZ_POW_TYPE = CZPowGate
_CZ_TYPE = qp.CZ
_FSWAP_POW_TYPE = FSwapPowGate
_FSWAP_TYPE = FSWAP


def noisy_circuit(
    circuit: qp.tape.qscript.QuantumScript,
    p_2q: Optional[float] = None,
    p_idle: Optional[float] = None,
) -> qp.tape.qscript.QuantumScript:
    """Adds noise to a circuit."""
    wire_name_lookup = {w: wire for w, wire in enumerate(circuit.wires)}
    all_indices = list(range(len(circuit.wires)))
    add_2q_noise = p_2q is not None and p_2q > 0
    add_idle_noise = p_idle is not None and p_idle > 0

    # schedule circuit, note we can't schedule it with noise since CommutationDAG does not support DepolarizingChannel
    schedule, scheduled_ids = greedy_schedule(circuit)

    # add 2-qubit depolarizing noise
    if add_2q_noise:
        two_qubit_cond = qp.noise.op_in([qp.CNOT, qp.CZ, FSWAP, CNotPowGate, CZPowGate, FSwapPowGate])

        def noise_two(op, **kwargs):
            # note there is no 2-qubit depolarizing channel, we just use two 1-qubit channels
            qp.DepolarizingChannel(p_2q, op.wires[0])
            qp.DepolarizingChannel(p_2q, op.wires[1])

        noise_model = qp.NoiseModel(
            {two_qubit_cond: noise_two}
        )

        [circuit], _ = qp.noise.add_noise(circuit, noise_model=noise_model)

    # add 1-qubit depolarizing noise in idle time
    if add_idle_noise:
        noisy_ops = circuit.operations
        for i, moment in enumerate(schedule):
            # get the list of qubits active in this moment
            moment_wires = reduce(lambda acc, next: acc + next, [n.wires for n in moment])

            # get the list of qubits idle in this moment
            idle_qubits = list(map(lambda wire_num: wire_name_lookup[wire_num], sorted(all_indices - moment_wires)))

            for qubit in idle_qubits:

                # trivial insertion case
                if i == 0:
                    noisy_ops.insert(0, qp.DepolarizingChannel(p_idle, qubit))
                    continue

                # non-trivial insertion case
                prev_index = i
                map_id = -1
                while prev_index > 0 and map_id == -1:
                    prev_index = prev_index - 1

                    # get the ids of the ops in the previous moment
                    prev_ops_ids = scheduled_ids[prev_index]
                    prev_moment = schedule[prev_index]

                    # find the previous op to append noise after
                    for j, op in enumerate(prev_moment):
                        if qubit in list(map(lambda w: wire_name_lookup[w], op.wires)):
                            map_id = prev_ops_ids[j]
                            break

                # index into the noisy circuit
                found_algo_ops = 0
                noisy_index = 0

                # the number of algorithmic ops into the circuit we want to put the idle noise is map_id
                while found_algo_ops < map_id:
                    if not isinstance(noisy_ops[noisy_index], qp.DepolarizingChannel):
                        found_algo_ops += 1
                    noisy_index += 1

                # insert the noise
                noisy_ops.insert(noisy_index, qp.DepolarizingChannel(p_idle, qubit))

        circuit = qp.tape.qscript.QuantumScript(noisy_ops)

    return circuit


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
    dev = qp.device("default.clifford", shots=shots, tableau=False)

    [forward_circuit], _ = qp.transforms.decompose(forward_circuit)

    noisy_fwd = noisy_circuit(forward_circuit, p_2q, p_idle)

    inverse_circuit = qp.tape.qscript.QuantumScript(forward_circuit.operations[::-1])
    noiseless_inv = noisy_circuit(inverse_circuit)

    combined = qp.tape.qscript.QuantumScript(noisy_fwd.operations + noiseless_inv.operations, [qp.expval(qp.PauliZ(q)) for q in qubit_order], shots=shots)

    results = qp.execute([combined], dev, diff_method=None)[0]

    all_zero = np.all([res == 0 for res in results])
    return float(np.mean(all_zero))
