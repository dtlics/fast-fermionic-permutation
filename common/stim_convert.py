"""Convert Cirq FP circuits to Stim and estimate their process fidelity.

The publication estimator is :func:`simulate_process_fidelity` (Pauli-frame
propagation with ``stim.FlipSimulator``).  The older all-zero return-probability
sampler is kept for reproducing the earlier data set.

Gate mapping (all gates in these circuits are Clifford):
    FSWAP  -> SWAP + CZ  (exact unitary; charged as two CNOT-equivalents)
    CNOT   -> CX
    CZ     -> CZ
    Z      -> Z

The default noise schedule is layered: FSWAP pairs receive two independent
``DEPOLARIZE2`` channels, and every qubit not active in the second FSWAP
layer receives a second-layer idle channel.  Set ``layered=False`` only to
reproduce the legacy one-error-event-per-Cirq-moment convention.

Performance: builds the Stim circuit as a string and parses once,
avoiding per-instruction append overhead (~3000x faster than append).
"""

from __future__ import annotations

from typing import Optional, Sequence

import cirq
import numpy as np
import stim
from openfermion.circuits.gates import FSwapPowGate


# Cache type objects for fast comparison
_FSWAP_TYPE = FSwapPowGate
_CNOT_TYPE = cirq.CNotPowGate
_CZ_TYPE = cirq.CZPowGate
_Z_TYPE = cirq.ZPowGate


def cirq_to_stim_circuit(
    circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
    p_2q: Optional[float] = None,
    p_idle: Optional[float] = None,
    p_1q: Optional[float] = None,
    layered: bool = True,
) -> stim.Circuit:
    """Convert a Cirq circuit to a Stim circuit, optionally with noise.

    Builds the circuit as a string for fast parsing.

    With ``layered=True`` (the default), the noise schedule uses the same
    CNOT-equivalent unit as :func:`common.metrics.count_resources`.  In a
    mixed FSWAP/CNOT moment, all pairs are active in layer one; only FSWAP
    pairs remain active in layer two.  The CNOT pairs and otherwise idle
    qubits therefore receive second-layer idle noise.

    ``p_1q`` optionally charges the explicit Pauli-Z corrections.  It is
    disabled by default because these gates are frame tracked in the named
    hardware and early-fault-tolerant models.
    """
    qmap = {q: i for i, q in enumerate(qubit_order)}
    n_qubits = len(qubit_order)
    all_indices = set(range(n_qubits))
    add_2q_noise = p_2q is not None and p_2q > 0
    add_idle_noise = p_idle is not None and p_idle > 0
    add_1q_noise = p_1q is not None and p_1q > 0

    lines = []

    for moment in circuit:
        swap_t = []
        cz_t = []
        cx_t = []
        z_t = []
        noise_t = []
        fswap_noise_t = []
        noise_1q_t = []
        fswap_active = set()
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
                    fswap_noise_t.extend((i0, i1))
                    fswap_active.update((i0, i1))
                elif gt is _CNOT_TYPE:
                    cx_t.extend((i0, i1))
                elif gt is _CZ_TYPE:
                    cz_t.extend((i0, i1))
                else:
                    raise ValueError(f"Unsupported 2-qubit gate: {gate}")

            elif nq == 1:
                i0 = qmap[qubits[0]]
                # ``cirq.Z`` is a private subclass of ZPowGate in current
                # Cirq releases, so exact-type comparison silently dropped
                # the Gamma circuits' final Pauli-frame repair.
                if isinstance(gate, _Z_TYPE) and abs(gate.exponent) == 1:
                    z_t.append(i0)
                    noise_1q_t.append(i0)

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
        if layered and has_2q and fswap_noise_t:
            # FSWAP occupies a second CNOT-equivalent layer.  Bare CNOT/CZ
            # pairs from a mixed moment have finished and idle in this layer.
            if add_2q_noise:
                lines.append(
                    f"DEPOLARIZE2({p_2q}) "
                    + " ".join(map(str, fswap_noise_t))
                )
            if add_idle_noise:
                idle_second_layer = sorted(all_indices - fswap_active)
                if idle_second_layer:
                    lines.append(
                        f"DEPOLARIZE1({p_idle}) "
                        + " ".join(map(str, idle_second_layer))
                    )
        if add_1q_noise and noise_1q_t:
            lines.append(
                f"DEPOLARIZE1({p_1q}) " + " ".join(map(str, noise_1q_t))
            )

        lines.append("TICK")

    return stim.Circuit("\n".join(lines))


def simulate_all_zero_return_probability(
    forward_circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
    shots: int = 1000,
    p_1q: Optional[float] = None,
    layered: bool = True,
    seed: Optional[int] = None,
) -> float:
    """Estimate the bit-error-sensitive all-zero return probability.

    Protocol: apply noisy forward circuit, then noiseless inverse circuit,
    and measure all qubits in the computational basis.  The returned quantity
    is P(measuring all zeros), not full state or circuit fidelity: phase-only
    faults are invisible for these computational-basis-preserving circuits.

    ``seed`` is passed directly to Stim's compiled sampler.  Supplying a
    stable row-specific seed makes Monte Carlo results independent of process
    scheduling and exactly repeatable with the same Stim version.

    """
    noisy_fwd = cirq_to_stim_circuit(
        forward_circuit,
        qubit_order,
        p_2q,
        p_idle,
        p_1q,
        layered=layered,
    )

    inverse_circuit = cirq.inverse(forward_circuit)
    noiseless_inv = cirq_to_stim_circuit(inverse_circuit, qubit_order)

    combined = noisy_fwd + noiseless_inv
    combined.append("M", list(range(len(qubit_order))))

    sampler = combined.compile_sampler(seed=seed)
    # Bit-packed sampling returns exactly the same measurement outcomes while
    # reducing the Python-side memory traffic by a factor of eight.  Mask the
    # unused high bits in the last byte explicitly so this remains correct
    # when the qubit count is not divisible by eight.
    results = sampler.sample(shots, bit_packed=True)
    all_zero = _packed_rows_are_zero(results, len(qubit_order))
    return float(np.mean(all_zero))


def simulate_clifford_fidelity(
    forward_circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
    shots: int = 1000,
    p_1q: Optional[float] = None,
    layered: bool = True,
    seed: Optional[int] = None,
) -> float:
    """Backward-compatible wrapper for all-zero return-probability sampling.

    New code should call :func:`simulate_all_zero_return_probability`; this
    protocol is insensitive to phase-only faults and is not a circuit fidelity.
    """
    return simulate_all_zero_return_probability(
        forward_circuit,
        qubit_order,
        p_2q=p_2q,
        p_idle=p_idle,
        shots=shots,
        p_1q=p_1q,
        layered=layered,
        seed=seed,
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p_hat + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def verify_ancilla_disentanglement(
    forward_circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
    ancilla_indices: Sequence[int],
) -> None:
    """Check that the ideal circuit returns the ancilla register to |0...0>.

    Two conditions, together sufficient for every data input:
    (i) in the Heisenberg picture each ancilla ``Z_a`` is mapped to a Z-type
        Pauli supported only on ancilla qubits, so the ancilla register's final
        state is fixed independently of the data state; and
    (ii) on the all-zero input every ancilla reads ``Z_a = +1`` at the end.
    Raises ``ValueError`` otherwise.
    """
    if not ancilla_indices:
        return
    ideal = cirq_to_stim_circuit(forward_circuit, qubit_order)
    anc = set(int(a) for a in ancilla_indices)
    tableau = stim.Tableau.from_circuit(ideal, ignore_noise=True)
    for a in anc:
        z_out = tableau.z_output(a)
        support = {k for k in range(len(z_out)) if z_out[k] != 0}
        if not support <= anc or any(z_out[k] != 3 for k in support):
            raise ValueError(
                f"ancilla qubit {a} does not disentangle from the data: Z_{a} -> {z_out}"
            )
    sim = stim.TableauSimulator()
    sim.do(ideal)
    for a in anc:
        ps = stim.PauliString(len(qubit_order))
        ps[a] = 3
        if sim.peek_observable_expectation(ps) != 1:
            raise ValueError(f"ancilla qubit {a} is not returned to |0> on the all-zero input")


def simulate_process_fidelity(
    forward_circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
    data_indices: Optional[Sequence[int]] = None,
    p_2q: float = 1e-3,
    p_idle: float = 1e-4,
    shots: int = 1000,
    p_1q: Optional[float] = None,
    layered: bool = True,
    seed: Optional[int] = None,
    batch_size: int = 1 << 18,
    extra_tail: Optional[str] = None,
) -> dict:
    """Estimate the process fidelity F_e = Pr(final Pauli on the data qubits is I).

    The sampled Pauli errors of the noisy forward circuit are propagated through
    that circuit alone with ``stim.FlipSimulator`` (no inverse, no final
    measurement).  A shot succeeds when both the X and the Z flip masks vanish on
    every data qubit at the end, so bit and phase errors are both detected and
    faults that cancel are correctly counted as no error.  Errors confined to
    working ancillas are ignored; call :func:`verify_ancilla_disentanglement`
    first when the circuit has any.

    ``disable_stabilizer_randomization=True`` keeps the frame deterministic for
    the noiseless circuit, which is what makes ``F_e = 1`` at ``p = 0``.
    ``extra_tail`` is a Stim snippet appended after the circuit (used by the
    tests to inject a deterministic final X_ERROR(1) or Z_ERROR(1)).

    Returns ``{"fidelity", "successes", "shots", "ci_lo", "ci_hi"}`` with a
    Wilson 95% interval.
    """
    noisy_fwd = cirq_to_stim_circuit(
        forward_circuit, qubit_order, p_2q, p_idle, p_1q, layered=layered,
    )
    if extra_tail:
        noisy_fwd += stim.Circuit(extra_tail)
    n = len(qubit_order)
    data = np.asarray(
        sorted(range(n) if data_indices is None else data_indices), dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    successes = 0
    done = 0
    while done < shots:
        b = min(batch_size, shots - done)
        sim = stim.FlipSimulator(
            batch_size=b,
            disable_stabilizer_randomization=True,
            num_qubits=n,
            seed=int(rng.integers(0, 2**63 - 1)),
        )
        sim.do(noisy_fwd)
        xs, zs, _ms, _ds, _os = sim.to_numpy(
            bit_packed=True, transpose=False, output_xs=True, output_zs=True,
        )
        # xs, zs: shape (n_qubits, ceil(b/8)); OR over the data qubits gives the
        # per-shot "some data Pauli is non-identity" bit mask.
        bad = np.bitwise_or.reduce(xs[data], axis=0) | np.bitwise_or.reduce(zs[data], axis=0)
        n_bad = int(np.unpackbits(bad, bitorder="little")[:b].sum())
        successes += b - n_bad
        done += b
    lo, hi = wilson_interval(successes, shots)
    return {
        "fidelity": successes / shots,
        "successes": successes,
        "shots": shots,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def _packed_rows_are_zero(samples: np.ndarray, n_bits: int) -> np.ndarray:
    """Return which little-endian bit-packed sample rows are all zero."""
    if samples.ndim != 2:
        raise ValueError("packed samples must be a two-dimensional array")
    expected_bytes = (n_bits + 7) // 8
    if samples.shape[1] != expected_bytes:
        raise ValueError(
            f"expected {expected_bytes} packed bytes per row for {n_bits} bits, "
            f"got {samples.shape[1]}"
        )

    full_bytes, remainder = divmod(n_bits, 8)
    if full_bytes:
        all_zero = np.all(samples[:, :full_bytes] == 0, axis=1)
    else:
        all_zero = np.ones(samples.shape[0], dtype=bool)
    if remainder:
        mask = (1 << remainder) - 1
        all_zero &= (samples[:, full_bytes] & mask) == 0
    return all_zero
