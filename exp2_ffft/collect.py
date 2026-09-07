"""Experiment 2: data collection for FFFT benchmarking.

Sweeps over grid sizes, comparing:
  - 1D baseline (OpenFermion ffft on snake-ordered chain)
  - 2D raster-input core (Gamma sandwich + twiddle + row FFT)
  - 2D proper (core + odd-row reversal + FP transpose, mode-preserving)
  - 2D proper with the ancilla-Gamma construction (mode-preserving)

Depth metric
------------
All depths are CNOT-equivalent (= CZ-equivalent, since 1 CNOT = 1 CZ up
to single-qubit gates).  Each 2-qubit gate is costed by its known KAK
decomposition into CZ + single-qubit:

    CNOT / CZ  (KAK: one nonzero axis)           -> 1 CNOT depth
    FSWAP      (KAK: (pi/4, pi/4, 0), z=0)       -> 2 CNOT depth
    _F0Gate    (KAK: (pi/8, pi/8, 0))             -> 2 CNOT depth
    PhasedISwapPowGate (generic 2-axis)           -> 2 CNOT depth

Note: FSWAP costs only 2 (not 3) because FSWAP = SWAP·CZ and the CZ
cancels SWAP's z-component in the Weyl chamber, dropping from 3-CZ to
2-CZ region.

Verification (statevector) for L <= 4.
No Stim simulation (FFFT is non-Clifford).

Noise accounting
----------------
The independent-location no-fault model is evaluated after expanding every two-qubit operation
into its CNOT-equivalent primitive layers.  Thus an FSWAP, F0/beam-splitter,
phased-iSWAP/Givens, or FSWAP-backed SwapPermutationGate contributes two
two-qubit error locations and two time layers.  For a Cirq moment m,

    I_m = Q d_m - 2 g_m,

where d_m is the largest gate cost in the moment and g_m is the sum of its
gate costs.  This also charges the otherwise hidden second-layer idles in a
    mixed FSWAP/CNOT moment.  Exact Pauli-Z corrections are frame tracked.  In
    the publication early-FT model, every non-entangling qubit-layer has the
    same error rate as a CNOT-equivalent location.  This exponent already
    includes qubits undergoing one-qubit gates, so their explicit inventory is
    not charged a second time.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, List, Sequence

import cirq
from cirq.contrib.acquaintance.permutation import SwapPermutationGate
import matplotlib
import networkx as nx
import numpy as np
import openfermion
import pandas as pd
from openfermion.circuits.primitives.ffft import _F0Gate

from common.fp_2d import GammaMethod
from common.fswap import is_fswap
from common.gamma_folded import (
    GAMMA_SCHEDULE_NOT_APPLICABLE,
    folded_gamma_schedule_model,
)
from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from common.metrics import (
    multiplicative_fidelity,
    no_fault_probability,
    spacetime_volume,
)

from exp2_ffft.ffft_baseline_1d import build_ffft_1d
from exp2_ffft.ffft_proper import build_ffft_core, build_ffft_proper
from exp2_ffft.ft_accounting import (
    ROTATION_INVENTORY_MODEL,
    count_ffft_magic_rotations,
)
from exp2_ffft.verify import verify_ffft_1d, verify_ffft_proper


ACCOUNTING_MODEL = "native_cnot_layers_uniform_nonentangling_v2"
HALL_NOT_APPLICABLE = "not_applicable"
PUBLICATION_L_VALUES = tuple(range(4, 21))
PUBLICATION_P_VALUES = (1e-3, 1e-4, 1e-5)


def _hall_model_for_method(method: str) -> str:
    """Return Hall-router provenance only for circuits that invoke it."""
    return (
        HALL_DECOMPOSITION_MODEL
        if method in {"gamma_2d_proper", "gamma_2d_ancilla"}
        else HALL_NOT_APPLICABLE
    )


def _gamma_schedule_model_for_method(gamma_method: str | None, L: int) -> str:
    """Return folded-scheduler provenance when this row uses it."""
    return (
        folded_gamma_schedule_model(L)
        if gamma_method == GammaMethod.FOLDED.value
        else GAMMA_SCHEDULE_NOT_APPLICABLE
    )


# ---------------------------------------------------------------------------
# CZ-equivalent depth counting
# ---------------------------------------------------------------------------

def _gate_cnot_cost(gate) -> int:
    """CNOT-equivalent cost of a 2-qubit gate (= CZ cost, since 1 CNOT = 1 CZ).

    Determined by KAK decomposition (Weyl chamber position):
        CNOT (CNotPowGate, exp=1)       -> 1  (one nonzero KAK axis)
        CZ   (CZPowGate, exp=1)         -> 1  (one nonzero KAK axis)
        FSWAP (FSwapPowGate)            -> 2  (KAK: pi/4, pi/4, 0)
        _F0Gate (beam splitter)         -> 2  (KAK: pi/8, pi/8, 0)
        PhasedISwapPowGate              -> 2  (two nonzero KAK axes)
        SwapPermutationGate (2q, =FSWAP)-> 2  (wraps FSWAP)

    Unrecognized gates are rejected so a future SWAP or higher-cost
    primitive cannot be silently undercounted.
    """
    if is_fswap(gate):
        return 2
    if isinstance(gate, cirq.CNotPowGate) and gate.exponent == 1:
        return 1
    if isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
        return 1
    if isinstance(gate, _F0Gate):
        return 2
    if isinstance(gate, cirq.PhasedISwapPowGate):
        return 2
    if isinstance(gate, SwapPermutationGate) and is_fswap(gate.swap_gate):
        return 2
    raise ValueError(
        "unrecognized two-qubit gate in CNOT-equivalent accounting: "
        f"{gate!r}"
    )


def _is_charged_1q(gate) -> bool:
    """Return whether a one-qubit gate is a charged explicit operation.

    Identity and exact Pauli-Z operations (up to global phase) are free.
    Arbitrary-angle Z rotations/twiddles, S gates, non-diagonal gates, and
    gates whose unitary cannot be determined are charged conservatively.
    """
    try:
        unitary = cirq.unitary(gate)
    except (TypeError, ValueError):
        return True

    if abs(unitary[0, 1]) < 1e-12 and abs(unitary[1, 0]) < 1e-12:
        phase_ratio = unitary[1, 1] / unitary[0, 0]
        if abs(phase_ratio - 1.0) < 1e-9:
            return False
        if abs(phase_ratio + 1.0) < 1e-9:
            return False
    return True


def count_cnot_resources(circuit: cirq.Circuit, L: int, n_ancillas: int = 0) -> Dict:
    """Count CNOT-equivalent depth and gate resources.

    Each moment's depth contribution = max CNOT cost of any 2q gate in
    that moment (gates in a moment run in parallel; the bottleneck is the
    most expensive gate).

    Returns dict with:
        L, N, n_ancillas, total_qubits,
        cnot_depth, total_2q_gates, total_cnot_equiv,
        total_idle_slots (legacy raw-moment count),
        total_idle_slots_layered (CNOT-equivalent idle qubit-layers),
        n_1q_nonz (inventory covered by non-entangling qubit-layers),
        n_exact_t (exact T/T-dagger rotations),
        n_synth_rz (arbitrary rotations synthesized to the target accuracy)
    """
    N = L * L
    total_qubits = N + n_ancillas

    cnot_depth = 0
    total_2q_gates = 0
    total_cnot_equiv = 0
    total_idle_slots = 0
    total_idle_slots_layered = 0
    n_1q_nonz = 0

    for moment in circuit:
        max_cost = 0
        n_2q = 0
        moment_cnot_equiv = 0
        for op in moment:
            if len(op.qubits) > 2:
                raise ValueError(
                    "multi-qubit primitive must be decomposed before "
                    f"CNOT-equivalent accounting: {op!r}"
                )
            if len(op.qubits) == 2:
                n_2q += 1
                cost = _gate_cnot_cost(op.gate)
                total_cnot_equiv += cost
                moment_cnot_equiv += cost
                max_cost = max(max_cost, cost)
            elif len(op.qubits) == 1 and _is_charged_1q(op.gate):
                n_1q_nonz += 1

        if n_2q > 0:
            cnot_depth += max_cost
            total_2q_gates += n_2q
            active_qubits = 2 * n_2q
            total_idle_slots += max(0, total_qubits - active_qubits)
            layered_idle = total_qubits * max_cost - 2 * moment_cnot_equiv
            if layered_idle < 0:
                raise ValueError(
                    "circuit moment uses more two-qubit resources than the "
                    f"declared {total_qubits} qubits"
                )
            total_idle_slots_layered += layered_idle

    expected_idle = total_qubits * cnot_depth - 2 * total_cnot_equiv
    if total_idle_slots_layered != expected_idle:
        raise AssertionError(
            "CNOT-equivalent idle accounting violated Q*D - 2*G: "
            f"{total_idle_slots_layered} != {expected_idle}"
        )

    magic_inventory = count_ffft_magic_rotations(circuit)
    return {
        "L": L,
        "N": N,
        "n_ancillas": n_ancillas,
        "total_qubits": total_qubits,
        "cnot_depth": cnot_depth,
        "total_2q_gates": total_2q_gates,
        "total_cnot_equiv": total_cnot_equiv,
        "total_idle_slots": total_idle_slots,
        "total_idle_slots_layered": total_idle_slots_layered,
        "n_1q_nonz": n_1q_nonz,
        **magic_inventory,
    }


def cnot_equivalent_no_fault(
    resources: Dict,
    p_2q: float,
    p_idle: float,
    p_1q: float | None = None,
) -> float:
    """Estimate no-fault probability without double-counting one-qubit slots.

    ``total_idle_slots_layered = QD-2G`` is more precisely the number of
    non-entangling qubit-layers: it includes qubits on which a one-qubit gate
    is scheduled.  The one-qubit inventory is therefore not an additional
    exponent in the uniform early-FT model.  ``p_1q`` remains accepted for API
    and CSV compatibility but does not create a second failure opportunity.
    """
    if p_1q is None:
        p_1q = p_2q
    return no_fault_probability(
        resources["total_cnot_equiv"],
        resources["total_idle_slots_layered"],
        p_2q=p_2q,
        p_idle=p_idle,
    )


def native_fidelity(
    resources: Dict,
    p_2q: float,
    p_idle: float,
    p_1q: float | None = None,
) -> float:
    """Backward-compatible alias for :func:`cnot_equivalent_no_fault`."""
    return cnot_equivalent_no_fault(resources, p_2q, p_idle, p_1q)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _log(msg: str, end="\n"):
    print(msg, end=end, flush=True)


def _append_rows(
    rows,
    L,
    N,
    method,
    res,
    p_values,
    p_idle_factor,
    verify_err,
    gamma_method=None,
):
    for p_2q in p_values:
        p_idle = p_2q * p_idle_factor
        p_1q = p_2q
        rows.append({
            "L": L, "N": N,
            "method": method,
            "gamma_method": gamma_method,
            "gamma_schedule_model": _gamma_schedule_model_for_method(
                gamma_method, L
            ),
            "cirq_version": cirq.__version__,
            "openfermion_version": openfermion.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "matplotlib_version": matplotlib.__version__,
            "networkx_version": nx.__version__,
            "accounting_model": ACCOUNTING_MODEL,
            "rotation_inventory_model": ROTATION_INVENTORY_MODEL,
            "hall_decomposition_model": _hall_model_for_method(method),
            **res,
            "spacetime_volume": spacetime_volume(
                res["total_qubits"], res["cnot_depth"]),
            "p_2q": p_2q, "p_idle": p_idle, "p_1q": p_1q,
            "mult_fidelity": cnot_equivalent_no_fault(
                res, p_2q=p_2q, p_idle=p_idle, p_1q=p_1q),
            # Retain the original raw-gate/raw-moment estimate solely as an
            # audit column.  Paper-facing plots use mult_fidelity above.
            "mult_fidelity_legacy": multiplicative_fidelity(
                res["total_2q_gates"],
                res["total_idle_slots"],
                p_2q=p_2q,
                p_idle=p_idle,
            ),
            "verification_error": verify_err,
        })


def collect_for_L(
    L: int,
    p_values: Sequence[float],
    p_idle_factor: float,
    verify: bool = True,
    gamma_method: GammaMethod = GammaMethod.FOLDED,
) -> List[Dict]:
    """Collect the FFFT baselines and both mode-preserving Gamma variants.

    ``gamma_method`` configures the arithmetic-core and no-ancilla proper rows;
    the official ancilla-Gamma proper row is always collected separately with
    explicit ancilla-aware resource accounting.  Reject ``ANCILLA`` as the
    family selector so those two roles cannot be conflated.
    """
    if gamma_method == GammaMethod.ANCILLA:
        raise ValueError(
            "gamma_method selects the core/no-ancilla proper family; "
            "the ancilla-Gamma proper comparator is collected automatically"
        )

    rows: List[Dict] = []
    N = L * L
    can_verify = verify and N <= 16

    # --- 1D baseline ---
    _log(f"  1d_baseline...", end=" ")
    r1d = build_ffft_1d(L)
    res_1d = count_cnot_resources(r1d.circuit, L, 0)
    v1d = float("nan")
    if can_verify:
        v1d, _, _ = verify_ffft_1d(r1d.circuit, r1d.sys_qubits, L)
        _log(f"err={v1d:.2e}", end=" ")
    _append_rows(rows, L, N, "1d_baseline", res_1d, p_values, p_idle_factor, v1d)
    _log("done")

    # --- 2D arithmetic core (raster input, no reordering) ---
    _log(f"  gamma_2d_core...", end=" ")
    r_core = build_ffft_core(L, gamma_method)
    res_core = count_cnot_resources(r_core.circuit, L, 0)
    _append_rows(rows, L, N, "gamma_2d_core", res_core,
                 p_values, p_idle_factor, float("nan"), gamma_method.value)
    _log("done")

    # --- 2D proper (mode-preserving) ---
    _log(f"  gamma_2d_proper...", end=" ")
    r_prop = build_ffft_proper(L, gamma_method)
    res_p = count_cnot_resources(r_prop.circuit, L, 0)
    vp = float("nan")
    if can_verify:
        vp, _, _ = verify_ffft_proper(r_prop.circuit, r_prop.sys_qubits, L)
        _log(f"err={vp:.2e}", end=" ")
    _append_rows(rows, L, N, "gamma_2d_proper", res_p,
                 p_values, p_idle_factor, vp, gamma_method.value)
    _log("done")

    # --- 2D proper with ancilla Gamma (mode-preserving) ---
    _log("  gamma_2d_ancilla...", end=" ")
    r_anc = build_ffft_proper(L, GammaMethod.ANCILLA)
    n_anc = len(r_anc.anc_qubits)
    if n_anc != L:
        raise AssertionError(
            f"ancilla-Gamma FFFT at L={L} uses {n_anc} ancillas, expected {L}"
        )
    res_anc = count_cnot_resources(r_anc.circuit, L, n_anc)
    # The current matrix verifier supplies a system-only qubit order and is
    # therefore not valid for circuits with explicit ancillas.  Dedicated
    # ancilla-statevector tests cover the small instances; keep this audit
    # field explicitly unavailable instead of mis-verifying the circuit.
    _append_rows(
        rows,
        L,
        N,
        "gamma_2d_ancilla",
        res_anc,
        p_values,
        p_idle_factor,
        float("nan"),
        GammaMethod.ANCILLA.value,
    )
    _log("done")

    return rows


def run_experiment(
    L_values: Sequence[int] = PUBLICATION_L_VALUES,
    p_values: Sequence[float] = PUBLICATION_P_VALUES,
    p_idle_factor: float = 1.0,
    output_dir: str = "exp2_ffft/results",
    verify: bool = True,
    gamma_method: GammaMethod = GammaMethod.FOLDED,
) -> pd.DataFrame:
    """Run the full FFFT experiment."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")

    all_rows: List[Dict] = []

    for L in L_values:
        _log(f"L={L} (N={L*L}):")
        rows = collect_for_L(
            L,
            p_values,
            p_idle_factor,
            verify=verify,
            gamma_method=gamma_method,
        )
        all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        _atomic_write_csv(df, csv_path)
        _log(f"  -> saved {len(all_rows)} rows to {csv_path}")

    return pd.DataFrame(all_rows)


def _atomic_write_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Replace a benchmark CSV without exposing a partial checkpoint."""
    output_dir = os.path.dirname(os.path.abspath(csv_path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(csv_path)}.", suffix=".tmp", dir=output_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            df.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp_path, csv_path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
