"""Experiment 2: data collection for FFFT benchmarking.

Sweeps over grid sizes, comparing:
  - 1D baseline (OpenFermion ffft on snake-ordered chain)
  - 2D core (Gamma sandwich + twiddle + row FFT, no input/output reordering)
  - 2D proper (core + odd-row reversal + FP transpose, mode-preserving)

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
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import cirq
import pandas as pd

from common.fp_2d import GammaMethod
from common.fswap import is_fswap
from common.metrics import multiplicative_fidelity, spacetime_volume

from exp2_ffft.ffft_baseline_1d import build_ffft_1d
from exp2_ffft.ffft_proper import build_ffft_core, build_ffft_proper
from exp2_ffft.verify import verify_ffft_1d, verify_ffft_proper


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
        Other                           -> 2  (conservative default)
    """
    if is_fswap(gate):
        return 2
    if isinstance(gate, cirq.CNotPowGate) and gate.exponent == 1:
        return 1
    if isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
        return 1
    # openfermion beam splitter, phased-iswap, 2q swap perm — all cost 2 CZ
    return 2


def count_cnot_resources(circuit: cirq.Circuit, L: int, n_ancillas: int = 0) -> Dict:
    """Count CNOT-equivalent depth and gate resources.

    Each moment's depth contribution = max CNOT cost of any 2q gate in
    that moment (gates in a moment run in parallel; the bottleneck is the
    most expensive gate).

    Returns dict with:
        L, N, n_ancillas, total_qubits,
        cnot_depth, total_2q_gates, total_cnot_equiv, total_idle_slots
    """
    N = L * L
    total_qubits = N + n_ancillas

    cnot_depth = 0
    total_2q_gates = 0
    total_cnot_equiv = 0
    total_idle_slots = 0

    for moment in circuit:
        max_cost = 0
        n_2q = 0
        for op in moment:
            if len(op.qubits) >= 2:
                n_2q += 1
                cost = _gate_cnot_cost(op.gate)
                total_cnot_equiv += cost
                max_cost = max(max_cost, cost)

        if n_2q > 0:
            cnot_depth += max_cost
            total_2q_gates += n_2q
            active_qubits = 2 * n_2q
            total_idle_slots += max(0, total_qubits - active_qubits)

    return {
        "L": L,
        "N": N,
        "n_ancillas": n_ancillas,
        "total_qubits": total_qubits,
        "cnot_depth": cnot_depth,
        "total_2q_gates": total_2q_gates,
        "total_cnot_equiv": total_cnot_equiv,
        "total_idle_slots": total_idle_slots,
    }


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _log(msg: str, end="\n"):
    print(msg, end=end, flush=True)


def _append_rows(rows, L, N, method, res, p_values, p_idle_factor, verify_err):
    for p_2q in p_values:
        p_idle = p_2q * p_idle_factor
        rows.append({
            "L": L, "N": N,
            "method": method,
            **res,
            "spacetime_volume": spacetime_volume(
                res["total_qubits"], res["cnot_depth"]),
            "p_2q": p_2q, "p_idle": p_idle,
            "mult_fidelity": multiplicative_fidelity(
                res["total_2q_gates"], res["total_idle_slots"],
                p_2q=p_2q, p_idle=p_idle),
            "verification_error": verify_err,
        })


def collect_for_L(
    L: int,
    p_values: Sequence[float],
    p_idle_factor: float,
    verify: bool = True,
) -> List[Dict]:
    """Collect metrics for one L value."""
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

    # --- 2D core (no reordering) ---
    _log(f"  gamma_2d_core...", end=" ")
    r_core = build_ffft_core(L, GammaMethod.PIPELINED)
    res_core = count_cnot_resources(r_core.circuit, L, 0)
    _append_rows(rows, L, N, "gamma_2d_core", res_core,
                 p_values, p_idle_factor, float("nan"))
    _log("done")

    # --- 2D proper (mode-preserving) ---
    _log(f"  gamma_2d_proper...", end=" ")
    r_prop = build_ffft_proper(L, GammaMethod.PIPELINED)
    res_p = count_cnot_resources(r_prop.circuit, L, 0)
    vp = float("nan")
    if can_verify:
        vp, _, _ = verify_ffft_proper(r_prop.circuit, r_prop.sys_qubits, L)
        _log(f"err={vp:.2e}", end=" ")
    _append_rows(rows, L, N, "gamma_2d_proper", res_p,
                 p_values, p_idle_factor, vp)
    _log("done")

    return rows


def run_experiment(
    L_values: Sequence[int] = (2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    p_values: Sequence[float] = (1e-3, 1e-4, 1e-5),
    p_idle_factor: float = 0.1,
    output_dir: str = "exp2_ffft/results",
    verify: bool = True,
) -> pd.DataFrame:
    """Run the full FFFT experiment."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")

    all_rows: List[Dict] = []

    for L in L_values:
        _log(f"L={L} (N={L*L}):")
        rows = collect_for_L(L, p_values, p_idle_factor, verify=verify)
        all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        _log(f"  -> saved {len(all_rows)} rows to {csv_path}")

    return pd.DataFrame(all_rows)
