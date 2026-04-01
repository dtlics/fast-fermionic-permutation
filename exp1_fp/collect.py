"""Experiment 1: data collection for FP benchmarking.

Sweeps over grid sizes, permutation types, and baselines.
Computes CNOT depth, spacetime volume, multiplicative fidelity, and
Stim noisy Clifford fidelity.

Saves incremental results after each L value to avoid losing progress.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.fp_1d import build_benchmark_permutation, build_fp_1d
from common.fp_2d import FPResult, GammaMethod, build_fp_2d
from common.metrics import count_resources, multiplicative_fidelity, spacetime_volume
from common.stim_convert import simulate_clifford_fidelity


BASELINE_CONFIGS = [
    ("1d", None),
    ("ancilla", GammaMethod.ANCILLA),
    ("primitive", GammaMethod.PRIMITIVE),
    ("pipelined", GammaMethod.PIPELINED),
]


def _build_circuit(L: int, perm: List[int], baseline: str, gamma: Optional[GammaMethod]) -> FPResult:
    if baseline == "1d":
        return build_fp_1d(L, perm)
    return build_fp_2d(L, perm, gamma)


def _qubit_order(result: FPResult) -> list:
    return list(result.sys_qubits) + list(result.anc_qubits)


def _log(msg: str, end="\n"):
    print(msg, end=end, flush=True)


def collect_instance(
    L: int,
    perm: List[int],
    perm_kind: str,
    perm_idx: int,
    p_values: Sequence[float],
    p_idle_factor: float,
    shots: int,
) -> List[Dict]:
    """Collect all metrics for one (L, perm) instance across all baselines and noise rates."""
    rows = []

    for baseline_name, gamma in BASELINE_CONFIGS:
        result = _build_circuit(L, perm, baseline_name, gamma)
        qo = _qubit_order(result)
        res = count_resources(result.circuit, L, len(result.anc_qubits))

        for p_2q in p_values:
            p_idle = p_2q * p_idle_factor

            mult_fid = multiplicative_fidelity(
                res["total_2q_gates"], res["total_idle_slots"],
                p_2q=p_2q, p_idle=p_idle,
            )

            stim_fid = simulate_clifford_fidelity(
                result.circuit, qo,
                p_2q=p_2q, p_idle=p_idle, shots=shots,
            )

            rows.append({
                "L": L,
                "N": L * L,
                "perm_kind": perm_kind,
                "perm_idx": perm_idx,
                "baseline": baseline_name,
                "n_ancillas": res["n_ancillas"],
                "total_qubits": res["total_qubits"],
                "cnot_depth": res["cnot_depth"],
                "total_2q_gates": res["total_2q_gates"],
                "total_cnots": res["total_cnots"],
                "total_idle_slots": res["total_idle_slots"],
                "spacetime_volume": spacetime_volume(res["total_qubits"], res["cnot_depth"]),
                "p_2q": p_2q,
                "p_idle": p_idle,
                "mult_fidelity": mult_fid,
                "stim_fidelity": stim_fid,
                "stim_shots": shots,
            })

    return rows


def run_experiment(
    L_values: Sequence[int] = (4, 6, 8, 10, 12, 14, 16),
    perm_kinds: Sequence[str] = ("reverse", "transpose", "random"),
    n_random: int = 20,
    p_values: Sequence[float] = (1e-3, 1e-4, 1e-5),
    p_idle_factor: float = 0.1,
    shots: int = 1000,
    output_dir: str = "exp1_fp/results",
) -> pd.DataFrame:
    """Run the full Experiment 1 sweep."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")
    all_rows = []

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_Ls = set(existing["L"].unique())
        _log(f"Found existing data for L = {sorted(done_Ls)}, will skip these.")
        all_rows = existing.to_dict("records")
        L_values = [L for L in L_values if L not in done_Ls]

    for L in L_values:
        t0 = time.time()
        _log(f"\n{'='*60}")
        _log(f"L = {L}  (N = {L*L})")
        _log(f"{'='*60}")

        L_rows = []

        for kind in perm_kinds:
            if kind == "random":
                instances = []
                for i in range(n_random):
                    rng = np.random.default_rng(seed=i)
                    instances.append((i, build_benchmark_permutation(L, "random", rng=rng)))
            else:
                instances = [(0, build_benchmark_permutation(L, kind))]

            for idx, perm in instances:
                label = kind if kind != "random" else f"random[{idx}]"
                _log(f"  {label} ...", end=" ")
                t1 = time.time()

                rows = collect_instance(
                    L, perm, kind, idx,
                    p_values, p_idle_factor, shots,
                )
                L_rows.extend(rows)
                _log(f"done ({time.time()-t1:.1f}s)")

        all_rows.extend(L_rows)
        elapsed = time.time() - t0
        _log(f"  L={L} total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        _log(f"  Saved {len(all_rows)} rows to {csv_path}")

    df = pd.DataFrame(all_rows)
    _log(f"\nExperiment complete. {len(df)} total rows.")
    return df
