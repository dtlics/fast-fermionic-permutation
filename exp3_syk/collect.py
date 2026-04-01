"""Experiment 3: data collection for sparse SYK Trotter step benchmarking.

Sweeps over grid sizes, sparsity parameters, and random instances.
For each instance, builds one Trotter step with 5 baselines and
computes CNOT depth, spacetime volume, and multiplicative fidelity.

No Stim simulation (non-Clifford Pauli rotations).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.fp_2d import GammaMethod
from common.metrics import count_resources, multiplicative_fidelity, spacetime_volume

from exp3_syk.syk_instance import SYKInstance, generate_sparse_syk
from exp3_syk.trotter import build_trotter_step_fp, build_trotter_step_naive


BASELINE_CONFIGS = [
    ("naive_pauli", None),
    ("1d", None),
    ("ancilla", GammaMethod.ANCILLA),
    ("primitive", GammaMethod.PRIMITIVE),
    ("pipelined", GammaMethod.PIPELINED),
]


def _log(msg: str, end="\n"):
    print(msg, end=end, flush=True)


def collect_instance(
    instance: SYKInstance,
    instance_idx: int,
    p_values: Sequence[float],
    p_idle_factor: float,
) -> List[Dict]:
    """Collect all metrics for one SYK instance across all baselines and noise rates."""
    L = instance.L
    rows = []

    for baseline_name, gamma in BASELINE_CONFIGS:
        if baseline_name == "naive_pauli":
            circuit, n_anc, fp_depth, int_depth = build_trotter_step_naive(instance)
        else:
            circuit, n_anc, fp_depth, int_depth = build_trotter_step_fp(
                instance, baseline_name, gamma,
            )

        res = count_resources(circuit, L, n_anc)

        for p_2q in p_values:
            p_idle = p_2q * p_idle_factor
            mult_fid = multiplicative_fidelity(
                res["total_2q_gates"], res["total_idle_slots"],
                p_2q=p_2q, p_idle=p_idle,
            )

            rows.append({
                "L": L,
                "N": L * L,
                "k": instance.k,
                "instance_idx": instance_idx,
                "baseline": baseline_name,
                "n_colors": instance.n_colors,
                "n_quartets": len(instance.quartets),
                "n_ancillas": res["n_ancillas"],
                "total_qubits": res["total_qubits"],
                "cnot_depth": res["cnot_depth"],
                "fp_cnot_depth": fp_depth,
                "interaction_cnot_depth": int_depth,
                "total_2q_gates": res["total_2q_gates"],
                "total_cnots": res["total_cnots"],
                "total_idle_slots": res["total_idle_slots"],
                "spacetime_volume": spacetime_volume(
                    res["total_qubits"], res["cnot_depth"]
                ),
                "p_2q": p_2q,
                "p_idle": p_idle,
                "mult_fidelity": mult_fid,
            })

    return rows


def run_experiment(
    L_values: Sequence[int] = (4, 6, 8, 10, 12, 14, 16, 18, 20),
    k_values: Sequence[float] = (1.0,),
    n_instances: int = 10,
    p_values: Sequence[float] = (1e-3, 1e-4, 1e-5),
    p_idle_factor: float = 0.1,
    output_dir: str = "exp3_syk/results",
) -> pd.DataFrame:
    """Run the full Experiment 3 sweep."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")
    all_rows: List[Dict] = []

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_keys = set(
            zip(existing["L"], existing["k"])
        )
        done_Lk = {(int(l), float(kv)) for l, kv in done_keys}
        _log(f"Found existing data for (L,k) = {sorted(done_Lk)}, will skip these.")
        all_rows = existing.to_dict("records")
    else:
        done_Lk = set()

    for L in L_values:
        for k in k_values:
            if (L, k) in done_Lk:
                continue

            t0 = time.time()
            _log(f"\n{'='*60}")
            _log(f"L = {L}  (N = {L*L}), k = {k}")
            _log(f"{'='*60}")

            Lk_rows: List[Dict] = []

            for idx in range(n_instances):
                rng = np.random.default_rng(seed=idx)
                _log(f"  instance {idx} ...", end=" ")
                t1 = time.time()

                instance = generate_sparse_syk(L, k, rng)
                rows = collect_instance(instance, idx, p_values, p_idle_factor)
                Lk_rows.extend(rows)
                _log(f"done ({time.time()-t1:.1f}s, {len(instance.quartets)} terms, {instance.n_colors} colors)")

            all_rows.extend(Lk_rows)
            elapsed = time.time() - t0
            _log(f"  L={L}, k={k} total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

            df = pd.DataFrame(all_rows)
            df.to_csv(csv_path, index=False)
            _log(f"  Saved {len(all_rows)} rows to {csv_path}")

    df = pd.DataFrame(all_rows)
    _log(f"\nExperiment complete. {len(df)} total rows.")
    return df
