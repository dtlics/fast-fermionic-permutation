"""Experiment 1: data collection for FP benchmarking.

Sweeps over grid sizes, permutation types, and baselines.
Computes CNOT-equivalent entangling depth, spacetime volume, an independent-location
no-fault estimate, and the Stim process fidelity with its 95% interval.

Saves incremental results after each L value to avoid losing progress.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version as distribution_version
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.fp_1d import build_benchmark_permutation, build_fp_1d
from common.fp_2d import FPResult, GammaMethod, build_fp_2d
from common.gamma_folded import (
    GAMMA_SCHEDULE_NOT_APPLICABLE,
    folded_gamma_schedule_model,
)
from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from common.metrics import count_resources, multiplicative_fidelity, spacetime_volume
from common.stim_convert import (
    simulate_process_fidelity,
    verify_ancilla_disentanglement,
)


BASELINE_CONFIGS = [
    ("1d", None),
    ("ancilla", GammaMethod.ANCILLA),
    ("primitive", GammaMethod.PRIMITIVE),
    ("pipelined", GammaMethod.PIPELINED),
    ("folded", GammaMethod.FOLDED),
]
PUBLICATION_L_VALUES = tuple(range(4, 21))
ACCOUNTING_MODEL = "native_cnot_layers_stim_uniform_v2"
# v3: stim_fidelity is the PROCESS fidelity F_e = Pr(final data Pauli = I),
# estimated by Pauli-frame propagation (stim.FlipSimulator) through the noisy
# forward circuit; v2 was the all-zero return probability after a noiseless
# inverse, which is blind to phase-only faults.
SAMPLING_MODEL = "stim_flipsim_process_fidelity_sha256_row_key_v3"
HALL_NOT_APPLICABLE = "not_applicable_1d"
PERMUTATION_MODEL = "build_benchmark_permutation_raster_v1"
DETERMINISTIC_PERMUTATION_SEED = -1
DEPENDENCY_VERSIONS = {
    "numpy_version": distribution_version("numpy"),
    "cirq_version": distribution_version("cirq"),
    "openfermion_version": distribution_version("openfermion"),
    "stim_version": distribution_version("stim"),
}
REPRODUCIBILITY_COLUMNS = {
    "permutation_model",
    "permutation_seed",
    "permutation_sha256",
    *DEPENDENCY_VERSIONS,
}


def _hall_model_for_baseline(baseline: str) -> str:
    return HALL_NOT_APPLICABLE if baseline == "1d" else HALL_DECOMPOSITION_MODEL


def _gamma_schedule_model_for_baseline(baseline: str, L: int) -> str:
    """Return the Gamma implementation provenance for an Experiment 1 baseline."""
    if baseline == "folded":
        return folded_gamma_schedule_model(L)
    if baseline in {"1d", "ancilla", "primitive", "pipelined"}:
        return GAMMA_SCHEDULE_NOT_APPLICABLE
    raise ValueError(f"unknown Experiment 1 baseline: {baseline}")


def _benchmark_permutation_for_key(
    L: int, perm_kind: str, perm_idx: int
) -> List[int]:
    """Rebuild the exact permutation identified by an Experiment 1 row key."""
    if perm_kind == "random":
        if perm_idx < 0:
            raise ValueError("random permutation indices must be nonnegative")
        rng = np.random.default_rng(seed=int(perm_idx))
        return build_benchmark_permutation(L, perm_kind, rng=rng)
    if perm_idx != 0:
        raise ValueError("deterministic permutation families require perm_idx=0")
    return build_benchmark_permutation(L, perm_kind)


def _permutation_sha256(perm: Sequence[int]) -> str:
    """Hash destination indices encoded as fixed-width little-endian uint64."""
    encoded = np.asarray(perm, dtype=np.dtype("<u8")).tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def _permutation_provenance(
    L: int,
    perm_kind: str,
    perm_idx: int,
    perm: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    expected = _benchmark_permutation_for_key(L, perm_kind, perm_idx)
    if perm is not None and list(map(int, perm)) != expected:
        raise ValueError("permutation contents do not match their scientific row key")
    return {
        "permutation_model": PERMUTATION_MODEL,
        "permutation_seed": (
            int(perm_idx) if perm_kind == "random"
            else DETERMINISTIC_PERMUTATION_SEED
        ),
        "permutation_sha256": _permutation_sha256(expected),
    }


def add_reproducibility_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic permutation and dependency metadata without changing rows."""
    key_columns = ["L", "perm_kind", "perm_idx"]
    if not set(key_columns).issubset(df.columns):
        raise ValueError("data lack permutation-key columns")

    result = df.copy()
    keys = [
        (int(L), str(kind), int(idx))
        for L, kind, idx in result[key_columns].itertuples(index=False, name=None)
    ]
    expected_by_key = {
        key: _permutation_provenance(*key) for key in dict.fromkeys(keys)
    }
    expected_columns = {
        column: [expected_by_key[key][column] for key in keys]
        for column in (
            "permutation_model", "permutation_seed", "permutation_sha256"
        )
    }
    expected_columns.update(
        {column: [value] * len(result) for column, value in DEPENDENCY_VERSIONS.items()}
    )

    for column, expected in expected_columns.items():
        expected_series = pd.Series(expected, index=result.index)
        if column in result.columns and not result[column].equals(expected_series):
            raise ValueError(f"existing {column} provenance is inconsistent")
        result[column] = expected_series
    return result


def has_valid_reproducibility_provenance(df: pd.DataFrame) -> bool:
    """Return whether every row has exact permutation and library provenance."""
    if not REPRODUCIBILITY_COLUMNS.issubset(df.columns):
        return False
    try:
        expected = add_reproducibility_provenance(
            df.drop(columns=list(REPRODUCIBILITY_COLUMNS))
        )
    except (TypeError, ValueError):
        return False
    for column in REPRODUCIBILITY_COLUMNS:
        if not df[column].equals(expected[column]):
            return False
    return True


def _stim_seed_for_row(
    L: int,
    perm_kind: str,
    perm_idx: int,
    baseline: str,
    p_2q: float,
    p_idle: float,
) -> int:
    """Derive a stable 63-bit Stim seed from the full scientific row key."""
    payload = json.dumps(
        {
            "L": int(L),
            "perm_kind": str(perm_kind),
            "perm_idx": int(perm_idx),
            "baseline": str(baseline),
            "p_2q": format(float(p_2q), ".17g"),
            "p_idle": format(float(p_idle), ".17g"),
            "accounting_model": ACCOUNTING_MODEL,
            "hall_decomposition_model": _hall_model_for_baseline(baseline),
            "gamma_schedule_model": _gamma_schedule_model_for_baseline(baseline, L),
            "sampling_model": SAMPLING_MODEL,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    # Stay in signed int64 range for lossless CSV/pandas round trips.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _build_circuit(L: int, perm: List[int], baseline: str, gamma: Optional[GammaMethod]) -> FPResult:
    if baseline == "1d":
        return build_fp_1d(L, perm)
    return build_fp_2d(L, perm, gamma)


def _qubit_order(result: FPResult) -> list:
    return list(result.sys_qubits) + list(result.anc_qubits)


def _log(msg: str, end="\n"):
    print(msg, end=end, flush=True)


def _ordered_dataframe(
    rows: List[Dict],
    perm_kinds: Sequence[str],
    p_values: Sequence[float],
) -> pd.DataFrame:
    """Build a deterministic dataframe independent of worker completion order."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    perm_order = {kind: i for i, kind in enumerate(perm_kinds)}
    baseline_order = {name: i for i, (name, _) in enumerate(BASELINE_CONFIGS)}
    p_order = {round(float(p), 15): i for i, p in enumerate(p_values)}
    ordered = df.assign(
        _perm_order=df["perm_kind"].map(perm_order),
        _baseline_order=df["baseline"].map(baseline_order),
        _p_order=df["p_2q"].map(lambda p: p_order[round(float(p), 15)]),
    ).sort_values(
        ["L", "_perm_order", "perm_idx", "_baseline_order", "_p_order"],
        kind="stable",
    )
    return ordered.drop(columns=["_perm_order", "_baseline_order", "_p_order"]).reset_index(drop=True)


def _atomic_to_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Replace a checkpoint only after its complete CSV is on disk."""
    tmp_path = f"{csv_path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)


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
    permutation_provenance = _permutation_provenance(
        L, perm_kind, perm_idx, perm
    )

    for baseline_name, gamma in BASELINE_CONFIGS:
        result = _build_circuit(L, perm, baseline_name, gamma)
        qo = _qubit_order(result)
        res = count_resources(result.circuit, L, len(result.anc_qubits))
        n_data = len(result.sys_qubits)
        data_indices = list(range(n_data))
        anc_indices = list(range(n_data, len(qo)))
        if shots > 0:
            # errors confined to working ancillas are ignored by the estimator,
            # which is only legitimate if the ideal circuit disentangles them
            verify_ancilla_disentanglement(result.circuit, qo, anc_indices)

        for p_2q in p_values:
            p_idle = p_2q * p_idle_factor
            stim_seed = _stim_seed_for_row(
                L, perm_kind, perm_idx, baseline_name, p_2q, p_idle
            )

            mult_fid = multiplicative_fidelity(
                res["total_cnots"], res["total_idle_slots_layered"],
                p_2q=p_2q, p_idle=p_idle,
            )
            legacy_mult_fid = multiplicative_fidelity(
                res["total_2q_gates"], res["total_idle_slots"],
                p_2q=p_2q, p_idle=p_idle,
            )

            if shots > 0:
                est = simulate_process_fidelity(
                    result.circuit, qo, data_indices,
                    p_2q=p_2q, p_idle=p_idle, shots=shots, seed=stim_seed,
                )
                stim_fid, ci_lo, ci_hi = est["fidelity"], est["ci_lo"], est["ci_hi"]
            else:
                stim_fid = ci_lo = ci_hi = float("nan")

            rows.append({
                "L": L,
                "N": L * L,
                "perm_kind": perm_kind,
                "perm_idx": perm_idx,
                "baseline": baseline_name,
                "accounting_model": ACCOUNTING_MODEL,
                "hall_decomposition_model": _hall_model_for_baseline(baseline_name),
                "gamma_schedule_model": _gamma_schedule_model_for_baseline(
                    baseline_name, L
                ),
                "sampling_model": SAMPLING_MODEL,
                "stim_seed": stim_seed,
                **permutation_provenance,
                **DEPENDENCY_VERSIONS,
                "n_ancillas": res["n_ancillas"],
                "total_qubits": res["total_qubits"],
                "cnot_depth": res["cnot_depth"],
                "total_cnots": res["total_cnots"],
                "n_fswap": res["n_fswap"],
                "n_2q_cnot_cz": res["n_2q_cnot_cz"],
                "total_idle_slots_layered": res["total_idle_slots_layered"],
                # Explicit audit columns for the original per-moment model.
                "total_2q_gates_legacy": res["total_2q_gates"],
                "total_idle_slots_legacy": res["total_idle_slots"],
                "n_1q_hadamard": res["n_1q_hadamard"],
                "n_1q_other": res["n_1q_other"],
                "n_1q_s_sdag": res["n_1q_s_sdag"],
                "n_1q_rz": res["n_1q_rz"],
                "n_1q_pauli_z": res["n_1q_pauli_z"],
                "spacetime_volume": spacetime_volume(res["total_qubits"], res["cnot_depth"]),
                "p_2q": p_2q,
                "p_idle": p_idle,
                "mult_fidelity": mult_fid,
                "mult_fidelity_legacy": legacy_mult_fid,
                "stim_fidelity": stim_fid,
                "stim_fidelity_ci_lo": ci_lo,
                "stim_fidelity_ci_hi": ci_hi,
                "stim_shots": shots,
            })

    return rows


def _validated_resume_rows(
    existing: pd.DataFrame,
    perm_kinds: Sequence[str],
    n_random: int,
    p_values: Sequence[float],
    p_idle_factor: float,
    shots: int,
) -> tuple[set[int], pd.DataFrame]:
    """Return complete, provenance-compatible grid-size groups."""
    required = {
        "L", "N", "perm_kind", "perm_idx", "baseline", "p_2q", "p_idle",
        "accounting_model", "hall_decomposition_model", "gamma_schedule_model",
        "sampling_model", "stim_seed", "total_qubits", "cnot_depth",
        *REPRODUCIBILITY_COLUMNS,
        "total_cnots", "total_idle_slots_layered",
        "stim_fidelity", "stim_fidelity_ci_lo", "stim_fidelity_ci_hi", "stim_shots",
    }
    if not required.issubset(existing.columns):
        return set(), existing.iloc[0:0].copy()

    expected_perms = set()
    for kind in perm_kinds:
        if kind == "random":
            expected_perms.update((kind, i) for i in range(n_random))
        else:
            expected_perms.add((kind, 0))
    expected_baselines = {name for name, _ in BASELINE_CONFIGS}
    expected_p = {round(float(p), 15) for p in p_values}
    expected_rows = len(expected_perms) * len(expected_baselines) * len(expected_p)

    valid_L = set()
    keep_parts = []
    for L, group in existing.groupby("L", sort=False):
        combos = group[
            ["perm_kind", "perm_idx", "baseline", "p_2q"]
        ].drop_duplicates()
        observed_perms = set(
            zip(group.perm_kind.astype(str), group.perm_idx.astype(int))
        )
        idle_expected = (
            group.total_qubits * group.cnot_depth - 2 * group.total_cnots
        )
        p_idle_expected = group.p_2q * p_idle_factor
        hall_expected = group.baseline.astype(str).map(_hall_model_for_baseline)
        gamma_schedule_expected = group.apply(
            lambda row: _gamma_schedule_model_for_baseline(
                str(row.baseline), int(row.L)
            ),
            axis=1,
        )
        seed_expected = group.apply(
            lambda row: _stim_seed_for_row(
                int(row.L), str(row.perm_kind), int(row.perm_idx),
                str(row.baseline), float(row.p_2q), float(row.p_idle),
            ),
            axis=1,
        )
        valid = (
            len(group) == expected_rows
            and len(combos) == expected_rows
            and observed_perms == expected_perms
            and set(group.baseline.astype(str)) == expected_baselines
            and {round(float(p), 15) for p in group.p_2q} == expected_p
            and np.allclose(group.p_idle, p_idle_expected, rtol=0, atol=1e-15)
            and (group.N == int(L) ** 2).all()
            and set(group.accounting_model.astype(str)) == {ACCOUNTING_MODEL}
            and (group.hall_decomposition_model.astype(str) == hall_expected).all()
            and (
                group.gamma_schedule_model.astype(str)
                == gamma_schedule_expected
            ).all()
            and set(group.sampling_model.astype(str)) == {SAMPLING_MODEL}
            and np.array_equal(group.stim_seed.astype("int64"), seed_expected.astype("int64"))
            and has_valid_reproducibility_provenance(group)
            and set(group.stim_shots.astype(int)) == {shots}
            and (group.total_idle_slots_layered == idle_expected).all()
            and (shots <= 0 or group.stim_fidelity.notna().all())
        )
        if valid:
            valid_L.add(int(L))
            keep_parts.append(group)

    kept = (
        pd.concat(keep_parts, ignore_index=True)
        if keep_parts else existing.iloc[0:0].copy()
    )
    return valid_L, kept


def run_experiment(
    L_values: Sequence[int] = PUBLICATION_L_VALUES,
    perm_kinds: Sequence[str] = ("reverse", "transpose", "random"),
    n_random: int = 20,
    p_values: Sequence[float] = (1e-4, 1e-5),
    p_idle_factor: float = 1.0,
    shots: int = 1_000_000,
    output_dir: str = "exp1_fp/results",
    workers: int = 1,
) -> pd.DataFrame:
    """Run the full Experiment 1 sweep."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    requested_L_values = tuple(dict.fromkeys(int(L) for L in L_values))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")
    all_rows = []

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[
            existing["L"].astype(int).isin(requested_L_values)
        ].copy()
        done_Ls, existing = _validated_resume_rows(
            existing, perm_kinds, n_random, p_values, p_idle_factor, shots
        )
        _log(
            "Found complete CNOT-equivalent-accounting data for L = "
            f"{sorted(done_Ls)}, will skip these."
        )
        all_rows = existing.to_dict("records")
        L_values = [L for L in requested_L_values if L not in done_Ls]
    else:
        L_values = list(requested_L_values)

    for L in L_values:
        t0 = time.time()
        _log(f"\n{'='*60}")
        _log(f"L = {L}  (N = {L*L})")
        _log(f"{'='*60}")

        L_rows = []

        instances = []
        for kind in perm_kinds:
            if kind == "random":
                for i in range(n_random):
                    instances.append(
                        (kind, i, _benchmark_permutation_for_key(L, kind, i))
                    )
            else:
                instances.append((kind, 0, _benchmark_permutation_for_key(L, kind, 0)))

        if workers == 1:
            for kind, idx, perm in instances:
                label = kind if kind != "random" else f"random[{idx}]"
                _log(f"  {label} ...", end=" ")
                t1 = time.time()

                rows = collect_instance(
                    L, perm, kind, idx,
                    p_values, p_idle_factor, shots,
                )
                L_rows.extend(rows)
                _log(f"done ({time.time()-t1:.1f}s)")
        else:
            max_workers = min(workers, len(instances))
            _log(f"  Collecting {len(instances)} permutations with {max_workers} workers.")
            completed = {}
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_to_index = {
                    pool.submit(
                        collect_instance,
                        L,
                        perm,
                        kind,
                        idx,
                        p_values,
                        p_idle_factor,
                        shots,
                    ): instance_index
                    for instance_index, (kind, idx, perm) in enumerate(instances)
                }
                for future in as_completed(future_to_index):
                    instance_index = future_to_index[future]
                    completed[instance_index] = future.result()
                    _log(
                        f"  Completed {len(completed)}/{len(instances)} permutations.",
                        end="\r" if len(completed) < len(instances) else "\n",
                    )
            for instance_index in range(len(instances)):
                L_rows.extend(completed[instance_index])

        all_rows.extend(L_rows)
        elapsed = time.time() - t0
        _log(f"  L={L} total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

        df = _ordered_dataframe(all_rows, perm_kinds, p_values)
        _atomic_to_csv(df, csv_path)
        all_rows = df.to_dict("records")
        _log(f"  Saved {len(all_rows)} rows to {csv_path}")

    df = _ordered_dataframe(all_rows, perm_kinds, p_values)
    _log(f"\nExperiment complete. {len(df)} total rows.")
    return df
