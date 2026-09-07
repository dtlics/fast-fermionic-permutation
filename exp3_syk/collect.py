"""Experiment 3: data collection for sparse SYK Trotter step benchmarking.

Sweeps over grid sizes, sparsity parameters, and random instances.
For each instance, builds one Trotter step with six baselines and
computes CNOT-equivalent entangling depth, spacetime volume, and an independent-location
no-fault estimate.

No Stim simulation (non-Clifford Pauli rotations).
"""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from importlib.metadata import version as distribution_version
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.fp_2d import GammaMethod
from common.gamma_folded import (
    GAMMA_SCHEDULE_NOT_APPLICABLE,
    folded_gamma_schedule_model,
)
from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from common.metrics import count_resources, multiplicative_fidelity, spacetime_volume
from common.noise_models import clifford_idle_fidelity

from exp3_syk.syk_instance import (
    SYKInstance,
    generate_sparse_syk,
    instance_reproducibility_provenance,
)
from exp3_syk.trotter import (
    PAULI_ROTATION_COMPILER_MODEL,
    assert_nearest_neighbor_circuit,
    build_trotter_step_fp,
    build_trotter_step_naive,
)


# Default k values for the colors-vs-N plot (lightweight, no circuit building)
COLORS_K_VALUES = (0.5, 1.0, 2.0, 3.0)
PUBLICATION_L_VALUES = tuple(range(4, 21))
ACCOUNTING_MODEL = "native_cnot_layers_uniform_nonentangling_v2"
HALL_NOT_APPLICABLE = "not_applicable"


BASELINE_CONFIGS = [
    ("naive_pauli", None),
    ("1d", None),
    ("ancilla", GammaMethod.ANCILLA),
    ("primitive", GammaMethod.PRIMITIVE),
    ("pipelined", GammaMethod.PIPELINED),
    ("folded", GammaMethod.FOLDED),
]


def no_fault_values_match(observed, expected) -> bool:
    """Compare no-fault values robustly at IEEE-754 underflow.

    Ordinary values must match at tight relative tolerance.  Below the
    smallest normal float, decimal CSV round-tripping can shift a value by one
    subnormal ULP, for which a relative tolerance is not representable; all
    nonnegative subnormal encodings are equivalent for the plotted zero floor.
    """

    observed_values = np.asarray(observed, dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    tiny = np.finfo(float).tiny
    normal = expected_values >= tiny
    matches = np.zeros(expected_values.shape, dtype=bool)
    matches[normal] = np.isclose(
        observed_values[normal],
        expected_values[normal],
        rtol=5e-12,
        atol=0,
    )
    matches[~normal] = (
        (observed_values[~normal] >= 0)
        & (observed_values[~normal] < tiny)
    )
    return bool(matches.all())


REPRODUCIBILITY_COLUMNS = (
    "rng_seed",
    "syk_generator_model",
    "quartet_sampling_model",
    "coupling_model",
    "parent_coloring_model",
    "numpy_version",
    "networkx_version",
    "quartets_sha256",
    "coloring_sha256",
)

# These libraries participate in circuit construction/resource accounting but
# not in the lightweight colors-only dataset.  Keep their provenance separate
# so colors_data.csv remains limited to its NumPy/NetworkX dependencies.
RESOURCE_DEPENDENCY_VERSIONS = {
    "cirq_version": distribution_version("cirq"),
    "openfermion_version": distribution_version("openfermion"),
    "pauli_rotation_compiler_model": PAULI_ROTATION_COMPILER_MODEL,
}
RESOURCE_REPRODUCIBILITY_COLUMNS = (
    *REPRODUCIBILITY_COLUMNS,
    *RESOURCE_DEPENDENCY_VERSIONS,
)


def _hall_model_for_baseline(baseline: str) -> str:
    """Return the Hall compiler provenance applicable to one baseline."""
    if baseline in {"ancilla", "primitive", "pipelined", "folded"}:
        return HALL_DECOMPOSITION_MODEL
    return HALL_NOT_APPLICABLE


def _gamma_schedule_model_for_baseline(baseline: str, L: int) -> str:
    """Return folded-scheduler provenance only for the folded baseline."""
    return (
        folded_gamma_schedule_model(L)
        if baseline == "folded"
        else GAMMA_SCHEDULE_NOT_APPLICABLE
    )


def validate_gamma_schedule_provenance(df: pd.DataFrame) -> None:
    """Reject rows produced by an older folded-Gamma scheduler."""
    required = {"L", "baseline", "gamma_schedule_model"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"SYK CSV lacks Gamma-schedule provenance columns {missing}; "
            "rerun collection"
        )
    expected = df.apply(
        lambda row: _gamma_schedule_model_for_baseline(
            str(row.baseline), int(row.L)
        ),
        axis=1,
    )
    observed = df["gamma_schedule_model"].astype(str)
    if not (observed == expected).all():
        raise ValueError(
            "SYK CSV mixes stale or inapplicable Gamma schedules; rerun "
            "collection"
        )


def validate_hall_provenance(df: pd.DataFrame) -> None:
    """Reject missing, stale, or incorrectly attributed Hall compiler rows."""
    required = {"baseline", "hall_decomposition_model"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"SYK CSV lacks Hall-decomposition provenance columns {missing}; "
            "rerun collection"
        )
    expected = df["baseline"].astype(str).map(_hall_model_for_baseline)
    observed = df["hall_decomposition_model"].astype(str)
    bad = observed != expected
    if bad.any():
        found = sorted(set(observed[bad]))
        raise ValueError(
            "SYK CSV mixes stale or inapplicable Hall decompositions; "
            f"found {found!r}, expected {HALL_DECOMPOSITION_MODEL!r} for "
            "every 2D FP row"
        )


@lru_cache(maxsize=None)
def _expected_instance_identity(L: int, k: float, instance_idx: int) -> Dict:
    """Recreate the lightweight structural identity of one seeded instance."""
    instance = generate_sparse_syk(
        int(L),
        float(k),
        np.random.default_rng(seed=int(instance_idx)),
    )
    return {
        **instance_reproducibility_provenance(instance, int(instance_idx)),
        "n_colors": int(instance.n_colors),
        "n_quartets": int(len(instance.quartets)),
    }


def add_reproducibility_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic identity and circuit-library provenance to rows."""
    required = {"L", "k", "instance_idx"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"cannot add SYK reproducibility columns: missing {missing}")

    result = df.copy()
    for column in REPRODUCIBILITY_COLUMNS:
        if column not in result:
            result[column] = None
    for (L, k, instance_idx), indices in result.groupby(
        ["L", "k", "instance_idx"], sort=False
    ).groups.items():
        identity = _expected_instance_identity(int(L), float(k), int(instance_idx))
        for column in REPRODUCIBILITY_COLUMNS:
            result.loc[indices, column] = identity[column]
    result["rng_seed"] = pd.to_numeric(result["rng_seed"], errors="raise").astype(int)
    for column, expected in RESOURCE_DEPENDENCY_VERSIONS.items():
        result[column] = expected
    return result


def validate_resource_dependency_provenance(df: pd.DataFrame) -> None:
    """Require the exact circuit-library versions used by this runtime."""
    missing = sorted(set(RESOURCE_DEPENDENCY_VERSIONS) - set(df.columns))
    if missing:
        raise ValueError(
            f"SYK resource CSV lacks dependency provenance columns {missing}; "
            "rerun or postprocess collection"
        )
    for column, expected in RESOURCE_DEPENDENCY_VERSIONS.items():
        if not (df[column].astype(str) == str(expected)).all():
            raise ValueError(
                "SYK resource dependency provenance mismatch for "
                f"{column!r}: expected {expected!r}"
            )


def validate_reproducibility_provenance(df: pd.DataFrame) -> None:
    """Recompute and verify every saved seed-indexed structural fingerprint."""
    required = {
        "L", "k", "instance_idx", "n_colors", "n_quartets",
        *REPRODUCIBILITY_COLUMNS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"SYK CSV lacks reproducibility provenance columns {missing}; "
            "rerun or postprocess collection"
        )

    for (L, k, instance_idx), group in df.groupby(
        ["L", "k", "instance_idx"], sort=False
    ):
        identity = _expected_instance_identity(int(L), float(k), int(instance_idx))
        for column in (*REPRODUCIBILITY_COLUMNS, "n_colors", "n_quartets"):
            observed = group[column]
            if column in {"rng_seed", "n_colors", "n_quartets"}:
                numeric = pd.to_numeric(observed, errors="coerce")
                valid = (
                    numeric.notna().all()
                    and (numeric == np.floor(numeric)).all()
                    and (
                        numeric.astype(int) == int(identity[column])
                    ).all()
                )
            else:
                valid = (observed.astype(str) == str(identity[column])).all()
            if not valid:
                raise ValueError(
                    "SYK reproducibility provenance mismatch for "
                    f"(L={L}, k={k}, instance={instance_idx}) column {column!r}"
                )


def validate_colors_data(
    df: pd.DataFrame,
    L_values: Sequence[int],
    k_values: Sequence[float],
    n_instances: int,
) -> None:
    """Require exact colors-data key coverage and structural provenance."""
    required = {
        "L", "N", "k", "instance_idx", "n_colors", "n_quartets",
        *REPRODUCIBILITY_COLUMNS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"colors_data.csv lacks columns {missing}")
    expected_keys = {
        (int(L), float(k), int(instance_idx))
        for L in L_values
        for k in k_values
        for instance_idx in range(n_instances)
    }
    observed_keys = {
        (int(row.L), float(row.k), int(row.instance_idx))
        for row in df.itertuples(index=False)
    }
    if len(df) != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError("colors_data.csv does not have exact requested key coverage")
    if df.duplicated(["L", "k", "instance_idx"]).any():
        raise ValueError("colors_data.csv contains duplicate instance keys")
    if not (pd.to_numeric(df["N"], errors="coerce") == df["L"] ** 2).all():
        raise ValueError("colors_data.csv violates N=L^2")
    validate_reproducibility_provenance(df)


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
    reproducibility = instance_reproducibility_provenance(instance, instance_idx)

    for baseline_name, gamma in BASELINE_CONFIGS:
        if baseline_name == "naive_pauli":
            circuit, n_anc, fp_depth, int_depth = build_trotter_step_naive(instance)
        else:
            circuit, n_anc, fp_depth, int_depth = build_trotter_step_fp(
                instance, baseline_name, gamma,
            )

        # Publication rows are valid only for circuits that are physically
        # local on the declared square-grid architecture.  Keep this check in
        # the collection path so future compiler changes fail before a
        # checkpoint can be accepted or promoted.
        assert_nearest_neighbor_circuit(circuit)
        res = count_resources(circuit, L, n_anc)
        counts = {
            "total_cnots": res["total_cnots"],
            "total_idle_slots_layered": res["total_idle_slots_layered"],
            "hadamard": res["n_1q_hadamard"],
            "other_1q": res["n_1q_other"],
            "s_sdag": res["n_1q_s_sdag"],
            "pauli_z": res["n_1q_pauli_z"],
            "rz": res["n_1q_rz"],
        }

        for p_2q in p_values:
            p_idle = p_2q * p_idle_factor

            rows.append({
                "L": L,
                "N": L * L,
                "k": instance.k,
                "instance_idx": instance_idx,
                "baseline": baseline_name,
                "accounting_model": ACCOUNTING_MODEL,
                "hall_decomposition_model": _hall_model_for_baseline(
                    baseline_name
                ),
                "gamma_schedule_model": _gamma_schedule_model_for_baseline(
                    baseline_name, L
                ),
                **reproducibility,
                **RESOURCE_DEPENDENCY_VERSIONS,
                "n_colors": instance.n_colors,
                "n_quartets": len(instance.quartets),
                "n_ancillas": res["n_ancillas"],
                "total_qubits": res["total_qubits"],
                "cnot_depth": res["cnot_depth"],
                "fp_cnot_depth": fp_depth,
                "interaction_cnot_depth": int_depth,
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
                "spacetime_volume": spacetime_volume(
                    res["total_qubits"], res["cnot_depth"]
                ),
                "p_2q": p_2q,
                "p_idle": p_idle,
                # Uniform early-FT model: native CNOT-equivalent operations
                # and all remaining qubit-layers have the same logical rate.
                # The latter already cover one-qubit-active slots, so the
                # explicit one-qubit inventory must not be charged again.
                "mult_fidelity": clifford_idle_fidelity(
                    counts, "modelB", p_2q, p_idle=p_idle
                ),
                "mult_fidelity_legacy": multiplicative_fidelity(
                    res["total_2q_gates"],
                    res["total_idle_slots"],
                    p_2q=p_2q,
                    p_idle=p_idle,
                ),
            })

    return rows


def _validated_instance_resume_rows(
    existing: pd.DataFrame,
    p_values: Sequence[float],
    p_idle_factor: float = 1.0,
) -> tuple[set[tuple[int, float, int]], pd.DataFrame]:
    """Return complete CNOT-equivalent-accounting groups safe to resume.

    The checkpoint unit is ``(L, k, instance_idx)``.  It is retained only if
    it contains every baseline/noise-rate pair exactly once, records the
    current accounting-model provenance, the requested idle-noise ratio, and
    all resource/fidelity invariants used by the publication plots.  Invalid
    or interrupted instance groups are discarded in full so they can be
    recomputed safely.
    """
    required = {
        "L", "N", "k", "instance_idx", "baseline", "p_2q", "p_idle",
        "accounting_model", "hall_decomposition_model",
        "gamma_schedule_model",
        *RESOURCE_REPRODUCIBILITY_COLUMNS,
        "n_colors", "n_quartets",
        "n_ancillas", "total_qubits", "cnot_depth", "fp_cnot_depth",
        "interaction_cnot_depth",
        "total_cnots", "n_fswap", "n_2q_cnot_cz",
        "total_idle_slots_layered", "spacetime_volume",
        "n_1q_hadamard", "n_1q_other", "n_1q_s_sdag", "n_1q_rz",
        "mult_fidelity",
    }
    if not required.issubset(existing.columns):
        return set(), existing.iloc[0:0].copy()

    expected_baselines = {name for name, _ in BASELINE_CONFIGS}
    expected_p = {round(float(p), 15) for p in p_values}
    expected_rows = len(expected_baselines) * len(expected_p)

    valid_keys: set[tuple[int, float, int]] = set()
    keep_parts = []
    for (L, k, instance_idx), group in existing.groupby(
        ["L", "k", "instance_idx"], sort=False
    ):
        combos = group[["baseline", "p_2q"]].drop_duplicates()
        total_qubits = pd.to_numeric(group["total_qubits"], errors="coerce")
        N = pd.to_numeric(group["N"], errors="coerce")
        n_ancillas = pd.to_numeric(group["n_ancillas"], errors="coerce")
        cnot_depth = pd.to_numeric(group["cnot_depth"], errors="coerce")
        fp_cnot_depth = pd.to_numeric(
            group["fp_cnot_depth"], errors="coerce"
        )
        interaction_cnot_depth = pd.to_numeric(
            group["interaction_cnot_depth"], errors="coerce"
        )
        total_cnots = pd.to_numeric(group["total_cnots"], errors="coerce")
        n_fswap = pd.to_numeric(group["n_fswap"], errors="coerce")
        n_2q_cnot_cz = pd.to_numeric(
            group["n_2q_cnot_cz"], errors="coerce"
        )
        idle_slots = pd.to_numeric(
            group["total_idle_slots_layered"], errors="coerce"
        )
        volume = pd.to_numeric(group["spacetime_volume"], errors="coerce")
        p_2q = pd.to_numeric(group["p_2q"], errors="coerce")
        p_idle = pd.to_numeric(group["p_idle"], errors="coerce")
        n_h = pd.to_numeric(group["n_1q_hadamard"], errors="coerce")
        n_other = pd.to_numeric(group["n_1q_other"], errors="coerce")
        n_s = pd.to_numeric(group["n_1q_s_sdag"], errors="coerce")
        n_rz = pd.to_numeric(group["n_1q_rz"], errors="coerce")
        fidelity = pd.to_numeric(group["mult_fidelity"], errors="coerce")
        idle_expected = total_qubits * cnot_depth - 2 * total_cnots
        fidelity_expected = (
            (1.0 - p_2q) ** total_cnots
            * (1.0 - p_idle) ** idle_slots
        )
        expected_hall = group["baseline"].astype(str).map(
            _hall_model_for_baseline
        )
        observed_hall = group["hall_decomposition_model"].astype(str)
        expected_gamma_schedule = group.apply(
            lambda row: _gamma_schedule_model_for_baseline(
                str(row.baseline), int(row.L)
            ),
            axis=1,
        )
        observed_gamma_schedule = group["gamma_schedule_model"].astype(str)
        resource_columns = [
            "N", "n_ancillas", "total_qubits", "cnot_depth",
            "fp_cnot_depth", "interaction_cnot_depth", "total_cnots",
            "n_fswap", "n_2q_cnot_cz", "total_idle_slots_layered",
            "spacetime_volume", "n_1q_hadamard", "n_1q_other",
            "n_1q_s_sdag", "n_1q_rz",
        ]
        resource_counts = group.groupby("baseline", sort=False)[
            resource_columns
        ].nunique(dropna=False)
        resource_noise_invariant = (resource_counts == 1).all().all()

        try:
            L_key = int(L)
            k_key = float(k)
            instance_key = int(instance_idx)
            integral_keys = (
                float(L) == L_key and float(instance_idx) == instance_key
            )
            observed_p = {round(float(p), 15) for p in group["p_2q"]}
            identity = _expected_instance_identity(L_key, k_key, instance_key)
            identity_ok = True
            for column in (*REPRODUCIBILITY_COLUMNS, "n_colors", "n_quartets"):
                observed = group[column]
                if column in {"rng_seed", "n_colors", "n_quartets"}:
                    numeric = pd.to_numeric(observed, errors="coerce")
                    matches = (
                        numeric.notna().all()
                        and (numeric == np.floor(numeric)).all()
                        and (
                            numeric.astype(int) == int(identity[column])
                        ).all()
                    )
                else:
                    matches = (
                        observed.astype(str) == str(identity[column])
                    ).all()
                identity_ok = identity_ok and matches
            dependency_ok = all(
                (
                    group[column].astype(str)
                    == str(expected)
                ).all()
                for column, expected in RESOURCE_DEPENDENCY_VERSIONS.items()
            )
        except (TypeError, ValueError, OverflowError):
            integral_keys = False
            observed_p = set()
            identity_ok = False
            dependency_ok = False

        valid = (
            integral_keys
            and len(group) == expected_rows
            and len(combos) == expected_rows
            and set(group["baseline"].astype(str)) == expected_baselines
            and observed_p == expected_p
            and set(group["accounting_model"].astype(str)) == {ACCOUNTING_MODEL}
            and (observed_hall == expected_hall).all()
            and (observed_gamma_schedule == expected_gamma_schedule).all()
            and identity_ok
            and dependency_ok
            and resource_noise_invariant
            and not pd.concat([
                N, n_ancillas, total_qubits, cnot_depth, fp_cnot_depth,
                interaction_cnot_depth, total_cnots, n_fswap,
                n_2q_cnot_cz, idle_slots, volume, p_2q, p_idle,
                n_h, n_other, n_s, n_rz, fidelity,
            ]).isna().any()
            and (N == L_key * L_key).all()
            and (total_qubits == N + n_ancillas).all()
            and (
                cnot_depth == fp_cnot_depth + interaction_cnot_depth
            ).all()
            and (total_cnots == 2 * n_fswap + n_2q_cnot_cz).all()
            and (idle_slots == idle_expected).all()
            and (volume == total_qubits * cnot_depth).all()
            and np.isclose(
                p_idle, p_2q * float(p_idle_factor), rtol=1e-12, atol=0,
            ).all()
            and no_fault_values_match(fidelity, fidelity_expected)
        )
        if valid:
            key = (L_key, k_key, instance_key)
            valid_keys.add(key)
            keep_parts.append(group)

    if keep_parts:
        kept = pd.concat(keep_parts, ignore_index=True)
    else:
        kept = existing.iloc[0:0].copy()
    return valid_keys, kept


def _validated_resume_rows(
    existing: pd.DataFrame,
    n_instances: int,
    p_values: Sequence[float],
    p_idle_factor: float = 1.0,
) -> tuple[set[tuple[int, float]], pd.DataFrame]:
    """Return complete ``(L, k)`` groups, preserving the legacy API.

    Per-instance validation is deliberately stricter and reusable by the
    checkpointing path below.  This wrapper retains the previous all-or-none
    ``(L, k)`` behavior for callers and tests that use it directly.
    """
    _, instance_rows = _validated_instance_resume_rows(
        existing, p_values, p_idle_factor
    )
    if instance_rows.empty:
        return set(), instance_rows

    expected_instances = set(range(n_instances))
    expected_p = {round(float(p), 15) for p in p_values}
    expected_rows = n_instances * len(BASELINE_CONFIGS) * len(expected_p)
    valid_keys: set[tuple[int, float]] = set()
    keep_parts = []

    for (L, k), group in instance_rows.groupby(["L", "k"], sort=False):
        instances = set(group["instance_idx"].astype(int))
        if len(group) == expected_rows and instances == expected_instances:
            key = (int(L), float(k))
            valid_keys.add(key)
            keep_parts.append(group)

    if keep_parts:
        kept = pd.concat(keep_parts, ignore_index=True)
    else:
        kept = existing.iloc[0:0].copy()
    return valid_keys, kept


def _atomic_write_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Durably replace ``csv_path`` without exposing a partial CSV."""
    output_dir = os.path.dirname(os.path.abspath(csv_path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(csv_path)}.",
        suffix=".tmp",
        dir=output_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            df.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        # On Windows, virus scanners and readers that did not request delete
        # sharing can hold the destination briefly.  Preserve atomic replace
        # semantics while tolerating that transient lock.
        for attempt in range(20):
            try:
                os.replace(tmp_path, csv_path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def run_experiment(
    L_values: Sequence[int] = PUBLICATION_L_VALUES,
    k_values: Sequence[float] = (1.0,),
    n_instances: int = 10,
    p_values: Sequence[float] = (1e-3, 1e-4, 1e-5),
    p_idle_factor: float = 1.0,
    output_dir: str = "exp3_syk/results",
) -> pd.DataFrame:
    """Run the full Experiment 3 sweep."""
    requested_L_values = tuple(dict.fromkeys(int(L) for L in L_values))
    requested_k_values = tuple(dict.fromkeys(float(k) for k in k_values))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")
    all_rows: List[Dict] = []

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_instances, existing = _validated_instance_resume_rows(
            existing, p_values, p_idle_factor
        )
        requested_instances = set(range(n_instances))
        done_instances = {
            key for key in done_instances
            if key[0] in requested_L_values
            and key[1] in requested_k_values
            and key[2] in requested_instances
        }
        if not existing.empty:
            existing = existing[
                existing["L"].astype(int).isin(requested_L_values)
                & existing["k"].astype(float).isin(requested_k_values)
                & existing["instance_idx"].astype(int).isin(requested_instances)
            ].copy()
        _log(
            "Found complete CNOT-equivalent-accounting data for instances = "
            f"{sorted(done_instances)}, will skip these."
        )
        all_rows = existing.to_dict("records")
    else:
        done_instances = set()

    for L in requested_L_values:
        for k in requested_k_values:
            L_key = int(L)
            k_key = float(k)
            pending_instances = [
                idx for idx in range(n_instances)
                if (L_key, k_key, idx) not in done_instances
            ]
            if not pending_instances:
                continue

            t0 = time.time()
            _log(f"\n{'='*60}")
            _log(f"L = {L}  (N = {L*L}), k = {k}")
            _log(f"{'='*60}")

            for idx in range(n_instances):
                instance_key = (L_key, k_key, idx)
                if instance_key in done_instances:
                    continue

                rng = np.random.default_rng(seed=idx)
                _log(f"  instance {idx} ...", end=" ")
                t1 = time.time()

                instance = generate_sparse_syk(L, k, rng)
                rows = collect_instance(instance, idx, p_values, p_idle_factor)
                new_keys, valid_rows = _validated_instance_resume_rows(
                    pd.DataFrame(rows), p_values, p_idle_factor
                )
                if new_keys != {instance_key} or len(valid_rows) != len(rows):
                    raise RuntimeError(
                        "collector produced an incomplete or invalid native-"
                        f"accounting instance group for {instance_key}"
                    )

                all_rows.extend(valid_rows.to_dict("records"))
                done_instances.add(instance_key)
                checkpoint = pd.DataFrame(all_rows)
                _atomic_write_csv(checkpoint, csv_path)
                _log(
                    f"done ({time.time()-t1:.1f}s, "
                    f"{len(instance.quartets)} terms, "
                    f"{instance.n_colors} colors); checkpointed "
                    f"{len(all_rows)} rows"
                )

            elapsed = time.time() - t0
            _log(f"  L={L}, k={k} total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    df = pd.DataFrame(all_rows)
    _log(f"\nExperiment complete. {len(df)} total rows.")
    return df


def collect_colors_data(
    L_values: Sequence[int] = PUBLICATION_L_VALUES,
    k_values: Sequence[float] = COLORS_K_VALUES,
    n_instances: int = 10,
    output_dir: str = "exp3_syk/results",
) -> pd.DataFrame:
    """Lightweight collection of n_colors for multiple k values.

    Only generates SYK instances and records coloring statistics —
    no circuit building.  Fast even for large L.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "colors_data.csv")
    rows: List[Dict] = []

    for L in L_values:
        for k in k_values:
            for idx in range(n_instances):
                rng = np.random.default_rng(seed=idx)
                instance = generate_sparse_syk(L, k, rng)
                reproducibility = instance_reproducibility_provenance(
                    instance, idx
                )
                rows.append({
                    "L": L,
                    "N": L * L,
                    "k": k,
                    "instance_idx": idx,
                    "n_colors": instance.n_colors,
                    "n_quartets": len(instance.quartets),
                    **reproducibility,
                })

    df = pd.DataFrame(rows)
    validate_colors_data(df, L_values, k_values, n_instances)
    _atomic_write_csv(df, csv_path)
    _log(f"Colors data: {len(df)} rows saved to {csv_path}")
    return df
