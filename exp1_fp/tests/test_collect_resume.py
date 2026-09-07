"""Regression tests for Experiment 1 CSV resume provenance."""

import numpy as np
import pandas as pd

from exp1_fp.collect import (
    ACCOUNTING_MODEL,
    BASELINE_CONFIGS,
    SAMPLING_MODEL,
    _gamma_schedule_model_for_baseline,
    _hall_model_for_baseline,
    _stim_seed_for_row,
    _validated_resume_rows,
    add_reproducibility_provenance,
)


P_VALUES = (1e-3, 1e-4)
PERM_KINDS = ("reverse", "transpose", "random")
N_RANDOM = 2
SHOTS = 1_000
P_IDLE_FACTOR = 1.0


def _complete_group(L=4):
    permutations = [("reverse", 0), ("transpose", 0)]
    permutations.extend(("random", idx) for idx in range(N_RANDOM))

    rows = []
    for perm_kind, perm_idx in permutations:
        for baseline, _ in BASELINE_CONFIGS:
            for p_2q in P_VALUES:
                rows.append({
                    "L": L,
                    "N": L * L,
                    "perm_kind": perm_kind,
                    "perm_idx": perm_idx,
                    "baseline": baseline,
                    "p_2q": p_2q,
                    "p_idle": p_2q * P_IDLE_FACTOR,
                    "accounting_model": ACCOUNTING_MODEL,
                    "hall_decomposition_model": _hall_model_for_baseline(baseline),
                    "gamma_schedule_model": _gamma_schedule_model_for_baseline(
                        baseline, L
                    ),
                    "sampling_model": SAMPLING_MODEL,
                    "stim_seed": _stim_seed_for_row(
                        L, perm_kind, perm_idx, baseline, p_2q,
                        p_2q * P_IDLE_FACTOR,
                    ),
                    "total_qubits": L * L,
                    "cnot_depth": 20,
                    "total_cnots": 100,
                    "total_idle_slots_layered": L * L * 20 - 200,
                    "stim_fidelity": 0.9,
                    "stim_fidelity_ci_lo": 0.9,
                    "stim_fidelity_ci_hi": 0.9,
                    "stim_shots": SHOTS,
                })
    return add_reproducibility_provenance(pd.DataFrame(rows))


def _validate(df, shots=SHOTS):
    return _validated_resume_rows(
        df,
        perm_kinds=PERM_KINDS,
        n_random=N_RANDOM,
        p_values=P_VALUES,
        p_idle_factor=P_IDLE_FACTOR,
        shots=shots,
    )


def test_resume_accepts_complete_native_group():
    df = _complete_group()

    grid_sizes, kept = _validate(df)

    assert grid_sizes == {4}
    pd.testing.assert_frame_equal(kept, df)


def test_resume_keeps_valid_groups_and_discards_incomplete_groups():
    complete = _complete_group(L=4)
    incomplete = _complete_group(L=6).iloc[:-1]

    grid_sizes, kept = _validate(pd.concat([complete, incomplete]))

    assert grid_sizes == {4}
    assert set(kept["L"]) == {4}
    assert len(kept) == len(complete)


def test_resume_invalidates_only_changed_boundary_schedule_rows():
    """An old boundary tag is stale without invalidating unchanged L>=7 data."""
    from common.gamma_folded import FOLDED_GAMMA_SCHEDULE_MODEL

    boundary = _complete_group(L=6)
    asymptotic = _complete_group(L=8)
    boundary.loc[
        boundary["baseline"] == "folded", "gamma_schedule_model"
    ] = FOLDED_GAMMA_SCHEDULE_MODEL

    grid_sizes, kept = _validate(pd.concat([boundary, asymptotic]))

    assert grid_sizes == {8}
    assert set(kept["L"]) == {8}
    assert len(kept) == len(asymptotic)


def test_resume_rejects_legacy_or_broken_idle_accounting():
    legacy = _complete_group().drop(columns=["accounting_model"])
    grid_sizes, kept = _validate(legacy)
    assert not grid_sizes
    assert kept.empty

    stale_hall = _complete_group()
    stale_hall["hall_decomposition_model"] = "networkx_unseeded"
    grid_sizes, kept = _validate(stale_hall)
    assert not grid_sizes
    assert kept.empty

    stale_sampling = _complete_group()
    stale_sampling["sampling_model"] = "stim_unseeded"
    grid_sizes, kept = _validate(stale_sampling)
    assert not grid_sizes
    assert kept.empty

    missing_gamma_schedule = _complete_group().drop(
        columns=["gamma_schedule_model"]
    )
    grid_sizes, kept = _validate(missing_gamma_schedule)
    assert not grid_sizes
    assert kept.empty

    stale_gamma_schedule = _complete_group()
    stale_gamma_schedule.loc[
        stale_gamma_schedule["baseline"] == "folded", "gamma_schedule_model"
    ] = "folded_stale"
    grid_sizes, kept = _validate(stale_gamma_schedule)
    assert not grid_sizes
    assert kept.empty

    wrong_seed = _complete_group()
    wrong_seed.loc[0, "stim_seed"] += 1
    grid_sizes, kept = _validate(wrong_seed)
    assert not grid_sizes
    assert kept.empty

    wrong_permutation = _complete_group()
    wrong_permutation.loc[0, "permutation_sha256"] = "0" * 64
    grid_sizes, kept = _validate(wrong_permutation)
    assert not grid_sizes
    assert kept.empty

    wrong_version = _complete_group()
    wrong_version["numpy_version"] = "0.0"
    grid_sizes, kept = _validate(wrong_version)
    assert not grid_sizes
    assert kept.empty

    broken = _complete_group()
    broken.loc[0, "total_idle_slots_layered"] += 1
    grid_sizes, kept = _validate(broken)
    assert not grid_sizes
    assert kept.empty


def test_resume_rejects_wrong_stim_provenance():
    wrong_shots = _complete_group()
    wrong_shots["stim_shots"] = SHOTS + 1
    grid_sizes, kept = _validate(wrong_shots)
    assert not grid_sizes
    assert kept.empty

    missing_fidelity = _complete_group()
    missing_fidelity.loc[0, "stim_fidelity"] = np.nan
    grid_sizes, kept = _validate(missing_fidelity)
    assert not grid_sizes
    assert kept.empty


def test_resume_allows_missing_stim_fidelity_when_simulation_is_disabled():
    df = _complete_group()
    df["stim_shots"] = 0
    df["stim_fidelity"] = np.nan

    grid_sizes, kept = _validate(df, shots=0)

    assert grid_sizes == {4}
    assert len(kept) == len(df)
