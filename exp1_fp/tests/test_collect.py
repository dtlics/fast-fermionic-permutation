"""Regression tests for Experiment 1 CNOT-equivalent accounting and resume."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from common.fp_1d import build_benchmark_permutation
from common.gamma_folded import (
    GAMMA_SCHEDULE_NOT_APPLICABLE,
    folded_gamma_schedule_model,
)
from exp1_fp.collect import (
    ACCOUNTING_MODEL,
    BASELINE_CONFIGS,
    HALL_NOT_APPLICABLE,
    SAMPLING_MODEL,
    _gamma_schedule_model_for_baseline,
    _hall_model_for_baseline,
    _stim_seed_for_row,
    _validated_resume_rows,
    add_reproducibility_provenance,
    collect_instance,
    run_experiment,
)


def _complete_group(p_values=(1e-3, 1e-4), shots=1000):
    rows = []
    permutations = [("reverse", 0), ("transpose", 0), ("random", 0)]
    for perm_kind, perm_idx in permutations:
        for baseline, _ in BASELINE_CONFIGS:
            for p_2q in p_values:
                rows.append({
                    "L": 4,
                    "N": 16,
                    "perm_kind": perm_kind,
                    "perm_idx": perm_idx,
                    "baseline": baseline,
                    "p_2q": p_2q,
                    "p_idle": p_2q,
                    "accounting_model": ACCOUNTING_MODEL,
                    "hall_decomposition_model": _hall_model_for_baseline(baseline),
                    "gamma_schedule_model": _gamma_schedule_model_for_baseline(
                        baseline, 4
                    ),
                    "sampling_model": SAMPLING_MODEL,
                    "stim_seed": _stim_seed_for_row(
                        4, perm_kind, perm_idx, baseline, p_2q, p_2q
                    ),
                    "total_qubits": 16,
                    "cnot_depth": 20,
                    "total_cnots": 100,
                    "total_idle_slots_layered": 120,
                    "stim_fidelity": 0.5,
                    "stim_fidelity_ci_lo": 0.5,
                    "stim_fidelity_ci_hi": 0.5,
                    "stim_shots": shots,
                })
    return add_reproducibility_provenance(pd.DataFrame(rows))


def test_collected_rows_use_folded_and_native_fswap_accounting():
    L = 2
    perm = build_benchmark_permutation(L, "reverse")
    rows = collect_instance(
        L, perm, "reverse", 0, p_values=(1e-3,),
        p_idle_factor=1.0, shots=0,
    )

    assert {row["baseline"] for row in rows} == {
        name for name, _ in BASELINE_CONFIGS
    }
    assert any(row["baseline"] == "folded" for row in rows)
    for row in rows:
        assert row["accounting_model"] == ACCOUNTING_MODEL
        assert row["sampling_model"] == SAMPLING_MODEL
        assert row["hall_decomposition_model"] == (
            HALL_NOT_APPLICABLE if row["baseline"] == "1d"
            else _hall_model_for_baseline(row["baseline"])
        )
        assert row["gamma_schedule_model"] == (
            folded_gamma_schedule_model(L)
            if row["baseline"] == "folded"
            else GAMMA_SCHEDULE_NOT_APPLICABLE
        )
        assert row["stim_seed"] == _stim_seed_for_row(
            row["L"], row["perm_kind"], row["perm_idx"], row["baseline"],
            row["p_2q"], row["p_idle"],
        )
        assert row["total_cnots"] == (
            2 * row["n_fswap"] + row["n_2q_cnot_cz"]
        )
        assert row["total_idle_slots_layered"] == (
            row["total_qubits"] * row["cnot_depth"]
            - 2 * row["total_cnots"]
        )
        expected = (
            (1 - row["p_2q"]) ** row["total_cnots"]
            * (1 - row["p_idle"]) ** row["total_idle_slots_layered"]
        )
        assert row["mult_fidelity"] == pytest.approx(expected)


def test_resume_accepts_only_complete_matching_native_group():
    df = _complete_group()
    done, kept = _validated_resume_rows(
        df, ("reverse", "transpose", "random"), 1,
        (1e-3, 1e-4), 1.0, 1000,
    )
    assert done == {4}
    assert len(kept) == len(df)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.drop(columns=["accounting_model"]),
        lambda df: df.drop(columns=["hall_decomposition_model"]),
        lambda df: df.drop(columns=["gamma_schedule_model"]),
        lambda df: df.drop(columns=["sampling_model"]),
        lambda df: df.drop(columns=["permutation_sha256"]),
        lambda df: df.iloc[:-1].copy(),
        lambda df: df.assign(accounting_model="legacy"),
        lambda df: df.assign(hall_decomposition_model="legacy"),
        lambda df: df.assign(gamma_schedule_model="legacy"),
        lambda df: df.assign(sampling_model="unseeded"),
        lambda df: df.assign(stim_seed=df.stim_seed + 1),
        lambda df: df.assign(permutation_model="legacy"),
        lambda df: df.assign(permutation_seed=999),
        lambda df: df.assign(permutation_sha256="0" * 64),
        lambda df: df.assign(stim_version="0.0"),
        lambda df: df.assign(p_idle=df.p_idle * 2),
        lambda df: df.assign(N=25),
        lambda df: df.assign(stim_shots=999),
    ],
)
def test_resume_rejects_incompatible_or_incomplete_group(mutate):
    df = mutate(_complete_group())
    done, kept = _validated_resume_rows(
        df, ("reverse", "transpose", "random"), 1,
        (1e-3, 1e-4), 1.0, 1000,
    )
    assert not done
    assert kept.empty


def test_resume_rejects_broken_idle_invariant_or_missing_stim_result():
    for column in ("total_idle_slots_layered", "stim_fidelity"):
        df = _complete_group()
        if column == "total_idle_slots_layered":
            df.loc[0, column] += 1
        else:
            df.loc[0, column] = np.nan
        done, kept = _validated_resume_rows(
            df, ("reverse", "transpose", "random"), 1,
            (1e-3, 1e-4), 1.0, 1000,
        )
        assert not done
        assert kept.empty


def test_parallel_collection_matches_serial_and_checkpoints_atomically(tmp_path):
    common = dict(
        L_values=(2,),
        perm_kinds=("reverse", "random"),
        n_random=2,
        p_values=(1e-3, 1e-4),
        p_idle_factor=1.0,
        shots=0,
    )
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = run_experiment(**common, output_dir=str(serial_dir), workers=1)
    parallel = run_experiment(**common, output_dir=str(parallel_dir), workers=2)

    assert_frame_equal(serial, parallel, check_exact=True)
    assert (serial_dir / "data.csv").is_file()
    assert (parallel_dir / "data.csv").is_file()
    assert not (serial_dir / "data.csv.tmp").exists()
    assert not (parallel_dir / "data.csv.tmp").exists()


def test_seeded_stim_collection_matches_across_worker_schedules(tmp_path):
    common = dict(
        L_values=(2,),
        perm_kinds=("reverse", "random"),
        n_random=1,
        p_values=(1e-2,),
        p_idle_factor=1.0,
        shots=256,
    )
    serial = run_experiment(
        **common, output_dir=str(tmp_path / "seeded_serial"), workers=1
    )
    parallel = run_experiment(
        **common, output_dir=str(tmp_path / "seeded_parallel"), workers=2
    )

    assert_frame_equal(serial, parallel, check_exact=True)


def test_stim_seed_is_repeatable_and_uses_every_row_key_field():
    base = dict(
        L=4, perm_kind="random", perm_idx=0, baseline="folded",
        p_2q=1e-3, p_idle=1e-4,
    )
    seed = _stim_seed_for_row(**base)
    assert seed == _stim_seed_for_row(**base)
    assert 0 <= seed < 2**63

    variants = [
        {**base, "L": 6},
        {**base, "perm_kind": "reverse"},
        {**base, "perm_idx": 1},
        {**base, "baseline": "ancilla"},
        {**base, "p_2q": 1e-4},
        {**base, "p_idle": 1e-5},
    ]
    assert all(_stim_seed_for_row(**variant) != seed for variant in variants)


def test_stim_seed_includes_folded_gamma_schedule_model(monkeypatch):
    base = dict(
        L=4, perm_kind="random", perm_idx=0, baseline="folded",
        p_2q=1e-3, p_idle=1e-4,
    )
    original = _stim_seed_for_row(**base)
    monkeypatch.setattr(
        "exp1_fp.collect.folded_gamma_schedule_model",
        lambda L: "stale_folded_schedule",
    )
    assert _stim_seed_for_row(**base) != original
